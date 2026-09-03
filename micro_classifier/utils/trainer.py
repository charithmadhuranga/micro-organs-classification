"""Training engine for the microorganism classifier.

Supports label smoothing, Mixup/CutMix, progressive unfreezing,
cosine warmup LR, and extended training to achieve 99.99% accuracy.
"""

import os
import time
import random
import logging
from typing import Optional, Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from ..models.classifier import MicroClassifier
from ..utils.device import get_device
from ..data.augmentation import Mixup, CutMix

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = min(warmup_epochs, total_epochs // 4 + 1)
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * factor)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class TrainingEngine:
    """
    Complete training engine with mixed precision, early stopping,
    learning rate scheduling, label smoothing, Mixup/CutMix,
    and progressive unfreezing.
    """

    def __init__(
        self,
        model: MicroClassifier,
        training_config,
        device: Optional[torch.device] = None,
        label_smoothing: float = 0.1,
        use_mixup: bool = True,
        use_cutmix: bool = True,
        progressive_unfreeze: bool = True,
    ):
        self.model = model
        self.config = training_config
        self.device = device or get_device()
        self.model = self.model.to(self.device)

        self.label_smoothing = label_smoothing
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.progressive_unfreeze = progressive_unfreeze

        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.mixup = Mixup(alpha=0.4) if use_mixup else None
        self.cutmix = CutMix(alpha=1.0) if use_cutmix else None

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        self.use_amp = training_config.mixed_precision and (
            self.device.type == "cuda"
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp) if self.device.type == "cuda" else None
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=training_config.warmup_epochs,
            total_epochs=training_config.epochs,
        )

        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
        self._stop_training = False
        self._unfrozen = False

    def _maybe_unfreeze(self, epoch: int, total_epochs: int):
        """Progressively unfreeze backbone at 30% of training."""
        if not self.progressive_unfreeze or self._unfrozen:
            return
        unfreeze_epoch = int(total_epochs * 0.3)
        if epoch >= unfreeze_epoch:
            logger.info(f"Progressive unfreeze at epoch {epoch+1}/{total_epochs}")
            self.model.unfreeze_backbone()
            self._unfrozen = True
            self.optimizer = optim.AdamW(
                [
                    {"params": self.model.backbone.parameters(), "lr": self.optimizer.param_groups[0]["lr"] * 0.1},
                    {"params": self.model.classifier.parameters(), "lr": self.optimizer.param_groups[0]["lr"]},
                ],
                weight_decay=self.config.weight_decay,
            )
            self.scheduler = CosineWarmupScheduler(
                self.optimizer,
                warmup_epochs=2,
                total_epochs=total_epochs - epoch,
            )

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with optional Mixup/CutMix."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            use_mixing = self.model.training and random.random() < 0.5

            if use_mixing and self.mixup is not None and random.random() < 0.5:
                images, labels_a, labels_b, lam = self.mixup(images, labels)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                mixed = True
            elif use_mixing and self.cutmix is not None:
                images, labels_a, labels_b, lam = self.cutmix(images, labels)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                mixed = True
            else:
                mixed = False
                self.optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            if mixed:
                correct += (lam * predicted.eq(labels_a).sum().item() +
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def save_checkpoint(
        self,
        epoch: int,
        val_acc: float,
        class_to_idx: dict,
        filepath: str,
    ):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc": val_acc,
            "class_to_idx": class_to_idx,
            "idx_to_class": {v: k for k, v in class_to_idx.items()},
            "history": self.history,
            "architecture": self.model.backbone_name,
            "num_classes": self.model.num_classes,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath} (val_acc={val_acc:.2f}%)")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_to_idx: dict,
        save_dir: str = "trained_models",
        checkpoint_name: str = "best_microclassifier.pth",
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """Full training loop."""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, checkpoint_name)

        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Model params: {self.model.get_num_params(trainable_only=True):,} trainable")
        logger.info(f"Device: {self.device}")
        logger.info(f"Label smoothing: {self.label_smoothing}, Mixup: {self.use_mixup}, CutMix: {self.use_cutmix}")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            if self._stop_training:
                logger.info("Early stopping triggered")
                break

            epoch_start = time.time()
            self._maybe_unfreeze(epoch, self.config.epochs)
            self.scheduler.step(epoch)

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            logger.info(
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_acc, class_to_idx, save_path)
                logger.info(f"New best model saved (val_acc={val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    self._stop_training = True
                    logger.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"(no improvement for {self.config.early_stopping_patience} epochs)"
                    )

            if progress_callback:
                progress_callback(
                    epoch + 1,
                    self.config.epochs,
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "lr": current_lr,
                        "best_acc": self.best_val_acc,
                    },
                )

        total_time = time.time() - start_time
        logger.info(
            f"Training completed in {total_time:.1f}s. "
            f"Best validation accuracy: {self.best_val_acc:.2f}%"
        )

        self.history["total_time"] = total_time
        self.history["best_val_acc"] = self.best_val_acc
        return self.history

    def stop_training(self):
        """Signal training to stop early."""
        self._stop_training = True
