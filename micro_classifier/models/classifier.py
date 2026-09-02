"""Microorganism classifier model based on EfficientNet-B4 with transfer learning."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Tuple

from ..utils.device import get_device


class MicroClassifier(nn.Module):
    """
    Professional microorganism classifier using EfficientNet-B4 backbone.

    Architecture:
        - EfficientNet-B4 pretrained on ImageNet (frozen or fine-tunable)
        - Global Average Pooling
        - Batch Normalization
        - Custom classification head with dropout regularization
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)

        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self._freeze_backbone(freeze_backbone)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _freeze_backbone(self, freeze: bool):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """Unfreeze backbone layers for fine-tuning."""
        layers = list(self.backbone.features)
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_model_info(self) -> Dict:
        total_params = self.get_num_params(trainable_only=False)
        trainable_params = self.get_num_params(trainable_only=True)
        return {
            "architecture": "EfficientNet-B4",
            "total_params": f"{total_params:,}",
            "trainable_params": f"{trainable_params:,}",
            "frozen_params": f"{total_params - trainable_params:,}",
            "num_classes": self.num_classes,
        }


def create_model(
    num_classes: int = 8,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = True,
    device: Optional[torch.device] = None,
) -> MicroClassifier:
    """Factory function to create a MicroClassifier model."""
    if device is None:
        device = get_device()

    model = MicroClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
    model = model.to(device)
    return model


def load_checkpoint(
    checkpoint_path: str,
    num_classes: int = 8,
    device: Optional[torch.device] = None,
) -> Tuple[MicroClassifier, Dict]:
    """Load a model from a checkpoint file."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = MicroClassifier(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "best_acc": checkpoint.get("best_acc", 0.0),
        "class_to_idx": checkpoint.get("class_to_idx", {}),
        "idx_to_class": checkpoint.get("idx_to_class", {}),
    }
    return model, metadata
