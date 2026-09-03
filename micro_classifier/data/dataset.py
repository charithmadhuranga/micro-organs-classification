"""Dataset and data loading utilities for microorganism classification."""

import os
import logging
from typing import Tuple, Optional, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

logger = logging.getLogger(__name__)

_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')


def _is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


class RobustImageFolder(ImageFolder):
    """ImageFolder that silently skips corrupt or unreadable images."""

    def __init__(self, root, transform=None, **kwargs):
        self._original_root = root
        self._valid_samples = None
        super().__init__(root, transform=transform, **kwargs)
        self._filter_samples()

    def _filter_samples(self):
        bad = []
        for idx, (path, _) in enumerate(self.samples):
            if not _is_valid_image(path):
                bad.append(idx)
        if bad:
            logger.warning("Skipping %d corrupt images out of %d total",
                           len(bad), len(self.samples))
            self.samples = [s for i, s in enumerate(self.samples) if i not in set(bad)]
            self.targets = [t for i, t in enumerate(self.targets) if i not in set(bad)]

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            if index + 1 < len(self):
                return super().__getitem__(index + 1)
            return super().__getitem__(0)


def get_training_transforms(cfg, use_advanced: bool = True) -> transforms.Compose:
    """Get training data augmentation pipeline."""
    from .augmentation import get_training_transforms as _get_adv, get_validation_transforms as _get_val
    if use_advanced:
        return _get_adv(cfg, use_advanced=True)
    return _get_val(cfg)


def get_validation_transforms(cfg) -> transforms.Compose:
    """Get validation/test inference transforms (deterministic)."""
    from .augmentation import get_validation_transforms as _get_val
    return _get_val(cfg)


class MicroorganismDataset:
    """Data manager for the microorganism image classification dataset."""

    def __init__(
        self,
        dataset_path: str,
        model_config,
        augment_config,
        training_config,
    ):
        self.dataset_path = dataset_path
        self.model_config = model_config
        self.augment_config = augment_config
        self.training_config = training_config
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, int] = {}
        self._class_names: List[str] = []

    def _discover_classes(self) -> List[str]:
        """Auto-discover class folders from dataset directory."""
        classes = sorted([
            d for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
            and not d.startswith(".")
        ])
        self._class_names = classes
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        return classes

    def get_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """
        Create training and validation DataLoaders.

        Returns:
            train_loader, val_loader, class_to_idx mapping
        """
        classes = self._discover_classes()
        num_classes = len(classes)

        self.model_config.num_classes = num_classes

        train_transforms = get_training_transforms(self.augment_config)
        val_transforms = get_validation_transforms(self.augment_config)

        full_dataset = RobustImageFolder(root=self.dataset_path, transform=train_transforms)

        total_size = len(full_dataset)
        val_size = int(total_size * self.training_config.validation_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        val_dataset.dataset = RobustImageFolder(
            root=self.dataset_path, transform=val_transforms
        )

        use_pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers,
            pin_memory=use_pin_memory,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers,
            pin_memory=use_pin_memory,
        )

        class_counts = {}
        for cls_name in classes:
            cls_path = os.path.join(self.dataset_path, cls_name)
            class_counts[cls_name] = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith(_EXTENSIONS)
            ])

        stats = {
            "num_classes": num_classes,
            "total_images": total_size,
            "train_images": train_size,
            "val_images": val_size,
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "class_counts": class_counts,
            "class_names": classes,
        }

        return train_loader, val_loader, stats

    def get_inference_transform(self) -> transforms.Compose:
        """Get transforms for single-image inference."""
        return get_validation_transforms(self.augment_config)

    def get_class_names(self) -> List[str]:
        return self._class_names
