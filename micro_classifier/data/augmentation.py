"""Advanced data augmentation for microorganism classification.

Includes RandAugment, Mixup, CutMix, and aggressive training transforms
to push validation accuracy toward 99.99%.
"""

import random
from typing import Tuple

import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image, ImageFilter, ImageEnhance


class Mixup:
    """Mixup: mix two samples and their labels."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class CutMix:
    """CutMix: cut a patch from one image and paste to another."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _rand_bbox(self, size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        return x1, y1, x2, y2

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(x.size(0), device=x.device)
        x1, y1, x2, y2 = self._rand_bbox(x.size(), lam)
        x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
        return x, y, y[index], lam


class GridMask:
    """GridMask: randomly drop grid regions of the image."""

    def __init__(self, d: int = 96, r: float = 0.6, prob: float = 0.3):
        self.d = d
        self.r = r
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        w, h = img.size
        mask = Image.new("L", (w, h), 255)
        draw = np.array(mask)
        for y in range(0, h, self.d):
            for x in range(0, w, self.d):
                if random.random() < self.r:
                    draw[y:y + self.d // 2, x:x + self.d // 2] = 0
        mask = Image.fromarray(draw)
        return Image.composite(img, Image.new("RGB", (w, h), 0), mask)


def get_training_transforms(cfg, use_advanced: bool = True) -> transforms.Compose:
    """Get training data augmentation pipeline.

    Args:
        cfg: AugmentationConfig
        use_advanced: Enable RandAugment + extra augmentations
    """
    img_size = cfg.input_size
    if use_advanced:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.RandAugment(num_ops=3, magnitude=9),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.33)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=cfg.horizontal_flip),
        transforms.RandomVerticalFlip(p=cfg.vertical_flip),
        transforms.RandomRotation(cfg.rotation),
        transforms.ColorJitter(
            brightness=cfg.color_jitter_brightness,
            contrast=cfg.color_jitter_contrast,
            saturation=cfg.color_jitter_saturation,
            hue=cfg.color_jitter_hue,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
        transforms.RandomErasing(p=cfg.random_erasing_prob),
    ])


def get_validation_transforms(cfg) -> transforms.Compose:
    """Get validation/test inference transforms (deterministic)."""
    return transforms.Compose([
        transforms.Resize((cfg.input_size, cfg.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
    ])


def get_test_time_augmentation_transforms(cfg, num_views: int = 5) -> transforms.Compose:
    """TTA: horizontal flip + slight scale variations."""
    return transforms.Compose([
        transforms.Resize((cfg.input_size + 16, cfg.input_size + 16)),
        transforms.CenterCrop(cfg.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
    ])
