"""Microorganism classifier model based on EfficientNet-B5 backbone via timm."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from ..utils.device import get_device


class MicroClassifier(nn.Module):
    """
    EfficientNet-B5 classifier with improved classification head.

    Architecture:
        - EfficientNet-B5 pretrained on ImageNet-1K (frozen or fine-tunable)
        - Global Average Pooling + Global Max Pooling (concatenated)
        - Batch Normalization
        - Multi-layer classification head with dropout
    """

    def __init__(
        self,
        num_classes: int = 16,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = "tf_efficientnet_b5"

        import timm
        self.backbone = timm.create_model(
            self.backbone_name, pretrained=pretrained, num_classes=0
        )
        num_features = self.backbone.num_features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self._freeze_backbone(freeze_backbone)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, num_classes),
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
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            features = list(self.backbone.children())
            for child in features[-num_layers:]:
                for param in child.parameters():
                    param.requires_grad = True

    def freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        avg_feat = self.pool(features)
        max_feat = self.maxpool(features)
        features = torch.cat([avg_feat, max_feat], dim=1)
        return self.classifier(features)

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_model_info(self) -> Dict:
        total_params = self.get_num_params(trainable_only=False)
        trainable_params = self.get_num_params(trainable_only=True)
        return {
            "architecture": "EfficientNet-B5",
            "total_params": f"{total_params:,}",
            "trainable_params": f"{trainable_params:,}",
            "frozen_params": f"{total_params - trainable_params:,}",
            "num_classes": self.num_classes,
        }


def create_model(
    num_classes: int = 16,
    pretrained: bool = True,
    dropout_rate: float = 0.4,
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
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[MicroClassifier, Dict]:
    """Load a model from a checkpoint file."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    idx_to_class = checkpoint.get("idx_to_class", {})
    if num_classes is None:
        num_classes = len(idx_to_class) if idx_to_class else 16

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
        "idx_to_class": idx_to_class,
    }
    return model, metadata
