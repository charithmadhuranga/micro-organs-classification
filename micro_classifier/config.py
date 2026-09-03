"""Configuration module for MicroClassify."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "efficientnet_b4"
    num_classes: int = 16
    input_size: int = 380
    pretrained: bool = True
    dropout_rate: float = 0.3
    freeze_backbone: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    early_stopping_patience: int = 7
    mixed_precision: bool = True
    num_workers: int = 4
    validation_split: float = 0.15
    save_dir: str = "trained_models"
    checkpoint_name: str = "best_microclassifier.pth"


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    input_size: int = 380
    rotation: int = 20
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.2
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    random_erasing_prob: float = 0.2
    normalize_mean: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    normalize_std: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )


@dataclass
class StreamConfig:
    """Video stream configuration."""
    source_type: str = "file"  # file, webcam, rtsp, rtmp
    source_path: str = ""
    fps: int = 30
    buffer_size: int = 3
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0
    inference_width: int = 380
    inference_height: int = 380


@dataclass
class GUIConfig:
    """GUI configuration."""
    title: str = "MicroClassify - Microorganism Classification System"
    window_width: int = 1400
    window_height: int = 850
    primary_color: str = "#1a2332"
    secondary_color: str = "#2d3748"
    accent_color: str = "#00b4d8"
    success_color: str = "#2ecc71"
    warning_color: str = "#f39c12"
    danger_color: str = "#e74c3c"
    text_color: str = "#ecf0f1"
    font_size: int = 14
    title_font_size: int = 20


CLASSES: List[str] = [
    "Amoeba",
    "Chlamydomonas",
    "Diatom",
    "Euglena",
    "Hydra",
    "Nematode",
    "Paramecium",
    "Penicillium",
    "Rod_bacteria",
    "Rotifer",
    "Spherical_bacteria",
    "Spiral_bacteria",
    "Spirogyra",
    "Stentor",
    "Volvox",
    "Yeast",
]

CLASS_DESCRIPTIONS = {
    "Amoeba": "Single-celled protozoan that moves using pseudopods",
    "Chlamydomonas": "Single-cell green algae with two flagella",
    "Diatom": "Photosynthetic algae with silica cell walls",
    "Euglena": "Flagellated protozoan with both plant and animal traits",
    "Hydra": "Freshwater polyp of the phylum Cnidaria",
    "Nematode": "Unsegmented roundworms (microscopic species)",
    "Paramecium": "Slipper-shaped ciliated protozoan",
    "Penicillium": "Saprophytic fungus (mold) with brush-like conidiophores",
    "Rod_bacteria": "Rod-shaped bacteria (Bacillus, E. coli, etc.)",
    "Rotifer": "Microscopic multicellular animals with ciliated corona",
    "Spherical_bacteria": "Spherical bacteria (Staphylococcus, Streptococcus)",
    "Spiral_bacteria": "Spiral-shaped bacteria (Spirillum, Spirochete)",
    "Spirogyra": "Filamentous green algae with spiral chloroplasts",
    "Stentor": "Large trumpet-shaped ciliated protist",
    "Volvox": "Colonial green algae forming spherical colonies",
    "Yeast": "Single-celled eukaryotic fungus",
}


def get_dataset_classes(dataset_path: str) -> List[str]:
    """Auto-discover class folders from the dataset directory."""
    if not os.path.isdir(dataset_path):
        return CLASSES
    classes = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
        and not d.startswith(".")
    ])
    return classes if classes else CLASSES


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "trained_models", "best_microclassifier.pth")
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
