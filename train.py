"""Standalone training script for the microorganism classifier."""

import os
import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Microorganism Classifier (EfficientNet-B4)"
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset",
        help="Path to dataset directory",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--input-size", type=int, default=380, help="Input image size")
    parser.add_argument("--save-dir", type=str, default="trained_models", help="Checkpoint save directory")
    parser.add_argument("--checkpoint", type=str, default="best_microclassifier.pth", help="Checkpoint filename")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights")
    parser.add_argument("--no-freeze", action="store_true", help="Do not freeze backbone initially")
    parser.add_argument("--finetune-layers", type=int, default=None, help="Unfreeze last N layers for fine-tuning")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from micro_classifier.config import ModelConfig, TrainingConfig, AugmentationConfig
    from micro_classifier.models.classifier import create_model
    from micro_classifier.data.dataset import MicroorganismDataset
    from micro_classifier.utils.trainer import TrainingEngine
    from micro_classifier.utils.device import get_device, get_device_name

    device = get_device()
    logger.info(f"Device: {get_device_name()}")

    model_config = ModelConfig(
        input_size=args.input_size,
        pretrained=not args.no_pretrained,
        dropout_rate=args.dropout,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.workers,
        save_dir=args.save_dir,
        checkpoint_name=args.checkpoint,
    )
    augment_config = AugmentationConfig(input_size=args.input_size)

    dataset = MicroorganismDataset(args.dataset, model_config, augment_config, training_config)
    train_loader, val_loader, stats = dataset.get_dataloaders()

    logger.info(f"Dataset: {stats['total_images']} images, {stats['num_classes']} classes")
    for cls, count in stats["class_counts"].items():
        logger.info(f"  {cls}: {count} images")

    model = create_model(
        num_classes=stats["num_classes"],
        pretrained=not args.no_pretrained,
        dropout_rate=args.dropout,
        freeze_backbone=not args.no_freeze,
        device=device,
    )

    if args.finetune_layers:
        model.unfreeze_backbone(num_layers=args.finetune_layers)
        logger.info(f"Unfrozen last {args.finetune_layers} backbone layers")

    info = model.get_model_info()
    logger.info(f"Model: {info['architecture']}")
    logger.info(f"  Total params: {info['total_params']}")
    logger.info(f"  Trainable: {info['trainable_params']}")

    trainer = TrainingEngine(model, training_config, device)

    def progress_callback(epoch, total, metrics):
        logger.info(
            f"Progress: Epoch {epoch}/{total} - "
            f"Train Acc: {metrics['train_acc']:.1f}% - "
            f"Val Acc: {metrics['val_acc']:.1f}%"
        )

    history = trainer.train(
        train_loader, val_loader,
        class_to_idx=stats["class_to_idx"],
        save_dir=args.save_dir,
        checkpoint_name=args.checkpoint,
        progress_callback=progress_callback,
    )

    logger.info(f"Training complete! Best accuracy: {history['best_val_acc']:.2f}%")
    logger.info(f"Model saved to: {os.path.join(args.save_dir, args.checkpoint)}")

    if args.finetune_layers is None and history["best_val_acc"] < 80:
        logger.info(
            "Consider running fine-tuning for better accuracy:\n"
            f"  python train.py --dataset {args.dataset} --finetune-layers 20 --epochs 30 --lr 1e-4"
        )


if __name__ == "__main__":
    main()
