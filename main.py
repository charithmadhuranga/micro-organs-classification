"""Main entry point for the MicroClassify application."""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("microclassify.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Launch the MicroClassify GUI application."""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        logger.error("PyTorch not found. Install with: uv add torch torchvision")
        sys.exit(1)

    from micro_classifier.utils.device import get_device_name
    logger.info(f"Compute device: {get_device_name()}")

    try:
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        logger.error("OpenCV not found. Install with: pip install opencv-python")
        sys.exit(1)

    try:
        import kivy
        logger.info(f"Kivy version: {kivy.__version__}")
    except ImportError:
        logger.error("Kivy not found. Install with: pip install kivy")
        sys.exit(1)

    from micro_classifier.gui.app import MicroClassifyApp

    logger.info("Starting MicroClassify application...")
    app = MicroClassifyApp()
    app.run()


if __name__ == "__main__":
    main()
