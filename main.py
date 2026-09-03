"""Main entry point for the MicroClassify application."""

import os
import sys
import logging

os.environ["KIVY_LOG_LEVEL"] = "warning"
os.environ["KIVY_INPUT"] = "sdl2"
os.environ["KIVY_CLIPBOARD"] = "sdl2"
os.environ["KIVY_CUTBUFFER"] = "sdl2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("microclassify.log"),
    ],
)
logger = logging.getLogger(__name__)

_NOISY_PATTERNS = (
    "MTDev is not supported",
    "libmtdev.so",
    "Unable to find any valuable Cutbuffer",
    "xclip",
    "xsel",
    "clipboard_xclip",
    "clipboard_xsel",
    "clipboard_dbusklipper",
    "clipboard_gtk3",
    "Window.minimum_width",
)


class _KivyNoiseFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in _NOISY_PATTERNS)


for _name in ("kivy", "kivy.input", "kivy.input.providers",
              "kivy.core", "kivy.core.clipboard", "kivy.core.cutbuffer",
              "kivy.core.window"):
    logging.getLogger(_name).addFilter(_KivyNoiseFilter())


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
