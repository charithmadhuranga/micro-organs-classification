"""MicroClassify - Professional Microorganism Image Classification System."""

__version__ = "1.0.0"
__author__ = "MicroClassify Team"

import os
import sys


def main():
    from micro_classifier.gui.app import MicroClassifyApp
    app = MicroClassifyApp()
    app.run()


if __name__ == "__main__":
    main()
