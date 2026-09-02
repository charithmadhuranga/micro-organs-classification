# MicroClassify — Microorganism Classification System

A professional, real-time **microorganism image classification system** with a complete training pipeline, real-time video classification, and a polished Kivy GUI.

This project analyzes and reimplements the concepts from
`microorganisms-image-classification-inceptionv3.ipynb` (transfer-learning based
microorganism classification) into a production-grade, extensible application.

## Features

- **Deep Learning Classifier** — EfficientNet-B4 pretrained on ImageNet (higher
  accuracy and efficiency than InceptionV3 for small datasets), fine-tuned via
  transfer learning for 8 microorganism classes.
- **Real-Time Video Classification** — Live classification from multiple sources:
  - USB webcam
  - RTSP / RTMP / HTTP network streams
  - Local video files (`.mp4`, `.avi`, `.mov`, `.mkv`, ...)
- **Professional Training System** — CLI training with:
  - Data augmentation (rotation, flips, color jitter, random erasing)
  - Mixed-precision training, gradient clipping
  - Cosine warmup learning-rate scheduling
  - Early stopping, best-checkpoint saving
  - Optional backbone fine-tuning stage
- **Professional Kivy GUI** — Dark professional UI with:
  - Live video preview
  - On-screen classification results and confidence probability bars
  - Source selection (file / webcam / RTSP)
  - In-app model training controls and progress
  - Model loading, device/FPS status bar

## Dataset

Uses the **Microorganism Image Classification** dataset from Kaggle:
https://www.kaggle.com/datasets/mdwaquarazam/microorganism-image-classification

8 classes:
`Amoeba, Euglena, Hydra, Paramecium, Rod Bacteria, Spherical Bacteria, Spiral Bacteria, Yeast`

Download it and place the `Micro_Organism` folder contents such that each class
is a sub-directory under `dataset/`:

```
dataset/
├── Amoeba/
├── Euglena/
├── Hydra/
├── Paramecium/
├── Rod_bacteria/
├── Spherical_bacteria/
├── Spiral_bacteria/
└── Yeast/
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) and ships a `Makefile` with
all common commands. Run `make help` for the full list.

```bash
make setup          # create venv + install dependencies
make data           # generate synthetic test dataset + video
make train          # train on the real dataset
make run            # launch the Kivy GUI
make check          # compile-check all source files
make help           # list all targets
```

Or manually:

```bash
# Install dependencies (torch, torchvision, opencv, kivy, etc.)
uv add torch torchvision opencv-python kivy Pillow numpy

# Activate environment
source .venv/bin/activate
```

## GPU Acceleration

The system automatically selects the best available compute device:
1. **CUDA** (NVIDIA GPU)
2. **MPS / Apple Silicon GPU** (Mac — uses `torch.backends.mps`)
3. **CPU** (fallback)

Mixed-precision (AMP) is used on CUDA; MPS runs in standard precision.
The active device is logged at startup and shown in the GUI status bar.

## Testing without a webcam

Since no webcam may be available, the system supports streaming a **video file**
for testing. Either point the GUI source at a microorganism video, or generate
test fixtures with the included script:

```bash
# Generate a synthetic dataset + test video (no downloading required)
uv run python tests/make_test_data.py --dataset dataset --per-class 10

# Train a quick model on it
uv run python train.py --dataset dataset --epochs 10 --batch-size 8 --lr 0.001

# Run CLI inference e2e test (loads model + classifies video frames on MPS/CUDA)
uv run python tests/e2e_inference.py

# Run GUI smoke test (opens a window briefly)
uv run python tests/gui_smoke.py
```

## Training (CLI)

```bash
# Basic transfer learning
uv run python train.py --dataset dataset --epochs 50 --batch-size 32 --lr 0.001

# With backbone fine-tuning for higher accuracy (recommended second stage)
uv run python train.py --dataset dataset --epochs 30 --lr 0.0001 --finetune-layers 20

# Full help
uv run python train.py --help
```

Checkpoints are saved to `trained_models/best_microclassifier.pth`.

## Running the GUI

```bash
uv run python main.py
# or
uv run microclassify
```

### Using the GUI

1. **Model** — In the TRAINING panel, click **Load Model** and point to a trained
   checkpoint (e.g. `trained_models/best_microclassifier.pth`), *or* train fresh
   directly from the GUI.
2. **Source** — Set the source type (Video File / USB Webcam / RTSP) and path,
   then click **START STREAM**.
   - Local file: e.g. `dataset/Yeast/sample.mp4`
   - Webcam: `0`, `1`, ...
   - RTSP: `rtsp://192.168.1.10:554/stream`
3. Live classification results appear in the right panel with confidence bars.

## Testing without a webcam

Since no webcam may be available, the system supports streaming a **video file**
for testing. Either point the GUI source at a microorganism video, or generate
test fixtures with the included script:

```bash
# Generate a synthetic dataset + test video (no downloading required)
uv run python tests/make_test_data.py --dataset dataset --per-class 10

# Train a quick model on it
uv run python train.py --dataset dataset --epochs 10 --batch-size 8 --lr 0.001

# Run CLI inference e2e test (loads model + classifies video frames on MPS/CUDA)
uv run python tests/e2e_inference.py

# Run GUI smoke test (opens a window briefly)
uv run python tests/gui_smoke.py
```

You can also generate a test video from any image folder with ffmpeg:
```bash
ffmpeg -framerate 10 -pattern_type glob -i 'dataset/*/*.jpg' -vf scale=640:-1 -c:v libx264 test_micro.mp4
```

## Project Structure

```
micro-classifier/
├── main.py                          # GUI entry point
├── train.py                         # CLI training script
├── tests/
│   ├── make_test_data.py            # synthetic dataset + video generator
│   ├── e2e_inference.py             # CLI inference end-to-end test
│   └── gui_smoke.py                 # GUI smoke test
├── pyproject.toml                   # uv project config & dependencies
├── requirements.txt
├── dataset/                         # place Kaggle dataset here
├── trained_models/                  # model checkpoints
└── micro_classifier/
    ├── config.py                    # model/training/stream/GUI config
    ├── models/classifier.py         # EfficientNet-B4 classifier
    ├── data/dataset.py              # dataset + augmentation
    ├── utils/device.py              # CUDA/MPS/CPU device selection
    ├── utils/trainer.py             # training engine
    ├── utils/inference.py           # real-time inference engine
    ├── utils/stream.py              # video stream handler (webcam/RTSP/file)
    └── gui/app.py                   # Kivy application
```

## Notes

- **Model choice:** EfficientNet-B4 is used instead of InceptionV3 from the
  notebook because it provides a better accuracy/parameter trade-off, is well
  suited to small datasets via transfer learning, and is fast enough for
  real-time inference.
- Class probabilities are displayed with a softmax head; a confidence threshold
  can be tuned in `micro_classifier/utils/inference.py`.
