# ==============================================================================
# MicroClassify — Makefile
# Professional Microorganism Classification System
# ------------------------------------------------------------------------------
# Usage:
#   make setup         Create uv venv + install all dependencies
#   make deps          Install/update all dependencies
#   make sync          Sync project to uv.lock
#   make data          Generate synthetic test dataset + video (no download)
#   make train         Train model on real dataset (CLI)
#   make train-quick   Quick training smoke test
#   make train-finetune  Second-stage backbone fine-tuning
#   make run           Launch the Kivy GUI
#   make inference     Run CLI inference e2e test on a video stream
#   make gui-smoke     Run GUI smoke test (opens window briefly)
#   make stream        Test video stream handler (mic-check)
#   make check         Compile-check all Python source files
#   make clean         Remove generated artifacts & caches
#   make help          Show this help
# ==============================================================================

SHELL := /bin/bash
UV    := uv
PY    := $(UV) run python

# Training defaults (override on the command line, e.g. make train DATASET=... )
DATASET      ?= dataset
EPOCHS       ?= 50
BATCH_SIZE   ?= 32
LR           ?= 0.001
WEIGHT_DECAY ?= 1e-4
DROPOUT      ?= 0.3
INPUT_SIZE   ?= 380
SAVE_DIR     ?= trained_models
CHECKPOINT   ?= best_microclassifier.pth
WORKERS      ?= 4
MODEL_PATH   ?= trained_models/best_microclassifier.pth
VIDEO        ?= test_micro.mp4

.PHONY: help setup deps sync data train train-quick train-finetune run
.PHONY: inference gui-smoke stream check clean

help:
	@printf '\n\033[1;36mMicroClassify — available targets\033[0m\n\n'
	@printf '  \033[1;32msetup\033[0m           Create uv venv + install all dependencies\n'
	@printf '  \033[1;32mdeps\033[0m            Install/update all core dependencies\n'
	@printf '  \033[1;32msync\033[0m            Sync the project to uv.lock\n'
	@printf '  \033[1;32mdata\033[0m            Generate synthetic test dataset + video\n'
	@printf '  \033[1;32mtrain\033[0m           Train the model (CLI, defaults: $(EPOCHS) epochs, lr $(LR))\n'
	@printf '  \033[1;32mtrain-quick\033[0m     Quick training smoke test (3 epochs)\n'
	@printf '  \033[1;32mtrain-finetune\033[0m  Second-stage backbone fine-tuning for higher accuracy\n'
	@printf '  \033[1;32mrun\033[0m             Launch the Kivy GUI\n'
	@printf '  \033[1;32minference\033[0m      Run CLI inference e2e test on a video stream\n'
	@printf '  \033[1;32mgui-smoke\033[0m      Run GUI smoke test (opens window briefly)\n'
	@printf '  \033[1;32mstream\033[0m         Sanity-check the video stream handler\n'
	@printf '  \033[1;32mcheck\033[0m          Compile-check all Python source files\n'
	@printf '  \033[1;32mclean\033[0m          Remove generated artifacts & caches\n\n'
	@printf '  Override training args inline: '
	@printf '`make train DATASET=... EPOCHS=... LR=... BATCH_SIZE=...`\n\n'

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

setup:
	$(UV) venv --python 3.11 .venv
	$(UV) add torch torchvision opencv-python kivy Pillow numpy
	@printf '\n\033[1;32mSetup complete. Activate with: source .venv/bin/activate\033[0m\n'

deps:
	$(UV) add torch torchvision opencv-python kivy Pillow numpy

sync:
	$(UV) sync

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

data:
	$(PY) tests/make_test_data.py --dataset $(DATASET) --per-class 10

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

train:
	$(PY) train.py --dataset $(DATASET) --epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) --lr $(LR) --weight-decay $(WEIGHT_DECAY) \
		--dropout $(DROPOUT) --input-size $(INPUT_SIZE) \
		--save-dir $(SAVE_DIR) --checkpoint $(CHECKPOINT) --workers $(WORKERS)

train-quick:
	$(PY) train.py --dataset $(DATASET) --epochs 3 --batch-size 8 --lr 0.001

train-finetune:
	$(PY) train.py --dataset $(DATASET) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) \
		--lr 1e-4 --finetune-layers 20 --save-dir $(SAVE_DIR) --checkpoint $(CHECKPOINT)

# ------------------------------------------------------------------------------
# Run / GUI
# ------------------------------------------------------------------------------

run:
	$(PY) main.py

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

inference:
	$(PY) tests/e2e_inference.py

gui-smoke:
	$(PY) tests/gui_smoke.py

stream:
	$(PY) -c "import logging; logging.disable(logging.CRITICAL); \
from micro_classifier.utils.stream import VideoStream; \
s=VideoStream('$(VIDEO)'); assert s.start(); \
import time; \
[time.sleep(0.05) for _ in range(20)]; \
f=s.get_frame(); \
print('opened source:', s.source_type, '| frame:', 'NONE' if f is None else str(f.shape)); \
s.stop()"

# ------------------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------------------

check:
	$(PY) -m py_compile \
		main.py train.py \
		micro_classifier/__init__.py micro_classifier/__main__.py micro_classifier/config.py \
		micro_classifier/models/classifier.py \
		micro_classifier/data/dataset.py \
		micro_classifier/utils/device.py micro_classifier/utils/trainer.py \
		micro_classifier/utils/inference.py micro_classifier/utils/stream.py \
		micro_classifier/gui/__init__.py micro_classifier/gui/theme.py \
		micro_classifier/gui/widgets.py micro_classifier/gui/panels.py \
		micro_classifier/gui/layout.py micro_classifier/gui/app.py
	@printf '\n\033[1;32mAll source files compile OK\033[0m\n'

clean:
	rm -rf __pycache__ micro_classifier/__pycache__ \
		micro_classifier/*/__pycache__ tests/__pycache__
	rm -f *.pyc microclassify.log
	rm -f $(DATASET)/*.jpg test_micro.mp4
	rm -rf .venv
	@printf '\n\033[1;33mCleaned project artifacts (note: venv and dataset removed)\033[0m\n'