# LinkedIn Post: MicroClassify

---

**Just shipped a real-time microorganism classification system using deep learning and computer vision**

Over the past few weeks, I built MicroClassify — an end-to-end AI system that identifies 16 types of microorganisms from live video feeds in real-time.

**The Problem:**
Manual identification of microorganisms under a microscope is time-consuming, requires expertise, and is prone to human error. I wanted to see how far I could push a modern deep learning pipeline to automate this.

---

**System Features:**

**Deep Learning & Model**
- EfficientNet-B5 backbone (33M parameters) via timm, pretrained on ImageNet
- Dual pooling strategy: Adaptive Average + Max Pooling concatenated for richer feature representation
- Wide classification head: 4096 -> 1024 -> 512 -> 16 classes with SiLU activation
- Progressive unfreezing — backbone unfreezes at 30% of training with separate LR groups
- Label smoothing cross-entropy loss (0.1) for better calibration
- Kaiming weight initialization for classification head
- Mixed-precision (AMP) training and inference via PyTorch autocast

**Data Augmentation Pipeline**
- RandAugment (3 ops, magnitude 9) for automated augmentation
- Mixup (alpha=0.4) and CutMix (alpha=1.0) applied during training
- RandomResizedCrop with scale (0.5-1.0) and aspect ratio (0.8-1.2) jitter
- GaussianBlur, RandomGrayscale, double RandomErasing
- Aggressive color jitter (brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)
- Vertical flip (0.3) and 30-degree rotation
- Test-Time Augmentation (TTA) support for inference

**Training Pipeline**
- Cosine warmup LR scheduler (5 epoch warmup)
- Early stopping with configurable patience (15 epochs)
- AdamW optimizer with weight decay regularization
- Gradient clipping (max_norm=1.0) for stability
- Periodic CUDA cache clearing for memory efficiency
- Background training thread with real-time progress callbacks
- Full checkpoint saving: model weights, optimizer state, class mappings, training history
- CLI with 14 configurable arguments for experiments

**Real-Time Inference & Video**
- Thread-safe video stream handler with lock-protected frame access
- Multi-source support: USB webcam, RTSP/RTMP/HTTP streams, local video files
- Automatic reconnection for network streams (5 retries, 2s delay)
- Video looping for file sources
- Real-time FPS measurement
- BGR -> RGB -> PIL -> Tensor -> Model -> Softmax pipeline
- OpenCV overlay with per-class probability bars, sorted by confidence
- Model warmup for JIT/CUDA kernel optimization

**Dataset & Data Management**
- Web crawler with Wikimedia Commons API and Pixabay API integration
- Verified taxonomy for 16 microorganism classes with search terms and categories
- Image validation: size checks, format verification, blank image detection
- Perceptual hash deduplication (16x16 average hashing)
- RobustImageFolder that silently skips corrupt or unreadable images
- Auto class discovery from dataset directory structure
- Reproducible train/val split (seed=42, 15% validation)
- Per-class image counting and dataset statistics reporting

**GUI (Kivy)**
- Professional dark theme with 15 named color constants
- Responsive 3-column layout (switches to stacked on narrow screens, breakpoint at 1100dp)
- HeaderBar with status pill showing system state
- StatusBar with stream state, mode, and live FPS counter
- SourcePanel: video source selection with type cycling (File/Webcam/RTSP), path input, Browse
- TrainingPanel: dataset/model path inputs, hyperparameter fields (Epochs/LR/Batch), progress bar, live stats log
- ClassificationPanel: color-coded detection label (green >80%, yellow >50%, red <50%), per-class probability bars with dynamic class rebuild
- ModelInfoPanel: architecture details, parameter counts, class list with descriptions
- VideoDisplay: full-bleed preview with aspect ratio preservation, placeholder text
- OS-like FileBrowser: back/forward/up/home navigation, path bar, list/icon view toggle, file type filtering, selection info with file size
- Custom styled widgets: StyledLabel, StyledButton, StyledTextInput, SectionHeader, PanelCard
- Rounded card backgrounds with border lines on all panels
- Adaptive window sizing with macOS Retina DPI detection
- Error popup dialogs with themed styling
- Graceful shutdown (stops stream and training on close)

**Configuration & Build**
- 5 typed dataclasses: ModelConfig, TrainingConfig, AugmentationConfig, StreamConfig, GUIConfig
- pyproject.toml with package metadata, dependencies, and CLI entry point
- Makefile with 14 targets: setup, deps, train, run, check, clean, etc.
- Dual logging: stdout + file (microclassify.log)
- Kivy noise filter suppressing 7 verbose Kivy loggers
- Dependency version checks on startup
- Multi-device support: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU

**Current Results:**
- 16 classes, 4,059 images, 200+ per class
- Best validation accuracy: 68.4% (training in progress)
- ~50s per epoch with full backbone fine-tuning on RTX 3080 Ti

**Tech Stack:**
Python, PyTorch, timm, OpenCV, Kivy, CUDA (RTX 3080 Ti)

**What's Next:**
- Push validation accuracy toward 99.99% with longer training
- Test-time augmentation for inference boost
- ONNX export for deployment

This project reinforced my belief that the gap between research and production is where the real engineering happens — robust data pipelines, real-time inference, and a usable interface matter just as much as the model architecture.

#DeepLearning #ComputerVision #MachineLearning #PyTorch #AI #Microbiology #TransferLearning #EfficientNet #timm #Kivy #OpenCV
