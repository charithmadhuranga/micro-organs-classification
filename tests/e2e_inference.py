"""End-to-end inference test: load trained model + classify video file frames."""

import time
import logging
import os

logging.disable(logging.CRITICAL)

import cv2

from micro_classifier.config import AugmentationConfig
from micro_classifier.data.dataset import get_validation_transforms
from micro_classifier.utils.inference import InferenceEngine
from micro_classifier.utils.stream import VideoStream
from micro_classifier.utils.device import get_device, get_device_name


def main():
    model_path = os.path.join("trained_models", "best_microclassifier.pth")
    video_path = "test_micro.mp4"

    transform = get_validation_transforms(AugmentationConfig())
    device = get_device()
    print(f"Device: {get_device_name()}")

    print("Loading model...")
    t0 = time.time()
    engine = InferenceEngine.from_checkpoint(model_path, transform, device)
    engine.warmup()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    print("Opening video stream...")
    stream = VideoStream(video_path)
    assert stream.start(), "Failed to open video"
    print(f"Source type: {stream.source_type}")
    print(f"Frame dims: {stream.get_frame_dimensions()}")

    frame = None
    wait_start = time.time()
    while frame is None and time.time() - wait_start < 5.0:
        frame = stream.get_frame()
        time.sleep(0.02)
    if frame is None:
        print("No frames received from stream")
        stream.stop()
        return False

    print("Classifying frames...")
    pred_counts = {}
    n_frames = 0
    total_inf = 0.0
    t0 = time.time()
    for _ in range(30):
        frame = stream.get_frame()
        if frame is None:
            continue
        cls, conf, probs, inf_ms = engine.predict_frame_with_timing(frame)
        pred_counts[cls] = pred_counts.get(cls, 0) + 1
        total_inf += inf_ms
        n_frames += 1
        annotated = engine.annotate_frame(frame, cls, conf, probs, inf_ms)
        time.sleep(0.03)

    elapsed = time.time() - t0
    stream.stop()

    print(f"Processed {n_frames} frames in {elapsed:.1f}s")
    print(f"Average inference: {total_inf/max(1,n_frames):.1f}ms/frame "
          f"({max(1,n_frames)/max(elapsed,1e-6):.1f} fps)")
    print("Prediction distribution:")
    for k, v in sorted(pred_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    return n_frames > 0


if __name__ == "__main__":
    ok = main()
    print("E2E_INFERENCE_OK" if ok else "E2E_INFERENCE_FAILED")