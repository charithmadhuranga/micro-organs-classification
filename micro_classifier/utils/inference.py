"""Inference engine for real-time microorganism classification."""

import time
import logging
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..models.classifier import MicroClassifier, load_checkpoint
from ..utils.device import get_device

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Real-time inference engine for classifying microorganisms from video frames.
    Handles frame preprocessing, model inference, and result post-processing.
    """

    def __init__(
        self,
        model: MicroClassifier,
        class_names: List[str],
        transform: transforms.Compose,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.1,
    ):
        self.model = model
        self.class_names = class_names
        self.transform = transform
        self.device = device or get_device()
        self.confidence_threshold = confidence_threshold
        self.model.eval()
        self._warmup_done = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        transform: transforms.Compose,
        device: Optional[torch.device] = None,
    ) -> "InferenceEngine":
        """Create an InferenceEngine from a saved checkpoint."""
        model, metadata = load_checkpoint(checkpoint_path, device=device)
        idx_to_class = metadata.get("idx_to_class", {})
        if not idx_to_class:
            raise ValueError("Checkpoint missing class mapping")
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        return cls(model, class_names, transform, device)

    def warmup(self, input_size: int = 380):
        """Run a dummy forward pass to warm up the model (JIT/CUDA kernels)."""
        if self._warmup_done:
            return
        dummy = torch.randn(1, 3, input_size, input_size).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        self._warmup_done = True
        logger.info("Model warmup complete")

    @torch.no_grad()
    def predict_frame(
        self, frame: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a single video frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            (predicted_class, confidence, all_class_probabilities)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)

        probs = probabilities.cpu().numpy()[0]
        class_probs = {
            self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
        }

        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])
        predicted_class = self.class_names[max_idx]

        return predicted_class, confidence, class_probs

    def predict_frame_with_timing(
        self, frame: np.ndarray
    ) -> Tuple[str, float, Dict[str, float], float]:
        """Predict with inference time measurement."""
        start = time.perf_counter()
        pred_class, confidence, all_probs = self.predict_frame(frame)
        inference_time = (time.perf_counter() - start) * 1000
        return pred_class, confidence, all_probs, inference_time

    def annotate_frame(
        self,
        frame: np.ndarray,
        predicted_class: str,
        confidence: float,
        all_probs: Dict[str, float],
        inference_time_ms: float = 0.0,
        show_bars: bool = True,
    ) -> np.ndarray:
        """
        Draw classification results on a video frame.

        Args:
            frame: Original BGR frame
            predicted_class: Top predicted class name
            confidence: Confidence score for the prediction
            all_probs: All class probabilities
            inference_time_ms: Inference time in milliseconds
            show_bars: Whether to show probability bars

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        overlay = annotated.copy()
        panel_h = 280 if show_bars else 80
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        cv2.putText(
            annotated, "MICROORGANISM CLASSIFIER",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 216), 2,
        )
        cv2.putText(
            annotated,
            f"Detected: {predicted_class}  ({confidence*100:.1f}%)",
            (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (46, 204, 113), 2,
        )

        if inference_time_ms > 0:
            cv2.putText(
                annotated,
                f"Inference: {inference_time_ms:.1f}ms",
                (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
            )

        if show_bars and len(all_probs) > 0:
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            bar_x_start = 15
            bar_y_start = 90
            bar_width = int(w * 0.4)
            bar_height = 20
            bar_spacing = 28

            for idx, (cls_name, prob) in enumerate(sorted_probs):
                y = bar_y_start + idx * bar_spacing
                bar_fill = int(bar_width * prob)

                cv2.putText(
                    annotated, f"{cls_name[:18]}",
                    (bar_x_start, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                )

                cv2.rectangle(
                    annotated,
                    (bar_x_start, y + 2),
                    (bar_x_start + bar_width, y + 2 + bar_height),
                    (50, 50, 50), -1,
                )

                color = (0, 180, 216) if prob == max(all_probs.values()) else (80, 80, 100)
                cv2.rectangle(
                    annotated,
                    (bar_x_start, y + 2),
                    (bar_x_start + bar_fill, y + 2 + bar_height),
                    color, -1,
                )

                cv2.putText(
                    annotated,
                    f"{prob*100:.1f}%",
                    (bar_x_start + bar_width + 10, y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1,
                )

        return annotated
