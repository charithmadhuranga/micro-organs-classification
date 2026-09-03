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

        # --------------------------------------------------------------- #
        # Top overlay panel. Its height is computed from the number of
        # classes so every class label + bar always fits with comfortable
        # spacing and never overlaps.
        # --------------------------------------------------------------- #
        title_h = 62                     # room for title + detected lines
        n_classes = len(all_probs) if (show_bars and all_probs) else 0

        if n_classes > 0:
            bar_height = 16
            label_gap = 17               # vertical gap between bar and its label
            bar_gap = 7                  # gap between one bar and next label
            slot = bar_height + label_gap + bar_gap
            bars_start = title_h + 4
            panel_h = int(bars_start + n_classes * slot + 14)
        else:
            panel_h = title_h + 18

        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        cv2.putText(
            annotated, "MICROORGANISM CLASSIFIER",
            (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 216), 2,
        )
        cv2.putText(
            annotated,
            f"Detected: {predicted_class}  ({confidence*100:.1f}%)",
            (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (46, 204, 113), 2,
        )

        if inference_time_ms > 0:
            cv2.putText(
                annotated,
                f"Inference: {inference_time_ms:.1f}ms",
                (w - 240, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
            )

        if n_classes > 0:
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            max_prob = max(prob for _, prob in sorted_probs)
            bar_x_start = 15
            label_width = 150
            track_start = 175
            track_width = int(w * 0.30)
            pct_x = track_start + track_width + 10

            for idx, (cls_name, prob) in enumerate(sorted_probs):
                slot_top = bars_start + idx * slot
                label_baseline = slot_top + label_gap
                bar_top = slot_top + label_gap
                bar_fill_w = max(1, int(track_width * (prob / max_prob if max_prob else 0)))

                # Class name on a dark background so it is always legible.
                label_w_meas = cv2.getTextSize(
                    cls_name[:18], cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
                )[0][0]
                cv2.rectangle(
                    annotated,
                    (bar_x_start, label_baseline - 13),
                    (bar_x_start + max(label_w_meas + 8, label_width),
                     label_baseline + 3),
                    (32, 32, 40), -1,
                )
                cv2.putText(
                    annotated, cls_name[:18],
                    (bar_x_start + 5, label_baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
                )

                # Bar track + fill.
                cv2.rectangle(
                    annotated,
                    (track_start, bar_top),
                    (track_start + track_width, bar_top + bar_height),
                    (50, 50, 50), -1,
                )
                fill_color = (0, 180, 216) if prob == max_prob else (80, 80, 100)
                cv2.rectangle(
                    annotated,
                    (track_start, bar_top),
                    (track_start + bar_fill_w, bar_top + bar_height),
                    fill_color, -1,
                )

                # Percentage right of the bar.
                cv2.putText(
                    annotated,
                    f"{prob*100:.1f}%",
                    (pct_x, bar_top + bar_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
                )

        return annotated
