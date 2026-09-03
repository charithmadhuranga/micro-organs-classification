"""Panels for the MicroClassify GUI.

Each panel is a self-contained, full-width card. In every vertical BoxLayout the
first child added appears at the BOTTOM and the last-added appears at the TOP,
so the section header is always the LAST widget added.
"""

import os

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image

from ..config import CLASSES
from .theme import THEME
from .widgets import (
    PanelCard,
    SectionHeader,
    StyledButton,
    StyledLabel,
    StyledTextInput,
    paint_card,
)


class SourcePanel(PanelCard):
    """Controls for choosing and starting a video source."""

    def __init__(self, app_controller, **kwargs):
        super().__init__(**kwargs)
        self.app = app_controller
        self.source_types = ["Video File", "USB Webcam", "RTSP Stream"]
        self._current_type_idx = 0

        # Build controls from bottom to top. In a vertical BoxLayout the first
        # added lands at the BOTTOM, so the last control (Start/Stop) is added
        # first and the header is added last to land on top.
        btn_row = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(38))
        self.start_btn = StyledButton(
            text="START STREAM",
            font_size=sp(11),
            background_color=THEME["success"],
            size_hint_x=0.6,
        )
        self.start_btn.bind(on_press=self._start_stream)
        btn_row.add_widget(self.start_btn)
        self.stop_btn = StyledButton(
            text="STOP",
            font_size=sp(11),
            background_color=THEME["danger"],
            size_hint_x=0.4,
        )
        self.stop_btn.bind(on_press=self._stop_stream)
        btn_row.add_widget(self.stop_btn)
        self.add_widget(btn_row)

        row = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(36))
        self.path_input = StyledTextInput(
            hint_text="Enter path, device index or URL...",
            font_size=sp(11),
            size_hint_x=0.7,
        )
        row.add_widget(self.path_input)
        browse_btn = StyledButton(
            text="Browse",
            size_hint_x=0.3,
            height=dp(34),
            font_size=sp(11),
            background_color=THEME["accent_dim"],
        )
        browse_btn.bind(on_press=self._browse_file)
        row.add_widget(browse_btn)
        self.add_widget(row)

        self.source_type_btn = StyledButton(
            text="TYPE: Video File",
            font_size=sp(11),
            size_hint_x=1.0,
            height=dp(34),
            background_color=THEME["bg_input"],
        )
        self.source_type_btn.bind(on_press=self._cycle_source_type)
        self.add_widget(self.source_type_btn)

        # Header added LAST so it sits on top.
        self.add_widget(SectionHeader("VIDEO SOURCE", "Input stream"))

    def _cycle_source_type(self, *args):
        self._current_type_idx = (self._current_type_idx + 1) % len(self.source_types)
        type_name = self.source_types[self._current_type_idx]
        self.source_type_btn.text = f"TYPE: {type_name}"
        hints = {
            "Video File": "Enter video file path...",
            "USB Webcam": "Enter device index (0, 1, 2...)",
            "RTSP Stream": "Enter RTSP URL...",
        }
        self.path_input.hint_text = hints[type_name]

    def _browse_file(self, *args):
        self.app.open_file_chooser()

    def _start_stream(self, *args):
        source = self.path_input.text.strip()
        if not source:
            self.app.show_error("Please enter a source path or URL")
            return
        self.app.start_stream(source)

    def _stop_stream(self, *args):
        self.app.stop_stream()


class TrainingPanel(PanelCard):
    """Model training controls, progress and live metrics."""

    def __init__(self, app_controller, **kwargs):
        super().__init__(**kwargs)
        self.app = app_controller
        self.is_training = False

        # Build controls and add them bottom-to-top (first added = bottom).
        # Desired top-to-bottom: header, dataset, model, hyperparams, progress,
        # buttons, stats (fills remaining space).

        self.stats_label = StyledLabel(
            text="Ready.\n1. Enter dataset path and click START TRAINING.\n"
                 "2. Load a trained model checkpoint.\n"
                 "3. Select a source and press Start Stream.\n\n"
                 "Training metrics will appear here.",
            font_size=sp(11),
            color=THEME["text_muted"],
            halign="left",
            valign="top",
            size_hint_y=1.0,
            text_size=(None, None),
        )
        self.add_widget(self.stats_label)

        btn_row = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(38))
        self.train_btn = StyledButton(
            text="START TRAINING",
            font_size=sp(11),
            background_color=THEME["success"],
            size_hint_x=0.6,
        )
        self.train_btn.bind(on_press=self._start_training)
        btn_row.add_widget(self.train_btn)
        self.stop_train_btn = StyledButton(
            text="STOP",
            font_size=sp(11),
            background_color=THEME["danger"],
            size_hint_x=0.4,
        )
        self.stop_train_btn.bind(on_press=self._stop_training)
        btn_row.add_widget(self.stop_train_btn)
        self.add_widget(btn_row)

        self.progress_label = StyledLabel(
            text="Ready",
            font_size=sp(10),
            color=THEME["text_dim"],
            halign="left",
            size_hint_y=None,
            height=dp(18),
        )
        self.add_widget(self.progress_label)
        self.progress_bar = TrainingProgress()
        self.add_widget(self.progress_bar)

        params_row = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(38))
        self.epochs_input = StyledTextInput(text="50", font_size=sp(11), size_hint_x=0.33)
        self.lr_input = StyledTextInput(text="0.001", font_size=sp(11), size_hint_x=0.33)
        self.batch_input = StyledTextInput(text="32", font_size=sp(11), size_hint_x=0.34)
        params_row.add_widget(self._field("Epochs", self.epochs_input, size_hint_x=0.33))
        params_row.add_widget(self._field("LR", self.lr_input, size_hint_x=0.33))
        params_row.add_widget(self._field("Batch", self.batch_input, size_hint_x=0.34))
        self.add_widget(params_row)

        mp_row = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(36))
        self.model_path_input = StyledTextInput(
            text=app_controller.default_model_path, font_size=sp(11), size_hint_x=0.50
        )
        mp_row.add_widget(self.model_path_input)
        browse_btn = StyledButton(
            text="Browse",
            size_hint_x=0.20,
            height=dp(34),
            font_size=sp(10),
            background_color=THEME["accent_dim"],
        )
        browse_btn.bind(on_press=self._browse_model)
        mp_row.add_widget(browse_btn)
        load_btn = StyledButton(
            text="Load",
            size_hint_x=0.30,
            height=dp(34),
            font_size=sp(10),
            background_color=THEME["accent"],
        )
        load_btn.bind(on_press=self._load_model)
        mp_row.add_widget(load_btn)
        self.add_widget(mp_row)

        ds_row = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(36))
        ds_row.add_widget(StyledLabel(
            text="Dataset path",
            font_size=sp(10),
            color=THEME["text_dim"],
            size_hint_x=0.34,
            halign="left",
        ))
        self.dataset_path_input = StyledTextInput(
            text=app_controller.default_dataset_path, font_size=sp(11), size_hint_x=0.46
        )
        ds_row.add_widget(self.dataset_path_input)
        ds_browse_btn = StyledButton(
            text="Browse",
            size_hint_x=0.20,
            height=dp(34),
            font_size=sp(10),
            background_color=THEME["accent_dim"],
        )
        ds_browse_btn.bind(on_press=self._browse_dataset)
        ds_row.add_widget(ds_browse_btn)
        self.add_widget(ds_row)

        # Header added LAST so it sits on top.
        self.add_widget(SectionHeader("MODEL TRAINING", "Hyperparameters & progress"))

    def _field(self, label, widget, size_hint_x=1.0):
        box = BoxLayout(
            orientation="vertical",
            spacing=dp(2),
            size_hint_x=size_hint_x,
        )
        lbl = StyledLabel(
            text=label,
            font_size=sp(9),
            color=THEME["text_muted"],
            halign="center",
            valign="center",
            size_hint_y=None,
            height=dp(12),
        )
        lbl.bind(
            size=lambda w, *a: setattr(w, "text_size", (w.width, w.height))
        )
        box.add_widget(lbl)
        box.add_widget(widget)
        return box

    def _load_model(self, *args):
        self.app.load_model(self.model_path_input.text.strip())

    def _browse_model(self, *args):
        self.app.open_model_chooser()

    def _browse_dataset(self, *args):
        self.app.open_dataset_chooser()

    def _start_training(self, *args):
        if self.is_training:
            return
        self.app.start_training(
            dataset_path=self.dataset_path_input.text.strip(),
            epochs=int(self.epochs_input.text.strip() or "50"),
            lr=float(self.lr_input.text.strip() or "0.001"),
            batch_size=int(self.batch_input.text.strip() or "32"),
        )

    def _stop_training(self, *args):
        self.app.stop_training()

    def update_progress(self, epoch, total, metrics):
        self.progress_bar.set_value(epoch / total if total else 0.0)
        self.progress_label.text = (
            f"Epoch {epoch}/{total}  "
            f"train {metrics.get('train_acc', 0):.1f}%  "
            f"val {metrics.get('val_acc', 0):.1f}%  "
            f"loss {metrics.get('val_loss', 0):.3f}"
        )
        self._append_stats(
            f"[Epoch {epoch}/{total}] "
            f"train_acc={metrics.get('train_acc', 0):.1f}% "
            f"val_acc={metrics.get('val_acc', 0):.1f}% "
            f"loss={metrics.get('val_loss', 0):.3f}"
        )

    def _append_stats(self, line):
        if self.stats_label.text.startswith("Ready."):
            self.stats_label.text = line
        else:
            lines = self.stats_label.text.split("\n")
            lines.append(line)
            self.stats_label.text = "\n".join(lines[-14:])

    def set_training(self, training):
        self.is_training = training
        if training:
            self.progress_bar.reset()
            self.train_btn.background_color = THEME["text_muted"]
            self.train_btn.text = "TRAINING..."
        else:
            self.train_btn.background_color = THEME["success"]
            self.train_btn.text = "START TRAINING"


class TrainingProgress(BoxLayout):
    """A simple rounded progress track with a fill that follows resize."""

    def __init__(self, **kwargs):
        super().__init__(size_hint_y=None, height=dp(8), size_hint_x=1.0, **kwargs)
        self._fill = 0.0
        self._track = None
        self._bar = None
        self.bind(pos=self._repaint, size=self._repaint)

    def reset(self):
        self._fill = 0.0
        self._repaint()

    def set_value(self, value):
        self._fill = max(0.0, min(1.0, value))
        self._repaint()

    def _repaint(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(*THEME["bg_input"])
            self._track = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(4)]
            )
            if self._fill > 0:
                Color(*THEME["accent"])
                self._bar = RoundedRectangle(
                    pos=self.pos,
                    size=(self.width * self._fill, self.height),
                    radius=[dp(4)],
                )


class ClassificationPanel(PanelCard):
    """Live prediction results: detection header + per-class probability bars."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prob_bars = {}
        self._build_ui()

    def _build_ui(self):
        # Add widgets bottom-to-top: bars (fill), result card, then header on top.
        # Probability bars fill the remaining vertical space at the bottom.
        bars_box = BoxLayout(orientation="vertical", spacing=dp(4), size_hint_y=1.0)
        for cls_name in CLASSES:
            row = BoxLayout(orientation="horizontal", spacing=dp(6), size_hint_y=1.0)
            name = StyledLabel(
                text=cls_name,
                font_size=sp(9.5),
                color=THEME["text_dim"],
                size_hint_x=0.38,
                halign="right",
            )
            row.add_widget(name)
            bar = BarWidget(size_hint_x=0.46, size_hint_y=1.0)
            row.add_widget(bar)
            pct = StyledLabel(
                text="0.0%",
                font_size=sp(9),
                color=THEME["text_muted"],
                size_hint_x=0.16,
                halign="left",
            )
            row.add_widget(pct)
            bars_box.add_widget(row)
            self._prob_bars[cls_name] = (bar, pct)
        self.add_widget(bars_box)

        # Detection summary card just below the header.
        result_card = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            height=dp(76),
            padding=[dp(8), dp(4)],
            spacing=dp(1),
        )
        self.detection_label = StyledLabel(
            text="No detection",
            font_size=sp(15),
            bold=True,
            color=THEME["text"],
            halign="left",
            size_hint_y=0.55,
        )
        result_card.add_widget(self.detection_label)
        status_line = BoxLayout(spacing=dp(8), size_hint_y=0.45)
        self.confidence_label = StyledLabel(
            text="Confidence  --",
            font_size=sp(10),
            color=THEME["text_dim"],
            halign="left",
            size_hint_x=0.62,
        )
        status_line.add_widget(self.confidence_label)
        self.inference_label = StyledLabel(
            text="-- ms",
            font_size=sp(10),
            color=THEME["text_muted"],
            halign="right",
            size_hint_x=0.38,
        )
        status_line.add_widget(self.inference_label)
        result_card.add_widget(status_line)
        self.add_widget(result_card)

        # Header added LAST so it sits on top.
        self.add_widget(SectionHeader("LIVE DETECTION", "Per-class confidence"))

    def update_results(self, predicted_class, confidence, all_probs, inference_time):
        self.detection_label.text = predicted_class
        self.detection_label.color = (
            THEME["success"] if confidence > 0.8
            else THEME["warning"] if confidence > 0.5
            else THEME["danger"]
        )
        self.confidence_label.text = f"Confidence  {confidence * 100:.1f}%"
        self.inference_label.text = f"{inference_time:.1f} ms"
        max_prob = max(all_probs.values()) if all_probs else 1.0
        for cls_name, (bar, pct) in self._prob_bars.items():
            prob = all_probs.get(cls_name, 0.0)
            pct.text = f"{prob * 100:.1f}%"
            bar.set_fill(prob / max_prob if max_prob else 0.0,
                         active=(cls_name == predicted_class))

    def clear_results(self):
        self.detection_label.text = "No detection"
        self.detection_label.color = THEME["text"]
        self.confidence_label.text = "Confidence  --"
        self.inference_label.text = "-- ms"
        for cls_name, (bar, pct) in self._prob_bars.items():
            bar.set_fill(0.0, active=False)
            pct.text = "0.0%"


class BarWidget(BoxLayout):
    """A probability bar: rounded track + fill sized by a stored ratio."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fill = 0.0
        self._active = False
        self.bind(pos=self._repaint, size=self._repaint)

    def set_fill(self, value, active=False):
        self._fill = max(0.0, min(1.0, value))
        self._active = active
        self._repaint()

    def _repaint(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(*THEME["bg_input"])
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(3)])
            if self._fill > 0:
                Color(*THEME["accent"] if self._active else THEME["accent_dim"])
                RoundedRectangle(
                    pos=self.pos,
                    size=(self.width * self._fill, self.height),
                    radius=[dp(3)],
                )


class VideoDisplay(BoxLayout):
    """Full-bleed video preview that stretches frames to fill the column.

    Uses a FloatLayout (not a BoxLayout) so the Image child's size/position
    are controlled directly and never re-positioned by layout enforcement.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding = dp(8)
        paint_card(self, radius=12, bg_color=THEME["bg_dark"], border_color=THEME["border"])

        self._inner = FloatLayout()
        self.add_widget(self._inner)

        self._preview = Image(
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(None, None),
        )
        self._preview.texture = None
        self._inner.add_widget(self._preview)

        self.placeholder = StyledLabel(
            text="Video source appears here",
            color=THEME["text_muted"],
            font_size=sp(13),
            halign="center",
            valign="center",
            size_hint_x=0.6,
            size_hint_y=0.2,
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.placeholder.bind(
            size=lambda w, *a: setattr(w, "text_size", (w.width, None))
        )
        self._inner.add_widget(self.placeholder)

        self._frame_size = (0, 0)
        self._inner.bind(pos=self._layout_preview, size=self._layout_preview)

    def _layout_preview(self, *args):
        w = self._inner.width
        h = self._inner.height
        fw, fh = self._frame_size
        if fw and fh:
            scale = min(w / fw, h / fh)
            pw, ph = max(1, int(fw * scale)), max(1, int(fh * scale))
        else:
            pw, ph = max(1, w), max(1, h)
        self._preview.size = (pw, ph)
        self._preview.pos = (
            self._inner.x + (w - pw) / 2,
            self._inner.y + (h - ph) / 2,
        )

    def update_frame(self, frame):
        if frame is None:
            return
        frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        buf = frame.tobytes()
        texture = Texture.create(size=(w, h), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        texture.flip_vertical()
        self._preview.texture = texture
        if self.placeholder in self._inner.children:
            self._inner.remove_widget(self.placeholder)
        self._layout_preview()

    def clear_display(self):
        self._preview.texture = None
        self._frame_size = (0, 0)
        if self.placeholder not in self._inner.children:
            self._inner.add_widget(self.placeholder)
