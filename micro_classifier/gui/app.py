"""Professional Kivy-based GUI for the Microorganism Classification System."""

import os
import time
import logging
import threading
from typing import Optional, Dict

import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, RoundedRectangle, Line
from kivy.metrics import dp, sp
from kivy.properties import (
    StringProperty, NumericProperty, BooleanProperty,
    ListProperty, DictProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.image import Image as KivyImage
from kivy.graphics.texture import Texture
from kivy.utils import get_color_from_hex
from kivy.core.image import Image as CoreImage
from io import BytesIO

from ..config import (
    ModelConfig, TrainingConfig, AugmentationConfig,
    StreamConfig, GUIConfig, CLASSES, CLASS_DESCRIPTIONS,
    DEFAULT_MODEL_PATH, DEFAULT_DATASET_PATH,
)
from ..models.classifier import create_model, load_checkpoint
from ..data.dataset import MicroorganismDataset, get_validation_transforms
from ..utils.trainer import TrainingEngine
from ..utils.inference import InferenceEngine
from ..utils.stream import VideoStream
from ..utils.device import get_device, get_device_name

logger = logging.getLogger(__name__)

THEME = {
    "bg_dark": get_color_from_hex("#0f1923"),
    "bg_panel": get_color_from_hex("#1a2a3a"),
    "bg_card": get_color_from_hex("#223344"),
    "bg_input": get_color_from_hex("#2a3a4a"),
    "accent": get_color_from_hex("#00b4d8"),
    "accent_hover": get_color_from_hex("#0096c7"),
    "accent_dim": get_color_from_hex("#006d8a"),
    "success": get_color_from_hex("#2ecc71"),
    "warning": get_color_from_hex("#f39c12"),
    "danger": get_color_from_hex("#e74c3c"),
    "text": get_color_from_hex("#ecf0f1"),
    "text_dim": get_color_from_hex("#8899aa"),
    "text_muted": get_color_from_hex("#556677"),
    "border": get_color_from_hex("#334455"),
    "white": get_color_from_hex("#ffffff"),
}


class StyledLabel(Label):
    """A styled label with consistent theming."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = kwargs.get("color", THEME["text"])
        self.font_size = kwargs.get("font_size", sp(13))
        self.bold = kwargs.get("bold", False)


class StyledButton(Button):
    """A styled button with hover effect support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_down = ""
        self.background_disabled_normal = ""
        self.color = THEME["white"]
        self.bold = True
        self.font_size = kwargs.get("font_size", sp(13))
        self.size_hint_y = kwargs.get("size_hint_y", None)
        self.height = kwargs.get("height", dp(40))

    def on_press(self):
        self.background_color = THEME["accent_hover"]

    def on_release(self):
        self.background_color = THEME["accent"]


class StyledTextInput(TextInput):
    """A styled text input with consistent theming."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = THEME["bg_input"]
        self.foreground_color = THEME["text"]
        self.cursor_color = THEME["accent"]
        self.font_size = sp(13)
        self.padding = [dp(10), dp(8), dp(10), dp(8)]
        self.multiline = False


class PanelCard(BoxLayout):
    """A styled panel card with background and padding."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = kwargs.get("orientation", "vertical")
        with self.canvas.before:
            Color(*THEME["bg_card"])
            self._bg = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(8)]
            )
            Color(*THEME["border"])
            self._border = Line(
                rounded_rectangle=(*self.pos, *self.size, dp(8)),
                width=dp(0.5),
            )
        self.bind(pos=self._update_bg, size=self._update_bg)
        padding = kwargs.get("padding", dp(12))
        self.padding = [padding, padding, padding, padding]
        spacing = kwargs.get("spacing", dp(6))
        self.spacing = spacing

    def _update_bg(self, *args):
        self._bg.pos = self.pos
        self._bg.size = self.size
        self._border.rounded_rectangle = (*self.pos, *self.size, dp(8))


class SourcePanel(PanelCard):
    """Video source selection panel."""

    def __init__(self, app_controller, **kwargs):
        super().__init__(**kwargs)
        self.app = app_controller
        self.spacing = dp(8)
        self.add_widget(StyledLabel(
            text="VIDEO SOURCE",
            font_size=sp(14),
            bold=True,
            color=THEME["accent"],
            size_hint_y=None,
            height=dp(28),
            halign="left",
        ))

        row1 = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(40))
        self.source_type_btn = Button(
            text="TYPE: Video File",
            background_normal="",
            background_color=THEME["bg_input"],
            color=THEME["text"],
            font_size=sp(12),
            bold=True,
        )
        self.source_type_btn.bind(on_press=self._cycle_source_type)
        row1.add_widget(self.source_type_btn)

        self.source_types = ["Video File", "USB Webcam", "RTSP Stream"]
        self._current_type_idx = 0

        row2 = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(40))
        self.path_input = StyledTextInput(
            hint_text="Enter file path or URL...",
            size_hint_x=0.7,
        )
        row2.add_widget(self.path_input)

        browse_btn = StyledButton(
            text="Browse",
            size_hint_x=0.3,
            height=dp(36),
            background_color=THEME["accent_dim"],
        )
        browse_btn.bind(on_press=self._browse_file)
        row2.add_widget(browse_btn)

        btn_row = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(42))

        self.start_btn = StyledButton(
            text="START STREAM",
            background_color=THEME["success"],
            font_size=sp(12),
        )
        self.start_btn.bind(on_press=self._start_stream)
        btn_row.add_widget(self.start_btn)

        self.stop_btn = StyledButton(
            text="STOP",
            background_color=THEME["danger"],
            font_size=sp(12),
        )
        self.stop_btn.bind(on_press=self._stop_stream)
        btn_row.add_widget(self.stop_btn)

        self.add_widget(row1)
        self.add_widget(row2)
        self.add_widget(btn_row)

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
        source_type = self.source_types[self._current_type_idx]
        source = self.path_input.text.strip()
        if not source:
            self.app.show_error("Please enter a source path or URL")
            return
        if source_type == "USB Webcam" and source.isdigit():
            pass
        self.app.start_stream(source)

    def _stop_stream(self, *args):
        self.app.stop_stream()


class ClassificationPanel(PanelCard):
    """Classification results display panel."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spacing = dp(4)
        self._prob_bars = {}
        self._build_ui()

    def _build_ui(self):
        self.add_widget(StyledLabel(
            text="CLASSIFICATION RESULTS",
            font_size=sp(14),
            bold=True,
            color=THEME["accent"],
            size_hint_y=None,
            height=dp(28),
            halign="left",
        ))

        self.detection_label = StyledLabel(
            text="No detection",
            font_size=sp(16),
            bold=True,
            color=THEME["text"],
            size_hint_y=None,
            height=dp(30),
            halign="left",
            text_size=(None, None),
        )
        self.add_widget(self.detection_label)

        self.confidence_label = StyledLabel(
            text="Confidence: --",
            font_size=sp(12),
            color=THEME["text_dim"],
            size_hint_y=None,
            height=dp(22),
            halign="left",
        )
        self.add_widget(self.confidence_label)

        self.inference_label = StyledLabel(
            text="Inference: --ms",
            font_size=sp(11),
            color=THEME["text_muted"],
            size_hint_y=None,
            height=dp(20),
            halign="left",
        )
        self.add_widget(self.inference_label)

        sep = BoxLayout(size_hint_y=None, height=dp(4))
        with sep.canvas:
            Color(*THEME["border"])
            Line(points=[0, 0, dp(300), 0], width=dp(0.5))
        self.add_widget(sep)

        for cls_name in CLASSES:
            row = BoxLayout(
                orientation="horizontal",
                size_hint_y=None,
                height=dp(22),
                spacing=dp(6),
            )
            name_lbl = StyledLabel(
                text=cls_name[:16],
                font_size=sp(10),
                color=THEME["text_dim"],
                size_hint_x=0.35,
                halign="left",
            )
            name_lbl.bind(size=name_lbl.setter("text_size"))

            bar_container = BoxLayout(
                size_hint_x=0.5,
                size_hint_y=None,
                height=dp(14),
            )
            with bar_container.canvas:
                Color(*THEME["bg_input"])
                RoundedRectangle(
                    pos=bar_container.pos,
                    size=bar_container.size,
                    radius=[dp(3)],
                )
            bar_widget = BoxLayout(size_hint_x=0.0)
            with bar_widget.canvas.before:
                Color(*THEME["accent"])
                self._rect = RoundedRectangle(
                    pos=bar_widget.pos,
                    size=bar_widget.size,
                    radius=[dp(3)],
                )
            bar_widget.bind(pos=self._update_rect, size=self._update_rect)
            bar_container.add_widget(bar_widget)

            pct_lbl = StyledLabel(
                text="0.0%",
                font_size=sp(10),
                color=THEME["text_muted"],
                size_hint_x=0.15,
                halign="left",
            )
            row.add_widget(name_lbl)
            row.add_widget(bar_container)
            row.add_widget(pct_lbl)
            self.add_widget(row)
            self._prob_bars[cls_name] = (bar_widget, pct_lbl)

    def _update_rect(self, *args):
        pass

    def update_results(
        self, predicted_class: str, confidence: float,
        all_probs: Dict[str, float], inference_time: float,
    ):
        self.detection_label.text = predicted_class
        if confidence > 0.8:
            self.detection_label.color = THEME["success"]
        elif confidence > 0.5:
            self.detection_label.color = THEME["warning"]
        else:
            self.detection_label.color = THEME["danger"]

        self.confidence_label.text = f"Confidence: {confidence*100:.1f}%"
        self.inference_label.text = f"Inference: {inference_time:.1f}ms"

        max_prob = max(all_probs.values()) if all_probs else 1.0

        for cls_name, (bar_widget, pct_lbl) in self._prob_bars.items():
            prob = all_probs.get(cls_name, 0.0)
            pct_lbl.text = f"{prob*100:.1f}%"

            bar_width = max(0.01, prob / max_prob) if max_prob > 0 else 0
            bar_widget.size_hint_x = bar_width
            if cls_name == predicted_class:
                bar_widget.canvas.before.clear()
                with bar_widget.canvas.before:
                    Color(*THEME["accent"])
                    RoundedRectangle(
                        pos=bar_widget.pos, size=bar_widget.size, radius=[dp(3)],
                    )
            else:
                bar_widget.canvas.before.clear()
                with bar_widget.canvas.before:
                    Color(*THEME["accent_dim"])
                    RoundedRectangle(
                        pos=bar_widget.pos, size=bar_widget.size, radius=[dp(3)],
                    )

    def clear_results(self):
        self.detection_label.text = "No detection"
        self.detection_label.color = THEME["text"]
        self.confidence_label.text = "Confidence: --"
        self.inference_label.text = "Inference: --ms"
        for cls_name, (bar_widget, pct_lbl) in self._prob_bars.items():
            pct_lbl.text = "0.0%"
            bar_widget.size_hint_x = 0.01


class TrainingPanel(PanelCard):
    """Model training control panel."""

    def __init__(self, app_controller, **kwargs):
        super().__init__(**kwargs)
        self.app = app_controller
        self.spacing = dp(6)
        self.is_training = False

        self.add_widget(StyledLabel(
            text="MODEL TRAINING",
            font_size=sp(14),
            bold=True,
            color=THEME["accent"],
            size_hint_y=None,
            height=dp(28),
            halign="left",
        ))

        path_row = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(38))
        self.dataset_path_input = StyledTextInput(
            hint_text="Dataset path...",
            size_hint_x=0.7,
            text=DEFAULT_DATASET_PATH,
        )
        path_row.add_widget(self.dataset_path_input)
        browse_ds_btn = StyledButton(
            text="Browse",
            size_hint_x=0.3,
            height=dp(34),
            background_color=THEME["accent_dim"],
            font_size=sp(11),
        )
        browse_ds_btn.bind(on_press=self._browse_dataset)
        path_row.add_widget(browse_ds_btn)
        self.add_widget(path_row)

        params_row = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(38))
        self.epochs_input = StyledTextInput(
            hint_text="Epochs", text="50", size_hint_x=0.33,
        )
        self.lr_input = StyledTextInput(
            hint_text="Learning Rate", text="0.001", size_hint_x=0.33,
        )
        self.batch_input = StyledTextInput(
            hint_text="Batch Size", text="32", size_hint_x=0.34,
        )
        params_row.add_widget(self.epochs_input)
        params_row.add_widget(self.lr_input)
        params_row.add_widget(self.batch_input)
        self.add_widget(params_row)

        btn_row = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(42))
        self.train_btn = StyledButton(
            text="START TRAINING",
            background_color=THEME["success"],
            font_size=sp(12),
        )
        self.train_btn.bind(on_press=self._start_training)
        btn_row.add_widget(self.train_btn)

        self.stop_train_btn = StyledButton(
            text="STOP",
            background_color=THEME["danger"],
            font_size=sp(12),
        )
        self.stop_train_btn.bind(on_press=self._stop_training)
        btn_row.add_widget(self.stop_train_btn)
        self.add_widget(btn_row)

        self.progress_label = StyledLabel(
            text="Ready to train",
            font_size=sp(11),
            color=THEME["text_dim"],
            size_hint_y=None,
            height=dp(20),
            halign="left",
        )
        self.add_widget(self.progress_label)

        self.stats_label = StyledLabel(
            text="",
            font_size=sp(10),
            color=THEME["text_muted"],
            size_hint_y=None,
            height=dp(40),
            halign="left",
            text_size=(dp(320), None),
            valign="top",
        )
        self.add_widget(self.stats_label)

        model_row = BoxLayout(spacing=dp(6), size_hint_y=None, height=dp(38))
        self.model_path_input = StyledTextInput(
            hint_text="Model checkpoint path...",
            size_hint_x=0.65,
            text=DEFAULT_MODEL_PATH,
        )
        model_row.add_widget(self.model_path_input)
        load_btn = StyledButton(
            text="Load Model",
            size_hint_x=0.35,
            height=dp(34),
            background_color=THEME["accent_dim"],
            font_size=sp(11),
        )
        load_btn.bind(on_press=self._load_model)
        model_row.add_widget(load_btn)
        self.add_widget(model_row)

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

    def _load_model(self, *args):
        path = self.model_path_input.text.strip()
        if not path:
            self.app.show_error("Enter model checkpoint path")
            return
        self.app.load_model(path)

    def update_progress(self, epoch: int, total: int, metrics: dict):
        self.progress_label.text = f"Epoch {epoch}/{total}"
        self.stats_label.text = (
            f"Train Loss: {metrics['train_loss']:.4f}  Acc: {metrics['train_acc']:.1f}%\n"
            f"Val Loss: {metrics['val_loss']:.4f}  Acc: {metrics['val_acc']:.1f}%\n"
            f"Best: {metrics['best_acc']:.1f}%  LR: {metrics['lr']:.6f}"
        )


class StatusBar(BoxLayout):
    """Bottom status bar."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(30)
        self.padding = [dp(12), dp(4)]
        with self.canvas.before:
            Color(*THEME["bg_dark"])
            Rectangle(pos=self.pos, size=self.size)
            Color(*THEME["border"])
            Line(points=[0, self.height, Window.width, self.height], width=dp(0.5))
        self.bind(pos=self._update_bg, size=self._update_bg)

        self.status_label = StyledLabel(
            text="Ready",
            font_size=sp(11),
            color=THEME["text_dim"],
            size_hint_x=0.5,
            halign="left",
        )
        self.fps_label = StyledLabel(
            text="FPS: --",
            font_size=sp(11),
            color=THEME["text_muted"],
            size_hint_x=0.25,
        )
        self.device_label = StyledLabel(
            text="Device: CPU",
            font_size=sp(11),
            color=THEME["text_muted"],
            size_hint_x=0.25,
        )
        self.add_widget(self.status_label)
        self.add_widget(self.fps_label)
        self.add_widget(self.device_label)

    def _update_bg(self, *args):
        pass

    def set_status(self, text: str, color=None):
        self.status_label.text = text
        if color:
            self.status_label.color = color

    def set_fps(self, fps: float):
        self.fps_label.text = f"FPS: {fps:.1f}"

    def set_device(self, device: str):
        self.device_label.text = f"Device: {device}"


class VideoDisplay(FloatLayout):
    """Video display area with overlay."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(*THEME["bg_dark"])
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)

        self.video_image = KivyImage(
            allow_stretch=True,
            keep_ratio=True,
        )
        self.add_widget(self.video_image)

        self.placeholder_label = StyledLabel(
            text="[ No Video Source ]",
            font_size=sp(18),
            color=THEME["text_muted"],
            halign="center",
        )
        self.add_widget(self.placeholder_label)

    def _update_bg(self, *args):
        self._bg.pos = self.pos
        self._bg.size = self.size

    def update_frame(self, frame: np.ndarray):
        """Update the displayed frame from an OpenCV BGR numpy array."""
        buf = cv2.flip(frame, 0)
        buf = buf.tobytes()
        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
        )
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.video_image.texture = texture
        self.placeholder_label.opacity = 0

    def clear_display(self):
        self.video_image.texture = None
        self.placeholder_label.opacity = 1


class MicroClassifyApp(App):
    """Main application class."""

    title = "MicroClassify - Microorganism Classification System"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.augment_config = AugmentationConfig()
        self.stream_config = StreamConfig()

        self.video_stream: Optional[VideoStream] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.model: Optional[object] = None
        self.training_engine: Optional[TrainingEngine] = None
        self._training_thread: Optional[threading.Thread] = None
        self._is_streaming = False
        self._stream_event = None

    def build(self):
        Window.size = (dp(1400), dp(850))
        Window.minimum_width = dp(1100)
        Window.minimum_height = dp(700)
        Window.clearcolor = THEME["bg_dark"]

        root = BoxLayout(
            orientation="vertical",
            padding=dp(4),
            spacing=dp(4),
        )

        header = self._build_header()
        root.add_widget(header)

        main_content = BoxLayout(spacing=dp(4))

        left_panel = BoxLayout(
            orientation="vertical",
            size_hint_x=0.25,
            spacing=dp(4),
        )
        source_panel = SourcePanel(self, size_hint_y=None, height=dp(200))
        self.source_panel = source_panel
        left_panel.add_widget(source_panel)

        training_panel = TrainingPanel(self)
        self.training_panel = training_panel
        left_panel.add_widget(training_panel)

        main_content.add_widget(left_panel)

        center = BoxLayout(orientation="vertical", size_hint_x=0.50, spacing=dp(4))
        self.video_display = VideoDisplay()
        center.add_widget(self.video_display)
        main_content.add_widget(center)

        right_panel = BoxLayout(
            orientation="vertical",
            size_hint_x=0.25,
            spacing=dp(4),
        )
        self.classification_panel = ClassificationPanel()
        right_panel.add_widget(self.classification_panel)
        main_content.add_widget(right_panel)

        root.add_widget(main_content)

        self.status_bar = StatusBar()
        device_name = get_device_name()
        self.status_bar.set_device(device_name)
        root.add_widget(self.status_bar)

        Clock.schedule_interval(self._update_stream, 1.0 / 30)

        return root

    def _build_header(self):
        header = BoxLayout(
            size_hint_y=None,
            height=dp(52),
            padding=[dp(16), dp(8), dp(16), dp(8)],
        )
        with header.canvas.before:
            Color(*THEME["bg_panel"])
            Rectangle(pos=header.pos, size=header.size)
            Color(*THEME["accent"])
            Line(points=[0, 0, Window.width, 0], width=dp(1.5))
        header.bind(
            pos=lambda inst, val: header.canvas.before.clear()
            or header.canvas.before.add(Color(*THEME["bg_panel"]))
            or header.canvas.before.add(Rectangle(pos=inst.pos, size=inst.size)),
            size=lambda inst, val: header.canvas.before.clear()
            or header.canvas.before.add(Color(*THEME["bg_panel"]))
            or header.canvas.before.add(Rectangle(pos=inst.pos, size=inst.size)),
        )

        title_label = StyledLabel(
            text="MICROCLASSIFY",
            font_size=sp(20),
            bold=True,
            color=THEME["accent"],
            size_hint_x=0.4,
            halign="left",
        )
        subtitle_label = StyledLabel(
            text="Microorganism Image Classification System",
            font_size=sp(12),
            color=THEME["text_dim"],
            size_hint_x=0.4,
            halign="left",
        )
        version_label = StyledLabel(
            text="v1.0 | EfficientNet-B4",
            font_size=sp(11),
            color=THEME["text_muted"],
            size_hint_x=0.2,
            halign="right",
        )
        header.add_widget(title_label)
        header.add_widget(subtitle_label)
        header.add_widget(version_label)
        return header

    def open_file_chooser(self):
        content = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(10))
        file_chooser = FileChooserIconView(
            path=os.path.expanduser("~"),
            filters=["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv"],
        )
        file_chooser.bind(on_submit=self._on_file_selected)
        content.add_widget(file_chooser)

        btn_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        cancel_btn = StyledButton(
            text="Cancel", background_color=THEME["danger"], font_size=sp(12),
        )
        btn_row.add_widget(cancel_btn)
        content.add_widget(btn_row)

        popup = Popup(
            title="Select Video File",
            content=content,
            size_hint=(0.85, 0.85),
            auto_dismiss=True,
        )
        cancel_btn.bind(on_press=popup.dismiss)
        file_chooser.bind(on_submit=lambda *a: popup.dismiss())
        self._file_popup = popup
        popup.open()

    def _on_file_selected(self, file_chooser, selection, *args):
        if selection:
            self.source_panel.path_input.text = selection[0]
            self.source_panel._current_type_idx = 0
            self.source_panel.source_type_btn.text = "TYPE: Video File"

    def open_dataset_chooser(self):
        content = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(10))
        file_chooser = FileChooserIconView(
            path=os.path.expanduser("~"),
            dirselect=True,
        )
        file_chooser.bind(on_submit=self._on_dataset_selected)
        content.add_widget(file_chooser)

        btn_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        cancel_btn = StyledButton(
            text="Cancel", background_color=THEME["danger"], font_size=sp(12),
        )
        btn_row.add_widget(cancel_btn)
        content.add_widget(btn_row)

        popup = Popup(
            title="Select Dataset Directory",
            content=content,
            size_hint=(0.85, 0.85),
            auto_dismiss=True,
        )
        cancel_btn.bind(on_press=popup.dismiss)
        file_chooser.bind(on_submit=lambda *a: popup.dismiss())
        self._ds_popup = popup
        popup.open()

    def _on_dataset_selected(self, file_chooser, selection, *args):
        if selection:
            self.training_panel.dataset_path_input.text = selection[0]

    def start_stream(self, source: str):
        self.stop_stream()
        self.video_stream = VideoStream(source=source)
        if not self.video_stream.start():
            self.show_error(f"Failed to open: {source}")
            return
        self._is_streaming = True
        self.status_bar.set_status(
            f"Streaming: {os.path.basename(source) if not source.startswith(('rtsp', 'rtmp')) else source[:40]}",
            THEME["success"],
        )

    def stop_stream(self):
        self._is_streaming = False
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None
        self.video_display.clear_display()
        self.classification_panel.clear_results()
        self.status_bar.set_status("Stream stopped", THEME["warning"])

    def load_model(self, checkpoint_path: str):
        try:
            self.status_bar.set_status("Loading model...", THEME["warning"])
            transform = get_validation_transforms(self.augment_config)
            self.inference_engine = InferenceEngine.from_checkpoint(
                checkpoint_path, transform
            )
            self.inference_engine.warmup(self.model_config.input_size)
            self.model = self.inference_engine.model
            self.status_bar.set_status("Model loaded successfully", THEME["success"])
            logger.info(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            self.show_error(f"Failed to load model: {str(e)}")
            logger.error(f"Model load error: {e}")

    def start_training(self, dataset_path: str, epochs: int, lr: float, batch_size: int):
        if self.training_engine and hasattr(self.training_engine, '_stop_training'):
            self.show_error("Training already in progress")
            return
        self.training_config.epochs = epochs
        self.training_config.learning_rate = lr
        self.training_config.batch_size = batch_size

        self.training_panel.is_training = True
        self.training_panel.train_btn.background_color = THEME["text_muted"]
        self.training_panel.progress_label.text = "Initializing training..."

        self._training_thread = threading.Thread(
            target=self._run_training, args=(dataset_path,), daemon=True,
        )
        self._training_thread.start()

    def _run_training(self, dataset_path: str):
        try:
            device = get_device()

            dataset = MicroorganismDataset(
                dataset_path, self.model_config, self.augment_config, self.training_config,
            )
            train_loader, val_loader, stats = dataset.get_dataloaders()

            self.model = create_model(
                num_classes=stats["num_classes"],
                pretrained=True,
                dropout_rate=self.model_config.dropout_rate,
                device=device,
            )

            self.training_engine = TrainingEngine(
                self.model, self.training_config, device,
            )

            def progress_cb(epoch, total, metrics):
                Clock.schedule_once(
                    lambda dt: self.training_panel.update_progress(epoch, total, metrics),
                )

            history = self.training_engine.train(
                train_loader, val_loader,
                class_to_idx=stats["class_to_idx"],
                save_dir=self.training_config.save_dir,
                checkpoint_name=self.training_config.checkpoint_name,
                progress_callback=progress_cb,
            )

            Clock.schedule_once(lambda dt: self._on_training_complete(history, stats))

        except Exception as e:
            logger.error(f"Training error: {e}")
            Clock.schedule_once(
                lambda dt: self.show_error(f"Training failed: {str(e)}"),
            )
            Clock.schedule_once(
                lambda dt: setattr(self.training_panel, "is_training", False),
            )

    def _on_training_complete(self, history: dict, stats: dict):
        self.training_panel.is_training = False
        self.training_panel.train_btn.background_color = THEME["success"]
        best_acc = history.get("best_val_acc", 0)
        self.training_panel.progress_label.text = f"Training complete! Best: {best_acc:.1f}%"
        self.status_bar.set_status(
            f"Training complete. Best accuracy: {best_acc:.1f}%", THEME["success"],
        )
        model_path = os.path.join(
            self.training_config.save_dir, self.training_config.checkpoint_name,
        )
        self.training_panel.model_path_input.text = model_path
        self.load_model(model_path)

    def stop_training(self):
        if self.training_engine:
            self.training_engine.stop_training()
            self.training_panel.progress_label.text = "Stopping..."
            self.status_bar.set_status("Stopping training...", THEME["warning"])

    def _update_stream(self, dt):
        if not self._is_streaming or self.video_stream is None:
            return
        frame = self.video_stream.get_frame()
        if frame is None:
            return

        fps = self.video_stream.fps
        self.status_bar.set_fps(fps)

        if self.inference_engine is not None:
            pred_class, confidence, all_probs, inf_time = (
                self.inference_engine.predict_frame_with_timing(frame)
            )
            annotated = self.inference_engine.annotate_frame(
                frame, pred_class, confidence, all_probs, inf_time,
            )
            self.classification_panel.update_results(
                pred_class, confidence, all_probs, inf_time,
            )
            display_frame = annotated
        else:
            display_frame = frame

        h, w = display_frame.shape[:2]
        max_display_w = int(Window.width * 0.50)
        max_display_h = int(Window.height * 0.60)
        scale = min(max_display_w / w, max_display_h / h)
        if scale < 1:
            display_frame = cv2.resize(
                display_frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        self.video_display.update_frame(display_frame)

    def show_error(self, message: str):
        content = BoxLayout(
            orientation="vertical",
            padding=dp(20),
            spacing=dp(12),
        )
        content.add_widget(StyledLabel(
            text=message,
            font_size=sp(13),
            color=THEME["text"],
            size_hint_y=0.7,
            text_size=(dp(400), None),
        ))
        btn = StyledButton(
            text="OK",
            size_hint_y=None,
            height=dp(38),
            size_hint_x=0.4,
            background_color=THEME["accent"],
        )
        content.add_widget(btn)

        popup = Popup(
            title="Error",
            content=content,
            size_hint=(0.5, 0.35),
            auto_dismiss=True,
            title_color=THEME["danger"],
        )
        btn.bind(on_press=popup.dismiss)
        popup.open()

    def on_stop(self):
        self.stop_stream()
        if self.training_engine:
            self.training_engine.stop_training()
