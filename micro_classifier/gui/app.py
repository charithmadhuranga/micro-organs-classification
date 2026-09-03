"""MicroClassifyApp: Kivy application integrating the classification backend."""

import logging
import os
import threading

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup

from ..config import (
    AugmentationConfig, DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, GUIConfig,
    ModelConfig, StreamConfig, TrainingConfig,
)
from ..utils.device import get_device_name
from ..utils.inference import InferenceEngine
from ..utils.stream import VideoStream
from ..utils.trainer import TrainingEngine
from .layout import HeaderBar, ResponsiveWorkspace, StatusBar
from .panels import (
    ClassificationPanel, SourcePanel, TrainingPanel, VideoDisplay,
)
from .theme import THEME
from .widgets import StyledButton, StyledLabel

logger = logging.getLogger(__name__)


class MicroClassifyApp(App):
    """Main application entry point."""

    title = "MicroClassify - Microorganism Classification System"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gui_config = GUIConfig()
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.augment_config = AugmentationConfig()
        self.stream_config = StreamConfig()

        self.default_model_path = DEFAULT_MODEL_PATH
        self.default_dataset_path = DEFAULT_DATASET_PATH

        self.video_stream = None
        self.inference_engine = None
        self.model = None
        self.training_engine = None
        self._training_thread = None
        self._is_streaming = False
        self._stream_event = None

    def build(self):
        self._size_window_to_fit_screen()
        Window.clearcolor = THEME["bg_dark"]

        root = BoxLayout(
            orientation="vertical",
            padding=[dp(14), dp(12), dp(14), dp(10)],
            spacing=dp(8),
        )

        self.header = HeaderBar(self.title)
        root.add_widget(self.header)

        self.status_bar = StatusBar()
        root.add_widget(self.status_bar)

        self.source_panel = SourcePanel(self, size_hint_y=None, height=dp(200))
        self.training_panel = TrainingPanel(self)
        self.classification_panel = ClassificationPanel()
        self.video_display = VideoDisplay()

        self.workspace = ResponsiveWorkspace(
            self.source_panel,
            self.video_display,
            self.training_panel,
            self.classification_panel,
            size_hint_y=1.0,
        )
        root.add_widget(self.workspace)

        return root

    # ------------------------------------------------------------------ #
    # Window sizing
    # ------------------------------------------------------------------ #
    def _size_window_to_fit_screen(self):
        try:
            sw, sh = Window.system_size
        except Exception:
            sw, sh = 0, 0
        import sys
        if sys.platform == "darwin" and (sw < 1024 or sh < 768):
            try:
                import subprocess, re
                out = subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType"],
                    timeout=5, text=True,
                )
                m = re.search(r"Resolution:\s+(\d+)\s+x\s+(\d+)", out)
                if m:
                    sw, sh = int(m.group(1)), int(m.group(2))
            except Exception:
                sw, sh = 2560, 1600
        if not sw or not sh:
            sw, sh = 1920, 1080
        if sys.platform == "darwin":
            win_w, win_h = sw, sh
        else:
            scale = max(1.0, Window.dpi / 96.0) if Window.dpi else 1.0
            win_w = max(int(sw / scale), int(dp(1000)))
            win_h = max(int(sh / scale), int(dp(600)))
        try:
            Window.size = (win_w, win_h)
        except Exception:
            pass
        Window.minimum_width = dp(820)
        Window.minimum_height = dp(600)

    # ------------------------------------------------------------------ #
    # Stream control
    # ------------------------------------------------------------------ #
    def start_stream(self, source: str):
        self.stop_stream()
        self.video_stream = VideoStream(source=source)
        if not self.video_stream.start():
            self.show_error(f"Failed to open: {source}")
            return
        self._is_streaming = True
        source_type = self.video_stream.source_type.upper()
        self.status_bar.set_stream(
            f"{source_type} | {os.path.basename(source) or source}", THEME["success"]
        )
        self.header.set_status("STREAMING", THEME["success"])

    def stop_stream(self):
        self._is_streaming = False
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None
        self.video_display.clear_display()
        self.classification_panel.clear_results()
        self.status_bar.set_stream("idle", THEME["text_muted"])
        self.header.set_status("READY", THEME["text_dim"])

    def _update_stream(self, dt):
        if not self._is_streaming or self.video_stream is None:
            return
        frame = self.video_stream.get_frame()
        if frame is None:
            return
        self.status_bar.set_fps(self.video_stream.fps or 0.0)

        if self.inference_engine is not None:
            try:
                pred_class, confidence, all_probs, inf_time = (
                    self.inference_engine.predict_frame_with_timing(frame)
                )
                self.classification_panel.update_results(
                    pred_class, confidence, all_probs, inf_time,
                )
            except Exception:
                logger.exception("Inference failed")

        self.video_display.update_frame(frame)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    def load_model(self, checkpoint_path: str):
        try:
            from ..data.dataset import get_validation_transforms
            self.status_bar.set_mode("Loading model...", THEME["warning"])
            transform = get_validation_transforms(self.augment_config)
            self.inference_engine = InferenceEngine.from_checkpoint(
                checkpoint_path, transform
            )
            self.inference_engine.warmup(self.model_config.input_size)
            self.model = self.inference_engine.model
            self.status_bar.set_mode(
                f"Inference | {get_device_name()}", THEME["accent"]
            )
            self.header.set_status("MODEL READY", THEME["success"])
            logger.info(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            self.show_error(f"Failed to load model: {e}")
            logger.exception("Model load error")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def start_training(self, dataset_path, epochs, lr, batch_size):
        if self.training_engine and self._training_thread and self._training_thread.is_alive():
            self.show_error("Training already in progress")
            return

        self.training_config.epochs = epochs
        self.training_config.learning_rate = lr
        self.training_config.batch_size = batch_size

        self.training_panel.set_training(True)
        self.training_panel.progress_label.text = "Preparing dataset..."

        def run_training():
            try:
                self._train(dataset_path)
            except Exception:
                logger.exception("Training failed")
                Clock.schedule_once(
                    lambda dt: self._training_done(error=True), 0
                )

        self._training_thread = threading.Thread(target=run_training, daemon=True)
        self._training_thread.start()

    def _train(self, dataset_path):
        from ..data.dataset import MicroorganismDataset
        from ..models.classifier import create_model

        ds = MicroorganismDataset(
            dataset_path, self.model_config, self.augment_config,
            self.training_config,
        )
        train_loader, val_loader, stats = ds.get_dataloaders()

        self.model_config.num_classes = stats["num_classes"]
        model = create_model(self.model_config)
        self.training_engine = TrainingEngine(model, self.training_config)
        class_to_idx = stats["class_to_idx"]

        self.training_engine.train(
            train_loader, val_loader, class_to_idx,
            save_dir=self.training_config.save_dir,
            checkpoint_name=self.training_config.checkpoint_name,
            progress_callback=lambda ep, tot, metrics: Clock.schedule_once(
                lambda dt, e=ep, t=tot, m=metrics: self.training_panel.update_progress(
                    e, t, m
                ),
                0,
            ),
        )
        Clock.schedule_once(lambda dt: self._training_done(), 0)

    def _training_done(self, error=False):
        self.training_panel.set_training(False)
        if error:
            self.training_panel.progress_label.text = "Training failed"
        else:
            best = getattr(self.training_engine, "best_val_acc", 0.0)
            self.training_panel.progress_label.text = f"Training complete! Best: {best:.1f}%"
        self.training_engine = None

    def stop_training(self):
        if self.training_engine:
            self.training_engine.stop_training()
            self.training_panel.progress_label.text = "Stopping..."

    # ------------------------------------------------------------------ #
    # Dialogs
    # ------------------------------------------------------------------ #
    def open_file_chooser(self):
        self._open_chooser(self._on_file_chosen)

    def open_dataset_chooser(self):
        self._open_chooser(self._on_dataset_chosen)

    def open_model_chooser(self):
        self._open_chooser(self._on_model_chosen, filters=["*.pth", "*.pt", "*.onnx"])

    def _open_chooser(self, on_select, filters=None):
        from kivy.uix.filechooser import FileChooserIconView

        chooser = FileChooserIconView(size_hint_y=0.9)
        if filters:
            chooser.filters = filters
        box = BoxLayout(orientation="vertical", padding=dp(8))
        box.add_widget(chooser)
        btn_row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(8))
        ok = StyledButton(text="Select", background_color=THEME["accent"], size_hint_x=0.5)
        cancel = StyledButton(text="Cancel", background_color=THEME["text_muted"], size_hint_x=0.5)
        btn_row.add_widget(ok)
        btn_row.add_widget(cancel)
        box.add_widget(btn_row)

        popup = Popup(title="Choose", content=box, size_hint=(0.6, 0.7))
        ok.bind(on_press=lambda *a: (on_select(chooser.selection), popup.dismiss()))
        cancel.bind(on_press=popup.dismiss)
        popup.open()

    def _on_file_chosen(self, selection):
        if selection:
            self.source_panel.path_input.text = selection[0]

    def _on_dataset_chosen(self, selection):
        if selection:
            self.training_panel.dataset_path_input.text = selection[0]

    def _on_model_chosen(self, selection):
        if selection:
            path = selection[0]
            self.training_panel.model_path_input.text = path
            self.load_model(path)

    def show_error(self, message: str):
        content = BoxLayout(orientation="vertical", padding=dp(18), spacing=dp(12))
        content.add_widget(StyledLabel(
            text=message, font_size=sp(13), color=THEME["text"], halign="left",
            valign="middle", size_hint_y=0.7, text_size=(dp(380), None),
        ))
        btn = StyledButton(text="OK", size_hint_y=None, height=dp(38), size_hint_x=0.4,
                           background_color=THEME["accent"])
        content.add_widget(btn)
        popup = Popup(title="Error", content=content, size_hint=(0.5, 0.35),
                      title_color=THEME["danger"])
        btn.bind(on_press=popup.dismiss)
        popup.open()

    # ------------------------------------------------------------------ #
    def on_stop(self):
        self.stop_stream()
        if self.training_engine:
            self.training_engine.stop_training()
