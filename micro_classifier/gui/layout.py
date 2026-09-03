"""Top-level layout widgets: header, status bar and responsive workspace."""

from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle, Line
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout

from .theme import THEME
from .widgets import StyledLabel, paint_card


class HeaderBar(BoxLayout):
    """App title bar with a status pill on the right."""

    def __init__(self, title, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None,
                         height=dp(52), padding=[dp(6), dp(4)], spacing=dp(10),
                         **kwargs)
        paint_card(self, radius=10, bg_color=THEME["bg_panel"], border_color=THEME["border"])

        self.add_widget(StyledLabel(
            text=title,
            font_size=sp(18),
            bold=True,
            color=THEME["accent"],
            halign="left",
            size_hint_x=0.7,
        ))

        self.status_pill = StyledLabel(
            text="READY",
            font_size=sp(11),
            bold=True,
            color=THEME["text_dim"],
            halign="right",
            size_hint_x=0.3,
        )
        self.add_widget(self.status_pill)

    def set_status(self, text, color=None):
        self.status_pill.text = text
        self.status_pill.color = color or THEME["text_dim"]


class StatusBar(BoxLayout):
    """Thin strip showing stream state, mode and FPS."""

    def __init__(self, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None,
                         height=dp(28), spacing=dp(16), **kwargs)

        self.stream_label = StyledLabel(
            text="Stream: idle",
            font_size=sp(11),
            color=THEME["text_muted"],
            size_hint_x=0.34,
            halign="left",
        )
        self.add_widget(self.stream_label)

        self.mode_label = StyledLabel(
            text="Mode: Inference",
            font_size=sp(11),
            color=THEME["text_muted"],
            size_hint_x=0.33,
            halign="center",
        )
        self.add_widget(self.mode_label)

        self.fps_label = StyledLabel(
            text="FPS: 0",
            font_size=sp(11),
            color=THEME["text_muted"],
            size_hint_x=0.33,
            halign="right",
        )
        self.add_widget(self.fps_label)

    def set_stream(self, text, color=None):
        self.stream_label.text = f"Stream: {text}"
        self.stream_label.color = color or THEME["text_muted"]

    def set_mode(self, text, color=None):
        self.mode_label.text = f"Mode: {text}"
        self.mode_label.color = color or THEME["text_muted"]

    def set_fps(self, fps):
        self.fps_label.text = f"FPS: {fps:.1f}"


class ResponsiveWorkspace(BoxLayout):
    """Switches between a 3-column desktop layout and a stacked layout.

    The panels are added to the workspace EXACTLY ONCE and are never
    re-parented. Resizing only mutates orientation, size_hints and fixed
    heights, so the BoxLayout's own layout algorithm always stays consistent
    (no "already has a parent" and no layout corruption).
    """

    BREAKPOINT = dp(1100)

    def __init__(self, source_panel, video_display, training_panel,
                 classification_panel, **kwargs):
        super().__init__(spacing=dp(12), **kwargs)
        self.source_panel = source_panel
        self.video_display = video_display
        self.training_panel = training_panel
        self.classification_panel = classification_panel

        # Left column groups source (top) and training (bottom). Created once.
        self.left = BoxLayout(orientation="vertical", spacing=dp(10), size_hint_x=0.26)
        self.left.add_widget(self.source_panel)
        self.left.add_widget(self.training_panel)

        # The workspace ALWAYS owns these three children in this order.
        self.add_widget(self.left)
        self.add_widget(self.video_display)
        self.add_widget(self.classification_panel)

        self.bind(size=self._responsive_layout)
        Clock.schedule_once(lambda dt: self._responsive_layout(), 0)

    def _responsive_layout(self, *args):
        wide = self.width >= self.BREAKPOINT
        if wide:
            self.orientation = "horizontal"
            self.spacing = dp(12)
            self.left.size_hint_x = 0.26
            self.left.orientation = "vertical"

            self.source_panel.size_hint_x = 1.0
            self.source_panel.size_hint_y = None
            self.source_panel.height = dp(200)
            self.training_panel.size_hint_x = 1.0
            self.training_panel.size_hint_y = 1.0

            self.video_display.size_hint_x = 0.48
            self.video_display.size_hint_y = 1.0

            self.classification_panel.size_hint_x = 0.26
            self.classification_panel.size_hint_y = 1.0
        else:
            self.orientation = "vertical"
            self.spacing = dp(10)
            self.left.size_hint_x = 1.0
            self.left.orientation = "vertical"

            self.source_panel.size_hint_x = 1.0
            self.source_panel.size_hint_y = None
            self.source_panel.height = dp(200)
            self.training_panel.size_hint_x = 1.0
            self.training_panel.size_hint_y = None
            self.training_panel.height = dp(340)

            self.video_display.size_hint_x = 1.0
            self.video_display.size_hint_y = None
            self.video_display.height = max(dp(300), self.width * 0.56)

            self.classification_panel.size_hint_x = 1.0
            self.classification_panel.size_hint_y = None
            self.classification_panel.height = dp(360)
