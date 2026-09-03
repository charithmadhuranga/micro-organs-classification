"""Reusable styled widgets and card helpers for the MicroClassify GUI."""

from kivy.graphics import Color, RoundedRectangle, Line
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput

from .theme import THEME


def paint_card(widget, radius=8, bg_color=None, border_color=None):
    """Draw a rounded card background + border on a widget's canvas.before.

    Binds itself to pos/size so it tracks the widget automatically. Returns the
    background RoundedRectangle for direct manipulation if needed.
    """
    widget._card_rect = None
    widget._card_radius = radius
    widget._card_bg = bg_color or THEME["bg_card"]
    widget._card_border = border_color or THEME["border"]

    def _repaint(*args):
        widget.canvas.before.clear()
        with widget.canvas.before:
            Color(*widget._card_bg)
            widget._card_rect = RoundedRectangle(
                pos=widget.pos, size=widget.size, radius=[dp(widget._card_radius)]
            )
            Color(*widget._card_border)
            Line(
                rounded_rectangle=(
                    widget.x, widget.y, widget.width, widget.height,
                    dp(widget._card_radius),
                ),
                width=dp(1),
            )

    widget.bind(pos=_repaint, size=_repaint)
    _repaint()
    return widget._card_rect


class StyledLabel(Label):
    """Label with theme-friendly defaults."""

    def __init__(
        self,
        text="",
        font_size=sp(13),
        color=None,
        bold=False,
        halign="left",
        valign="center",
        **kwargs,
    ):
        kwargs.setdefault("size_hint_x", 1.0)
        super().__init__(
            text=text,
            font_size=font_size,
            color=color or THEME["text"],
            bold=bold,
            halign=halign,
            valign=valign,
            **kwargs,
        )
        if halign in ("left", "right"):
            self.bind(size=self._set_text_size)

    def _set_text_size(self, *args):
        self.text_size = (self.width, None)


class StyledButton(Button):
    """Rounded button with theme-friendly colors."""

    def __init__(
        self,
        text="",
        background_color=None,
        color=None,
        font_size=sp(12),
        bold=True,
        size_hint_x=0.5,
        size_hint_y=None,
        height=dp(38),
        **kwargs,
    ):
        if size_hint_y is None:
            kwargs.setdefault("height", height)
        super().__init__(
            text=text,
            background_normal="",
            background_down="",
            background_color=background_color or THEME["accent"],
            color=color or THEME["text"],
            font_size=font_size,
            bold=bold,
            size_hint_x=size_hint_x,
            size_hint_y=size_hint_y,
            **kwargs,
        )


class StyledTextInput(TextInput):
    """Text input with rounded, theme-friendly background."""

    def __init__(self, hint_text="", text="", font_size=sp(12), **kwargs):
        super().__init__(
            hint_text=hint_text,
            text=text,
            background_normal="",
            background_active="",
            foreground_color=THEME["white"],
            hint_text_color=THEME["text_muted"],
            cursor_color=THEME["accent"],
            font_size=font_size,
            padding=[dp(8), dp(6)],
            multiline=False,
            **kwargs,
        )
        self.bind(pos=self._redraw, size=self._redraw)
        self._redraw()

    def _redraw(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*THEME["bg_input"])
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(6)])
            Color(*THEME["accent_dim"])
            Line(
                rounded_rectangle=(self.x, self.y, self.width, self.height, dp(6)),
                width=dp(1.2),
            )


class SectionHeader(BoxLayout):
    """Accent title row used at the top of each panel.

    Added LAST to a panel (a vertical BoxLayout) so it renders on top.
    """

    def __init__(self, title, subtitle="", **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None, height=dp(28), **kwargs)
        self.spacing = dp(6)

        accent = BoxLayout(size_hint_x=None, width=dp(4), size_hint_y=1.0)
        with accent.canvas.after:
            Color(*THEME["accent"])
            RoundedRectangle(pos=accent.pos, size=accent.size, radius=[dp(2)])
        accent.bind(
            pos=lambda w, *a: _accent_paint(w),
            size=lambda w, *a: _accent_paint(w),
        )
        self.add_widget(accent)

        title_box = BoxLayout(orientation="vertical", spacing=0)
        title_box.add_widget(StyledLabel(
            text=title,
            font_size=sp(13),
            bold=True,
            color=THEME["text"],
            halign="left",
            size_hint_y=0.6,
        ))
        if subtitle:
            title_box.add_widget(StyledLabel(
                text=subtitle,
                font_size=sp(9),
                color=THEME["text_muted"],
                halign="left",
                size_hint_y=0.4,
            ))
        self.add_widget(title_box)


def _accent_paint(widget):
    widget.canvas.after.clear()
    with widget.canvas.after:
        Color(*THEME["accent"])
        RoundedRectangle(pos=widget.pos, size=widget.size, radius=[dp(2)])


class PanelCard(BoxLayout):
    """A themed card container with a fixed internal height budget.

    Children must set size_hint_y=None + height (fixed controls) or be a single
    size_hint_y=1.0 filler (scroll area / bars column) that absorbs leftover
    space. This keeps the card from floating or showing empty regions.
    """

    _is_card_enabled = True

    def __init__(self, padding=(dp(10), dp(8)), spacing=dp(8), **kwargs):
        super().__init__(orientation="vertical", padding=padding, spacing=spacing, **kwargs)
        paint_card(self, radius=10, bg_color=THEME["bg_panel"], border_color=THEME["border"])
