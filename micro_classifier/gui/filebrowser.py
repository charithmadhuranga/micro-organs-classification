"""OS-like file browser dialog for the MicroClassify GUI."""

import os

from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserIconView, FileChooserListView
from kivy.uix.popup import Popup

from .theme import THEME
from .widgets import StyledButton, StyledLabel, StyledTextInput


class PathBar(BoxLayout):
    """Editable path bar."""

    __events__ = ("on_navigate",)

    def __init__(self, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None,
                         height=dp(34), spacing=dp(4), **kwargs)

        self._path_input = StyledTextInput(
            hint_text="Enter path...",
            font_size=sp(11),
            size_hint_x=1.0,
        )
        self._path_input.bind(on_text_validate=self._on_enter)
        self.add_widget(self._path_input)

        go_btn = StyledButton(
            text="Go",
            font_size=sp(10),
            size_hint_x=None,
            width=dp(48),
            background_color=THEME["accent"],
        )
        go_btn.bind(on_press=self._on_enter)
        self.add_widget(go_btn)

    def set_path(self, path):
        self._path_input.text = path

    def get_path(self):
        return self._path_input.text.strip()

    def _on_enter(self, *args):
        path = self.get_path()
        if path and os.path.isdir(path):
            self.dispatch("on_navigate", path)

    def on_navigate(self, path):
        pass


class NavToolbar(BoxLayout):
    """Navigation toolbar with back, forward, up, home buttons."""

    __events__ = ("on_back", "on_forward", "on_up", "on_home")

    def __init__(self, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None,
                         height=dp(36), spacing=dp(3), **kwargs)

        self._history = []
        self._history_index = -1

        btn_style = dict(
            font_size=sp(11),
            size_hint_x=None,
            width=dp(36),
            background_color=THEME["bg_input"],
        )

        self.back_btn = StyledButton(text="<", **btn_style)
        self.back_btn.bind(on_press=lambda *a: self.dispatch("on_back"))
        self.add_widget(self.back_btn)

        self.forward_btn = StyledButton(text=">", **btn_style)
        self.forward_btn.bind(on_press=lambda *a: self.dispatch("on_forward"))
        self.add_widget(self.forward_btn)

        self.up_btn = StyledButton(text="^", **btn_style)
        self.up_btn.bind(on_press=lambda *a: self.dispatch("on_up"))
        self.add_widget(self.up_btn)

        self.home_btn = StyledButton(text="~", **btn_style)
        self.home_btn.bind(on_press=lambda *a: self.dispatch("on_home"))
        self.add_widget(self.home_btn)

        self.add_widget(StyledLabel(text="", size_hint_x=None, width=dp(8)))

        self.path_bar = PathBar(size_hint_x=1.0)
        self.add_widget(self.path_bar)

    def push_history(self, path):
        self._history = self._history[:self._history_index + 1]
        self._history.append(path)
        self._history_index = len(self._history) - 1
        self._update_buttons()

    def can_go_back(self):
        return self._history_index > 0

    def can_go_forward(self):
        return self._history_index < len(self._history) - 1

    def _update_buttons(self):
        self.back_btn.disabled = not self.can_go_back()
        self.forward_btn.disabled = not self.can_go_forward()
        self.back_btn.background_color = (
            THEME["accent_dim"] if self.can_go_back() else THEME["bg_input"]
        )
        self.forward_btn.background_color = (
            THEME["accent_dim"] if self.can_go_forward() else THEME["bg_input"]
        )

    def on_back(self):
        pass

    def on_forward(self):
        pass

    def on_up(self):
        pass

    def on_home(self):
        pass


class FilterBar(BoxLayout):
    """Filter bar with view toggle and file type selector."""

    __events__ = ("on_view_change", "on_filter")

    def __init__(self, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None,
                         height=dp(32), spacing=dp(6), **kwargs)

        self._view_mode = "list"

        self.list_btn = StyledButton(
            text="[ ] List",
            font_size=sp(9),
            size_hint_x=None,
            width=dp(72),
            background_color=THEME["accent"],
        )
        self.list_btn.bind(on_press=lambda *a: self._set_view("list"))
        self.add_widget(self.list_btn)

        self.icon_btn = StyledButton(
            text="[][] Icons",
            font_size=sp(9),
            size_hint_x=None,
            width=dp(72),
            background_color=THEME["bg_input"],
        )
        self.icon_btn.bind(on_press=lambda *a: self._set_view("icon"))
        self.add_widget(self.icon_btn)

        self.add_widget(StyledLabel(text="", size_hint_x=None, width=dp(6)))

        self.add_widget(StyledLabel(
            text="Filter:",
            font_size=sp(9),
            color=THEME["text_muted"],
            size_hint_x=None,
            width=dp(40),
        ))

        self.filter_input = StyledTextInput(
            hint_text="*.jpg *.png...",
            font_size=sp(9),
            size_hint_x=0.5,
        )
        self.filter_input.bind(on_text_validate=self._on_filter)
        self.add_widget(self.filter_input)

        filter_btn = StyledButton(
            text="Apply",
            font_size=sp(9),
            size_hint_x=None,
            width=dp(50),
            background_color=THEME["accent_dim"],
        )
        filter_btn.bind(on_press=self._on_filter)
        self.add_widget(filter_btn)

        self.status_label = StyledLabel(
            text="",
            font_size=sp(9),
            color=THEME["text_muted"],
            size_hint_x=0.3,
            halign="right",
        )
        self.add_widget(self.status_label)

    def _set_view(self, mode):
        self._view_mode = mode
        if mode == "list":
            self.list_btn.background_color = THEME["accent"]
            self.icon_btn.background_color = THEME["bg_input"]
        else:
            self.list_btn.background_color = THEME["bg_input"]
            self.icon_btn.background_color = THEME["accent"]
        self.dispatch("on_view_change", mode)

    def _on_filter(self, *args):
        self.dispatch("on_filter", self.filter_input.text.strip())

    def set_status(self, text):
        self.status_label.text = text

    def on_view_change(self, mode):
        pass

    def on_filter(self, text):
        pass


class FileBrowserContent(BoxLayout):
    """Main content area combining toolbar, file list, and status."""

    def __init__(self, initial_path=None, filters=None, dirselect=False, **kwargs):
        super().__init__(orientation="vertical", spacing=dp(4), **kwargs)
        self._dirselect = dirselect
        self._filters = filters
        self._current_path = initial_path or os.path.expanduser("~")
        self._chooser = None
        self._on_submit = None

        self.nav_toolbar = NavToolbar()
        self.nav_toolbar.bind(on_back=self._go_back)
        self.nav_toolbar.bind(on_forward=self._go_forward)
        self.nav_toolbar.bind(on_up=self._go_up)
        self.nav_toolbar.bind(on_home=self._go_home)
        self.nav_toolbar.path_bar.bind(on_navigate=self._on_path_bar_navigate)
        self.add_widget(self.nav_toolbar)

        self.filter_bar = FilterBar()
        self.filter_bar.bind(on_view_change=self._switch_view)
        self.filter_bar.bind(on_filter=self._apply_filter)
        self.add_widget(self.filter_bar)

        self._file_area = BoxLayout()
        self.add_widget(self._file_area)

        btn_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        self.selected_label = StyledLabel(
            text="No selection",
            font_size=sp(10),
            color=THEME["text_muted"],
            size_hint_x=0.6,
            halign="left",
        )
        btn_row.add_widget(self.selected_label)

        self._cancel_btn = StyledButton(
            text="Cancel",
            font_size=sp(10),
            background_color=THEME["text_muted"],
            size_hint_x=0.2,
        )
        btn_row.add_widget(self._cancel_btn)

        self._select_btn = StyledButton(
            text="Select",
            font_size=sp(10),
            background_color=THEME["accent"],
            size_hint_x=0.2,
        )
        btn_row.add_widget(self._select_btn)

        self.add_widget(btn_row)

        self._create_chooser("list")
        self._navigate_to(self._current_path, push_history=True)

    def _create_chooser(self, view_mode):
        if self._chooser:
            self._file_area.clear_widgets()

        if view_mode == "list":
            self._chooser = FileChooserListView(
                size_hint=(1, 1),
                path=self._current_path,
                dirselect=self._dirselect,
            )
        else:
            self._chooser = FileChooserIconView(
                size_hint=(1, 1),
                path=self._current_path,
                dirselect=self._dirselect,
            )

        if self._filters:
            self._chooser.filters = self._filters

        self._chooser.bind(selection=self._on_selection)
        self._chooser.bind(path=self._on_path_changed)
        self._chooser.bind(on_submit=self._on_submit_event)
        self._file_area.add_widget(self._chooser)

    def _on_submit_event(self, chooser, selection, touch=None):
        if selection:
            path = selection[0]
            if os.path.isdir(path):
                self._navigate_to(path)
            elif self._on_submit:
                self._on_submit(path)

    def _navigate_to(self, path, push_history=True):
        if not os.path.isdir(path):
            return
        self._current_path = path
        self.nav_toolbar.path_bar.set_path(path)
        if push_history:
            self.nav_toolbar.push_history(path)
        if self._chooser:
            self._chooser.path = path
        self._update_status()

    def _on_path_bar_navigate(self, inst, path):
        Clock.schedule_once(lambda dt: self._navigate_to(path), 0)

    def _go_back(self, *args):
        if self.nav_toolbar.can_go_back():
            self.nav_toolbar._history_index -= 1
            path = self.nav_toolbar._history[self.nav_toolbar._history_index]
            self._navigate_to(path, push_history=False)
            self.nav_toolbar._update_buttons()

    def _go_forward(self, *args):
        if self.nav_toolbar.can_go_forward():
            self.nav_toolbar._history_index += 1
            path = self.nav_toolbar._history[self.nav_toolbar._history_index]
            self._navigate_to(path, push_history=False)
            self.nav_toolbar._update_buttons()

    def _go_up(self, *args):
        parent = os.path.dirname(self._current_path)
        if parent and parent != self._current_path:
            self._navigate_to(parent)

    def _go_home(self, *args):
        self._navigate_to(os.path.expanduser("~"))

    def _on_path_changed(self, chooser, path):
        self._current_path = path
        self.nav_toolbar.path_bar.set_path(path)
        self._update_status()

    def _on_selection(self, chooser, selection):
        if selection:
            path = selection[0]
            name = os.path.basename(path)
            if os.path.isdir(path):
                self.selected_label.text = "[DIR] %s" % name
            else:
                size = os.path.getsize(path) if os.path.isfile(path) else 0
                if size > 1024 * 1024:
                    size_str = "%.1f MB" % (size / (1024 * 1024))
                elif size > 1024:
                    size_str = "%.1f KB" % (size / 1024)
                else:
                    size_str = "%d B" % size
                self.selected_label.text = "%s  (%s)" % (name, size_str)
        else:
            self.selected_label.text = "No selection"

    def _update_status(self):
        if self._chooser:
            try:
                entries = len(self._chooser.files) if self._chooser.files else 0
            except Exception:
                entries = 0
            self.filter_bar.set_status("%d items" % entries)

    def _switch_view(self, instance, mode):
        path = self._chooser.path if self._chooser else self._current_path
        self._create_chooser(mode)
        self._navigate_to(path, push_history=False)

    def _apply_filter(self, instance, text):
        if not self._chooser:
            return
        if text:
            self._chooser.filters = [p.strip() for p in text.split() if p.strip()]
        elif self._filters:
            self._chooser.filters = self._filters
        else:
            self._chooser.filters = []
        Clock.schedule_once(lambda dt: self._update_status(), 0.1)

    def get_selected(self):
        if self._chooser and self._chooser.selection:
            sel = self._chooser.selection[0]
            if os.path.isfile(sel):
                return sel
        return None

    def get_selected_path(self):
        if self._chooser and self._chooser.selection:
            return self._chooser.selection[0]
        return None


class FileBrowserDialog:
    """High-level dialog wrapper around FileBrowserContent."""

    def __init__(self, title="Browse", initial_path=None, filters=None,
                 dirselect=False, on_select=None):
        self._on_select = on_select
        self._dirselect = dirselect
        self._content = FileBrowserContent(
            initial_path=initial_path,
            filters=filters,
            dirselect=dirselect,
        )
        self._content._on_submit = self._file_submitted

        self.popup = Popup(
            title=title,
            content=self._content,
            size_hint=(0.8, 0.85),
            auto_dismiss=False,
        )

        self._content._select_btn.bind(on_press=self._on_select_press)
        self._content._cancel_btn.bind(on_press=self._on_cancel_press)

    def _file_submitted(self, path):
        if self._on_select:
            self._on_select(path)
        self.popup.dismiss()

    def _on_select_press(self, *args):
        if self._dirselect:
            selected = self._content.get_selected_path()
            if not selected:
                selected = self._content._current_path
        else:
            selected = self._content.get_selected()
        if selected and self._on_select:
            self._on_select(selected)
        self.popup.dismiss()

    def _on_cancel_press(self, *args):
        self.popup.dismiss()

    def open(self):
        self.popup.open()
