"""Theme and shared visual constants for the MicroClassify GUI."""

from kivy.utils import get_color_from_hex


THEME = {
    "bg_dark": get_color_from_hex("#0d1520"),
    "bg_panel": get_color_from_hex("#182433"),
    "bg_card": get_color_from_hex("#223140"),
    "bg_input": get_color_from_hex("#2b3a4c"),
    "accent": get_color_from_hex("#00b4d8"),
    "accent_hover": get_color_from_hex("#0096c7"),
    "accent_dim": get_color_from_hex("#0a5c74"),
    "success": get_color_from_hex("#2ecc71"),
    "warning": get_color_from_hex("#f39c12"),
    "danger": get_color_from_hex("#e74c3c"),
    "text": get_color_from_hex("#ecf0f1"),
    "text_dim": get_color_from_hex("#93a5b5"),
    "text_muted": get_color_from_hex("#5f7285"),
    "border": get_color_from_hex("#2e4057"),
    "white": get_color_from_hex("#ffffff"),
}
