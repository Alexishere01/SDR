"""
Theme definitions for GeminiSDR GUI

This module contains theme configurations and styling for the application.
"""

from typing import Dict, Any

# Dark theme configuration
DARK_THEME = {
    "name": "Dark",
    "colors": {
        "background": "#2b2b2b",
        "foreground": "#ffffff",
        "accent": "#0078d4",
        "border": "#555555",
        "hover": "#505050",
        "pressed": "#353535",
        "disabled": "#666666",
    },
    "stylesheet": """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #404040;
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 5px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #505050;
        }
        QPushButton:pressed {
            background-color: #353535;
        }
        QPushButton:disabled {
            background-color: #666666;
            color: #999999;
        }
        QSlider::groove:horizontal {
            border: 1px solid #555555;
            height: 8px;
            background: #404040;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #0078d4;
            border: 1px solid #555555;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }
        QProgressBar {
            border: 1px solid #555555;
            border-radius: 3px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 2px;
        }
    """
}

# Light theme configuration
LIGHT_THEME = {
    "name": "Light",
    "colors": {
        "background": "#ffffff",
        "foreground": "#000000",
        "accent": "#0078d4",
        "border": "#cccccc",
        "hover": "#e5e5e5",
        "pressed": "#d0d0d0",
        "disabled": "#999999",
    },
    "stylesheet": """
        QMainWindow {
            background-color: #ffffff;
            color: #000000;
        }
        QWidget {
            background-color: #ffffff;
            color: #000000;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
            border-radius: 3px;
            padding: 5px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #e5e5e5;
        }
        QPushButton:pressed {
            background-color: #d0d0d0;
        }
        QPushButton:disabled {
            background-color: #999999;
            color: #666666;
        }
    """
}

# High contrast theme for accessibility
HIGH_CONTRAST_THEME = {
    "name": "High Contrast",
    "colors": {
        "background": "#000000",
        "foreground": "#ffffff",
        "accent": "#ffff00",
        "border": "#ffffff",
        "hover": "#333333",
        "pressed": "#666666",
        "disabled": "#808080",
    },
    "stylesheet": """
        QMainWindow {
            background-color: #000000;
            color: #ffffff;
        }
        QWidget {
            background-color: #000000;
            color: #ffffff;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #ffffff;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #000000;
            border: 2px solid #ffffff;
            border-radius: 3px;
            padding: 5px;
            min-width: 80px;
            color: #ffffff;
        }
        QPushButton:hover {
            background-color: #333333;
        }
        QPushButton:pressed {
            background-color: #666666;
        }
    """
}

# Available themes
THEMES = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
    "high_contrast": HIGH_CONTRAST_THEME,
}

def get_theme(theme_name: str) -> Dict[str, Any]:
    """Get theme configuration by name."""
    return THEMES.get(theme_name.lower(), DARK_THEME)

def get_available_themes() -> list:
    """Get list of available theme names."""
    return list(THEMES.keys())