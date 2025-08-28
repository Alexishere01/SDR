"""
GeminiSDR GUI Package

This package provides a comprehensive graphical user interface for the GeminiSDR system,
including real-time spectrum visualization, SDR control, ML model monitoring, and system status display.
"""

__version__ = "1.0.0"
__author__ = "GeminiSDR Team"

# Import main application components
from .main_application import GeminiSDRApplication, MainWindow

__all__ = [
    'GeminiSDRApplication',
    'MainWindow',
]