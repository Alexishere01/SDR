#!/usr/bin/env python3
"""
GeminiSDR GUI Launcher

Quick launcher for the GeminiSDR GUI application.
"""

import sys
import os
from pathlib import Path

# Add GUI path
gui_path = Path(__file__).parent / "gui"
sys.path.insert(0, str(gui_path))

if __name__ == "__main__":
    from main_application import main
    sys.exit(main())