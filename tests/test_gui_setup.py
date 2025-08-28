"""
GUI Setup Tests

This module contains tests for the GUI development environment setup.
"""

import pytest
import sys
import os
from pathlib import Path

# Add GUI module to path
gui_path = Path(__file__).parent.parent / "gui"
sys.path.insert(0, str(gui_path))

try:
    from PySide6.QtWidgets import QApplication
    from main_application import GeminiSDRApplication, MainWindow
    from gui_config import GUIConfigManager
    from resources.themes import get_theme, get_available_themes
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI dependencies not available")
class TestGUISetup:
    """Test GUI setup and basic functionality."""
    
    def test_gui_imports(self):
        """Test that all GUI imports work correctly."""
        # Test PySide6 imports
        from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
        from PySide6.QtCore import QTimer, Signal, Slot, Qt
        from PySide6.QtGui import QIcon, QPalette, QColor
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
        
        # Test PyQtGraph
        import pyqtgraph as pg
        assert hasattr(pg, '__version__')
        
        # Test OpenGL
        import OpenGL.GL as gl
        
        # Test pytest-qt
        import pytestqt
    
    def test_application_creation(self, qtbot):
        """Test GeminiSDR application creation."""
        # Use existing QApplication instance from qtbot
        app = QApplication.instance()
        if app is None:
            app = GeminiSDRApplication([])
        
        # Test application properties by setting them
        app.setApplicationName("GeminiSDR")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("GeminiSDR")
        
        assert app.applicationName() == "GeminiSDR"
        assert app.applicationVersion() == "1.0.0"
        assert app.organizationName() == "GeminiSDR"
    
    def test_main_window_creation(self, qtbot):
        """Test main window creation."""
        # Don't create new app, use existing one
        window = MainWindow()
        qtbot.addWidget(window)
        
        assert window.windowTitle() == "GeminiSDR - Intelligent SDR Control"
        assert window.minimumSize().width() == 1200
        assert window.minimumSize().height() == 800
        
        # Test that window has required attributes
        assert hasattr(window, 'sdr_config_changed')
        assert hasattr(window, 'spectrum_data_ready')
        assert hasattr(window, 'model_metrics_updated')
        assert hasattr(window, 'system_status_changed')
    
    def test_config_manager(self):
        """Test GUI configuration manager."""
        config_manager = GUIConfigManager()
        
        # Test default configurations
        gui_config = config_manager.get_gui_config()
        assert gui_config.theme == "dark"
        assert gui_config.update_rate_fps == 30
        assert gui_config.max_history_points == 1000
        
        viz_config = config_manager.get_visualization_config()
        assert viz_config.fft_size == 2048
        assert viz_config.window_function == "hann"
        assert viz_config.averaging_factor == 0.8
        
        control_config = config_manager.get_control_config()
        assert control_config.auto_apply_changes == True
        assert control_config.fine_tune_step_hz == 1000.0
    
    def test_theme_system(self):
        """Test theme system functionality."""
        themes = get_available_themes()
        assert "dark" in themes
        assert "light" in themes
        assert "high_contrast" in themes
        
        # Test theme retrieval
        dark_theme = get_theme("dark")
        assert dark_theme["name"] == "Dark"
        assert "colors" in dark_theme
        assert "stylesheet" in dark_theme
        
        light_theme = get_theme("light")
        assert light_theme["name"] == "Light"
        
        # Test invalid theme returns default
        invalid_theme = get_theme("nonexistent")
        assert invalid_theme["name"] == "Dark"  # Should return default
    
    def test_config_persistence(self, tmp_path):
        """Test configuration saving and loading."""
        # Create config manager with temporary directory
        config_manager = GUIConfigManager(str(tmp_path))
        
        # Modify configuration
        config_manager.update_gui_config(theme="light", update_rate_fps=60)
        config_manager.update_visualization_config(fft_size=4096)
        
        # Create new config manager with same directory
        new_config_manager = GUIConfigManager(str(tmp_path))
        
        # Verify configuration was loaded
        gui_config = new_config_manager.get_gui_config()
        assert gui_config.theme == "light"
        assert gui_config.update_rate_fps == 60
        
        viz_config = new_config_manager.get_visualization_config()
        assert viz_config.fft_size == 4096


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI dependencies not available")
def test_gui_environment_complete(qtbot):
    """Test that the complete GUI environment is set up correctly."""
    # Test that all required modules exist
    gui_modules = [
        "main_application",
        "gui_config", 
        "resources.themes",
        "widgets",
        "utils",
        "resources"
    ]
    
    for module_name in gui_modules:
        try:
            __import__(module_name)
        except ImportError as e:
            pytest.fail(f"Required GUI module '{module_name}' not available: {e}")
    
    # Test that GUI can be created without errors
    app = QApplication.instance()
    window = MainWindow()
    qtbot.addWidget(window)
    config_manager = GUIConfigManager()
    
    # Basic functionality test
    assert app is not None
    assert window is not None
    assert config_manager is not None