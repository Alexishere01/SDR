#!/usr/bin/env python3
"""
GUI Setup Test Script

This script tests the GUI development environment setup and verifies that all
dependencies are properly installed and working.
"""

import sys
import traceback
from typing import List, Tuple


def test_imports() -> List[Tuple[str, bool, str]]:
    """Test all required GUI imports."""
    results = []
    
    # Test PySide6 core
    try:
        from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
        from PySide6.QtCore import QTimer, Signal, Slot, Qt
        from PySide6.QtGui import QIcon, QPalette, QColor
        results.append(("PySide6 Core", True, "OK"))
    except ImportError as e:
        results.append(("PySide6 Core", False, str(e)))
    
    # Test PySide6 OpenGL
    try:
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
        results.append(("PySide6 OpenGL", True, "OK"))
    except ImportError as e:
        results.append(("PySide6 OpenGL", False, str(e)))
    
    # Test PyQtGraph
    try:
        import pyqtgraph as pg
        results.append(("PyQtGraph", True, f"Version {pg.__version__}"))
    except ImportError as e:
        results.append(("PyQtGraph", False, str(e)))
    
    # Test OpenGL
    try:
        import OpenGL.GL as gl
        import OpenGL.arrays.vbo as vbo
        results.append(("PyOpenGL", True, "OK"))
    except ImportError as e:
        results.append(("PyOpenGL", False, str(e)))
    
    # Test pytest-qt
    try:
        import pytestqt
        results.append(("pytest-qt", True, "OK"))
    except ImportError as e:
        results.append(("pytest-qt", False, str(e)))
    
    # Test qtawesome
    try:
        import qtawesome as qta
        results.append(("QtAwesome", True, "OK"))
    except ImportError as e:
        results.append(("QtAwesome", False, str(e)))
    
    return results


def test_gui_creation():
    """Test basic GUI creation."""
    try:
        # Import our GUI modules
        from main_application import GeminiSDRApplication, MainWindow
        from gui_config import GUIConfigManager
        from resources.themes import get_theme, get_available_themes
        
        print("✓ GUI modules imported successfully")
        
        # Test configuration manager
        config_manager = GUIConfigManager()
        gui_config = config_manager.get_gui_config()
        print(f"✓ Configuration manager created (theme: {gui_config.theme})")
        
        # Test theme system
        themes = get_available_themes()
        dark_theme = get_theme("dark")
        print(f"✓ Theme system working (available: {themes})")
        
        # Test application creation (without showing)
        app = GeminiSDRApplication([])
        print("✓ Application created successfully")
        
        # Test main window creation (without showing)
        window = MainWindow()
        print("✓ Main window created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ GUI creation failed: {e}")
        traceback.print_exc()
        return False


def test_qt_tools():
    """Test Qt development tools availability."""
    results = []
    
    # Test Qt Designer availability
    try:
        import subprocess
        result = subprocess.run(['pyside6-designer', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            results.append(("Qt Designer", True, "Available"))
        else:
            results.append(("Qt Designer", False, "Not found"))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        results.append(("Qt Designer", False, "Not found or timeout"))
    
    # Test resource compiler
    try:
        result = subprocess.run(['pyside6-rcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            results.append(("Resource Compiler", True, "Available"))
        else:
            results.append(("Resource Compiler", False, "Not found"))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        results.append(("Resource Compiler", False, "Not found or timeout"))
    
    return results


def main():
    """Main test function."""
    print("GeminiSDR GUI Setup Test")
    print("=" * 50)
    
    # Test imports
    print("\n1. Testing imports...")
    import_results = test_imports()
    
    for name, success, message in import_results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}: {message}")
    
    # Check if all imports succeeded
    all_imports_ok = all(result[1] for result in import_results)
    
    if not all_imports_ok:
        print("\n❌ Some imports failed. Please check your installation.")
        return False
    
    # Test GUI creation
    print("\n2. Testing GUI creation...")
    gui_ok = test_gui_creation()
    
    if not gui_ok:
        print("\n❌ GUI creation failed.")
        return False
    
    # Test Qt tools
    print("\n3. Testing Qt development tools...")
    tool_results = test_qt_tools()
    
    for name, success, message in tool_results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}: {message}")
    
    print("\n" + "=" * 50)
    
    if all_imports_ok and gui_ok:
        print("🎉 GUI development environment setup successful!")
        print("\nNext steps:")
        print("- Run 'python gui/main_application.py' to test the basic GUI")
        print("- Use Qt Designer for UI design: 'pyside6-designer'")
        print("- Run GUI tests with: 'pytest tests/gui/ -v'")
        return True
    else:
        print("❌ Setup incomplete. Please resolve the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)