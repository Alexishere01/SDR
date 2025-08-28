#!/usr/bin/env python3
"""
Basic GUI Test - Non-interactive

This script tests GUI creation without showing windows.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_gui():
    """Test basic GUI functionality without showing windows."""
    try:
        from main_application import GeminiSDRApplication, MainWindow
        from gui_config import GUIConfigManager
        
        # Create application
        app = GeminiSDRApplication([])
        
        # Create main window
        window = MainWindow()
        
        # Test configuration
        config_manager = GUIConfigManager()
        gui_config = config_manager.get_gui_config()
        
        print("✓ GUI components created successfully")
        print(f"✓ Theme: {gui_config.theme}")
        print(f"✓ Update rate: {gui_config.update_rate_fps} FPS")
        print(f"✓ Window size: {gui_config.window_geometry['width']}x{gui_config.window_geometry['height']}")
        
        # Clean up
        window.close()
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_gui()
    print("\n" + "="*50)
    if success:
        print("🎉 Basic GUI test passed!")
    else:
        print("❌ Basic GUI test failed!")
    sys.exit(0 if success else 1)