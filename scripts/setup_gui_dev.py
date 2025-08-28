#!/usr/bin/env python3
"""
GUI Development Setup Script

This script sets up the GUI development environment for GeminiSDR.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required for GUI development")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_gui_dependencies():
    """Install GUI development dependencies."""
    print("\n🔧 Installing GUI Dependencies")
    print("=" * 50)
    
    # Install main GUI requirements
    if not run_command("pip install -r requirements-gui-dev.txt", 
                      "Installing GUI development dependencies"):
        return False
    
    # Install additional development tools
    dev_packages = [
        "black",  # Code formatting
        "flake8",  # Linting
        "mypy",   # Type checking
    ]
    
    for package in dev_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package} (optional)")
    
    return True


def verify_installation():
    """Verify that the GUI installation is working."""
    print("\n🧪 Verifying Installation")
    print("=" * 50)
    
    # Run the GUI setup test
    gui_path = Path(__file__).parent.parent / "gui"
    test_script = gui_path / "test_setup.py"
    
    if not test_script.exists():
        print("✗ GUI test script not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(test_script)], 
                              cwd=str(gui_path), check=True, 
                              capture_output=True, text=True)
        print("✓ GUI setup verification passed")
        return True
    except subprocess.CalledProcessError as e:
        print("✗ GUI setup verification failed")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def run_gui_tests():
    """Run GUI unit tests."""
    print("\n🧪 Running GUI Tests")
    print("=" * 50)
    
    return run_command("python -m pytest tests/test_gui_setup.py -v", 
                      "Running GUI unit tests")


def create_development_shortcuts():
    """Create development shortcuts and scripts."""
    print("\n🔗 Creating Development Shortcuts")
    print("=" * 50)
    
    # Create a simple launcher script
    launcher_script = """#!/usr/bin/env python3
'''
GeminiSDR GUI Launcher

Quick launcher for the GeminiSDR GUI application.
'''

import sys
import os
from pathlib import Path

# Add GUI path
gui_path = Path(__file__).parent / "gui"
sys.path.insert(0, str(gui_path))

if __name__ == "__main__":
    from main_application import main
    sys.exit(main())
"""
    
    with open("launch_gui.py", "w") as f:
        f.write(launcher_script)
    
    # Make it executable on Unix systems
    if os.name != 'nt':
        os.chmod("launch_gui.py", 0o755)
    
    print("✓ Created launch_gui.py script")
    
    return True


def print_next_steps():
    """Print next steps for developers."""
    print("\n🎉 GUI Development Environment Setup Complete!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Test the GUI: python launch_gui.py")
    print("2. Run GUI tests: pytest tests/test_gui_setup.py -v")
    print("3. Start developing widgets in gui/widgets/")
    print("4. Use Qt Designer: pyside6-designer (if available)")
    print("5. Check the design document: .kiro/specs/gui-monitoring-control/design.md")
    
    print("\n📁 Project Structure:")
    print("gui/")
    print("├── main_application.py    # Main application and window")
    print("├── gui_config.py         # Configuration management")
    print("├── widgets/              # Custom GUI widgets")
    print("├── utils/                # GUI utilities")
    print("└── resources/            # Themes, icons, UI files")
    
    print("\n🔧 Development Tools:")
    print("- Code formatting: black gui/")
    print("- Linting: flake8 gui/")
    print("- Type checking: mypy gui/")
    print("- GUI testing: pytest tests/test_gui_setup.py")


def main():
    """Main setup function."""
    print("GeminiSDR GUI Development Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_gui_dependencies():
        print("❌ Failed to install GUI dependencies")
        return False
    
    # Verify installation
    if not verify_installation():
        print("❌ GUI installation verification failed")
        return False
    
    # Run tests
    if not run_gui_tests():
        print("⚠️  GUI tests failed, but installation may still work")
    
    # Create development shortcuts
    if not create_development_shortcuts():
        print("⚠️  Failed to create development shortcuts")
    
    # Print next steps
    print_next_steps()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)