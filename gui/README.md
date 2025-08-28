# GeminiSDR GUI Development

This directory contains the graphical user interface (GUI) for the GeminiSDR system, providing real-time spectrum visualization, SDR control, ML model monitoring, and system status display.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment activated
- GUI dependencies installed

### Installation

1. Install GUI dependencies:
```bash
pip install -r requirements-gui-dev.txt
```

2. Verify installation:
```bash
cd gui
python test_setup.py
```

3. Run the GUI:
```bash
python main_application.py
```

Or use the launcher from the project root:
```bash
python launch_gui.py
```

## Project Structure

```
gui/
├── main_application.py      # Main application and window
├── gui_config.py           # Configuration management
├── widgets/                # Custom GUI widgets
│   └── __init__.py
├── utils/                  # GUI utilities
│   └── __init__.py
├── resources/              # Themes, icons, UI files
│   ├── __init__.py
│   └── themes.py          # Theme definitions
├── test_setup.py          # Setup verification script
├── test_gui_basic.py      # Basic functionality test
└── README.md              # This file
```

## Architecture

The GUI is built using:

- **PySide6/Qt6**: Cross-platform GUI framework
- **PyQtGraph**: High-performance plotting and visualization
- **OpenGL**: Hardware-accelerated rendering
- **pytest-qt**: GUI testing framework

### Key Components

1. **GeminiSDRApplication**: Main application class with cross-platform setup
2. **MainWindow**: Main window with dockable widgets and signal management
3. **GUIConfigManager**: Configuration management and persistence
4. **Theme System**: Support for dark, light, and high-contrast themes

## Development Workflow

### 1. Setting Up Development Environment

```bash
# Run the automated setup script
python scripts/setup_gui_dev.py

# Or manually install dependencies
pip install -r requirements-gui-dev.txt
```

### 2. Running Tests

```bash
# Run GUI-specific tests
pytest tests/test_gui_setup.py -v

# Run all tests
pytest -v
```

### 3. Code Quality

```bash
# Format code
black gui/

# Lint code
flake8 gui/

# Type checking
mypy gui/
```

### 4. Creating New Widgets

1. Create widget file in `gui/widgets/`
2. Inherit from appropriate Qt widget class
3. Follow the existing patterns for signals and configuration
4. Add tests in `tests/test_gui_*.py`
5. Update `gui/widgets/__init__.py` to export the widget

Example widget structure:
```python
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal, Slot

class MyWidget(QWidget):
    # Define signals
    data_changed = Signal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_connections()
    
    def _setup_ui(self):
        """Setup the user interface."""
        pass
    
    def _setup_connections(self):
        """Connect signals and slots."""
        pass
    
    @Slot(object)
    def update_data(self, data):
        """Update widget with new data."""
        pass
```

## Configuration

The GUI uses a hierarchical configuration system:

- **GUIConfig**: General GUI settings (theme, window geometry, update rates)
- **VisualizationConfig**: Visualization parameters (FFT size, color maps, etc.)
- **ControlConfig**: SDR control settings (auto-apply, confirmation, etc.)

Configuration is automatically saved to `~/.geminisdr/gui_config.json`.

## Themes

The GUI supports multiple themes:

- **Dark**: Default dark theme for low-light environments
- **Light**: Light theme for bright environments  
- **High Contrast**: Accessibility theme with high contrast colors

Themes are defined in `gui/resources/themes.py` and can be switched at runtime.

## Testing

### Unit Tests

GUI components are tested using pytest-qt:

```bash
pytest tests/test_gui_setup.py -v
```

### Manual Testing

1. **Setup Verification**: `python gui/test_setup.py`
2. **Basic Functionality**: `python gui/test_gui_basic.py`
3. **Interactive Testing**: `python gui/main_application.py`

### Visual Testing

For visual regression testing, screenshots can be captured and compared:

```python
def test_widget_appearance(qtbot):
    widget = MyWidget()
    qtbot.addWidget(widget)
    widget.show()
    
    # Capture screenshot for comparison
    pixmap = widget.grab()
    # Compare with reference image
```

## Performance Considerations

### High-Performance Rendering

- Use PyQtGraph for real-time plotting
- Leverage OpenGL for hardware acceleration
- Implement level-of-detail rendering for large datasets
- Use efficient data structures and caching

### Threading

- Keep GUI updates on the main thread
- Use QTimer for periodic updates
- Implement background processing with QThread
- Use signals/slots for thread communication

### Memory Management

- Implement proper cleanup in closeEvent()
- Use weak references where appropriate
- Monitor memory usage in long-running operations
- Implement data streaming for large datasets

## Cross-Platform Compatibility

The GUI is designed to work on:

- **Windows**: Tested on Windows 10/11
- **macOS**: Tested on macOS 10.15+
- **Linux**: Tested on Ubuntu 20.04+

Platform-specific considerations:

- High-DPI display support
- Native look and feel
- Platform-specific shortcuts and conventions
- File dialog and system integration

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Qt Application Singleton**: Only one QApplication instance allowed
3. **OpenGL Issues**: Check graphics drivers and OpenGL support
4. **Theme Not Applied**: Verify theme files and stylesheet syntax

### Debug Mode

Enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export QT_LOGGING_RULES="*.debug=true"
```

### Performance Profiling

Profile GUI performance:

```python
import cProfile
import pstats

# Profile GUI operations
profiler = cProfile.Profile()
profiler.enable()
# ... GUI operations ...
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Contributing

1. Follow the existing code style and patterns
2. Add tests for new functionality
3. Update documentation
4. Test on multiple platforms if possible
5. Consider accessibility and usability

## Resources

- [PySide6 Documentation](https://doc.qt.io/qtforpython/)
- [PyQtGraph Documentation](https://pyqtgraph.readthedocs.io/)
- [Qt Designer Tutorial](https://doc.qt.io/qt-6/qtdesigner-manual.html)
- [GUI Design Patterns](https://doc.qt.io/qt-6/topics-ui.html)

## Next Steps

See the implementation plan in `.kiro/specs/gui-monitoring-control/tasks.md` for the next development tasks:

1. Real-time spectrum visualization system
2. SDR hardware control interface  
3. ML model monitoring and control interface
4. System status and health monitoring
5. Advanced visualization and analysis features