"""
Main GUI Application Module

This module contains the main application class and window management for GeminiSDR GUI.
"""

from PySide6.QtWidgets import (QApplication, QMainWindow, QSplitter, QVBoxLayout, QWidget,
                               QDockWidget, QLabel, QHBoxLayout, QTabWidget)
from PySide6.QtCore import QTimer, QThread, Signal, Slot, Qt, QSettings, QStandardPaths
from PySide6.QtGui import QIcon, QPixmap, QPalette, QColor, QFont
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from typing import Optional, Dict, Any
import sys
import os
import platform
import logging
from pathlib import Path

# Import theme management
from gui.resources.themes import get_theme, get_available_themes


class GeminiSDRApplication(QApplication):
    """Main application class with cross-platform setup."""
    
    # Signals for application-wide events
    theme_changed = Signal(str)
    settings_changed = Signal(dict)
    shutdown_requested = Signal()
    
    def __init__(self, argv: list):
        super().__init__(argv)
        
        # Application metadata
        self.setApplicationName("GeminiSDR")
        self.setApplicationVersion("1.0.0")
        self.setOrganizationName("GeminiSDR")
        self.setOrganizationDomain("geminisdr.org")
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize settings
        self.settings = QSettings()
        self.config_dir = self._get_config_directory()
        
        # Current theme
        self.current_theme = "dark"
        
        # Platform-specific setup
        self._setup_platform_specific()
        
        # Initialize theme management
        self._setup_theme_management()
        
        # Load application settings
        self._load_settings()
        
        # Setup cleanup handlers
        self.aboutToQuit.connect(self._cleanup)
        
        self.logger.info(f"GeminiSDR Application initialized on {platform.system()}")
        
    def _setup_logging(self) -> None:
        """Setup application logging."""
        self.logger = logging.getLogger("GeminiSDR.Application")
        
    def _get_config_directory(self) -> Path:
        """Get platform-specific configuration directory."""
        if sys.platform == "darwin":  # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "GeminiSDR"
        elif sys.platform == "win32":  # Windows
            config_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
        else:  # Linux and others
            config_dir = Path.home() / ".config" / "GeminiSDR"
            
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
        
    def _setup_platform_specific(self) -> None:
        """Configure platform-specific settings."""
        # Enable high DPI support
        self.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        self.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # Platform-specific configurations
        if sys.platform == "darwin":  # macOS
            self.setAttribute(Qt.AA_DontShowIconsInMenus, True)
            # macOS specific font settings
            font = QFont("SF Pro Display", 13)
            self.setFont(font)
            
        elif sys.platform == "win32":  # Windows
            self.setStyle("Fusion")
            # Windows specific font settings
            font = QFont("Segoe UI", 9)
            self.setFont(font)
            
        elif sys.platform.startswith("linux"):  # Linux
            self.setStyle("Fusion")
            # Linux specific font settings
            font = QFont("Ubuntu", 10)
            self.setFont(font)
            
        self.logger.info(f"Platform-specific setup completed for {platform.system()}")
            
    def _setup_theme_management(self) -> None:
        """Setup comprehensive theme management system."""
        self.available_themes = get_available_themes()
        self.theme_configs = {}
        
        # Load all theme configurations
        for theme_name in self.available_themes:
            self.theme_configs[theme_name] = get_theme(theme_name)
            
        self.logger.info(f"Theme management initialized with themes: {self.available_themes}")
        
    def apply_theme(self, theme_name: str) -> None:
        """Apply a theme to the application."""
        if theme_name not in self.available_themes:
            self.logger.warning(f"Unknown theme: {theme_name}, using default")
            theme_name = "dark"
            
        theme_config = self.theme_configs[theme_name]
        self.setStyleSheet(theme_config["stylesheet"])
        self.current_theme = theme_name
        
        # Save theme preference
        self.settings.setValue("appearance/theme", theme_name)
        
        # Emit theme changed signal
        self.theme_changed.emit(theme_name)
        
        self.logger.info(f"Applied theme: {theme_name}")
        
    def get_current_theme(self) -> str:
        """Get the currently active theme name."""
        return self.current_theme
        
    def get_theme_colors(self, theme_name: Optional[str] = None) -> Dict[str, str]:
        """Get color palette for a theme."""
        if theme_name is None:
            theme_name = self.current_theme
            
        if theme_name in self.theme_configs:
            return self.theme_configs[theme_name]["colors"]
        return self.theme_configs["dark"]["colors"]
        
    def _load_settings(self) -> None:
        """Load application settings."""
        # Load theme preference
        saved_theme = self.settings.value("appearance/theme", "dark")
        self.apply_theme(saved_theme)
        
        # Load other settings
        self.update_rate = int(self.settings.value("performance/update_rate", 30))
        self.enable_opengl = self.settings.value("performance/enable_opengl", True, type=bool)
        self.auto_save_interval = int(self.settings.value("general/auto_save_interval", 300))
        
        self.logger.info("Application settings loaded")
        
    def save_settings(self) -> None:
        """Save current application settings."""
        self.settings.setValue("appearance/theme", self.current_theme)
        self.settings.setValue("performance/update_rate", self.update_rate)
        self.settings.setValue("performance/enable_opengl", self.enable_opengl)
        self.settings.setValue("general/auto_save_interval", self.auto_save_interval)
        
        self.settings.sync()
        self.logger.info("Application settings saved")
        
    def get_setting(self, key: str, default_value: Any = None) -> Any:
        """Get a setting value."""
        return self.settings.value(key, default_value)
        
    def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value."""
        self.settings.setValue(key, value)
        self.settings_changed.emit({key: value})
        
    def reset_settings(self) -> None:
        """Reset all settings to defaults."""
        self.settings.clear()
        self._load_settings()
        self.logger.info("Settings reset to defaults")
        
    def _cleanup(self) -> None:
        """Cleanup resources before application exit."""
        self.logger.info("Starting application cleanup")
        
        # Save settings
        self.save_settings()
        
        # Emit shutdown signal for components to cleanup
        self.shutdown_requested.emit()
        
        self.logger.info("Application cleanup completed")


class MainWindow(QMainWindow):
    """Main application window with dockable widgets."""
    
    # Signals for inter-component communication
    sdr_config_changed = Signal(dict)
    spectrum_data_ready = Signal(object)
    model_metrics_updated = Signal(dict)
    system_status_changed = Signal(dict)
    
    # Window management signals
    layout_changed = Signal()
    dock_visibility_changed = Signal(str, bool)
    
    def __init__(self, app: GeminiSDRApplication):
        super().__init__()
        self.app = app
        self.logger = logging.getLogger("GeminiSDR.MainWindow")
        
        # Window properties
        self.setWindowTitle("GeminiSDR - Intelligent SDR Control")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)
        
        # Initialize core components
        self.sdr_controller = None
        self.data_pipeline = None
        self.config_manager = None
        
        # Dock widgets storage
        self.dock_widgets = {}
        
        # Status tracking
        self.connection_status = "Disconnected"
        self.hardware_status = "Not Detected"
        self.processing_status = "Idle"
        
        # Setup UI components
        self._setup_ui()
        self._setup_menu_system()
        self._setup_dockable_widgets()
        self._setup_status_bar()
        self._setup_connections()
        self._setup_timers()
        
        # Load window state
        self._load_window_state()
        
        self.logger.info("MainWindow initialized")
        
    def _setup_ui(self) -> None:
        """Initialize the user interface layout."""
        # Enable docking
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.AllowNestedDocks |
            QMainWindow.AllowTabbedDocks |
            QMainWindow.AnimatedDocks
        )
        
        # Create central widget - main spectrum display area
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Central layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Placeholder for main spectrum visualization
        # This will be replaced with actual spectrum widgets in later tasks
        self.main_display = QWidget()
        self.main_display.setStyleSheet("""
            QWidget {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.main_display)
        
    def _setup_menu_system(self) -> None:
        """Setup comprehensive menu system with keyboard shortcuts."""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        # New/Open/Save actions
        new_action = file_menu.addAction("&New Session")
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_session)
        
        open_action = file_menu.addAction("&Open Session")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_session)
        
        save_action = file_menu.addAction("&Save Session")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_session)
        
        file_menu.addSeparator()
        
        # Import/Export
        import_menu = file_menu.addMenu("&Import")
        import_menu.addAction("Import I/Q Data").triggered.connect(self._import_iq_data)
        import_menu.addAction("Import Configuration").triggered.connect(self._import_config)
        
        export_menu = file_menu.addMenu("&Export")
        export_menu.addAction("Export Spectrum Data").triggered.connect(self._export_spectrum)
        export_menu.addAction("Export Screenshot").triggered.connect(self._export_screenshot)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        
        # Theme submenu
        theme_menu = view_menu.addMenu("&Theme")
        if hasattr(self.app, 'available_themes'):
            for theme_name in self.app.available_themes:
                action = theme_menu.addAction(theme_name.replace("_", " ").title())
                action.triggered.connect(lambda checked, name=theme_name: self.app.apply_theme(name))
        
        view_menu.addSeparator()
        
        # Dock widget visibility toggles (will be populated when docks are created)
        self.dock_menu = view_menu.addMenu("&Dock Widgets")
        
        view_menu.addSeparator()
        
        # Layout actions
        view_menu.addAction("Reset Layout").triggered.connect(self._reset_layout)
        view_menu.addAction("Save Layout").triggered.connect(self._save_layout)
        view_menu.addAction("Load Layout").triggered.connect(self._load_layout)
        
        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        
        tools_menu.addAction("SDR Configuration").triggered.connect(self._show_sdr_config)
        tools_menu.addAction("Signal Generator").triggered.connect(self._show_signal_generator)
        tools_menu.addAction("Measurement Tools").triggered.connect(self._show_measurement_tools)
        
        tools_menu.addSeparator()
        
        tools_menu.addAction("Performance Monitor").triggered.connect(self._show_performance_monitor)
        tools_menu.addAction("System Diagnostics").triggered.connect(self._show_diagnostics)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        
        help_menu.addAction("User Manual").triggered.connect(self._show_user_manual)
        help_menu.addAction("Keyboard Shortcuts").triggered.connect(self._show_shortcuts)
        help_menu.addAction("About GeminiSDR").triggered.connect(self._show_about)
        
    def _setup_dockable_widgets(self) -> None:
        """Setup dockable widget system."""
        
        # SDR Control Panel (Left side)
        sdr_dock = QDockWidget("SDR Control", self)
        sdr_dock.setObjectName("SDRControlDock")
        
        # Import and create SDR control components
        try:
            from gui.widgets.sdr_control_panel import SDRControlPanel
            from gui.widgets.preset_manager import PresetManager
            from gui.widgets.hardware_manager import HardwareManager
            
            # Create tabbed widget for SDR controls
            sdr_tabs = QTabWidget()
            
            # SDR Control Panel
            self.sdr_control_panel = SDRControlPanel()
            sdr_tabs.addTab(self.sdr_control_panel, "Control")
            
            # Hardware Manager
            self.hardware_manager = HardwareManager()
            sdr_tabs.addTab(self.hardware_manager, "Hardware")
            
            # Preset Manager
            self.preset_manager = PresetManager()
            sdr_tabs.addTab(self.preset_manager, "Presets")
            
            # Connect components
            self._setup_sdr_connections()
            
            sdr_dock.setWidget(sdr_tabs)
            
        except ImportError as e:
            self.logger.warning(f"SDR control components not available: {e}")
            sdr_widget = QLabel("SDR Control Panel\n(Components not available)")
            sdr_widget.setStyleSheet("padding: 10px; background-color: #353535;")
            sdr_dock.setWidget(sdr_widget)
            
        self.addDockWidget(Qt.LeftDockWidgetArea, sdr_dock)
        self.dock_widgets["sdr_control"] = sdr_dock
        
        # Model Monitor (Right side)
        model_dock = QDockWidget("ML Model Monitor", self)
        model_dock.setObjectName("ModelMonitorDock")
        
        try:
            from gui.widgets.model_monitor import ModelMonitorWidget
            self.model_monitor = ModelMonitorWidget()
            
            # Connect model monitor signals
            self.model_monitor.model_load_requested.connect(self._on_model_load_requested)
            self.model_monitor.model_unload_requested.connect(self._on_model_unload_requested)
            self.model_monitor.model_selected.connect(self._on_model_selected)
            self.model_monitor.training_control_requested.connect(self._on_training_control_requested)
            
            model_dock.setWidget(self.model_monitor)
            
        except ImportError as e:
            self.logger.warning(f"Model monitor widget not available: {e}")
            model_widget = QLabel("ML Model Monitor\n(Widget not available)")
            model_widget.setStyleSheet("padding: 10px; background-color: #353535;")
            model_dock.setWidget(model_widget)
            self.model_monitor = None
            
        self.addDockWidget(Qt.RightDockWidgetArea, model_dock)
        self.dock_widgets["model_monitor"] = model_dock
        
        # System Status (Bottom)
        status_dock = QDockWidget("System Status", self)
        status_dock.setObjectName("SystemStatusDock")
        status_widget = QLabel("System Status Monitor\n(To be implemented)")
        status_widget.setStyleSheet("padding: 10px; background-color: #353535;")
        status_dock.setWidget(status_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, status_dock)
        self.dock_widgets["system_status"] = status_dock
        
        # Data Analysis Tools (Bottom, tabbed with system status)
        analysis_dock = QDockWidget("Data Analysis", self)
        analysis_dock.setObjectName("DataAnalysisDock")
        analysis_widget = QLabel("Data Analysis Tools\n(To be implemented)")
        analysis_widget.setStyleSheet("padding: 10px; background-color: #353535;")
        analysis_dock.setWidget(analysis_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, analysis_dock)
        self.tabifyDockWidget(status_dock, analysis_dock)
        self.dock_widgets["data_analysis"] = analysis_dock
        
        # Waterfall Display (Right side, tabbed with model monitor)
        waterfall_dock = QDockWidget("Waterfall Display", self)
        waterfall_dock.setObjectName("WaterfallDock")
        waterfall_widget = QLabel("Waterfall Display\n(To be implemented)")
        waterfall_widget.setStyleSheet("padding: 10px; background-color: #353535;")
        waterfall_dock.setWidget(waterfall_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, waterfall_dock)
        self.tabifyDockWidget(model_dock, waterfall_dock)
        self.dock_widgets["waterfall"] = waterfall_dock
        
        # Populate dock menu with visibility toggles
        for name, dock in self.dock_widgets.items():
            action = self.dock_menu.addAction(dock.windowTitle())
            action.setCheckable(True)
            action.setChecked(dock.isVisible())
            action.toggled.connect(lambda checked, d=dock: d.setVisible(checked))
            dock.visibilityChanged.connect(lambda visible, a=action: a.setChecked(visible))
            dock.visibilityChanged.connect(
                lambda visible, n=name: self.dock_visibility_changed.emit(n, visible)
            )
        
    def _setup_status_bar(self) -> None:
        """Setup comprehensive status bar with system information."""
        status_bar = self.statusBar()
        
        # Main status message
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)
        
        # Connection status
        self.connection_label = QLabel(f"Connection: {self.connection_status}")
        self.connection_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        status_bar.addPermanentWidget(self.connection_label)
        
        # Hardware status
        self.hardware_label = QLabel(f"Hardware: {self.hardware_status}")
        self.hardware_label.setStyleSheet("color: #ffa500; font-weight: bold;")
        status_bar.addPermanentWidget(self.hardware_label)
        
        # Processing status
        self.processing_label = QLabel(f"Processing: {self.processing_status}")
        self.processing_label.setStyleSheet("color: #4ecdc4; font-weight: bold;")
        status_bar.addPermanentWidget(self.processing_label)
        
        # Memory usage
        self.memory_label = QLabel("Memory: 0%")
        status_bar.addPermanentWidget(self.memory_label)
        
    def _setup_connections(self) -> None:
        """Connect signals and slots between components."""
        # Connect application signals
        self.app.theme_changed.connect(self._on_theme_changed)
        self.app.settings_changed.connect(self._on_settings_changed)
        self.app.shutdown_requested.connect(self._prepare_shutdown)
        
        # Connect dock widget signals
        for dock in self.dock_widgets.values():
            dock.dockLocationChanged.connect(self._on_dock_location_changed)
            
        self.logger.info("Signal-slot connections established")
        
    def _setup_sdr_connections(self) -> None:
        """Setup connections between SDR components."""
        try:
            # Connect hardware manager to control panel
            self.hardware_manager.hardware_info_updated.connect(
                self.sdr_control_panel.set_hardware_info
            )
            self.hardware_manager.device_selected.connect(
                self._on_hardware_device_selected
            )
            self.hardware_manager.connection_requested.connect(
                self._on_hardware_connection_requested
            )
            
            # Connect control panel to preset manager
            self.sdr_control_panel.config_changed.connect(
                self.preset_manager.set_current_config
            )
            self.sdr_control_panel.connection_requested.connect(
                self._on_sdr_connection_requested
            )
            self.sdr_control_panel.disconnection_requested.connect(
                self._on_sdr_disconnection_requested
            )
            
            # Connect preset manager to control panel
            self.preset_manager.preset_loaded.connect(
                self._on_preset_loaded
            )
            
            self.logger.info("SDR component connections established")
            
        except AttributeError as e:
            self.logger.warning(f"SDR connections not available: {e}")
            
    @Slot(object)
    def _on_hardware_device_selected(self, device_info) -> None:
        """Handle hardware device selection."""
        self.logger.info(f"Hardware device selected: {device_info.name}")
        self.update_hardware_status(device_info.name)
        
    @Slot(object)
    def _on_hardware_connection_requested(self, device_info) -> None:
        """Handle hardware connection request."""
        self.logger.info(f"Connection requested to: {device_info.name}")
        
        # Update status to connecting
        self.sdr_control_panel.set_connection_status(
            self.sdr_control_panel.ConnectionStatus.CONNECTING
        )
        self.update_connection_status("Connecting")
        
        # Simulate connection process (in real implementation, this would connect to actual hardware)
        QTimer.singleShot(2000, lambda: self._simulate_connection_complete(device_info))
        
    def _simulate_connection_complete(self, device_info) -> None:
        """Simulate connection completion."""
        try:
            # Update connection status
            self.sdr_control_panel.set_connection_status(
                self.sdr_control_panel.ConnectionStatus.CONNECTED
            )
            self.update_connection_status("Connected")
            self.update_hardware_status(f"{device_info.name} (Connected)")
            
            # Update hardware manager device status
            self.hardware_manager.update_device_status(
                device_info.device_id,
                self.hardware_manager.DeviceStatus.CONNECTED
            )
            
            self.show_status_message(f"Connected to {device_info.name}")
            self.logger.info(f"Connected to {device_info.name}")
            
        except Exception as e:
            self.logger.error(f"Connection simulation failed: {e}")
            self._simulate_connection_error(device_info, str(e))
            
    def _simulate_connection_error(self, device_info, error_msg: str) -> None:
        """Simulate connection error."""
        self.sdr_control_panel.set_connection_status(
            self.sdr_control_panel.ConnectionStatus.ERROR
        )
        self.update_connection_status("Error")
        
        # Update hardware manager device status
        self.hardware_manager.update_device_status(
            device_info.device_id,
            self.hardware_manager.DeviceStatus.ERROR,
            error_msg
        )
        
        self.show_status_message(f"Connection failed: {error_msg}")
        
    @Slot()
    def _on_sdr_connection_requested(self) -> None:
        """Handle SDR connection request from control panel."""
        # Get current device from hardware manager
        current_device = self.hardware_manager.get_current_device()
        if current_device:
            self._on_hardware_connection_requested(current_device)
        else:
            self.sdr_control_panel.show_error_message(
                "No Device Selected",
                "Please select a hardware device first."
            )
            
    @Slot()
    def _on_sdr_disconnection_requested(self) -> None:
        """Handle SDR disconnection request."""
        self.logger.info("Disconnection requested")
        
        # Update status
        self.sdr_control_panel.set_connection_status(
            self.sdr_control_panel.ConnectionStatus.DISCONNECTED
        )
        self.update_connection_status("Disconnected")
        self.update_hardware_status("Not Connected")
        
        # Update hardware manager
        current_device = self.hardware_manager.get_current_device()
        if current_device:
            self.hardware_manager.update_device_status(
                current_device.device_id,
                self.hardware_manager.DeviceStatus.DETECTED
            )
            
        self.show_status_message("Disconnected from hardware")
        
    @Slot(object)
    def _on_preset_loaded(self, preset) -> None:
        """Handle preset loading."""
        self.logger.info(f"Preset loaded: {preset.metadata.name}")
        self.sdr_control_panel.set_config(preset.config)
        self.show_status_message(f"Loaded preset: {preset.metadata.name}")
    
    # Model monitor signal handlers
    @Slot(str, str)
    def _on_model_load_requested(self, model_name: str, version: str) -> None:
        """Handle model load request from model monitor."""
        self.logger.info(f"Model load requested: {model_name} v{version}")
        self.show_status_message(f"Loading model: {model_name} v{version}")
        
        # In a real implementation, this would interface with the model manager
        # For now, simulate successful loading
        if self.model_monitor:
            QTimer.singleShot(2000, lambda: self._simulate_model_loaded(model_name, version))
    
    @Slot(str)
    def _on_model_unload_requested(self, model_name: str) -> None:
        """Handle model unload request from model monitor."""
        self.logger.info(f"Model unload requested: {model_name}")
        self.show_status_message(f"Unloading model: {model_name}")
        
        # Simulate unloading
        if self.model_monitor:
            self.model_monitor.remove_model(model_name)
            self.show_status_message(f"Model {model_name} unloaded")
    
    @Slot(str)
    def _on_model_selected(self, model_name: str) -> None:
        """Handle model selection from model monitor."""
        self.logger.info(f"Model selected: {model_name}")
        self.show_status_message(f"Selected model: {model_name}")
    
    @Slot(str, str)
    def _on_training_control_requested(self, action: str, model_name: str) -> None:
        """Handle training control request from model monitor."""
        self.logger.info(f"Training control requested: {action} for {model_name}")
        self.show_status_message(f"Training {action} for {model_name}")
        
        # Simulate training control
        if action == "start" and self.model_monitor:
            self._simulate_training_start(model_name)
        elif action == "stop" and self.model_monitor:
            self._simulate_training_stop(model_name)
    
    def _simulate_model_loaded(self, model_name: str, version: str) -> None:
        """Simulate successful model loading."""
        if self.model_monitor:
            # Create mock metadata
            from datetime import datetime
            
            # Add model to monitor
            self.model_monitor.add_model(model_name, version)
            
            # Update with some initial metrics
            self.model_monitor.update_model_metrics(model_name, {
                'accuracy': 0.85,
                'loss': 0.25,
                'memory_usage_mb': 128.0,
                'device': 'cpu'
            })
            
            self.show_status_message(f"Model {model_name} v{version} loaded successfully")
    
    def _simulate_training_start(self, model_name: str) -> None:
        """Simulate training start."""
        if self.model_monitor:
            # Update model status
            self.model_monitor.update_model_status(model_name, "training")
            
            # Start simulated training progress
            self._training_simulation_timer = QTimer()
            self._training_epoch = 0
            self._training_model = model_name
            
            def update_training():
                if self._training_epoch < 10:  # Simulate 10 epochs
                    self._training_epoch += 1
                    loss = 0.5 * (0.9 ** self._training_epoch)  # Decreasing loss
                    accuracy = 0.6 + 0.3 * (1 - 0.9 ** self._training_epoch)  # Increasing accuracy
                    
                    self.model_monitor.update_training_progress(
                        model_name, self._training_epoch, 10, 
                        25, 50, loss, accuracy,
                        val_loss=loss * 1.1, val_accuracy=accuracy * 0.95
                    )
                    
                    # Update model metrics
                    self.model_monitor.update_model_metrics(model_name, {
                        'accuracy': accuracy,
                        'loss': loss,
                        'memory_usage_mb': 256.0,
                        'device': 'cpu'
                    })
                else:
                    # Training complete
                    self._training_simulation_timer.stop()
                    self.model_monitor.update_model_status(model_name, "loaded")
                    self.show_status_message(f"Training completed for {model_name}")
            
            self._training_simulation_timer.timeout.connect(update_training)
            self._training_simulation_timer.start(2000)  # Update every 2 seconds
    
    def _simulate_training_stop(self, model_name: str) -> None:
        """Simulate training stop."""
        if hasattr(self, '_training_simulation_timer'):
            self._training_simulation_timer.stop()
        
        if self.model_monitor:
            self.model_monitor.update_model_status(model_name, "loaded")
            self.show_status_message(f"Training stopped for {model_name}")
        
    def _setup_timers(self) -> None:
        """Setup periodic update timers."""
        # Main update timer for GUI refresh
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._periodic_update)
        update_rate = self.app.get_setting("performance/update_rate", 30)
        self.update_timer.start(1000 // update_rate)  # Convert FPS to ms
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(1000)  # Update status every second
        
    @Slot()
    def _periodic_update(self) -> None:
        """Periodic update for GUI components."""
        # This will be expanded as components are added
        pass
        
    @Slot()
    def _update_status(self) -> None:
        """Update status bar information."""
        import psutil
        
        # Update memory usage
        memory_percent = psutil.virtual_memory().percent
        self.memory_label.setText(f"Memory: {memory_percent:.1f}%")
        
        # Update status colors based on values
        if memory_percent > 80:
            self.memory_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        elif memory_percent > 60:
            self.memory_label.setStyleSheet("color: #ffa500; font-weight: bold;")
        else:
            self.memory_label.setStyleSheet("color: #4ecdc4; font-weight: bold;")
            
    def update_connection_status(self, status: str) -> None:
        """Update connection status display."""
        self.connection_status = status
        self.connection_label.setText(f"Connection: {status}")
        
        # Update color based on status
        if status.lower() in ["connected", "active"]:
            self.connection_label.setStyleSheet("color: #4ecdc4; font-weight: bold;")
        elif status.lower() in ["connecting", "initializing"]:
            self.connection_label.setStyleSheet("color: #ffa500; font-weight: bold;")
        else:
            self.connection_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
            
    def update_hardware_status(self, status: str) -> None:
        """Update hardware status display."""
        self.hardware_status = status
        self.hardware_label.setText(f"Hardware: {status}")
        
        # Update color based on status
        if status.lower() in ["detected", "ready", "operational"]:
            self.hardware_label.setStyleSheet("color: #4ecdc4; font-weight: bold;")
        elif status.lower() in ["simulation", "virtual"]:
            self.hardware_label.setStyleSheet("color: #ffa500; font-weight: bold;")
        else:
            self.hardware_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
            
    def update_processing_status(self, status: str) -> None:
        """Update processing status display."""
        self.processing_status = status
        self.processing_label.setText(f"Processing: {status}")
        
        # Update color based on status
        if status.lower() in ["active", "processing", "running"]:
            self.processing_label.setStyleSheet("color: #4ecdc4; font-weight: bold;")
        elif status.lower() in ["paused", "waiting"]:
            self.processing_label.setStyleSheet("color: #ffa500; font-weight: bold;")
        else:
            self.processing_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
            
    def show_status_message(self, message: str, timeout: int = 5000) -> None:
        """Show temporary status message."""
        self.status_label.setText(message)
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self.status_label.setText("Ready"))
            
    # Slot implementations for menu actions
    @Slot()
    def _new_session(self) -> None:
        """Create new session."""
        self.show_status_message("New session created")
        
    @Slot()
    def _open_session(self) -> None:
        """Open existing session."""
        self.show_status_message("Open session dialog (to be implemented)")
        
    @Slot()
    def _save_session(self) -> None:
        """Save current session."""
        self.show_status_message("Session saved")
        
    @Slot()
    def _import_iq_data(self) -> None:
        """Import I/Q data."""
        self.show_status_message("Import I/Q data (to be implemented)")
        
    @Slot()
    def _import_config(self) -> None:
        """Import configuration."""
        self.show_status_message("Import configuration (to be implemented)")
        
    @Slot()
    def _export_spectrum(self) -> None:
        """Export spectrum data."""
        self.show_status_message("Export spectrum data (to be implemented)")
        
    @Slot()
    def _export_screenshot(self) -> None:
        """Export screenshot."""
        self.show_status_message("Export screenshot (to be implemented)")
        
    @Slot()
    def _show_sdr_config(self) -> None:
        """Show SDR configuration dialog."""
        self.show_status_message("SDR configuration (to be implemented)")
        
    @Slot()
    def _show_signal_generator(self) -> None:
        """Show signal generator."""
        self.show_status_message("Signal generator (to be implemented)")
        
    @Slot()
    def _show_measurement_tools(self) -> None:
        """Show measurement tools."""
        self.show_status_message("Measurement tools (to be implemented)")
        
    @Slot()
    def _show_performance_monitor(self) -> None:
        """Show performance monitor."""
        self.show_status_message("Performance monitor (to be implemented)")
        
    @Slot()
    def _show_diagnostics(self) -> None:
        """Show system diagnostics."""
        self.show_status_message("System diagnostics (to be implemented)")
        
    @Slot()
    def _show_user_manual(self) -> None:
        """Show user manual."""
        self.show_status_message("User manual (to be implemented)")
        
    @Slot()
    def _show_shortcuts(self) -> None:
        """Show keyboard shortcuts."""
        self.show_status_message("Keyboard shortcuts (to be implemented)")
        
    @Slot()
    def _show_about(self) -> None:
        """Show about dialog."""
        self.show_status_message("About GeminiSDR (to be implemented)")
        
    @Slot()
    def _reset_layout(self) -> None:
        """Reset window layout to default."""
        # Reset all dock widgets to default positions
        for dock in self.dock_widgets.values():
            dock.setFloating(False)
            dock.show()
            
        # Re-arrange docks
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_widgets["sdr_control"])
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_widgets["model_monitor"])
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_widgets["system_status"])
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_widgets["data_analysis"])
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_widgets["waterfall"])
        
        # Re-create tabs
        self.tabifyDockWidget(self.dock_widgets["system_status"], self.dock_widgets["data_analysis"])
        self.tabifyDockWidget(self.dock_widgets["model_monitor"], self.dock_widgets["waterfall"])
        
        self.show_status_message("Layout reset to default")
        
    @Slot()
    def _save_layout(self) -> None:
        """Save current window layout."""
        self.app.set_setting("window/geometry", self.saveGeometry())
        self.app.set_setting("window/state", self.saveState())
        self.show_status_message("Layout saved")
        
    @Slot()
    def _load_layout(self) -> None:
        """Load saved window layout."""
        self._load_window_state()
        self.show_status_message("Layout loaded")
        
    def _load_window_state(self) -> None:
        """Load saved window state and geometry."""
        geometry = self.app.get_setting("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)
            
        state = self.app.get_setting("window/state")
        if state:
            self.restoreState(state)
            
    def _save_window_state(self) -> None:
        """Save current window state and geometry."""
        self.app.set_setting("window/geometry", self.saveGeometry())
        self.app.set_setting("window/state", self.saveState())
        
    # Signal handlers
    @Slot(str)
    def _on_theme_changed(self, theme_name: str) -> None:
        """Handle theme change."""
        self.show_status_message(f"Theme changed to {theme_name}")
        
    @Slot(dict)
    def _on_settings_changed(self, settings: Dict[str, Any]) -> None:
        """Handle settings change."""
        # Update timers if update rate changed
        if "performance/update_rate" in settings:
            update_rate = settings["performance/update_rate"]
            self.update_timer.start(1000 // update_rate)
            
    @Slot(Qt.DockWidgetArea)
    def _on_dock_location_changed(self, area: Qt.DockWidgetArea) -> None:
        """Handle dock widget location change."""
        self.layout_changed.emit()
        
    @Slot()
    def _prepare_shutdown(self) -> None:
        """Prepare for application shutdown."""
        self.logger.info("Preparing for shutdown")
        self._save_window_state()
        
    def closeEvent(self, event):
        """Handle application close event."""
        self.logger.info("MainWindow close event")
        
        # Save window state
        self._save_window_state()
        
        # Cleanup resources
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
            
        # Accept the close event
        event.accept()
        
        self.logger.info("MainWindow closed")


def main():
    """Main entry point for the GUI application."""
    app = GeminiSDRApplication(sys.argv)
    
    # Create and show main window
    window = MainWindow(app)
    window.show()
    
    # Start the application event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())