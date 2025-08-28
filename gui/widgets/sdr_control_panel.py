"""
SDR Control Panel Widget

This module provides intuitive parameter control widgets for SDR hardware configuration,
including frequency, gain, sample rate controls with real-time validation and preset management.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QSpinBox, QDoubleSpinBox, QSlider, QLabel,
                               QPushButton, QComboBox, QCheckBox, QLineEdit,
                               QFrame, QSplitter, QProgressBar, QTabWidget,
                               QFormLayout, QGridLayout, QButtonGroup,
                               QMessageBox, QToolTip)
from PySide6.QtCore import Signal, Slot, QTimer, Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QPalette, QColor, QIcon, QPixmap, QPainter
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging


class SDRMode(Enum):
    """SDR operation modes."""
    HARDWARE = "hardware"
    SIMULATION = "simulation"
    OFFLINE = "offline"


class ConnectionStatus(Enum):
    """SDR connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class SDRConfig:
    """SDR configuration parameters."""
    center_freq_hz: float = 100e6
    sample_rate_hz: float = 2e6
    bandwidth_hz: float = 2e6
    gain_db: float = 30.0
    agc_enabled: bool = False
    mode: SDRMode = SDRMode.SIMULATION
    device_id: str = "ip:192.168.4.1"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['mode'] = self.mode.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SDRConfig':
        """Create from dictionary."""
        if 'mode' in data and isinstance(data['mode'], str):
            data['mode'] = SDRMode(data['mode'])
        return cls(**data)


class StatusIndicator(QWidget):
    """Visual status indicator widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.status = ConnectionStatus.DISCONNECTED
        self.colors = {
            ConnectionStatus.DISCONNECTED: QColor(200, 60, 60),  # Red
            ConnectionStatus.CONNECTING: QColor(255, 165, 0),    # Orange
            ConnectionStatus.CONNECTED: QColor(60, 200, 60),     # Green
            ConnectionStatus.ERROR: QColor(200, 60, 200)         # Magenta
        }
        
    def set_status(self, status: ConnectionStatus) -> None:
        """Set the status and update display."""
        self.status = status
        self.update()
        
    def paintEvent(self, event):
        """Paint the status indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw circle
        color = self.colors.get(self.status, self.colors[ConnectionStatus.DISCONNECTED])
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, 12, 12)


class ParameterControl(QWidget):
    """Base class for parameter control widgets with validation."""
    
    value_changed = Signal(object)  # Emits the new value
    validation_failed = Signal(str)  # Emits error message
    
    def __init__(self, name: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.name = name
        self.unit = unit
        self.min_value = None
        self.max_value = None
        self.validation_enabled = True
        
    def set_range(self, min_val: float, max_val: float) -> None:
        """Set valid range for the parameter."""
        self.min_value = min_val
        self.max_value = max_val
        
    def validate_value(self, value: float) -> Tuple[bool, str]:
        """Validate parameter value."""
        if not self.validation_enabled:
            return True, ""
            
        if self.min_value is not None and value < self.min_value:
            return False, f"{self.name} must be >= {self.min_value} {self.unit}"
            
        if self.max_value is not None and value > self.max_value:
            return False, f"{self.name} must be <= {self.max_value} {self.unit}"
            
        return True, ""
        
    def set_validation_enabled(self, enabled: bool) -> None:
        """Enable or disable validation."""
        self.validation_enabled = enabled


class FrequencyControl(ParameterControl):
    """Frequency control with fine tuning capabilities."""
    
    def __init__(self, parent=None):
        super().__init__("Frequency", "Hz", parent)
        self.current_freq = 100e6
        self.fine_tune_step = 1000.0  # 1 kHz steps
        
        self._setup_ui()
        self._setup_connections()
        
        # Set PlutoSDR frequency range
        self.set_range(70e6, 6e9)
        
    def _setup_ui(self) -> None:
        """Setup frequency control UI."""
        layout = QVBoxLayout(self)
        
        # Main frequency input
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Center Freq:"))
        
        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setRange(70, 6000)  # MHz
        self.freq_spinbox.setValue(100.0)
        self.freq_spinbox.setSuffix(" MHz")
        self.freq_spinbox.setDecimals(6)
        self.freq_spinbox.setSingleStep(0.001)  # 1 kHz steps
        self.freq_spinbox.setMinimumWidth(120)
        freq_layout.addWidget(self.freq_spinbox)
        
        layout.addLayout(freq_layout)
        
        # Fine tuning slider
        fine_layout = QVBoxLayout()
        fine_layout.addWidget(QLabel("Fine Tune (±1 MHz):"))
        
        self.fine_slider = QSlider(Qt.Horizontal)
        self.fine_slider.setRange(-1000, 1000)  # ±1000 steps of 1 kHz = ±1 MHz
        self.fine_slider.setValue(0)
        self.fine_slider.setTickPosition(QSlider.TicksBelow)
        self.fine_slider.setTickInterval(250)
        fine_layout.addWidget(self.fine_slider)
        
        # Fine tune value display
        self.fine_label = QLabel("0 kHz")
        self.fine_label.setAlignment(Qt.AlignCenter)
        self.fine_label.setStyleSheet("QLabel { color: #888888; font-size: 10px; }")
        fine_layout.addWidget(self.fine_label)
        
        layout.addLayout(fine_layout)
        
        # Quick frequency buttons
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel("Quick:"))
        
        quick_freqs = [
            ("FM", 100e6),
            ("Air", 121.5e6),
            ("ISM", 433e6),
            ("WiFi", 2.4e9),
            ("GPS", 1.575e9)
        ]
        
        self.quick_buttons = []
        for name, freq in quick_freqs:
            btn = QPushButton(name)
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda checked, f=freq: self.set_frequency(f))
            quick_layout.addWidget(btn)
            self.quick_buttons.append(btn)
            
        layout.addLayout(quick_layout)
        
    def _setup_connections(self) -> None:
        """Setup signal connections."""
        self.freq_spinbox.valueChanged.connect(self._on_freq_changed)
        self.fine_slider.valueChanged.connect(self._on_fine_tune_changed)
        
    @Slot(float)
    def _on_freq_changed(self, freq_mhz: float) -> None:
        """Handle main frequency change."""
        freq_hz = freq_mhz * 1e6
        
        # Validate frequency
        valid, error_msg = self.validate_value(freq_hz)
        if not valid:
            self.validation_failed.emit(error_msg)
            # Reset to previous valid value
            self.freq_spinbox.blockSignals(True)
            self.freq_spinbox.setValue(self.current_freq / 1e6)
            self.freq_spinbox.blockSignals(False)
            return
            
        # Reset fine tune when main frequency changes
        self.fine_slider.blockSignals(True)
        self.fine_slider.setValue(0)
        self.fine_label.setText("0 kHz")
        self.fine_slider.blockSignals(False)
        
        self.current_freq = freq_hz
        self.value_changed.emit(freq_hz)
        
    @Slot(int)
    def _on_fine_tune_changed(self, steps: int) -> None:
        """Handle fine tuning change."""
        fine_offset = steps * self.fine_tune_step
        total_freq = self.current_freq + fine_offset
        
        # Validate total frequency
        valid, error_msg = self.validate_value(total_freq)
        if not valid:
            self.validation_failed.emit(error_msg)
            return
            
        # Update fine tune display
        if abs(fine_offset) >= 1e6:
            self.fine_label.setText(f"{fine_offset/1e6:.3f} MHz")
        elif abs(fine_offset) >= 1e3:
            self.fine_label.setText(f"{fine_offset/1e3:.1f} kHz")
        else:
            self.fine_label.setText(f"{fine_offset:.0f} Hz")
            
        self.value_changed.emit(total_freq)
        
    def set_frequency(self, freq_hz: float) -> None:
        """Set frequency programmatically."""
        freq_mhz = freq_hz / 1e6
        
        self.freq_spinbox.blockSignals(True)
        self.freq_spinbox.setValue(freq_mhz)
        self.freq_spinbox.blockSignals(False)
        
        self.fine_slider.blockSignals(True)
        self.fine_slider.setValue(0)
        self.fine_label.setText("0 kHz")
        self.fine_slider.blockSignals(False)
        
        self.current_freq = freq_hz
        self.value_changed.emit(freq_hz)
        
    def get_frequency(self) -> float:
        """Get current frequency in Hz."""
        fine_offset = self.fine_slider.value() * self.fine_tune_step
        return self.current_freq + fine_offset


class GainControl(ParameterControl):
    """Gain control with AGC support."""
    
    def __init__(self, parent=None):
        super().__init__("Gain", "dB", parent)
        self.agc_enabled = False
        
        self._setup_ui()
        self._setup_connections()
        
        # Set PlutoSDR gain range
        self.set_range(0, 70)
        
    def _setup_ui(self) -> None:
        """Setup gain control UI."""
        layout = QVBoxLayout(self)
        
        # AGC checkbox
        self.agc_checkbox = QCheckBox("Automatic Gain Control (AGC)")
        self.agc_checkbox.setChecked(False)
        layout.addWidget(self.agc_checkbox)
        
        # Manual gain control
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Manual Gain:"))
        
        self.gain_spinbox = QSpinBox()
        self.gain_spinbox.setRange(0, 70)
        self.gain_spinbox.setValue(30)
        self.gain_spinbox.setSuffix(" dB")
        self.gain_spinbox.setMinimumWidth(80)
        gain_layout.addWidget(self.gain_spinbox)
        
        layout.addLayout(gain_layout)
        
        # Gain slider
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 70)
        self.gain_slider.setValue(30)
        self.gain_slider.setTickPosition(QSlider.TicksBelow)
        self.gain_slider.setTickInterval(10)
        layout.addWidget(self.gain_slider)
        
        # Gain presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        
        gain_presets = [
            ("Low", 10),
            ("Med", 30),
            ("High", 50),
            ("Max", 70)
        ]
        
        for name, gain in gain_presets:
            btn = QPushButton(name)
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda checked, g=gain: self.set_gain(g))
            preset_layout.addWidget(btn)
            
        layout.addLayout(preset_layout)
        
    def _setup_connections(self) -> None:
        """Setup signal connections."""
        self.agc_checkbox.toggled.connect(self._on_agc_toggled)
        self.gain_spinbox.valueChanged.connect(self._on_gain_changed)
        self.gain_slider.valueChanged.connect(self._on_slider_changed)
        
    @Slot(bool)
    def _on_agc_toggled(self, enabled: bool) -> None:
        """Handle AGC toggle."""
        self.agc_enabled = enabled
        
        # Enable/disable manual controls
        self.gain_spinbox.setEnabled(not enabled)
        self.gain_slider.setEnabled(not enabled)
        
        # Emit AGC status change
        self.value_changed.emit({"agc_enabled": enabled, "gain_db": self.gain_spinbox.value()})
        
    @Slot(int)
    def _on_gain_changed(self, gain: int) -> None:
        """Handle gain spinbox change."""
        if not self.agc_enabled:
            # Update slider
            self.gain_slider.blockSignals(True)
            self.gain_slider.setValue(gain)
            self.gain_slider.blockSignals(False)
            
            self.value_changed.emit({"agc_enabled": False, "gain_db": gain})
            
    @Slot(int)
    def _on_slider_changed(self, gain: int) -> None:
        """Handle gain slider change."""
        if not self.agc_enabled:
            # Update spinbox
            self.gain_spinbox.blockSignals(True)
            self.gain_spinbox.setValue(gain)
            self.gain_spinbox.blockSignals(False)
            
            self.value_changed.emit({"agc_enabled": False, "gain_db": gain})
            
    def set_gain(self, gain_db: float) -> None:
        """Set gain programmatically."""
        if not self.agc_enabled:
            self.gain_spinbox.blockSignals(True)
            self.gain_slider.blockSignals(True)
            
            self.gain_spinbox.setValue(int(gain_db))
            self.gain_slider.setValue(int(gain_db))
            
            self.gain_spinbox.blockSignals(False)
            self.gain_slider.blockSignals(False)
            
            self.value_changed.emit({"agc_enabled": False, "gain_db": gain_db})
            
    def set_agc_enabled(self, enabled: bool) -> None:
        """Set AGC enabled state."""
        self.agc_checkbox.blockSignals(True)
        self.agc_checkbox.setChecked(enabled)
        self.agc_checkbox.blockSignals(False)
        
        self._on_agc_toggled(enabled)
        
    def get_gain_config(self) -> Dict[str, Any]:
        """Get current gain configuration."""
        return {
            "agc_enabled": self.agc_enabled,
            "gain_db": self.gain_spinbox.value()
        }


class SampleRateControl(ParameterControl):
    """Sample rate control with bandwidth coupling."""
    
    def __init__(self, parent=None):
        super().__init__("Sample Rate", "Sps", parent)
        self.auto_bandwidth = True
        
        self._setup_ui()
        self._setup_connections()
        
        # Set PlutoSDR sample rate range
        self.set_range(520833, 61440000)
        
    def _setup_ui(self) -> None:
        """Setup sample rate control UI."""
        layout = QVBoxLayout(self)
        
        # Sample rate input
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Sample Rate:"))
        
        self.rate_spinbox = QDoubleSpinBox()
        self.rate_spinbox.setRange(0.52, 61.44)  # MSps
        self.rate_spinbox.setValue(2.0)
        self.rate_spinbox.setSuffix(" MSps")
        self.rate_spinbox.setDecimals(3)
        self.rate_spinbox.setSingleStep(0.1)
        self.rate_spinbox.setMinimumWidth(100)
        rate_layout.addWidget(self.rate_spinbox)
        
        layout.addLayout(rate_layout)
        
        # Bandwidth control
        bw_layout = QHBoxLayout()
        
        self.auto_bw_checkbox = QCheckBox("Auto Bandwidth")
        self.auto_bw_checkbox.setChecked(True)
        bw_layout.addWidget(self.auto_bw_checkbox)
        
        self.bw_spinbox = QDoubleSpinBox()
        self.bw_spinbox.setRange(0.2, 56.0)  # MHz
        self.bw_spinbox.setValue(1.6)  # 80% of 2 MSps
        self.bw_spinbox.setSuffix(" MHz")
        self.bw_spinbox.setDecimals(3)
        self.bw_spinbox.setEnabled(False)
        self.bw_spinbox.setMinimumWidth(100)
        bw_layout.addWidget(self.bw_spinbox)
        
        layout.addLayout(bw_layout)
        
        # Common sample rates
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        
        rate_presets = [
            ("1M", 1e6),
            ("2M", 2e6),
            ("5M", 5e6),
            ("10M", 10e6),
            ("20M", 20e6)
        ]
        
        for name, rate in rate_presets:
            btn = QPushButton(name)
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda checked, r=rate: self.set_sample_rate(r))
            preset_layout.addWidget(btn)
            
        layout.addLayout(preset_layout)
        
    def _setup_connections(self) -> None:
        """Setup signal connections."""
        self.rate_spinbox.valueChanged.connect(self._on_rate_changed)
        self.auto_bw_checkbox.toggled.connect(self._on_auto_bw_toggled)
        self.bw_spinbox.valueChanged.connect(self._on_bw_changed)
        
    @Slot(float)
    def _on_rate_changed(self, rate_msps: float) -> None:
        """Handle sample rate change."""
        rate_hz = rate_msps * 1e6
        
        # Validate sample rate
        valid, error_msg = self.validate_value(rate_hz)
        if not valid:
            self.validation_failed.emit(error_msg)
            return
            
        # Update bandwidth if auto mode
        if self.auto_bandwidth:
            bw_hz = rate_hz * 0.8  # 80% of sample rate
            bw_mhz = bw_hz / 1e6
            
            self.bw_spinbox.blockSignals(True)
            self.bw_spinbox.setValue(bw_mhz)
            self.bw_spinbox.blockSignals(False)
            
        self.value_changed.emit({
            "sample_rate_hz": rate_hz,
            "bandwidth_hz": self.bw_spinbox.value() * 1e6
        })
        
    @Slot(bool)
    def _on_auto_bw_toggled(self, enabled: bool) -> None:
        """Handle auto bandwidth toggle."""
        self.auto_bandwidth = enabled
        self.bw_spinbox.setEnabled(not enabled)
        
        if enabled:
            # Update bandwidth to 80% of sample rate
            rate_hz = self.rate_spinbox.value() * 1e6
            bw_hz = rate_hz * 0.8
            bw_mhz = bw_hz / 1e6
            
            self.bw_spinbox.blockSignals(True)
            self.bw_spinbox.setValue(bw_mhz)
            self.bw_spinbox.blockSignals(False)
            
            self.value_changed.emit({
                "sample_rate_hz": rate_hz,
                "bandwidth_hz": bw_hz
            })
            
    @Slot(float)
    def _on_bw_changed(self, bw_mhz: float) -> None:
        """Handle bandwidth change."""
        if not self.auto_bandwidth:
            self.value_changed.emit({
                "sample_rate_hz": self.rate_spinbox.value() * 1e6,
                "bandwidth_hz": bw_mhz * 1e6
            })
            
    def set_sample_rate(self, rate_hz: float) -> None:
        """Set sample rate programmatically."""
        rate_msps = rate_hz / 1e6
        
        self.rate_spinbox.blockSignals(True)
        self.rate_spinbox.setValue(rate_msps)
        self.rate_spinbox.blockSignals(False)
        
        self._on_rate_changed(rate_msps)
        
    def get_sample_rate_config(self) -> Dict[str, Any]:
        """Get current sample rate configuration."""
        return {
            "sample_rate_hz": self.rate_spinbox.value() * 1e6,
            "bandwidth_hz": self.bw_spinbox.value() * 1e6,
            "auto_bandwidth": self.auto_bandwidth
        }


class SDRControlPanel(QWidget):
    """Main SDR parameter control interface."""
    
    # Signals
    config_changed = Signal(SDRConfig)
    connection_requested = Signal()
    disconnection_requested = Signal()
    preset_save_requested = Signal(str, SDRConfig)
    preset_load_requested = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("GeminiSDR.SDRControlPanel")
        
        # Current configuration
        self.current_config = SDRConfig()
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.hardware_info = {}
        
        # Animation for status changes
        self.status_animation = None
        
        self._setup_ui()
        self._setup_connections()
        self._setup_validation_timer()
        
        self.logger.info("SDR Control Panel initialized")
        
    def _setup_ui(self) -> None:
        """Setup the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Connection status and controls
        connection_group = self._create_connection_group()
        layout.addWidget(connection_group)
        
        # Parameter controls in tabs
        self.tab_widget = QTabWidget()
        
        # Frequency tab
        freq_tab = QWidget()
        freq_layout = QVBoxLayout(freq_tab)
        self.freq_control = FrequencyControl()
        freq_layout.addWidget(self.freq_control)
        freq_layout.addStretch()
        self.tab_widget.addTab(freq_tab, "Frequency")
        
        # Gain tab
        gain_tab = QWidget()
        gain_layout = QVBoxLayout(gain_tab)
        self.gain_control = GainControl()
        gain_layout.addWidget(self.gain_control)
        gain_layout.addStretch()
        self.tab_widget.addTab(gain_tab, "Gain")
        
        # Sample Rate tab
        rate_tab = QWidget()
        rate_layout = QVBoxLayout(rate_tab)
        self.rate_control = SampleRateControl()
        rate_layout.addWidget(self.rate_control)
        rate_layout.addStretch()
        self.tab_widget.addTab(rate_tab, "Sample Rate")
        
        layout.addWidget(self.tab_widget)
        
        # Status and information
        status_group = self._create_status_group()
        layout.addWidget(status_group)
        
        layout.addStretch()
        
    def _create_connection_group(self) -> QGroupBox:
        """Create connection control group."""
        group = QGroupBox("Connection")
        layout = QVBoxLayout(group)
        
        # Status display
        status_layout = QHBoxLayout()
        
        self.status_indicator = StatusIndicator()
        status_layout.addWidget(self.status_indicator)
        
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("QLabel { font-weight: bold; }")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        # Mode selection
        self.mode_combo = QComboBox()
        for mode in SDRMode:
            self.mode_combo.addItem(mode.value.title(), mode)
        self.mode_combo.setCurrentText(self.current_config.mode.value.title())
        status_layout.addWidget(QLabel("Mode:"))
        status_layout.addWidget(self.mode_combo)
        
        layout.addLayout(status_layout)
        
        # Connection controls
        button_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.disconnect_btn)
        
        layout.addLayout(button_layout)
        
        return group
        
    def _create_status_group(self) -> QGroupBox:
        """Create status information group."""
        group = QGroupBox("Status")
        layout = QFormLayout(group)
        
        # Hardware information
        self.device_label = QLabel("Not Connected")
        layout.addRow("Device:", self.device_label)
        
        self.freq_label = QLabel("--")
        layout.addRow("Frequency:", self.freq_label)
        
        self.rate_label = QLabel("--")
        layout.addRow("Sample Rate:", self.rate_label)
        
        self.gain_label = QLabel("--")
        layout.addRow("Gain:", self.gain_label)
        
        return group
        
    def _setup_connections(self) -> None:
        """Setup signal connections."""
        # Connection controls
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        self.disconnect_btn.clicked.connect(self._on_disconnect_clicked)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        
        # Parameter controls
        self.freq_control.value_changed.connect(self._on_frequency_changed)
        self.freq_control.validation_failed.connect(self._on_validation_error)
        
        self.gain_control.value_changed.connect(self._on_gain_changed)
        self.gain_control.validation_failed.connect(self._on_validation_error)
        
        self.rate_control.value_changed.connect(self._on_sample_rate_changed)
        self.rate_control.validation_failed.connect(self._on_validation_error)
        
    def _setup_validation_timer(self) -> None:
        """Setup timer for parameter validation feedback."""
        self.validation_timer = QTimer()
        self.validation_timer.setSingleShot(True)
        self.validation_timer.timeout.connect(self._clear_validation_message)
        
    @Slot()
    def _on_connect_clicked(self) -> None:
        """Handle connect button click."""
        self.set_connection_status(ConnectionStatus.CONNECTING)
        self.connection_requested.emit()
        
    @Slot()
    def _on_disconnect_clicked(self) -> None:
        """Handle disconnect button click."""
        self.disconnection_requested.emit()
        
    @Slot(str)
    def _on_mode_changed(self, mode_text: str) -> None:
        """Handle mode change."""
        for mode in SDRMode:
            if mode.value.title() == mode_text:
                self.current_config.mode = mode
                self.config_changed.emit(self.current_config)
                break
                
    @Slot(object)
    def _on_frequency_changed(self, freq_hz: float) -> None:
        """Handle frequency change."""
        self.current_config.center_freq_hz = freq_hz
        self._update_status_display()
        self.config_changed.emit(self.current_config)
        
    @Slot(object)
    def _on_gain_changed(self, gain_config: Dict[str, Any]) -> None:
        """Handle gain change."""
        self.current_config.gain_db = gain_config["gain_db"]
        self.current_config.agc_enabled = gain_config["agc_enabled"]
        self._update_status_display()
        self.config_changed.emit(self.current_config)
        
    @Slot(object)
    def _on_sample_rate_changed(self, rate_config: Dict[str, Any]) -> None:
        """Handle sample rate change."""
        self.current_config.sample_rate_hz = rate_config["sample_rate_hz"]
        self.current_config.bandwidth_hz = rate_config["bandwidth_hz"]
        self._update_status_display()
        self.config_changed.emit(self.current_config)
        
    @Slot(str)
    def _on_validation_error(self, error_msg: str) -> None:
        """Handle validation error."""
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("QLabel { color: #ff6b6b; font-weight: bold; }")
        
        # Clear error message after 3 seconds
        self.validation_timer.start(3000)
        
    @Slot()
    def _clear_validation_message(self) -> None:
        """Clear validation error message."""
        self._update_connection_status_display()
        
    def set_connection_status(self, status: ConnectionStatus) -> None:
        """Set connection status and update UI."""
        self.connection_status = status
        self.status_indicator.set_status(status)
        self._update_connection_status_display()
        
        # Update button states
        if status == ConnectionStatus.CONNECTED:
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
        elif status == ConnectionStatus.CONNECTING:
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(False)
        else:
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            
    def _update_connection_status_display(self) -> None:
        """Update connection status display."""
        status_text = self.connection_status.value.title()
        
        if self.connection_status == ConnectionStatus.CONNECTED:
            self.status_label.setText(f"Connected ({self.current_config.mode.value.title()})")
            self.status_label.setStyleSheet("QLabel { color: #4ecdc4; font-weight: bold; }")
        elif self.connection_status == ConnectionStatus.CONNECTING:
            self.status_label.setText("Connecting...")
            self.status_label.setStyleSheet("QLabel { color: #ffa500; font-weight: bold; }")
        elif self.connection_status == ConnectionStatus.ERROR:
            self.status_label.setText("Connection Error")
            self.status_label.setStyleSheet("QLabel { color: #ff6b6b; font-weight: bold; }")
        else:
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("QLabel { color: #888888; font-weight: bold; }")
            
    def _update_status_display(self) -> None:
        """Update parameter status display."""
        # Frequency
        freq = self.current_config.center_freq_hz
        if freq >= 1e9:
            freq_str = f"{freq/1e9:.3f} GHz"
        elif freq >= 1e6:
            freq_str = f"{freq/1e6:.3f} MHz"
        elif freq >= 1e3:
            freq_str = f"{freq/1e3:.3f} kHz"
        else:
            freq_str = f"{freq:.0f} Hz"
        self.freq_label.setText(freq_str)
        
        # Sample rate
        rate = self.current_config.sample_rate_hz
        if rate >= 1e6:
            rate_str = f"{rate/1e6:.3f} MSps"
        elif rate >= 1e3:
            rate_str = f"{rate/1e3:.3f} kSps"
        else:
            rate_str = f"{rate:.0f} Sps"
        self.rate_label.setText(rate_str)
        
        # Gain
        if self.current_config.agc_enabled:
            gain_str = "AGC Enabled"
        else:
            gain_str = f"{self.current_config.gain_db:.1f} dB"
        self.gain_label.setText(gain_str)
        
    def set_hardware_info(self, info: Dict[str, Any]) -> None:
        """Set hardware information."""
        self.hardware_info = info
        
        device_name = info.get("device_name", "Unknown Device")
        device_id = info.get("device_id", "")
        
        if device_id:
            self.device_label.setText(f"{device_name} ({device_id})")
        else:
            self.device_label.setText(device_name)
            
    def get_config(self) -> SDRConfig:
        """Get current SDR configuration."""
        return self.current_config
        
    def set_config(self, config: SDRConfig) -> None:
        """Set SDR configuration."""
        self.current_config = config
        
        # Update controls
        self.freq_control.set_frequency(config.center_freq_hz)
        self.gain_control.set_gain(config.gain_db)
        self.gain_control.set_agc_enabled(config.agc_enabled)
        self.rate_control.set_sample_rate(config.sample_rate_hz)
        
        # Update mode combo
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText(config.mode.value.title())
        self.mode_combo.blockSignals(False)
        
        # Update status display
        self._update_status_display()
        
    def show_error_message(self, title: str, message: str) -> None:
        """Show error message dialog."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()
        
    def show_info_message(self, title: str, message: str) -> None:
        """Show information message dialog."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()