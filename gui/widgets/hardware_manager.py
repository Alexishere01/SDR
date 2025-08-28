"""
Hardware Detection and Management

This module provides hardware detection, simulation mode support, and automatic
fallback capabilities for SDR devices.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QLabel, QPushButton, QComboBox, QTextEdit,
                               QProgressBar, QFrame, QListWidget, QListWidgetItem,
                               QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
                               QLineEdit, QSpinBox, QCheckBox, QTabWidget)
from PySide6.QtCore import (Signal, Slot, QTimer, QThread, QObject, Qt, 
                           QPropertyAnimation, QEasingCurve, QRect)
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QPainter
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import time
import threading
from pathlib import Path

# Import existing SDR interface
try:
    from core.sdr_interface import PlutoSDRInterface
    SDR_INTERFACE_AVAILABLE = True
except ImportError:
    SDR_INTERFACE_AVAILABLE = False
    
from .sdr_control_panel import SDRConfig, SDRMode, ConnectionStatus


class HardwareType(Enum):
    """Supported hardware types."""
    PLUTO_SDR = "PlutoSDR"
    HACKRF = "HackRF"
    RTL_SDR = "RTL-SDR"
    USRP = "USRP"
    SIMULATION = "Simulation"
    UNKNOWN = "Unknown"


class DeviceStatus(Enum):
    """Device status states."""
    NOT_DETECTED = "not_detected"
    DETECTED = "detected"
    CONNECTED = "connected"
    ERROR = "error"
    BUSY = "busy"


@dataclass
class HardwareInfo:
    """Hardware device information."""
    device_type: HardwareType
    device_id: str
    name: str
    status: DeviceStatus
    capabilities: Dict[str, Any]
    connection_string: str = ""
    error_message: str = ""
    last_seen: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['device_type'] = self.device_type.value
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareInfo':
        """Create from dictionary."""
        if 'device_type' in data and isinstance(data['device_type'], str):
            data['device_type'] = HardwareType(data['device_type'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = DeviceStatus(data['status'])
        return cls(**data)


class HardwareDetector(QObject):
    """Background hardware detection worker."""
    
    # Signals
    device_detected = Signal(HardwareInfo)
    device_lost = Signal(str)  # device_id
    detection_complete = Signal(list)  # List of HardwareInfo
    detection_progress = Signal(int, str)  # progress, status
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("GeminiSDR.HardwareDetector")
        self.running = False
        self.detection_interval = 5.0  # seconds
        
    def start_detection(self) -> None:
        """Start hardware detection."""
        self.running = True
        
    def stop_detection(self) -> None:
        """Stop hardware detection."""
        self.running = False
        
    def detect_hardware(self) -> List[HardwareInfo]:
        """Detect available hardware devices."""
        devices = []
        
        self.detection_progress.emit(10, "Detecting PlutoSDR devices...")
        
        # Detect PlutoSDR
        pluto_devices = self._detect_pluto_sdr()
        devices.extend(pluto_devices)
        
        self.detection_progress.emit(30, "Detecting HackRF devices...")
        
        # Detect HackRF (placeholder)
        hackrf_devices = self._detect_hackrf()
        devices.extend(hackrf_devices)
        
        self.detection_progress.emit(50, "Detecting RTL-SDR devices...")
        
        # Detect RTL-SDR (placeholder)
        rtl_devices = self._detect_rtl_sdr()
        devices.extend(rtl_devices)
        
        self.detection_progress.emit(70, "Detecting USRP devices...")
        
        # Detect USRP (placeholder)
        usrp_devices = self._detect_usrp()
        devices.extend(usrp_devices)
        
        self.detection_progress.emit(90, "Adding simulation device...")
        
        # Always add simulation device
        sim_device = self._create_simulation_device()
        devices.append(sim_device)
        
        self.detection_progress.emit(100, "Detection complete")
        
        self.logger.info(f"Detected {len(devices)} devices")
        return devices
        
    def _detect_pluto_sdr(self) -> List[HardwareInfo]:
        """Detect PlutoSDR devices."""
        devices = []
        
        try:
            if not SDR_INTERFACE_AVAILABLE:
                self.logger.info("SDR interface not available, skipping PlutoSDR detection")
                return devices
                
            # Try common PlutoSDR addresses
            addresses = [
                "ip:192.168.4.1",
                "ip:pluto.local",
                "usb:1.2.5"
            ]
            
            for addr in addresses:
                try:
                    # Quick connection test
                    interface = PlutoSDRInterface()
                    interface.connect()
                    
                    if not interface.simulation_mode:
                        # Real device detected
                        capabilities = {
                            "frequency_range": (70e6, 6e9),
                            "sample_rate_range": (520833, 61440000),
                            "gain_range": (0, 70),
                            "rx_channels": 1,
                            "tx_channels": 1,
                            "full_duplex": True
                        }
                        
                        device = HardwareInfo(
                            device_type=HardwareType.PLUTO_SDR,
                            device_id=f"pluto_{addr}",
                            name=f"PlutoSDR ({addr})",
                            status=DeviceStatus.DETECTED,
                            capabilities=capabilities,
                            connection_string=addr,
                            last_seen=time.time()
                        )
                        
                        devices.append(device)
                        self.logger.info(f"Detected PlutoSDR at {addr}")
                        break  # Found one, stop looking
                        
                except Exception as e:
                    self.logger.debug(f"PlutoSDR not found at {addr}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error detecting PlutoSDR: {e}")
            
        return devices
        
    def _detect_hackrf(self) -> List[HardwareInfo]:
        """Detect HackRF devices (placeholder)."""
        devices = []
        
        try:
            # Placeholder for HackRF detection
            # In a real implementation, this would use libhackrf or similar
            self.logger.debug("HackRF detection not implemented")
            
        except Exception as e:
            self.logger.error(f"Error detecting HackRF: {e}")
            
        return devices
        
    def _detect_rtl_sdr(self) -> List[HardwareInfo]:
        """Detect RTL-SDR devices (placeholder)."""
        devices = []
        
        try:
            # Placeholder for RTL-SDR detection
            # In a real implementation, this would use rtlsdr library
            self.logger.debug("RTL-SDR detection not implemented")
            
        except Exception as e:
            self.logger.error(f"Error detecting RTL-SDR: {e}")
            
        return devices
        
    def _detect_usrp(self) -> List[HardwareInfo]:
        """Detect USRP devices (placeholder)."""
        devices = []
        
        try:
            # Placeholder for USRP detection
            # In a real implementation, this would use UHD
            self.logger.debug("USRP detection not implemented")
            
        except Exception as e:
            self.logger.error(f"Error detecting USRP: {e}")
            
        return devices
        
    def _create_simulation_device(self) -> HardwareInfo:
        """Create simulation device info."""
        capabilities = {
            "frequency_range": (1e6, 6e9),
            "sample_rate_range": (1e3, 100e6),
            "gain_range": (0, 100),
            "rx_channels": 1,
            "tx_channels": 1,
            "full_duplex": True,
            "signal_types": ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"],
            "noise_simulation": True
        }
        
        return HardwareInfo(
            device_type=HardwareType.SIMULATION,
            device_id="simulation_0",
            name="Simulation Mode",
            status=DeviceStatus.DETECTED,
            capabilities=capabilities,
            connection_string="simulation",
            last_seen=time.time()
        )


class DeviceListWidget(QListWidget):
    """Custom list widget for displaying detected devices."""
    
    device_selected = Signal(HardwareInfo)
    device_double_clicked = Signal(HardwareInfo)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setAlternatingRowColors(True)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        
    def add_device(self, device: HardwareInfo) -> None:
        """Add device to the list."""
        item = QListWidgetItem()
        
        # Create display text
        status_icon = self._get_status_icon(device.status)
        text = f"{status_icon} {device.name}"
        
        if device.device_type == HardwareType.SIMULATION:
            text += " (Always Available)"
        elif device.status == DeviceStatus.ERROR:
            text += f" - Error: {device.error_message}"
        elif device.status == DeviceStatus.BUSY:
            text += " - Device Busy"
            
        item.setText(text)
        item.setData(Qt.UserRole, device)
        
        # Set color based on status
        if device.status == DeviceStatus.DETECTED:
            item.setForeground(QColor(60, 200, 60))  # Green
        elif device.status == DeviceStatus.CONNECTED:
            item.setForeground(QColor(60, 150, 255))  # Blue
        elif device.status == DeviceStatus.ERROR:
            item.setForeground(QColor(200, 60, 60))  # Red
        elif device.status == DeviceStatus.BUSY:
            item.setForeground(QColor(255, 165, 0))  # Orange
        else:
            item.setForeground(QColor(128, 128, 128))  # Gray
            
        # Set tooltip
        tooltip = self._create_device_tooltip(device)
        item.setToolTip(tooltip)
        
        self.addItem(item)
        
    def update_device(self, device: HardwareInfo) -> None:
        """Update existing device in the list."""
        for i in range(self.count()):
            item = self.item(i)
            stored_device = item.data(Qt.UserRole)
            if stored_device and stored_device.device_id == device.device_id:
                # Update the item
                self.takeItem(i)
                self.add_device(device)
                break
                
    def remove_device(self, device_id: str) -> None:
        """Remove device from the list."""
        for i in range(self.count()):
            item = self.item(i)
            stored_device = item.data(Qt.UserRole)
            if stored_device and stored_device.device_id == device_id:
                self.takeItem(i)
                break
                
    def get_selected_device(self) -> Optional[HardwareInfo]:
        """Get currently selected device."""
        current_item = self.currentItem()
        if current_item:
            return current_item.data(Qt.UserRole)
        return None
        
    def _get_status_icon(self, status: DeviceStatus) -> str:
        """Get status icon character."""
        icons = {
            DeviceStatus.NOT_DETECTED: "⚫",
            DeviceStatus.DETECTED: "🟢",
            DeviceStatus.CONNECTED: "🔵",
            DeviceStatus.ERROR: "🔴",
            DeviceStatus.BUSY: "🟡"
        }
        return icons.get(status, "⚫")
        
    def _create_device_tooltip(self, device: HardwareInfo) -> str:
        """Create tooltip for device."""
        tooltip = f"""<b>{device.name}</b><br/>
<b>Type:</b> {device.device_type.value}<br/>
<b>Status:</b> {device.status.value.title()}<br/>
<b>Device ID:</b> {device.device_id}<br/>"""

        if device.connection_string:
            tooltip += f"<b>Connection:</b> {device.connection_string}<br/>"
            
        if device.error_message:
            tooltip += f"<b>Error:</b> {device.error_message}<br/>"
            
        # Add capabilities
        caps = device.capabilities
        if "frequency_range" in caps:
            freq_min, freq_max = caps["frequency_range"]
            tooltip += f"<b>Frequency Range:</b> {freq_min/1e6:.1f} - {freq_max/1e9:.1f} GHz<br/>"
            
        if "sample_rate_range" in caps:
            rate_min, rate_max = caps["sample_rate_range"]
            tooltip += f"<b>Sample Rate Range:</b> {rate_min/1e3:.1f} kSps - {rate_max/1e6:.1f} MSps<br/>"
            
        if "gain_range" in caps:
            gain_min, gain_max = caps["gain_range"]
            tooltip += f"<b>Gain Range:</b> {gain_min} - {gain_max} dB<br/>"
            
        if device.last_seen > 0:
            last_seen = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(device.last_seen))
            tooltip += f"<b>Last Seen:</b> {last_seen}"
            
        return tooltip
        
    @Slot()
    def _on_selection_changed(self) -> None:
        """Handle selection change."""
        device = self.get_selected_device()
        if device:
            self.device_selected.emit(device)
            
    @Slot(QListWidgetItem)
    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Handle item double click."""
        device = item.data(Qt.UserRole)
        if device:
            self.device_double_clicked.emit(device)


class HardwareManager(QWidget):
    """Hardware detection and management interface."""
    
    # Signals
    device_selected = Signal(HardwareInfo)
    connection_requested = Signal(HardwareInfo)
    hardware_info_updated = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("GeminiSDR.HardwareManager")
        
        # Hardware state
        self.detected_devices = {}  # device_id -> HardwareInfo
        self.current_device = None
        self.auto_detect_enabled = True
        
        # Detection worker
        self.detector_thread = QThread()
        self.detector = HardwareDetector()
        self.detector.moveToThread(self.detector_thread)
        self.detector_thread.start()
        
        self._setup_ui()
        self._setup_connections()
        
        # Start initial detection
        QTimer.singleShot(1000, self._start_detection)
        
        self.logger.info("Hardware Manager initialized")
        
    def _setup_ui(self) -> None:
        """Setup hardware manager UI."""
        layout = QVBoxLayout(self)
        
        # Detection controls
        controls_group = QGroupBox("Hardware Detection")
        controls_layout = QHBoxLayout(controls_group)
        
        self.detect_btn = QPushButton("Detect Hardware")
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        controls_layout.addWidget(self.detect_btn)
        
        self.auto_detect_cb = QCheckBox("Auto-detect")
        self.auto_detect_cb.setChecked(True)
        controls_layout.addWidget(self.auto_detect_cb)
        
        controls_layout.addStretch()
        
        # Detection progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)
        
        layout.addWidget(controls_group)
        
        # Device list
        device_group = QGroupBox("Detected Devices")
        device_layout = QVBoxLayout(device_group)
        
        self.device_list = DeviceListWidget()
        device_layout.addWidget(self.device_list)
        
        # Device controls
        device_controls = QHBoxLayout()
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setEnabled(False)
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
        device_controls.addWidget(self.connect_btn)
        
        self.refresh_btn = QPushButton("Refresh")
        device_controls.addWidget(self.refresh_btn)
        
        device_controls.addStretch()
        
        device_layout.addLayout(device_controls)
        
        layout.addWidget(device_group)
        
        # Device information
        info_group = QGroupBox("Device Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        
        # Simulation settings (initially hidden)
        self.sim_group = QGroupBox("Simulation Settings")
        self.sim_group.setVisible(False)
        sim_layout = QFormLayout(self.sim_group)
        
        self.sim_signal_combo = QComboBox()
        self.sim_signal_combo.addItems(["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "Mixed"])
        sim_layout.addRow("Signal Type:", self.sim_signal_combo)
        
        self.sim_snr_spinbox = QSpinBox()
        self.sim_snr_spinbox.setRange(-10, 40)
        self.sim_snr_spinbox.setValue(20)
        self.sim_snr_spinbox.setSuffix(" dB")
        sim_layout.addRow("SNR:", self.sim_snr_spinbox)
        
        self.sim_noise_cb = QCheckBox("Add Noise")
        self.sim_noise_cb.setChecked(True)
        sim_layout.addRow("", self.sim_noise_cb)
        
        layout.addWidget(self.sim_group)
        
    def _setup_connections(self) -> None:
        """Setup signal connections."""
        # Controls
        self.detect_btn.clicked.connect(self._start_detection)
        self.auto_detect_cb.toggled.connect(self._on_auto_detect_toggled)
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        self.refresh_btn.clicked.connect(self._start_detection)
        
        # Device list
        self.device_list.device_selected.connect(self._on_device_selected)
        self.device_list.device_double_clicked.connect(self._on_device_double_clicked)
        
        # Detector signals
        self.detector.device_detected.connect(self._on_device_detected)
        self.detector.device_lost.connect(self._on_device_lost)
        self.detector.detection_complete.connect(self._on_detection_complete)
        self.detector.detection_progress.connect(self._on_detection_progress)
        
        # Auto-detection timer
        self.auto_detect_timer = QTimer()
        self.auto_detect_timer.timeout.connect(self._start_detection)
        self.auto_detect_timer.start(10000)  # Every 10 seconds
        
    @Slot()
    def _start_detection(self) -> None:
        """Start hardware detection."""
        if not self.auto_detect_enabled and self.sender() == self.auto_detect_timer:
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Detecting hardware...")
        self.detect_btn.setEnabled(False)
        
        # Run detection in thread
        QTimer.singleShot(100, lambda: self._run_detection())
        
    def _run_detection(self) -> None:
        """Run hardware detection."""
        try:
            devices = self.detector.detect_hardware()
            self._on_detection_complete(devices)
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            self.status_label.setText(f"Detection failed: {e}")
            self.progress_bar.setVisible(False)
            self.detect_btn.setEnabled(True)
            
    @Slot(bool)
    def _on_auto_detect_toggled(self, enabled: bool) -> None:
        """Handle auto-detect toggle."""
        self.auto_detect_enabled = enabled
        if enabled:
            self.auto_detect_timer.start(10000)
        else:
            self.auto_detect_timer.stop()
            
    @Slot()
    def _on_connect_clicked(self) -> None:
        """Handle connect button click."""
        device = self.device_list.get_selected_device()
        if device:
            self.connection_requested.emit(device)
            
    @Slot(HardwareInfo)
    def _on_device_selected(self, device: HardwareInfo) -> None:
        """Handle device selection."""
        self.current_device = device
        self.connect_btn.setEnabled(device.status in [DeviceStatus.DETECTED, DeviceStatus.CONNECTED])
        
        # Show simulation settings for simulation device
        self.sim_group.setVisible(device.device_type == HardwareType.SIMULATION)
        
        # Update device information display
        self._update_device_info_display(device)
        
        self.device_selected.emit(device)
        
    @Slot(HardwareInfo)
    def _on_device_double_clicked(self, device: HardwareInfo) -> None:
        """Handle device double click - connect."""
        if device.status in [DeviceStatus.DETECTED, DeviceStatus.CONNECTED]:
            self.connection_requested.emit(device)
            
    @Slot(HardwareInfo)
    def _on_device_detected(self, device: HardwareInfo) -> None:
        """Handle device detection."""
        self.detected_devices[device.device_id] = device
        self.device_list.add_device(device)
        
    @Slot(str)
    def _on_device_lost(self, device_id: str) -> None:
        """Handle device loss."""
        if device_id in self.detected_devices:
            del self.detected_devices[device_id]
        self.device_list.remove_device(device_id)
        
    @Slot(list)
    def _on_detection_complete(self, devices: List[HardwareInfo]) -> None:
        """Handle detection completion."""
        # Clear existing devices
        self.detected_devices.clear()
        self.device_list.clear()
        
        # Add detected devices
        for device in devices:
            self.detected_devices[device.device_id] = device
            self.device_list.add_device(device)
            
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        
        device_count = len(devices)
        hardware_count = len([d for d in devices if d.device_type != HardwareType.SIMULATION])
        
        if hardware_count > 0:
            self.status_label.setText(f"Found {hardware_count} hardware device(s), {device_count} total")
        else:
            self.status_label.setText("No hardware detected - simulation mode available")
            
        self.logger.info(f"Detection complete: {device_count} devices found")
        
    @Slot(int, str)
    def _on_detection_progress(self, progress: int, status: str) -> None:
        """Handle detection progress update."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
    def _update_device_info_display(self, device: HardwareInfo) -> None:
        """Update device information display."""
        info_html = f"""<h3>{device.name}</h3>
<p><b>Type:</b> {device.device_type.value}</p>
<p><b>Status:</b> {device.status.value.title()}</p>
<p><b>Device ID:</b> {device.device_id}</p>"""

        if device.connection_string:
            info_html += f"<p><b>Connection:</b> {device.connection_string}</p>"
            
        if device.error_message:
            info_html += f"<p><b>Error:</b> <span style='color: red;'>{device.error_message}</span></p>"
            
        # Add capabilities
        caps = device.capabilities
        info_html += "<h4>Capabilities:</h4><ul>"
        
        if "frequency_range" in caps:
            freq_min, freq_max = caps["frequency_range"]
            info_html += f"<li><b>Frequency:</b> {freq_min/1e6:.1f} MHz - {freq_max/1e9:.1f} GHz</li>"
            
        if "sample_rate_range" in caps:
            rate_min, rate_max = caps["sample_rate_range"]
            info_html += f"<li><b>Sample Rate:</b> {rate_min/1e3:.1f} kSps - {rate_max/1e6:.1f} MSps</li>"
            
        if "gain_range" in caps:
            gain_min, gain_max = caps["gain_range"]
            info_html += f"<li><b>Gain:</b> {gain_min} - {gain_max} dB</li>"
            
        if "rx_channels" in caps:
            info_html += f"<li><b>RX Channels:</b> {caps['rx_channels']}</li>"
            
        if "tx_channels" in caps:
            info_html += f"<li><b>TX Channels:</b> {caps['tx_channels']}</li>"
            
        if caps.get("full_duplex", False):
            info_html += "<li><b>Full Duplex:</b> Yes</li>"
            
        if device.device_type == HardwareType.SIMULATION:
            if "signal_types" in caps:
                info_html += f"<li><b>Signal Types:</b> {', '.join(caps['signal_types'])}</li>"
            if caps.get("noise_simulation", False):
                info_html += "<li><b>Noise Simulation:</b> Yes</li>"
                
        info_html += "</ul>"
        
        if device.last_seen > 0:
            last_seen = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(device.last_seen))
            info_html += f"<p><b>Last Seen:</b> {last_seen}</p>"
            
        self.info_text.setHtml(info_html)
        
        # Emit hardware info for other components
        hardware_info = {
            "device_name": device.name,
            "device_type": device.device_type.value,
            "device_id": device.device_id,
            "status": device.status.value,
            "capabilities": device.capabilities,
            "connection_string": device.connection_string
        }
        self.hardware_info_updated.emit(hardware_info)
        
    def get_detected_devices(self) -> List[HardwareInfo]:
        """Get list of detected devices."""
        return list(self.detected_devices.values())
        
    def get_device_by_id(self, device_id: str) -> Optional[HardwareInfo]:
        """Get device by ID."""
        return self.detected_devices.get(device_id)
        
    def get_current_device(self) -> Optional[HardwareInfo]:
        """Get currently selected device."""
        return self.current_device
        
    def update_device_status(self, device_id: str, status: DeviceStatus, error_msg: str = "") -> None:
        """Update device status."""
        if device_id in self.detected_devices:
            device = self.detected_devices[device_id]
            device.status = status
            device.error_message = error_msg
            device.last_seen = time.time()
            
            self.device_list.update_device(device)
            
            if device == self.current_device:
                self._update_device_info_display(device)
                
    def get_simulation_settings(self) -> Dict[str, Any]:
        """Get simulation settings."""
        return {
            "signal_type": self.sim_signal_combo.currentText(),
            "snr_db": self.sim_snr_spinbox.value(),
            "add_noise": self.sim_noise_cb.isChecked()
        }
        
    def set_simulation_settings(self, settings: Dict[str, Any]) -> None:
        """Set simulation settings."""
        if "signal_type" in settings:
            self.sim_signal_combo.setCurrentText(settings["signal_type"])
        if "snr_db" in settings:
            self.sim_snr_spinbox.setValue(settings["snr_db"])
        if "add_noise" in settings:
            self.sim_noise_cb.setChecked(settings["add_noise"])
            
    def force_simulation_mode(self) -> None:
        """Force selection of simulation device."""
        for device in self.detected_devices.values():
            if device.device_type == HardwareType.SIMULATION:
                # Select simulation device
                for i in range(self.device_list.count()):
                    item = self.device_list.item(i)
                    stored_device = item.data(Qt.UserRole)
                    if stored_device and stored_device.device_id == device.device_id:
                        self.device_list.setCurrentItem(item)
                        break
                break