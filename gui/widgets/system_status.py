"""
System Status and Health Monitoring Widgets

This module provides comprehensive system monitoring widgets including:
- Real-time CPU, memory, and GPU monitoring
- Performance metrics visualization with historical trends
- Configurable alert thresholds and notifications
- System optimization recommendations
- Error logging and diagnostic tools
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import psutil
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QProgressBar, QLabel,
    QTextEdit, QTableWidget, QTableWidgetItem, QPushButton, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QTabWidget, QSplitter,
    QScrollArea, QFrame, QGridLayout, QMessageBox, QFileDialog,
    QLineEdit, QHeaderView
)
from PySide6.QtCore import (
    Signal, Slot, QTimer, QThread, QObject, Qt, QDateTime,
    QAbstractTableModel, QModelIndex, QSortFilterProxyModel
)
from PySide6.QtGui import QColor, QPalette, QFont, QPixmap, QIcon

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("Warning: PyQtGraph not available. Some visualization features will be disabled.")

try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """System performance metrics data structure."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_recv: Optional[int] = None


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    metric: str
    warning_threshold: float
    critical_threshold: float
    enabled: bool = True


class SystemMonitorThread(QThread):
    """Background thread for system monitoring."""
    
    metrics_updated = Signal(SystemMetrics)
    alert_triggered = Signal(str, str, str)  # severity, metric, message
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.update_interval = 1.0  # seconds
        self.alert_thresholds = {
            'cpu_percent': AlertThreshold('cpu_percent', 80.0, 95.0),
            'memory_percent': AlertThreshold('memory_percent', 85.0, 95.0),
            'disk_usage_percent': AlertThreshold('disk_usage_percent', 90.0, 98.0),
            'gpu_usage_percent': AlertThreshold('gpu_usage_percent', 90.0, 98.0),
            'gpu_memory_percent': AlertThreshold('gpu_memory_percent', 90.0, 98.0),
            'gpu_temperature': AlertThreshold('gpu_temperature', 80.0, 90.0),
        }
        self.last_alert_times = {}
        self.alert_cooldown = 30.0  # seconds between same alerts
        
    def run(self):
        """Main monitoring loop."""
        self.running = True
        last_network_stats = psutil.net_io_counters()
        
        while self.running:
            try:
                # Collect system metrics
                metrics = self._collect_metrics(last_network_stats)
                last_network_stats = psutil.net_io_counters()
                
                # Emit metrics update
                self.metrics_updated.emit(metrics)
                
                # Check alert thresholds
                self._check_alerts(metrics)
                
                # Sleep for update interval
                self.msleep(int(self.update_interval * 1000))
                
            except Exception as e:
                logging.error(f"Error in system monitoring thread: {e}")
                self.msleep(5000)  # Wait 5 seconds before retrying
                
    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        self.wait()
        
    def _collect_metrics(self, last_network_stats) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network stats
        current_network = psutil.net_io_counters()
        network_sent = current_network.bytes_sent - last_network_stats.bytes_sent
        network_recv = current_network.bytes_recv - last_network_stats.bytes_recv
        
        # GPU metrics (if available)
        gpu_usage = None
        gpu_memory_percent = None
        gpu_memory_used = None
        gpu_temperature = None
        
        if GPU_MONITORING_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_usage = gpu.load * 100
                    gpu_memory_percent = gpu.memoryUtil * 100
                    gpu_memory_used = gpu.memoryUsed
                    gpu_temperature = gpu.temperature
            except Exception as e:
                logging.debug(f"GPU monitoring error: {e}")
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            gpu_usage_percent=gpu_usage,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_temperature=gpu_temperature,
            network_bytes_sent=network_sent,
            network_bytes_recv=network_recv
        )
        
    def _check_alerts(self, metrics: SystemMetrics):
        """Check metrics against alert thresholds."""
        current_time = time.time()
        
        # Check each threshold
        for metric_name, threshold in self.alert_thresholds.items():
            if not threshold.enabled:
                continue
                
            value = getattr(metrics, metric_name, None)
            if value is None:
                continue
                
            # Check if we should send an alert
            alert_key = f"{metric_name}_{threshold.warning_threshold}"
            last_alert_time = self.last_alert_times.get(alert_key, 0)
            
            if current_time - last_alert_time < self.alert_cooldown:
                continue
                
            # Check thresholds
            if value >= threshold.critical_threshold:
                self.alert_triggered.emit(
                    AlertSeverity.CRITICAL.value,
                    metric_name,
                    f"Critical {metric_name}: {value:.1f}% (threshold: {threshold.critical_threshold}%)"
                )
                self.last_alert_times[alert_key] = current_time
            elif value >= threshold.warning_threshold:
                self.alert_triggered.emit(
                    AlertSeverity.WARNING.value,
                    metric_name,
                    f"Warning {metric_name}: {value:.1f}% (threshold: {threshold.warning_threshold}%)"
                )
                self.last_alert_times[alert_key] = current_time
                
    def set_alert_threshold(self, metric: str, warning: float, critical: float, enabled: bool = True):
        """Update alert threshold for a metric."""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric].warning_threshold = warning
            self.alert_thresholds[metric].critical_threshold = critical
            self.alert_thresholds[metric].enabled = enabled
            
    def set_update_interval(self, interval: float):
        """Set monitoring update interval in seconds."""
        self.update_interval = max(0.1, interval)


class SystemStatusWidget(QWidget):
    """Main system status and health monitoring widget."""
    
    # Signals
    alert_triggered = Signal(str, str, str)  # severity, metric, message
    optimization_recommended = Signal(str)  # recommendation message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self.metrics_history = []
        self.max_history_points = 1000
        self.alert_history = []
        
        # Monitoring thread
        self.monitor_thread = SystemMonitorThread()
        self.monitor_thread.metrics_updated.connect(self._update_metrics)
        self.monitor_thread.alert_triggered.connect(self._handle_alert)
        
        # UI setup
        self._setup_ui()
        self._setup_connections()
        
        # Start monitoring
        self.monitor_thread.start()
        
    def __del__(self):
        """Cleanup when widget is destroyed."""
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.stop()
            
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Resource monitoring tab
        resource_tab = self._create_resource_tab()
        self.tab_widget.addTab(resource_tab, "System Resources")
        
        # Performance metrics tab
        metrics_tab = self._create_metrics_tab()
        self.tab_widget.addTab(metrics_tab, "Performance Metrics")
        
        # Alert configuration tab
        alerts_tab = self._create_alerts_tab()
        self.tab_widget.addTab(alerts_tab, "Alert Configuration")
        
        # Optimization recommendations tab
        optimization_tab = self._create_optimization_tab()
        self.tab_widget.addTab(optimization_tab, "Optimization")
        
        layout.addWidget(self.tab_widget)
        
    def _create_resource_tab(self) -> QWidget:
        """Create the resource monitoring tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Current status group
        status_group = QGroupBox("Current System Status")
        status_layout = QGridLayout(status_group)
        
        # CPU usage
        status_layout.addWidget(QLabel("CPU Usage:"), 0, 0)
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        self.cpu_progress.setFormat("%p%")
        status_layout.addWidget(self.cpu_progress, 0, 1)
        self.cpu_label = QLabel("0.0%")
        status_layout.addWidget(self.cpu_label, 0, 2)
        
        # Memory usage
        status_layout.addWidget(QLabel("Memory Usage:"), 1, 0)
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        self.memory_progress.setFormat("%p%")
        status_layout.addWidget(self.memory_progress, 1, 1)
        self.memory_label = QLabel("0.0 GB / 0.0 GB")
        status_layout.addWidget(self.memory_label, 1, 2)
        
        # Disk usage
        status_layout.addWidget(QLabel("Disk Usage:"), 2, 0)
        self.disk_progress = QProgressBar()
        self.disk_progress.setRange(0, 100)
        self.disk_progress.setFormat("%p%")
        status_layout.addWidget(self.disk_progress, 2, 1)
        self.disk_label = QLabel("0.0 GB free")
        status_layout.addWidget(self.disk_label, 2, 2)
        
        # GPU usage (if available)
        if GPU_MONITORING_AVAILABLE:
            status_layout.addWidget(QLabel("GPU Usage:"), 3, 0)
            self.gpu_progress = QProgressBar()
            self.gpu_progress.setRange(0, 100)
            self.gpu_progress.setFormat("%p%")
            status_layout.addWidget(self.gpu_progress, 3, 1)
            self.gpu_label = QLabel("N/A")
            status_layout.addWidget(self.gpu_label, 3, 2)
            
            status_layout.addWidget(QLabel("GPU Memory:"), 4, 0)
            self.gpu_memory_progress = QProgressBar()
            self.gpu_memory_progress.setRange(0, 100)
            self.gpu_memory_progress.setFormat("%p%")
            status_layout.addWidget(self.gpu_memory_progress, 4, 1)
            self.gpu_memory_label = QLabel("N/A")
            status_layout.addWidget(self.gpu_memory_label, 4, 2)
            
            status_layout.addWidget(QLabel("GPU Temperature:"), 5, 0)
            self.gpu_temp_label = QLabel("N/A")
            status_layout.addWidget(self.gpu_temp_label, 5, 1, 1, 2)
        
        layout.addWidget(status_group)
        
        # System information group
        info_group = QGroupBox("System Information")
        info_layout = QGridLayout(info_group)
        
        # Get system info
        info_layout.addWidget(QLabel("Platform:"), 0, 0)
        info_layout.addWidget(QLabel(sys.platform), 0, 1)
        
        info_layout.addWidget(QLabel("CPU Count:"), 1, 0)
        info_layout.addWidget(QLabel(str(psutil.cpu_count())), 1, 1)
        
        memory_total = psutil.virtual_memory().total / (1024**3)
        info_layout.addWidget(QLabel("Total Memory:"), 2, 0)
        info_layout.addWidget(QLabel(f"{memory_total:.1f} GB"), 2, 1)
        
        layout.addWidget(info_group)
        
        # Network activity group
        network_group = QGroupBox("Network Activity")
        network_layout = QGridLayout(network_group)
        
        network_layout.addWidget(QLabel("Bytes Sent:"), 0, 0)
        self.network_sent_label = QLabel("0 B/s")
        network_layout.addWidget(self.network_sent_label, 0, 1)
        
        network_layout.addWidget(QLabel("Bytes Received:"), 1, 0)
        self.network_recv_label = QLabel("0 B/s")
        network_layout.addWidget(self.network_recv_label, 1, 1)
        
        layout.addWidget(network_group)
        
        return widget
        
    def _create_metrics_tab(self) -> QWidget:
        """Create the performance metrics visualization tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        if not PYQTGRAPH_AVAILABLE:
            layout.addWidget(QLabel("PyQtGraph not available. Install it for performance metrics visualization."))
            return widget
            
        # Control panel
        control_panel = QGroupBox("Display Controls")
        control_layout = QHBoxLayout(control_panel)
        
        # Time range selector
        control_layout.addWidget(QLabel("Time Range:"))
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["1 minute", "5 minutes", "15 minutes", "1 hour", "All"])
        self.time_range_combo.setCurrentText("5 minutes")
        self.time_range_combo.currentTextChanged.connect(self._update_time_range)
        control_layout.addWidget(self.time_range_combo)
        
        # Update interval
        control_layout.addWidget(QLabel("Update Interval:"))
        self.update_interval_spin = QDoubleSpinBox()
        self.update_interval_spin.setRange(0.1, 10.0)
        self.update_interval_spin.setValue(1.0)
        self.update_interval_spin.setSuffix(" s")
        self.update_interval_spin.valueChanged.connect(self._update_monitoring_interval)
        control_layout.addWidget(self.update_interval_spin)
        
        control_layout.addStretch()
        
        # Export button
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self._export_metrics_data)
        control_layout.addWidget(export_btn)
        
        layout.addWidget(control_panel)
        
        # Metrics plots
        self.metrics_plot = pg.PlotWidget(title="System Performance Metrics")
        self.metrics_plot.setLabel('left', 'Usage (%)')
        self.metrics_plot.setLabel('bottom', 'Time')
        self.metrics_plot.addLegend()
        self.metrics_plot.showGrid(True, True, alpha=0.3)
        
        # Create curves for different metrics
        self.cpu_curve = self.metrics_plot.plot(pen=pg.mkPen('r', width=2), name='CPU')
        self.memory_curve = self.metrics_plot.plot(pen=pg.mkPen('g', width=2), name='Memory')
        self.disk_curve = self.metrics_plot.plot(pen=pg.mkPen('b', width=2), name='Disk')
        
        if GPU_MONITORING_AVAILABLE:
            self.gpu_curve = self.metrics_plot.plot(pen=pg.mkPen('m', width=2), name='GPU')
            self.gpu_memory_curve = self.metrics_plot.plot(pen=pg.mkPen('c', width=2), name='GPU Memory')
        
        layout.addWidget(self.metrics_plot)
        
        return widget
        
    def _create_alerts_tab(self) -> QWidget:
        """Create the alert configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Alert configuration group
        config_group = QGroupBox("Alert Thresholds")
        config_layout = QGridLayout(config_group)
        
        # Headers
        config_layout.addWidget(QLabel("Metric"), 0, 0)
        config_layout.addWidget(QLabel("Warning (%)"), 0, 1)
        config_layout.addWidget(QLabel("Critical (%)"), 0, 2)
        config_layout.addWidget(QLabel("Enabled"), 0, 3)
        
        # Create threshold controls
        self.threshold_controls = {}
        metrics = ['cpu_percent', 'memory_percent', 'disk_usage_percent']
        if GPU_MONITORING_AVAILABLE:
            metrics.extend(['gpu_usage_percent', 'gpu_memory_percent', 'gpu_temperature'])
            
        for i, metric in enumerate(metrics, 1):
            threshold = self.monitor_thread.alert_thresholds.get(metric)
            if not threshold:
                continue
                
            # Metric name
            config_layout.addWidget(QLabel(metric.replace('_', ' ').title()), i, 0)
            
            # Warning threshold
            warning_spin = QDoubleSpinBox()
            warning_spin.setRange(0, 100)
            warning_spin.setValue(threshold.warning_threshold)
            warning_spin.setSuffix("%")
            config_layout.addWidget(warning_spin, i, 1)
            
            # Critical threshold
            critical_spin = QDoubleSpinBox()
            critical_spin.setRange(0, 100)
            critical_spin.setValue(threshold.critical_threshold)
            critical_spin.setSuffix("%")
            config_layout.addWidget(critical_spin, i, 2)
            
            # Enabled checkbox
            enabled_check = QCheckBox()
            enabled_check.setChecked(threshold.enabled)
            config_layout.addWidget(enabled_check, i, 3)
            
            self.threshold_controls[metric] = {
                'warning': warning_spin,
                'critical': critical_spin,
                'enabled': enabled_check
            }
            
            # Connect signals
            warning_spin.valueChanged.connect(lambda v, m=metric: self._update_threshold(m))
            critical_spin.valueChanged.connect(lambda v, m=metric: self._update_threshold(m))
            enabled_check.toggled.connect(lambda v, m=metric: self._update_threshold(m))
        
        layout.addWidget(config_group)
        
        # Alert history group
        history_group = QGroupBox("Recent Alerts")
        history_layout = QVBoxLayout(history_group)
        
        # Alert table
        self.alert_table = QTableWidget()
        self.alert_table.setColumnCount(4)
        self.alert_table.setHorizontalHeaderLabels(["Time", "Severity", "Metric", "Message"])
        self.alert_table.horizontalHeader().setStretchLastSection(True)
        self.alert_table.setAlternatingRowColors(True)
        self.alert_table.setSelectionBehavior(QTableWidget.SelectRows)
        history_layout.addWidget(self.alert_table)
        
        # Clear alerts button
        clear_btn = QPushButton("Clear Alert History")
        clear_btn.clicked.connect(self._clear_alert_history)
        history_layout.addWidget(clear_btn)
        
        layout.addWidget(history_group)
        
        return widget
        
    def _create_optimization_tab(self) -> QWidget:
        """Create the optimization recommendations tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Optimization recommendations
        recommendations_group = QGroupBox("System Optimization Recommendations")
        recommendations_layout = QVBoxLayout(recommendations_group)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        self.recommendations_text.setMaximumHeight(200)
        recommendations_layout.addWidget(self.recommendations_text)
        
        # Refresh recommendations button
        refresh_btn = QPushButton("Refresh Recommendations")
        refresh_btn.clicked.connect(self._generate_recommendations)
        recommendations_layout.addWidget(refresh_btn)
        
        layout.addWidget(recommendations_group)
        
        # System optimization actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QGridLayout(actions_group)
        
        # Memory cleanup
        memory_cleanup_btn = QPushButton("Clear System Cache")
        memory_cleanup_btn.clicked.connect(self._clear_system_cache)
        actions_layout.addWidget(memory_cleanup_btn, 0, 0)
        
        # Process monitor
        process_monitor_btn = QPushButton("Show Process Monitor")
        process_monitor_btn.clicked.connect(self._show_process_monitor)
        actions_layout.addWidget(process_monitor_btn, 0, 1)
        
        # System info
        system_info_btn = QPushButton("Detailed System Info")
        system_info_btn.clicked.connect(self._show_system_info)
        actions_layout.addWidget(system_info_btn, 1, 0)
        
        # Performance test
        perf_test_btn = QPushButton("Run Performance Test")
        perf_test_btn.clicked.connect(self._run_performance_test)
        actions_layout.addWidget(perf_test_btn, 1, 1)
        
        layout.addWidget(actions_group)
        
        # Auto-optimization settings
        auto_group = QGroupBox("Automatic Optimization")
        auto_layout = QVBoxLayout(auto_group)
        
        self.auto_optimize_check = QCheckBox("Enable automatic optimization recommendations")
        auto_layout.addWidget(self.auto_optimize_check)
        
        self.auto_cleanup_check = QCheckBox("Automatic memory cleanup when usage > 90%")
        auto_layout.addWidget(self.auto_cleanup_check)
        
        layout.addWidget(auto_group)
        
        return widget
        
    def _setup_connections(self):
        """Setup signal-slot connections."""
        # Connect internal signals
        self.alert_triggered.connect(self._show_alert_notification)
        
    @Slot(SystemMetrics)
    def _update_metrics(self, metrics: SystemMetrics):
        """Update UI with new system metrics."""
        # Store metrics in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_points:
            self.metrics_history.pop(0)
            
        # Update progress bars and labels
        self._update_resource_displays(metrics)
        
        # Update performance plots
        if PYQTGRAPH_AVAILABLE:
            self._update_performance_plots()
            
        # Check for optimization recommendations
        if self.auto_optimize_check.isChecked():
            self._check_optimization_triggers(metrics)
            
    def _update_resource_displays(self, metrics: SystemMetrics):
        """Update resource display widgets."""
        # CPU
        self.cpu_progress.setValue(int(metrics.cpu_percent))
        self.cpu_label.setText(f"{metrics.cpu_percent:.1f}%")
        
        # Set progress bar color based on usage
        if metrics.cpu_percent > 90:
            self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif metrics.cpu_percent > 80:
            self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            
        # Memory
        self.memory_progress.setValue(int(metrics.memory_percent))
        self.memory_label.setText(f"{metrics.memory_used_gb:.1f} GB / {metrics.memory_total_gb:.1f} GB")
        
        if metrics.memory_percent > 90:
            self.memory_progress.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif metrics.memory_percent > 80:
            self.memory_progress.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            self.memory_progress.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            
        # Disk
        self.disk_progress.setValue(int(metrics.disk_usage_percent))
        self.disk_label.setText(f"{metrics.disk_free_gb:.1f} GB free")
        
        if metrics.disk_usage_percent > 95:
            self.disk_progress.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif metrics.disk_usage_percent > 90:
            self.disk_progress.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            self.disk_progress.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            
        # GPU (if available)
        if GPU_MONITORING_AVAILABLE and hasattr(self, 'gpu_progress'):
            if metrics.gpu_usage_percent is not None:
                self.gpu_progress.setValue(int(metrics.gpu_usage_percent))
                self.gpu_label.setText(f"{metrics.gpu_usage_percent:.1f}%")
                
                if metrics.gpu_memory_percent is not None:
                    self.gpu_memory_progress.setValue(int(metrics.gpu_memory_percent))
                    self.gpu_memory_label.setText(f"{metrics.gpu_memory_used_mb:.0f} MB")
                    
                if metrics.gpu_temperature is not None:
                    self.gpu_temp_label.setText(f"{metrics.gpu_temperature:.1f}°C")
            else:
                self.gpu_label.setText("N/A")
                self.gpu_memory_label.setText("N/A")
                self.gpu_temp_label.setText("N/A")
                
        # Network
        if metrics.network_bytes_sent is not None:
            self.network_sent_label.setText(self._format_bytes(metrics.network_bytes_sent) + "/s")
        if metrics.network_bytes_recv is not None:
            self.network_recv_label.setText(self._format_bytes(metrics.network_bytes_recv) + "/s")
            
    def _update_performance_plots(self):
        """Update performance metric plots."""
        if not self.metrics_history or not PYQTGRAPH_AVAILABLE:
            return
            
        # Get time range
        time_range_text = self.time_range_combo.currentText()
        if time_range_text == "All":
            data = self.metrics_history
        else:
            # Parse time range
            if "minute" in time_range_text:
                minutes = int(time_range_text.split()[0])
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
            elif "hour" in time_range_text:
                hours = int(time_range_text.split()[0])
                cutoff_time = datetime.now() - timedelta(hours=hours)
            else:
                cutoff_time = datetime.now() - timedelta(minutes=5)
                
            data = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
        if not data:
            return
            
        # Prepare data arrays
        timestamps = [m.timestamp.timestamp() for m in data]
        cpu_data = [m.cpu_percent for m in data]
        memory_data = [m.memory_percent for m in data]
        disk_data = [m.disk_usage_percent for m in data]
        
        # Update curves
        self.cpu_curve.setData(timestamps, cpu_data)
        self.memory_curve.setData(timestamps, memory_data)
        self.disk_curve.setData(timestamps, disk_data)
        
        if GPU_MONITORING_AVAILABLE and hasattr(self, 'gpu_curve'):
            gpu_data = [m.gpu_usage_percent or 0 for m in data]
            gpu_memory_data = [m.gpu_memory_percent or 0 for m in data]
            self.gpu_curve.setData(timestamps, gpu_data)
            self.gpu_memory_curve.setData(timestamps, gpu_memory_data)
            
    @Slot(str, str, str)
    def _handle_alert(self, severity: str, metric: str, message: str):
        """Handle system alert."""
        # Add to alert history
        alert_time = datetime.now()
        self.alert_history.append({
            'time': alert_time,
            'severity': severity,
            'metric': metric,
            'message': message
        })
        
        # Update alert table
        self._update_alert_table()
        
        # Emit signal for external handling
        self.alert_triggered.emit(severity, metric, message)
        
    def _update_alert_table(self):
        """Update the alert history table."""
        self.alert_table.setRowCount(len(self.alert_history))
        
        for i, alert in enumerate(reversed(self.alert_history[-50:])):  # Show last 50 alerts
            self.alert_table.setItem(i, 0, QTableWidgetItem(alert['time'].strftime("%H:%M:%S")))
            
            # Color code severity
            severity_item = QTableWidgetItem(alert['severity'].upper())
            if alert['severity'] == AlertSeverity.CRITICAL.value:
                severity_item.setBackground(QColor(255, 0, 0, 100))
            elif alert['severity'] == AlertSeverity.WARNING.value:
                severity_item.setBackground(QColor(255, 165, 0, 100))
            else:
                severity_item.setBackground(QColor(0, 255, 0, 100))
                
            self.alert_table.setItem(i, 1, severity_item)
            self.alert_table.setItem(i, 2, QTableWidgetItem(alert['metric']))
            self.alert_table.setItem(i, 3, QTableWidgetItem(alert['message']))
            
    def _update_threshold(self, metric: str):
        """Update alert threshold for a metric."""
        controls = self.threshold_controls.get(metric)
        if not controls:
            return
            
        warning = controls['warning'].value()
        critical = controls['critical'].value()
        enabled = controls['enabled'].isChecked()
        
        self.monitor_thread.set_alert_threshold(metric, warning, critical, enabled)
        
    def _update_time_range(self, range_text: str):
        """Update the time range for performance plots."""
        self._update_performance_plots()
        
    def _update_monitoring_interval(self, interval: float):
        """Update the monitoring update interval."""
        self.monitor_thread.set_update_interval(interval)
        
    def _export_metrics_data(self):
        """Export metrics data to CSV file."""
        if not self.metrics_history:
            QMessageBox.information(self, "No Data", "No metrics data to export.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Metrics Data", "system_metrics.csv", "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='') as csvfile:
                    fieldnames = ['timestamp', 'cpu_percent', 'memory_percent', 'memory_used_gb',
                                'memory_total_gb', 'disk_usage_percent', 'disk_free_gb']
                    if GPU_MONITORING_AVAILABLE:
                        fieldnames.extend(['gpu_usage_percent', 'gpu_memory_percent', 
                                         'gpu_memory_used_mb', 'gpu_temperature'])
                    fieldnames.extend(['network_bytes_sent', 'network_bytes_recv'])
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for metrics in self.metrics_history:
                        row = {
                            'timestamp': metrics.timestamp.isoformat(),
                            'cpu_percent': metrics.cpu_percent,
                            'memory_percent': metrics.memory_percent,
                            'memory_used_gb': metrics.memory_used_gb,
                            'memory_total_gb': metrics.memory_total_gb,
                            'disk_usage_percent': metrics.disk_usage_percent,
                            'disk_free_gb': metrics.disk_free_gb,
                            'network_bytes_sent': metrics.network_bytes_sent,
                            'network_bytes_recv': metrics.network_bytes_recv
                        }
                        
                        if GPU_MONITORING_AVAILABLE:
                            row.update({
                                'gpu_usage_percent': metrics.gpu_usage_percent,
                                'gpu_memory_percent': metrics.gpu_memory_percent,
                                'gpu_memory_used_mb': metrics.gpu_memory_used_mb,
                                'gpu_temperature': metrics.gpu_temperature
                            })
                            
                        writer.writerow(row)
                        
                QMessageBox.information(self, "Export Complete", f"Metrics data exported to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")
                
    def _clear_alert_history(self):
        """Clear the alert history."""
        self.alert_history.clear()
        self.alert_table.setRowCount(0)
        
    def _generate_recommendations(self):
        """Generate system optimization recommendations."""
        if not self.metrics_history:
            self.recommendations_text.setText("No metrics data available for recommendations.")
            return
            
        recommendations = []
        latest_metrics = self.metrics_history[-1]
        
        # CPU recommendations
        if latest_metrics.cpu_percent > 80:
            recommendations.append("• High CPU usage detected. Consider closing unnecessary applications or upgrading CPU.")
            
        # Memory recommendations
        if latest_metrics.memory_percent > 85:
            recommendations.append("• High memory usage detected. Consider closing memory-intensive applications or adding more RAM.")
            
        # Disk recommendations
        if latest_metrics.disk_usage_percent > 90:
            recommendations.append("• Low disk space. Consider cleaning temporary files or moving data to external storage.")
            
        # GPU recommendations
        if GPU_MONITORING_AVAILABLE and latest_metrics.gpu_usage_percent and latest_metrics.gpu_usage_percent > 90:
            recommendations.append("• High GPU usage detected. Consider reducing graphics settings or closing GPU-intensive applications.")
            
        if GPU_MONITORING_AVAILABLE and latest_metrics.gpu_temperature and latest_metrics.gpu_temperature > 80:
            recommendations.append("• High GPU temperature detected. Check cooling system and clean dust from GPU fans.")
            
        # General recommendations
        avg_cpu = sum(m.cpu_percent for m in self.metrics_history[-10:]) / min(10, len(self.metrics_history))
        if avg_cpu > 70:
            recommendations.append("• Consistently high CPU usage. Consider background process optimization.")
            
        if not recommendations:
            recommendations.append("• System performance is optimal. No recommendations at this time.")
            
        self.recommendations_text.setText("\n".join(recommendations))
        
    def _check_optimization_triggers(self, metrics: SystemMetrics):
        """Check if automatic optimization should be triggered."""
        if self.auto_cleanup_check.isChecked() and metrics.memory_percent > 90:
            self._clear_system_cache()
            
    def _clear_system_cache(self):
        """Clear system cache (placeholder implementation)."""
        # This would implement actual cache clearing logic
        QMessageBox.information(self, "Cache Cleared", "System cache clearing initiated.")
        
    def _show_process_monitor(self):
        """Show process monitor dialog."""
        # This would open a process monitor dialog
        QMessageBox.information(self, "Process Monitor", "Process monitor would open here.")
        
    def _show_system_info(self):
        """Show detailed system information."""
        # This would show detailed system information
        QMessageBox.information(self, "System Info", "Detailed system information would be displayed here.")
        
    def _run_performance_test(self):
        """Run system performance test."""
        # This would run a performance benchmark
        QMessageBox.information(self, "Performance Test", "Performance test would run here.")
        
    @Slot(str, str, str)
    def _show_alert_notification(self, severity: str, metric: str, message: str):
        """Show alert notification to user."""
        if severity == AlertSeverity.CRITICAL.value:
            QMessageBox.critical(self, "Critical System Alert", message)
        elif severity == AlertSeverity.WARNING.value:
            QMessageBox.warning(self, "System Warning", message)
        else:
            QMessageBox.information(self, "System Information", message)
            
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes value to human readable string."""
        if bytes_value < 1024:
            return f"{bytes_value} B"
        elif bytes_value < 1024**2:
            return f"{bytes_value/1024:.1f} KB"
        elif bytes_value < 1024**3:
            return f"{bytes_value/(1024**2):.1f} MB"
        else:
            return f"{bytes_value/(1024**3):.1f} GB"
            
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
        
    def get_metrics_history(self, time_range_minutes: Optional[int] = None) -> List[SystemMetrics]:
        """Get metrics history, optionally filtered by time range."""
        if time_range_minutes is None:
            return self.metrics_history.copy()
            
        cutoff_time = datetime.now() - timedelta(minutes=time_range_minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
    def set_max_history_points(self, max_points: int):
        """Set maximum number of history points to keep."""
        self.max_history_points = max_points
        if len(self.metrics_history) > max_points:
            self.metrics_history = self.metrics_history[-max_points:]


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Log entry data structure."""
    timestamp: datetime
    level: LogLevel
    component: str
    message: str
    details: Optional[str] = None
    exception: Optional[str] = None


class LogTableModel(QAbstractTableModel):
    """Table model for log entries."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.log_entries = []
        self.headers = ["Time", "Level", "Component", "Message"]
        
    def rowCount(self, parent=QModelIndex()):
        return len(self.log_entries)
        
    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)
        
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self.log_entries):
            return None
            
        entry = self.log_entries[index.row()]
        
        if role == Qt.DisplayRole:
            if index.column() == 0:
                return entry.timestamp.strftime("%H:%M:%S.%f")[:-3]
            elif index.column() == 1:
                return entry.level.value
            elif index.column() == 2:
                return entry.component
            elif index.column() == 3:
                return entry.message
                
        elif role == Qt.BackgroundRole:
            if entry.level == LogLevel.CRITICAL:
                return QColor(255, 0, 0, 50)
            elif entry.level == LogLevel.ERROR:
                return QColor(255, 100, 100, 50)
            elif entry.level == LogLevel.WARNING:
                return QColor(255, 255, 0, 50)
            elif entry.level == LogLevel.DEBUG:
                return QColor(200, 200, 200, 30)
                
        elif role == Qt.ForegroundRole:
            if entry.level == LogLevel.CRITICAL:
                return QColor(255, 255, 255)
            elif entry.level == LogLevel.ERROR:
                return QColor(139, 0, 0)
            elif entry.level == LogLevel.WARNING:
                return QColor(255, 140, 0)
                
        return None
        
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return None
        
    def add_log_entry(self, entry: LogEntry):
        """Add a new log entry."""
        self.beginInsertRows(QModelIndex(), len(self.log_entries), len(self.log_entries))
        self.log_entries.append(entry)
        self.endInsertRows()
        
    def clear_logs(self):
        """Clear all log entries."""
        self.beginResetModel()
        self.log_entries.clear()
        self.endResetModel()
        
    def get_entry(self, index: int) -> Optional[LogEntry]:
        """Get log entry by index."""
        if 0 <= index < len(self.log_entries):
            return self.log_entries[index]
        return None


class LogFilterProxyModel(QSortFilterProxyModel):
    """Proxy model for filtering log entries."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level_filter = set(LogLevel)
        self.component_filter = ""
        self.message_filter = ""
        
    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        if not model:
            return True
            
        entry = model.get_entry(source_row)
        if not entry:
            return True
            
        # Level filter
        if entry.level not in self.level_filter:
            return False
            
        # Component filter
        if self.component_filter and self.component_filter.lower() not in entry.component.lower():
            return False
            
        # Message filter
        if self.message_filter and self.message_filter.lower() not in entry.message.lower():
            return False
            
        return True
        
    def set_level_filter(self, levels: set):
        """Set which log levels to show."""
        self.level_filter = levels
        self.invalidateFilter()
        
    def set_component_filter(self, component: str):
        """Set component filter."""
        self.component_filter = component
        self.invalidateFilter()
        
    def set_message_filter(self, message: str):
        """Set message filter."""
        self.message_filter = message
        self.invalidateFilter()


class ErrorLogWidget(QWidget):
    """Comprehensive error logging and diagnostic widget."""
    
    # Signals
    log_entry_added = Signal(LogEntry)
    diagnostic_requested = Signal(str)  # component name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data models
        self.log_model = LogTableModel()
        self.filter_model = LogFilterProxyModel()
        self.filter_model.setSourceModel(self.log_model)
        
        # Setup logging integration
        self._setup_logging_handler()
        
        # UI setup
        self._setup_ui()
        self._setup_connections()
        
    def _setup_ui(self):
        """Setup the error log widget UI."""
        layout = QVBoxLayout(self)
        
        # Filter controls
        filter_group = self._create_filter_controls()
        layout.addWidget(filter_group)
        
        # Log table
        self.log_table = QTableWidget()
        self.log_table.setModel(self.filter_model)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.log_table.setSortingEnabled(True)
        self.log_table.verticalHeader().setVisible(False)
        
        # Set column widths
        header = self.log_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.resizeSection(0, 100)  # Time
        header.resizeSection(1, 80)   # Level
        header.resizeSection(2, 120)  # Component
        
        layout.addWidget(self.log_table)
        
        # Log details
        details_group = self._create_details_panel()
        layout.addWidget(details_group)
        
        # Action buttons
        button_layout = self._create_action_buttons()
        layout.addLayout(button_layout)
        
    def _create_filter_controls(self) -> QGroupBox:
        """Create log filtering controls."""
        group = QGroupBox("Log Filters")
        layout = QGridLayout(group)
        
        # Level filter checkboxes
        layout.addWidget(QLabel("Log Levels:"), 0, 0)
        
        level_layout = QHBoxLayout()
        self.level_checkboxes = {}
        
        for level in LogLevel:
            checkbox = QCheckBox(level.value)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._update_level_filter)
            level_layout.addWidget(checkbox)
            self.level_checkboxes[level] = checkbox
            
        layout.addLayout(level_layout, 0, 1, 1, 2)
        
        # Component filter
        layout.addWidget(QLabel("Component:"), 1, 0)
        self.component_filter_edit = QLineEdit()
        self.component_filter_edit.setPlaceholderText("Filter by component name...")
        self.component_filter_edit.textChanged.connect(self._update_component_filter)
        layout.addWidget(self.component_filter_edit, 1, 1)
        
        # Message filter
        layout.addWidget(QLabel("Message:"), 1, 2)
        self.message_filter_edit = QLineEdit()
        self.message_filter_edit.setPlaceholderText("Filter by message content...")
        self.message_filter_edit.textChanged.connect(self._update_message_filter)
        layout.addWidget(self.message_filter_edit, 1, 3)
        
        # Auto-scroll checkbox
        self.auto_scroll_check = QCheckBox("Auto-scroll to new entries")
        self.auto_scroll_check.setChecked(True)
        layout.addWidget(self.auto_scroll_check, 2, 0, 1, 2)
        
        # Max entries
        layout.addWidget(QLabel("Max Entries:"), 2, 2)
        self.max_entries_spin = QSpinBox()
        self.max_entries_spin.setRange(100, 10000)
        self.max_entries_spin.setValue(1000)
        self.max_entries_spin.valueChanged.connect(self._update_max_entries)
        layout.addWidget(self.max_entries_spin, 2, 3)
        
        return group
        
    def _create_details_panel(self) -> QGroupBox:
        """Create log entry details panel."""
        group = QGroupBox("Log Entry Details")
        layout = QVBoxLayout(group)
        
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(150)
        self.details_text.setReadOnly(True)
        layout.addWidget(self.details_text)
        
        return group
        
    def _create_action_buttons(self) -> QHBoxLayout:
        """Create action buttons."""
        layout = QHBoxLayout()
        
        # Clear logs
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self._clear_logs)
        layout.addWidget(clear_btn)
        
        # Export logs
        export_btn = QPushButton("Export Logs")
        export_btn.clicked.connect(self._export_logs)
        layout.addWidget(export_btn)
        
        # Diagnostic tools
        diagnostic_btn = QPushButton("Run Diagnostics")
        diagnostic_btn.clicked.connect(self._run_diagnostics)
        layout.addWidget(diagnostic_btn)
        
        # System info
        sysinfo_btn = QPushButton("System Info")
        sysinfo_btn.clicked.connect(self._show_system_info)
        layout.addWidget(sysinfo_btn)
        
        layout.addStretch()
        
        # Log level indicator
        self.log_count_label = QLabel("0 entries")
        layout.addWidget(self.log_count_label)
        
        return layout
        
    def _setup_connections(self):
        """Setup signal-slot connections."""
        # Table selection
        self.log_table.selectionModel().currentRowChanged.connect(self._show_log_details)
        
        # Model updates
        self.log_model.rowsInserted.connect(self._update_log_count)
        self.log_model.modelReset.connect(self._update_log_count)
        
    def _setup_logging_handler(self):
        """Setup Python logging handler integration."""
        class GUILogHandler(logging.Handler):
            def __init__(self, widget):
                super().__init__()
                self.widget = widget
                
            def emit(self, record):
                try:
                    # Convert logging level to our LogLevel enum
                    level_map = {
                        logging.DEBUG: LogLevel.DEBUG,
                        logging.INFO: LogLevel.INFO,
                        logging.WARNING: LogLevel.WARNING,
                        logging.ERROR: LogLevel.ERROR,
                        logging.CRITICAL: LogLevel.CRITICAL
                    }
                    
                    level = level_map.get(record.levelno, LogLevel.INFO)
                    
                    # Create log entry
                    entry = LogEntry(
                        timestamp=datetime.fromtimestamp(record.created),
                        level=level,
                        component=record.name,
                        message=record.getMessage(),
                        details=self.format(record) if record.exc_info else None,
                        exception=record.exc_text if hasattr(record, 'exc_text') else None
                    )
                    
                    # Add to widget (thread-safe)
                    self.widget.add_log_entry(entry)
                    
                except Exception:
                    pass  # Don't let logging errors crash the application
                    
        # Add handler to root logger
        self.log_handler = GUILogHandler(self)
        self.log_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log_handler.setFormatter(formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        
    def add_log_entry(self, entry: LogEntry):
        """Add a log entry to the display."""
        # Ensure we're on the main thread
        if self.thread() != QThread.currentThread():
            # Use queued connection for thread safety
            QTimer.singleShot(0, lambda: self._add_log_entry_main_thread(entry))
        else:
            self._add_log_entry_main_thread(entry)
            
    def _add_log_entry_main_thread(self, entry: LogEntry):
        """Add log entry on main thread."""
        # Limit number of entries
        max_entries = self.max_entries_spin.value()
        if len(self.log_model.log_entries) >= max_entries:
            # Remove oldest entries
            entries_to_remove = len(self.log_model.log_entries) - max_entries + 1
            self.log_model.beginRemoveRows(QModelIndex(), 0, entries_to_remove - 1)
            self.log_model.log_entries = self.log_model.log_entries[entries_to_remove:]
            self.log_model.endRemoveRows()
            
        # Add new entry
        self.log_model.add_log_entry(entry)
        
        # Auto-scroll to new entry
        if self.auto_scroll_check.isChecked():
            self.log_table.scrollToBottom()
            
        # Emit signal
        self.log_entry_added.emit(entry)
        
    def _update_level_filter(self):
        """Update log level filter."""
        enabled_levels = set()
        for level, checkbox in self.level_checkboxes.items():
            if checkbox.isChecked():
                enabled_levels.add(level)
                
        self.filter_model.set_level_filter(enabled_levels)
        
    def _update_component_filter(self, text: str):
        """Update component filter."""
        self.filter_model.set_component_filter(text)
        
    def _update_message_filter(self, text: str):
        """Update message filter."""
        self.filter_model.set_message_filter(text)
        
    def _update_max_entries(self, max_entries: int):
        """Update maximum number of log entries."""
        current_count = len(self.log_model.log_entries)
        if current_count > max_entries:
            entries_to_remove = current_count - max_entries
            self.log_model.beginRemoveRows(QModelIndex(), 0, entries_to_remove - 1)
            self.log_model.log_entries = self.log_model.log_entries[entries_to_remove:]
            self.log_model.endRemoveRows()
            
    def _show_log_details(self, current, previous):
        """Show details for selected log entry."""
        if not current.isValid():
            self.details_text.clear()
            return
            
        # Get the actual row in the source model
        source_index = self.filter_model.mapToSource(current)
        entry = self.log_model.get_entry(source_index.row())
        
        if entry:
            details = f"Timestamp: {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
            details += f"Level: {entry.level.value}\n"
            details += f"Component: {entry.component}\n"
            details += f"Message: {entry.message}\n"
            
            if entry.details:
                details += f"\nDetails:\n{entry.details}"
                
            if entry.exception:
                details += f"\nException:\n{entry.exception}"
                
            self.details_text.setText(details)
        else:
            self.details_text.clear()
            
    def _update_log_count(self):
        """Update log count display."""
        total_count = len(self.log_model.log_entries)
        filtered_count = self.filter_model.rowCount()
        
        if total_count == filtered_count:
            self.log_count_label.setText(f"{total_count} entries")
        else:
            self.log_count_label.setText(f"{filtered_count} of {total_count} entries")
            
    def _clear_logs(self):
        """Clear all log entries."""
        reply = QMessageBox.question(
            self, "Clear Logs", 
            "Are you sure you want to clear all log entries?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.log_model.clear_logs()
            self.details_text.clear()
            
    def _export_logs(self):
        """Export log entries to file."""
        if not self.log_model.log_entries:
            QMessageBox.information(self, "No Data", "No log entries to export.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", f"system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;CSV Files (*.csv);;JSON Files (*.json)"
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    self._export_logs_csv(filename)
                elif filename.endswith('.json'):
                    self._export_logs_json(filename)
                else:
                    self._export_logs_text(filename)
                    
                QMessageBox.information(self, "Export Complete", f"Logs exported to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export logs: {str(e)}")
                
    def _export_logs_text(self, filename: str):
        """Export logs as text file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"GeminiSDR System Logs - Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for entry in self.log_model.log_entries:
                f.write(f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] ")
                f.write(f"{entry.level.value:8} {entry.component:20} {entry.message}\n")
                
                if entry.details:
                    f.write(f"Details: {entry.details}\n")
                if entry.exception:
                    f.write(f"Exception: {entry.exception}\n")
                f.write("\n")
                
    def _export_logs_csv(self, filename: str):
        """Export logs as CSV file."""
        import csv
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'level', 'component', 'message', 'details', 'exception']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in self.log_model.log_entries:
                writer.writerow({
                    'timestamp': entry.timestamp.isoformat(),
                    'level': entry.level.value,
                    'component': entry.component,
                    'message': entry.message,
                    'details': entry.details or '',
                    'exception': entry.exception or ''
                })
                
    def _export_logs_json(self, filename: str):
        """Export logs as JSON file."""
        import json
        
        logs_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_entries': len(self.log_model.log_entries),
            'entries': []
        }
        
        for entry in self.log_model.log_entries:
            logs_data['entries'].append({
                'timestamp': entry.timestamp.isoformat(),
                'level': entry.level.value,
                'component': entry.component,
                'message': entry.message,
                'details': entry.details,
                'exception': entry.exception
            })
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(logs_data, f, indent=2, ensure_ascii=False)
            
    def _run_diagnostics(self):
        """Run system diagnostics."""
        # This would implement comprehensive system diagnostics
        diagnostic_info = self._collect_diagnostic_info()
        
        # Show diagnostic results
        dialog = QMessageBox(self)
        dialog.setWindowTitle("System Diagnostics")
        dialog.setText("System diagnostic information:")
        dialog.setDetailedText(diagnostic_info)
        dialog.setIcon(QMessageBox.Information)
        dialog.exec()
        
    def _collect_diagnostic_info(self) -> str:
        """Collect system diagnostic information."""
        info = []
        info.append(f"System Diagnostics - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        info.append("=" * 60)
        
        # Python information
        info.append(f"Python Version: {sys.version}")
        info.append(f"Platform: {sys.platform}")
        
        # System information
        try:
            info.append(f"CPU Count: {psutil.cpu_count()}")
            info.append(f"Memory Total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
            info.append(f"Disk Total: {psutil.disk_usage('/').total / (1024**3):.1f} GB")
        except Exception as e:
            info.append(f"System info error: {e}")
            
        # GPU information
        if GPU_MONITORING_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    for i, gpu in enumerate(gpus):
                        info.append(f"GPU {i}: {gpu.name}")
                        info.append(f"  Memory: {gpu.memoryTotal} MB")
                        info.append(f"  Driver: {gpu.driver}")
                else:
                    info.append("No GPUs detected")
            except Exception as e:
                info.append(f"GPU info error: {e}")
        else:
            info.append("GPU monitoring not available")
            
        # Log statistics
        info.append(f"\nLog Statistics:")
        info.append(f"Total log entries: {len(self.log_model.log_entries)}")
        
        level_counts = {}
        for entry in self.log_model.log_entries:
            level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
            
        for level, count in level_counts.items():
            info.append(f"  {level.value}: {count}")
            
        return "\n".join(info)
        
    def _show_system_info(self):
        """Show detailed system information."""
        info = self._collect_diagnostic_info()
        
        # Create a dialog with system information
        dialog = QMessageBox(self)
        dialog.setWindowTitle("System Information")
        dialog.setText("Detailed system information:")
        dialog.setDetailedText(info)
        dialog.setIcon(QMessageBox.Information)
        dialog.exec()
        
    def get_log_entries(self, level_filter: Optional[set] = None, 
                       component_filter: Optional[str] = None,
                       time_range_minutes: Optional[int] = None) -> List[LogEntry]:
        """Get log entries with optional filtering."""
        entries = self.log_model.log_entries.copy()
        
        # Apply filters
        if level_filter:
            entries = [e for e in entries if e.level in level_filter]
            
        if component_filter:
            entries = [e for e in entries if component_filter.lower() in e.component.lower()]
            
        if time_range_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=time_range_minutes)
            entries = [e for e in entries if e.timestamp >= cutoff_time]
            
        return entries
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors and warnings."""
        summary = {
            'total_entries': len(self.log_model.log_entries),
            'level_counts': {},
            'component_counts': {},
            'recent_errors': []
        }
        
        # Count by level
        for entry in self.log_model.log_entries:
            level = entry.level.value
            summary['level_counts'][level] = summary['level_counts'].get(level, 0) + 1
            
            component = entry.component
            summary['component_counts'][component] = summary['component_counts'].get(component, 0) + 1
            
        # Get recent errors (last 10 errors/warnings)
        recent_cutoff = datetime.now() - timedelta(minutes=30)
        recent_errors = [
            e for e in self.log_model.log_entries 
            if e.timestamp >= recent_cutoff and e.level in [LogLevel.ERROR, LogLevel.CRITICAL, LogLevel.WARNING]
        ]
        
        summary['recent_errors'] = [
            {
                'timestamp': e.timestamp.isoformat(),
                'level': e.level.value,
                'component': e.component,
                'message': e.message
            }
            for e in recent_errors[-10:]
        ]
        
        return summary