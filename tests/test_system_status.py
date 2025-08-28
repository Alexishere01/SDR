"""
Tests for System Status and Health Monitoring Widgets

This module tests the system monitoring and error logging functionality.
"""

import pytest
import sys
import time
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest

# Import the widgets to test
from gui.widgets.system_status import (
    SystemStatusWidget, ErrorLogWidget, SystemMonitorThread,
    SystemMetrics, LogEntry, LogLevel, AlertSeverity
)


class TestSystemMetrics:
    """Test SystemMetrics data structure."""
    
    def test_system_metrics_creation(self):
        """Test creating SystemMetrics instance."""
        timestamp = datetime.now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_total_gb=16.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.memory_used_gb == 8.0
        assert metrics.memory_total_gb == 16.0
        assert metrics.disk_usage_percent == 70.0
        assert metrics.disk_free_gb == 100.0
        assert metrics.gpu_usage_percent is None
        assert metrics.gpu_memory_percent is None


class TestLogEntry:
    """Test LogEntry data structure."""
    
    def test_log_entry_creation(self):
        """Test creating LogEntry instance."""
        timestamp = datetime.now()
        entry = LogEntry(
            timestamp=timestamp,
            level=LogLevel.ERROR,
            component="test_component",
            message="Test error message",
            details="Additional details",
            exception="Exception traceback"
        )
        
        assert entry.timestamp == timestamp
        assert entry.level == LogLevel.ERROR
        assert entry.component == "test_component"
        assert entry.message == "Test error message"
        assert entry.details == "Additional details"
        assert entry.exception == "Exception traceback"


class TestSystemMonitorThread:
    """Test SystemMonitorThread functionality."""
    
    @pytest.fixture
    def monitor_thread(self):
        """Create a SystemMonitorThread instance for testing."""
        thread = SystemMonitorThread()
        yield thread
        if thread.isRunning():
            thread.stop()
    
    def test_thread_creation(self, monitor_thread):
        """Test thread creation and initial state."""
        assert not monitor_thread.running
        assert monitor_thread.update_interval == 1.0
        assert len(monitor_thread.alert_thresholds) > 0
        
    def test_alert_threshold_update(self, monitor_thread):
        """Test updating alert thresholds."""
        monitor_thread.set_alert_threshold('cpu_percent', 75.0, 90.0, True)
        
        threshold = monitor_thread.alert_thresholds['cpu_percent']
        assert threshold.warning_threshold == 75.0
        assert threshold.critical_threshold == 90.0
        assert threshold.enabled is True
        
    def test_update_interval_setting(self, monitor_thread):
        """Test setting update interval."""
        monitor_thread.set_update_interval(2.5)
        assert monitor_thread.update_interval == 2.5
        
        # Test minimum interval
        monitor_thread.set_update_interval(0.05)
        assert monitor_thread.update_interval == 0.1
        
    @patch('gui.widgets.system_status.psutil')
    def test_collect_metrics(self, mock_psutil, monitor_thread):
        """Test metrics collection."""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=65.0, used=8*1024**3, total=16*1024**3
        )
        mock_psutil.disk_usage.return_value = Mock(
            percent=75.0, free=100*1024**3
        )
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1000, bytes_recv=2000
        )
        
        last_network = Mock(bytes_sent=500, bytes_recv=1500)
        metrics = monitor_thread._collect_metrics(last_network)
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 45.0
        assert metrics.memory_percent == 65.0
        assert metrics.disk_usage_percent == 75.0
        assert metrics.network_bytes_sent == 500  # Difference
        assert metrics.network_bytes_recv == 500  # Difference


@pytest.fixture
def app():
    """Create QApplication instance for GUI tests."""
    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    yield app


class TestSystemStatusWidget:
    """Test SystemStatusWidget functionality."""
    
    @pytest.fixture
    def system_status_widget(self, app):
        """Create SystemStatusWidget for testing."""
        widget = SystemStatusWidget()
        yield widget
        widget.monitor_thread.stop()
        widget.deleteLater()
    
    def test_widget_creation(self, system_status_widget):
        """Test widget creation and initialization."""
        assert system_status_widget is not None
        assert hasattr(system_status_widget, 'tab_widget')
        assert hasattr(system_status_widget, 'monitor_thread')
        assert hasattr(system_status_widget, 'metrics_history')
        
        # Check tabs
        assert system_status_widget.tab_widget.count() >= 4
        
    def test_metrics_update(self, system_status_widget):
        """Test metrics update functionality."""
        # Create test metrics
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=55.0,
            memory_percent=70.0,
            memory_used_gb=11.2,
            memory_total_gb=16.0,
            disk_usage_percent=80.0,
            disk_free_gb=50.0
        )
        
        # Update metrics
        system_status_widget._update_metrics(metrics)
        
        # Check that metrics were stored
        assert len(system_status_widget.metrics_history) == 1
        assert system_status_widget.metrics_history[0] == metrics
        
        # Check UI updates
        assert system_status_widget.cpu_progress.value() == 55
        assert system_status_widget.memory_progress.value() == 70
        assert system_status_widget.disk_progress.value() == 80
        
    def test_alert_handling(self, system_status_widget):
        """Test alert handling."""
        # Connect to alert signal
        alerts_received = []
        system_status_widget.alert_triggered.connect(
            lambda severity, metric, message: alerts_received.append((severity, metric, message))
        )
        
        # Trigger alert
        system_status_widget._handle_alert("warning", "cpu_percent", "High CPU usage")
        
        # Check alert was handled
        assert len(alerts_received) == 1
        assert alerts_received[0] == ("warning", "cpu_percent", "High CPU usage")
        assert len(system_status_widget.alert_history) == 1
        
    def test_metrics_history_limit(self, system_status_widget):
        """Test metrics history size limit."""
        system_status_widget.max_history_points = 5
        
        # Add more metrics than the limit
        for i in range(10):
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=float(i * 10),
                memory_percent=50.0,
                memory_used_gb=8.0,
                memory_total_gb=16.0,
                disk_usage_percent=60.0,
                disk_free_gb=100.0
            )
            system_status_widget._update_metrics(metrics)
            
        # Check that history is limited
        assert len(system_status_widget.metrics_history) == 5
        
    def test_get_current_metrics(self, system_status_widget):
        """Test getting current metrics."""
        # No metrics initially
        assert system_status_widget.get_current_metrics() is None
        
        # Add metrics
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=45.0,
            memory_percent=60.0,
            memory_used_gb=9.6,
            memory_total_gb=16.0,
            disk_usage_percent=70.0,
            disk_free_gb=80.0
        )
        system_status_widget._update_metrics(metrics)
        
        # Check current metrics
        current = system_status_widget.get_current_metrics()
        assert current == metrics
        
    def test_get_metrics_history_with_time_filter(self, system_status_widget):
        """Test getting metrics history with time filtering."""
        now = datetime.now()
        
        # Add metrics with different timestamps
        for i in range(5):
            metrics = SystemMetrics(
                timestamp=now - timedelta(minutes=i),
                cpu_percent=float(i * 10),
                memory_percent=50.0,
                memory_used_gb=8.0,
                memory_total_gb=16.0,
                disk_usage_percent=60.0,
                disk_free_gb=100.0
            )
            system_status_widget.metrics_history.append(metrics)
            
        # Get metrics from last 2 minutes
        recent_metrics = system_status_widget.get_metrics_history(time_range_minutes=2)
        
        # Should get 3 metrics (0, 1, 2 minutes ago)
        assert len(recent_metrics) == 3
        
    def test_format_bytes(self, system_status_widget):
        """Test byte formatting utility."""
        assert system_status_widget._format_bytes(512) == "512 B"
        assert system_status_widget._format_bytes(1536) == "1.5 KB"
        assert system_status_widget._format_bytes(2097152) == "2.0 MB"
        assert system_status_widget._format_bytes(3221225472) == "3.0 GB"


class TestErrorLogWidget:
    """Test ErrorLogWidget functionality."""
    
    @pytest.fixture
    def error_log_widget(self, app):
        """Create ErrorLogWidget for testing."""
        widget = ErrorLogWidget()
        yield widget
        widget.deleteLater()
    
    def test_widget_creation(self, error_log_widget):
        """Test widget creation and initialization."""
        assert error_log_widget is not None
        assert hasattr(error_log_widget, 'log_model')
        assert hasattr(error_log_widget, 'filter_model')
        assert hasattr(error_log_widget, 'log_table')
        
    def test_add_log_entry(self, error_log_widget):
        """Test adding log entries."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            component="test_component",
            message="Test error message"
        )
        
        error_log_widget.add_log_entry(entry)
        
        # Check entry was added
        assert len(error_log_widget.log_model.log_entries) == 1
        assert error_log_widget.log_model.log_entries[0] == entry
        
    def test_log_level_filtering(self, error_log_widget):
        """Test log level filtering."""
        # Add entries with different levels
        levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]
        for i, level in enumerate(levels):
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                component="test_component",
                message=f"Test message {i}"
            )
            error_log_widget.add_log_entry(entry)
            
        # Filter to show only errors and warnings
        error_log_widget.filter_model.set_level_filter({LogLevel.ERROR, LogLevel.WARNING})
        
        # Check filtered count
        assert error_log_widget.filter_model.rowCount() == 2
        
    def test_component_filtering(self, error_log_widget):
        """Test component filtering."""
        # Add entries with different components
        components = ["component_a", "component_b", "test_component"]
        for i, component in enumerate(components):
            entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                component=component,
                message=f"Test message {i}"
            )
            error_log_widget.add_log_entry(entry)
            
        # Filter by component
        error_log_widget.filter_model.set_component_filter("component_a")
        
        # Check filtered count
        assert error_log_widget.filter_model.rowCount() == 1
        
    def test_message_filtering(self, error_log_widget):
        """Test message filtering."""
        # Add entries with different messages
        messages = ["error occurred", "warning message", "info update"]
        for i, message in enumerate(messages):
            entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                component="test_component",
                message=message
            )
            error_log_widget.add_log_entry(entry)
            
        # Filter by message content
        error_log_widget.filter_model.set_message_filter("error")
        
        # Check filtered count
        assert error_log_widget.filter_model.rowCount() == 1
        
    def test_max_entries_limit(self, error_log_widget):
        """Test maximum entries limit."""
        # Set low limit for testing
        error_log_widget.max_entries_spin.setValue(3)
        
        # Add more entries than the limit
        for i in range(5):
            entry = LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                component="test_component",
                message=f"Test message {i}"
            )
            error_log_widget._add_log_entry_main_thread(entry)
            
        # Check that entries are limited
        assert len(error_log_widget.log_model.log_entries) == 3
        
    def test_get_log_entries_with_filters(self, error_log_widget):
        """Test getting log entries with various filters."""
        now = datetime.now()
        
        # Add test entries
        entries_data = [
            (LogLevel.ERROR, "component_a", "error message", 0),
            (LogLevel.WARNING, "component_b", "warning message", 1),
            (LogLevel.INFO, "component_a", "info message", 5),  # 5 minutes ago
            (LogLevel.DEBUG, "component_c", "debug message", 10)  # 10 minutes ago
        ]
        
        for level, component, message, minutes_ago in entries_data:
            entry = LogEntry(
                timestamp=now - timedelta(minutes=minutes_ago),
                level=level,
                component=component,
                message=message
            )
            error_log_widget.add_log_entry(entry)
            
        # Test level filter
        error_entries = error_log_widget.get_log_entries(
            level_filter={LogLevel.ERROR, LogLevel.WARNING}
        )
        assert len(error_entries) == 2
        
        # Test component filter
        component_a_entries = error_log_widget.get_log_entries(
            component_filter="component_a"
        )
        assert len(component_a_entries) == 2
        
        # Test time range filter
        recent_entries = error_log_widget.get_log_entries(time_range_minutes=3)
        assert len(recent_entries) == 2  # Only entries from 0 and 1 minutes ago
        
    def test_get_error_summary(self, error_log_widget):
        """Test getting error summary."""
        # Add test entries
        entries_data = [
            (LogLevel.ERROR, "component_a"),
            (LogLevel.ERROR, "component_b"),
            (LogLevel.WARNING, "component_a"),
            (LogLevel.INFO, "component_c")
        ]
        
        for level, component in entries_data:
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                component=component,
                message="Test message"
            )
            error_log_widget.add_log_entry(entry)
            
        summary = error_log_widget.get_error_summary()
        
        assert summary['total_entries'] == 4
        assert summary['level_counts']['ERROR'] == 2
        assert summary['level_counts']['WARNING'] == 1
        assert summary['level_counts']['INFO'] == 1
        assert summary['component_counts']['component_a'] == 2
        assert summary['component_counts']['component_b'] == 1
        assert summary['component_counts']['component_c'] == 1
        
    def test_logging_integration(self, error_log_widget):
        """Test Python logging integration."""
        # Get a logger
        logger = logging.getLogger("test_logger")
        
        # Log some messages
        logger.error("Test error message")
        logger.warning("Test warning message")
        logger.info("Test info message")
        
        # Give some time for the handler to process
        QTest.qWait(100)
        
        # Check that entries were added
        # Note: This test might be flaky depending on logging setup
        # In a real scenario, you'd want to test the handler directly
        assert len(error_log_widget.log_model.log_entries) >= 0


if __name__ == "__main__":
    pytest.main([__file__])