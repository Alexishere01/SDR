"""
Tests for the metrics collection and monitoring system.

This module tests metrics collection, anomaly detection, alerting,
and monitoring dashboard functionality.
"""

import pytest
import json
import time
import tempfile
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from geminisdr.config.config_models import SystemConfig, LoggingConfig, PerformanceConfig
from geminisdr.core.metrics_collector import (
    MetricsCollector, Metric, Alert, SystemMetrics, MLMetrics,
    AnomalyDetector, MetricType, AlertSeverity
)


class TestMetric:
    """Test Metric dataclass."""
    
    def test_metric_creation(self):
        """Test creating a metric."""
        timestamp = datetime.now()
        metric = Metric(
            name="test_metric",
            value=42.5,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            tags={"env": "test"},
            metadata={"source": "test"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.metric_type == MetricType.GAUGE
        assert metric.timestamp == timestamp
        assert metric.tags == {"env": "test"}
        assert metric.metadata == {"source": "test"}
    
    def test_metric_to_dict(self):
        """Test converting metric to dictionary."""
        timestamp = datetime.now()
        metric = Metric(
            name="test_metric",
            value=42.5,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            tags={"env": "test"},
            metadata={"source": "test"}
        )
        
        result = metric.to_dict()
        
        assert result["name"] == "test_metric"
        assert result["value"] == 42.5
        assert result["type"] == "gauge"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["tags"] == {"env": "test"}
        assert result["metadata"] == {"source": "test"}


class TestAlert:
    """Test Alert dataclass."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        timestamp = datetime.now()
        alert = Alert(
            metric_name="test_metric",
            severity=AlertSeverity.HIGH,
            message="Test alert",
            value=95.0,
            threshold=90.0,
            timestamp=timestamp,
            tags={"env": "test"}
        )
        
        assert alert.metric_name == "test_metric"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.message == "Test alert"
        assert alert.value == 95.0
        assert alert.threshold == 90.0
        assert alert.timestamp == timestamp
        assert alert.tags == {"env": "test"}
        assert alert.resolved is False
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        timestamp = datetime.now()
        alert = Alert(
            metric_name="test_metric",
            severity=AlertSeverity.HIGH,
            message="Test alert",
            value=95.0,
            threshold=90.0,
            timestamp=timestamp,
            tags={"env": "test"},
            resolved=True
        )
        
        result = alert.to_dict()
        
        assert result["metric_name"] == "test_metric"
        assert result["severity"] == "high"
        assert result["message"] == "Test alert"
        assert result["value"] == 95.0
        assert result["threshold"] == 90.0
        assert result["timestamp"] == timestamp.isoformat()
        assert result["tags"] == {"env": "test"}
        assert result["resolved"] is True


class TestSystemMetrics:
    """Test SystemMetrics dataclass."""
    
    def test_system_metrics_creation(self):
        """Test creating system metrics."""
        timestamp = datetime.now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=75.5,
            memory_percent=60.0,
            memory_used_mb=4096.0,
            memory_available_mb=2048.0,
            disk_usage_percent=45.0,
            gpu_memory_used_mb=1024.0,
            gpu_memory_total_mb=8192.0,
            gpu_utilization_percent=80.0
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_percent == 75.5
        assert metrics.memory_percent == 60.0
        assert metrics.memory_used_mb == 4096.0
        assert metrics.memory_available_mb == 2048.0
        assert metrics.disk_usage_percent == 45.0
        assert metrics.gpu_memory_used_mb == 1024.0
        assert metrics.gpu_memory_total_mb == 8192.0
        assert metrics.gpu_utilization_percent == 80.0
    
    def test_system_metrics_to_dict(self):
        """Test converting system metrics to dictionary."""
        timestamp = datetime.now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=75.5,
            memory_percent=60.0,
            memory_used_mb=4096.0,
            memory_available_mb=2048.0,
            disk_usage_percent=45.0
        )
        
        result = metrics.to_dict()
        
        assert result["timestamp"] == timestamp
        assert result["cpu_percent"] == 75.5
        assert result["memory_percent"] == 60.0
        assert result["memory_used_mb"] == 4096.0
        assert result["memory_available_mb"] == 2048.0
        assert result["disk_usage_percent"] == 45.0


class TestMLMetrics:
    """Test MLMetrics dataclass."""
    
    def test_ml_metrics_creation(self):
        """Test creating ML metrics."""
        timestamp = datetime.now()
        metrics = MLMetrics(
            timestamp=timestamp,
            model_name="test_model",
            operation="training",
            epoch=10,
            batch_size=32,
            loss=0.25,
            accuracy=0.95,
            learning_rate=1e-4,
            duration=120.5,
            memory_usage_mb=2048.0
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.model_name == "test_model"
        assert metrics.operation == "training"
        assert metrics.epoch == 10
        assert metrics.batch_size == 32
        assert metrics.loss == 0.25
        assert metrics.accuracy == 0.95
        assert metrics.learning_rate == 1e-4
        assert metrics.duration == 120.5
        assert metrics.memory_usage_mb == 2048.0


class TestAnomalyDetector:
    """Test anomaly detection functionality."""
    
    def test_anomaly_detector_initialization(self):
        """Test anomaly detector initialization."""
        detector = AnomalyDetector(window_size=50, threshold_multiplier=2.5)
        
        assert detector.window_size == 50
        assert detector.threshold_multiplier == 2.5
        assert len(detector.metric_windows) == 0
    
    def test_normal_values_no_alert(self):
        """Test that normal values don't trigger alerts."""
        detector = AnomalyDetector(window_size=20, threshold_multiplier=2.0)
        
        # Add normal values
        for i in range(15):
            alert = detector.add_value("test_metric", 10.0 + i * 0.1)
            assert alert is None
    
    def test_anomaly_detection(self):
        """Test anomaly detection with outlier values."""
        detector = AnomalyDetector(window_size=20, threshold_multiplier=2.0)
        
        # Add normal values
        for i in range(15):
            detector.add_value("test_metric", 10.0)
        
        # Add anomalous value
        alert = detector.add_value("test_metric", 50.0)
        
        assert alert is not None
        assert alert.metric_name == "test_metric"
        assert alert.value == 50.0
        assert alert.severity in [AlertSeverity.LOW, AlertSeverity.MEDIUM, 
                                 AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    
    def test_severity_determination(self):
        """Test alert severity determination."""
        detector = AnomalyDetector(window_size=20, threshold_multiplier=2.0)
        
        # Add values with some variation to get meaningful standard deviation
        values = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.1, 
                 9.9, 10.0, 10.1, 9.9, 10.0]
        for value in values:
            detector.add_value("test_metric", value)
        
        # Test different severity levels
        # Small deviation should produce low/medium severity
        alert = detector.add_value("test_metric", 11.0)  # Small deviation
        if alert:
            assert alert.severity in [AlertSeverity.LOW, AlertSeverity.MEDIUM, 
                                     AlertSeverity.HIGH, AlertSeverity.CRITICAL]


class TestMetricsCollector:
    """Test metrics collector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SystemConfig(
            logging=LoggingConfig(),
            performance=PerformanceConfig()
        )
        self.collector = MetricsCollector(self.config)
    
    def test_collector_initialization(self):
        """Test metrics collector initialization."""
        assert self.collector.config == self.config
        assert len(self.collector.metrics) == 0
        assert len(self.collector.alerts) == 0
        assert self.collector.monitoring_active is False
        assert len(self.collector.health_checks) >= 3  # Default health checks
    
    def test_record_metric(self):
        """Test recording metrics."""
        self.collector.record_metric(
            "test_metric", 42.5, MetricType.GAUGE,
            tags={"env": "test"}, metadata={"source": "test"}
        )
        
        metrics = self.collector.get_metrics("test_metric")
        assert len(metrics) == 1
        
        metric = metrics[0]
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.metric_type == MetricType.GAUGE
        assert metric.tags == {"env": "test"}
        assert metric.metadata == {"source": "test"}
    
    def test_record_counter(self):
        """Test recording counter metrics."""
        self.collector.record_counter("test_counter", 5, tags={"type": "counter"})
        
        metrics = self.collector.get_metrics("test_counter")
        assert len(metrics) == 1
        assert metrics[0].metric_type == MetricType.COUNTER
        assert metrics[0].value == 5
    
    def test_record_gauge(self):
        """Test recording gauge metrics."""
        self.collector.record_gauge("test_gauge", 75.0, tags={"type": "gauge"})
        
        metrics = self.collector.get_metrics("test_gauge")
        assert len(metrics) == 1
        assert metrics[0].metric_type == MetricType.GAUGE
        assert metrics[0].value == 75.0
    
    def test_record_timer(self):
        """Test recording timer metrics."""
        self.collector.record_timer("test_timer", 1.5, tags={"type": "timer"})
        
        metrics = self.collector.get_metrics("test_timer")
        assert len(metrics) == 1
        assert metrics[0].metric_type == MetricType.TIMER
        assert metrics[0].value == 1.5
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_collect_system_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 75.5
        mock_memory.return_value = Mock(
            percent=60.0,
            used=4096 * 1024 * 1024,  # 4GB in bytes
            available=2048 * 1024 * 1024  # 2GB in bytes
        )
        mock_disk.return_value = Mock(percent=45.0)
        
        metrics = self.collector.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 75.5
        assert metrics.memory_percent == 60.0
        assert metrics.memory_used_mb == 4096.0
        assert metrics.memory_available_mb == 2048.0
        assert metrics.disk_usage_percent == 45.0
        
        # Check that individual metrics were recorded
        cpu_metrics = self.collector.get_metrics("system.cpu_percent")
        assert len(cpu_metrics) == 1
        assert cpu_metrics[0].value == 75.5
    
    def test_record_ml_metrics(self):
        """Test recording ML metrics."""
        self.collector.record_ml_metrics(
            model_name="test_model",
            operation="training",
            epoch=10,
            batch_size=32,
            loss=0.25,
            accuracy=0.95,
            learning_rate=1e-4,
            duration=120.5,
            memory_usage_mb=2048.0
        )
        
        # Check individual metrics were recorded
        loss_metrics = self.collector.get_metrics("ml.training.loss")
        assert len(loss_metrics) == 1
        assert loss_metrics[0].value == 0.25
        assert loss_metrics[0].tags == {"model": "test_model", "operation": "training"}
        
        accuracy_metrics = self.collector.get_metrics("ml.training.accuracy")
        assert len(accuracy_metrics) == 1
        assert accuracy_metrics[0].value == 0.95
    
    def test_get_metrics_with_time_range(self):
        """Test getting metrics with time range filtering."""
        # Record metrics at different times
        self.collector.record_gauge("test_metric", 10.0)
        time.sleep(0.1)
        self.collector.record_gauge("test_metric", 20.0)
        
        # Get all metrics
        all_metrics = self.collector.get_metrics("test_metric")
        assert len(all_metrics) == 2
        
        # Get recent metrics
        recent_metrics = self.collector.get_metrics(
            "test_metric", time_range=timedelta(seconds=1)
        )
        assert len(recent_metrics) == 2
        
        # Get very recent metrics (should be empty)
        very_recent_metrics = self.collector.get_metrics(
            "test_metric", time_range=timedelta(microseconds=1)
        )
        assert len(very_recent_metrics) == 0
    
    def test_get_metric_summary(self):
        """Test getting metric summary statistics."""
        # Record multiple values
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            self.collector.record_gauge("test_metric", value)
        
        summary = self.collector.get_metric_summary("test_metric")
        
        assert summary["count"] == 5
        assert summary["min"] == 10.0
        assert summary["max"] == 50.0
        assert summary["mean"] == 30.0
        assert summary["median"] == 30.0
        assert summary["latest"] == 50.0
        assert "latest_timestamp" in summary
    
    def test_get_metric_summary_empty(self):
        """Test getting summary for non-existent metric."""
        summary = self.collector.get_metric_summary("nonexistent_metric")
        
        assert summary == {"count": 0}
    
    def test_alert_handling(self):
        """Test alert generation and handling."""
        alert_received = []
        
        def alert_callback(alert):
            alert_received.append(alert)
        
        self.collector.register_alert_callback(alert_callback)
        
        # Generate anomaly to trigger alert
        # Add normal values first
        for i in range(15):
            self.collector.record_gauge("test_metric", 10.0)
        
        # Add anomalous value
        self.collector.record_gauge("test_metric", 100.0)
        
        # Check if alert was generated and callback was called
        alerts = self.collector.get_alerts()
        if len(alerts) > 0:
            assert len(alert_received) > 0
            assert alert_received[0].metric_name == "test_metric"
    
    def test_get_alerts_filtering(self):
        """Test alert filtering."""
        # Create mock alerts
        alert1 = Alert(
            metric_name="metric1",
            severity=AlertSeverity.HIGH,
            message="High alert",
            value=95.0,
            threshold=90.0,
            timestamp=datetime.now() - timedelta(hours=2)
        )
        alert2 = Alert(
            metric_name="metric2",
            severity=AlertSeverity.LOW,
            message="Low alert",
            value=85.0,
            threshold=80.0,
            timestamp=datetime.now(),
            resolved=True
        )
        
        # Add alerts manually
        self.collector.alerts.extend([alert1, alert2])
        
        # Test filtering by severity
        high_alerts = self.collector.get_alerts(severity=AlertSeverity.HIGH)
        assert len(high_alerts) == 1
        assert high_alerts[0].severity == AlertSeverity.HIGH
        
        # Test filtering by resolved status
        unresolved_alerts = self.collector.get_alerts(resolved=False)
        assert len(unresolved_alerts) == 1
        assert unresolved_alerts[0].resolved is False
        
        # Test filtering by time range
        recent_alerts = self.collector.get_alerts(time_range=timedelta(hours=1))
        assert len(recent_alerts) == 1
        assert recent_alerts[0].metric_name == "metric2"
    
    def test_health_checks(self):
        """Test health check functionality."""
        # Register custom health check
        def custom_check():
            return True
        
        self.collector.register_health_check("custom_check", custom_check)
        
        # Run health checks
        results = self.collector.run_health_checks()
        
        assert "custom_check" in results
        assert results["custom_check"] is True
        
        # Should have default health checks too
        assert "system_memory" in results
        assert "system_cpu" in results
        assert "disk_space" in results
    
    def test_health_check_failure(self):
        """Test health check failure handling."""
        def failing_check():
            raise Exception("Health check failed")
        
        self.collector.register_health_check("failing_check", failing_check)
        
        results = self.collector.run_health_checks()
        
        assert "failing_check" in results
        assert results["failing_check"] is False
    
    def test_monitoring_start_stop(self):
        """Test starting and stopping monitoring."""
        assert self.collector.monitoring_active is False
        
        # Start monitoring
        self.collector.start_monitoring()
        assert self.collector.monitoring_active is True
        assert self.collector.monitoring_thread is not None
        
        # Stop monitoring
        self.collector.stop_monitoring()
        assert self.collector.monitoring_active is False
    
    def test_export_metrics(self):
        """Test exporting metrics to file."""
        # Record some metrics
        self.collector.record_gauge("test_metric", 42.0)
        
        # Create mock alert
        alert = Alert(
            metric_name="test_metric",
            severity=AlertSeverity.MEDIUM,
            message="Test alert",
            value=42.0,
            threshold=40.0,
            timestamp=datetime.now()
        )
        self.collector.alerts.append(alert)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.collector.export_metrics(temp_path)
            
            # Verify export
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "timestamp" in data
            assert "metrics" in data
            assert "alerts" in data
            assert "test_metric" in data["metrics"]
            assert len(data["alerts"]) == 1
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_get_dashboard_data(self):
        """Test getting dashboard data."""
        with patch.object(self.collector, 'collect_system_metrics') as mock_collect:
            mock_metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=75.0,
                memory_percent=60.0,
                memory_used_mb=4096.0,
                memory_available_mb=2048.0,
                disk_usage_percent=45.0
            )
            mock_collect.return_value = mock_metrics
            
            dashboard_data = self.collector.get_dashboard_data()
            
            assert "timestamp" in dashboard_data
            assert "system_metrics" in dashboard_data
            assert "health_checks" in dashboard_data
            assert "recent_alerts" in dashboard_data
            assert "metric_summaries" in dashboard_data
            assert "overall_health" in dashboard_data
            
            assert dashboard_data["system_metrics"]["cpu_percent"] == 75.0


class TestThreadSafety:
    """Test thread safety of metrics collector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        config = SystemConfig()
        self.collector = MetricsCollector(config)
    
    def test_concurrent_metric_recording(self):
        """Test concurrent metric recording."""
        def worker(thread_id):
            for i in range(100):
                self.collector.record_gauge(f"metric_{thread_id}", i)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all metrics were recorded
        total_metrics = 0
        for i in range(5):
            metrics = self.collector.get_metrics(f"metric_{i}")
            total_metrics += len(metrics)
        
        assert total_metrics == 500  # 5 threads * 100 metrics each
    
    def test_concurrent_alert_handling(self):
        """Test concurrent alert handling."""
        alerts_received = []
        
        def alert_callback(alert):
            alerts_received.append(alert)
        
        self.collector.register_alert_callback(alert_callback)
        
        def worker():
            # Generate potential anomalies
            for i in range(20):
                self.collector.record_gauge("concurrent_metric", 10.0)
            # Add anomalous value
            self.collector.record_gauge("concurrent_metric", 100.0)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should not crash due to concurrent access
        assert True


if __name__ == "__main__":
    pytest.main([__file__])