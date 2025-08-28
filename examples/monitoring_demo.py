#!/usr/bin/env python3
"""
Monitoring and metrics collection demonstration.

This script demonstrates the comprehensive monitoring capabilities
including metrics collection, anomaly detection, alerting, and
dashboard data generation.
"""

import time
import json
from datetime import datetime, timedelta
from pathlib import Path

from geminisdr.config.config_models import SystemConfig, LoggingConfig, PerformanceConfig
from geminisdr.core.logging_manager import get_logger
from geminisdr.core.metrics_collector import MetricsCollector, AlertSeverity


def alert_callback(alert):
    """Example alert callback function."""
    print(f"🚨 ALERT: {alert.severity.value.upper()} - {alert.message}")
    print(f"   Metric: {alert.metric_name} = {alert.value} (threshold: {alert.threshold})")
    print(f"   Time: {alert.timestamp}")
    print()


def demonstrate_basic_metrics():
    """Demonstrate basic metrics collection."""
    print("=== Basic Metrics Collection ===")
    
    # Create system configuration
    config = SystemConfig(
        logging=LoggingConfig(),
        performance=PerformanceConfig()
    )
    
    # Initialize metrics collector
    collector = MetricsCollector(config)
    
    # Register alert callback
    collector.register_alert_callback(alert_callback)
    
    # Record various types of metrics
    print("Recording metrics...")
    
    # Counter metrics
    for i in range(5):
        collector.record_counter("requests.total", 1, tags={"endpoint": "/api/data"})
        collector.record_counter("requests.total", 1, tags={"endpoint": "/api/models"})
    
    # Gauge metrics
    collector.record_gauge("system.temperature", 65.5, tags={"sensor": "cpu"})
    collector.record_gauge("system.temperature", 45.2, tags={"sensor": "gpu"})
    collector.record_gauge("memory.usage_percent", 75.0)
    
    # Timer metrics
    collector.record_timer("api.response_time", 0.125, tags={"endpoint": "/api/data"})
    collector.record_timer("api.response_time", 0.089, tags={"endpoint": "/api/models"})
    
    print(f"Total metrics recorded: {sum(len(metrics) for metrics in collector.metrics.values())}")
    print()


def demonstrate_ml_metrics():
    """Demonstrate ML metrics recording."""
    print("=== ML Metrics Collection ===")
    
    config = SystemConfig()
    collector = MetricsCollector(config)
    
    # Simulate training metrics
    print("Recording training metrics...")
    for epoch in range(1, 6):
        loss = 1.0 - (epoch * 0.15)  # Decreasing loss
        accuracy = 0.6 + (epoch * 0.08)  # Increasing accuracy
        
        collector.record_ml_metrics(
            model_name="neural_amr_v1",
            operation="training",
            epoch=epoch,
            batch_size=32,
            loss=loss,
            accuracy=accuracy,
            learning_rate=1e-4,
            duration=120.5,
            memory_usage_mb=2048.0
        )
    
    # Simulate inference metrics
    print("Recording inference metrics...")
    for i in range(3):
        collector.record_ml_metrics(
            model_name="neural_amr_v1",
            operation="inference",
            batch_size=64,
            duration=0.05,
            memory_usage_mb=512.0
        )
    
    # Get ML metrics summary
    loss_summary = collector.get_metric_summary("ml.training.loss")
    accuracy_summary = collector.get_metric_summary("ml.training.accuracy")
    
    print(f"Training Loss - Min: {loss_summary.get('min', 'N/A'):.3f}, "
          f"Max: {loss_summary.get('max', 'N/A'):.3f}, "
          f"Mean: {loss_summary.get('mean', 'N/A'):.3f}")
    print(f"Training Accuracy - Min: {accuracy_summary.get('min', 'N/A'):.3f}, "
          f"Max: {accuracy_summary.get('max', 'N/A'):.3f}, "
          f"Mean: {accuracy_summary.get('mean', 'N/A'):.3f}")
    print()


def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection and alerting."""
    print("=== Anomaly Detection ===")
    
    config = SystemConfig()
    collector = MetricsCollector(config)
    collector.register_alert_callback(alert_callback)
    
    print("Recording normal values...")
    # Record normal values
    for i in range(20):
        collector.record_gauge("cpu.usage", 25.0 + (i % 5))  # 25-29% CPU usage
        time.sleep(0.01)  # Small delay to avoid overwhelming
    
    print("Introducing anomaly...")
    # Introduce anomaly
    collector.record_gauge("cpu.usage", 95.0)  # Spike to 95%
    
    print("Recording more normal values...")
    # More normal values
    for i in range(5):
        collector.record_gauge("cpu.usage", 26.0 + (i % 3))
        time.sleep(0.01)
    
    # Check alerts
    alerts = collector.get_alerts()
    print(f"Total alerts generated: {len(alerts)}")
    
    for alert in alerts:
        if not alert.resolved:
            print(f"Unresolved alert: {alert.metric_name} - {alert.message}")
    
    print()


def demonstrate_health_checks():
    """Demonstrate health check functionality."""
    print("=== Health Checks ===")
    
    config = SystemConfig()
    collector = MetricsCollector(config)
    
    # Add custom health checks
    def database_health():
        """Simulate database health check."""
        return True  # Assume healthy
    
    def api_health():
        """Simulate API health check."""
        return True  # Assume healthy
    
    def cache_health():
        """Simulate cache health check."""
        return False  # Simulate unhealthy cache
    
    collector.register_health_check("database", database_health)
    collector.register_health_check("api", api_health)
    collector.register_health_check("cache", cache_health)
    
    # Run health checks
    health_results = collector.run_health_checks()
    
    print("Health Check Results:")
    for check_name, is_healthy in health_results.items():
        status = "✅ HEALTHY" if is_healthy else "❌ UNHEALTHY"
        print(f"  {check_name}: {status}")
    
    overall_health = all(health_results.values())
    print(f"\nOverall System Health: {'✅ HEALTHY' if overall_health else '❌ UNHEALTHY'}")
    print()


def demonstrate_dashboard_data():
    """Demonstrate dashboard data generation."""
    print("=== Dashboard Data ===")
    
    config = SystemConfig()
    collector = MetricsCollector(config)
    
    # Record some sample data
    collector.record_gauge("system.cpu_percent", 45.0)
    collector.record_gauge("system.memory_percent", 60.0)
    collector.record_gauge("system.disk_usage_percent", 35.0)
    
    # Get dashboard data
    dashboard_data = collector.get_dashboard_data()
    
    print("Dashboard Data Structure:")
    print(f"  Timestamp: {dashboard_data['timestamp']}")
    print(f"  Overall Health: {dashboard_data['overall_health']}")
    print(f"  Health Checks: {len(dashboard_data['health_checks'])} checks")
    print(f"  Recent Alerts: {len(dashboard_data['recent_alerts'])} alerts")
    print(f"  Metric Summaries: {len(dashboard_data['metric_summaries'])} metrics")
    
    # Show system metrics
    sys_metrics = dashboard_data['system_metrics']
    print(f"  System Metrics:")
    print(f"    CPU: {sys_metrics['cpu_percent']:.1f}%")
    print(f"    Memory: {sys_metrics['memory_percent']:.1f}%")
    print(f"    Disk: {sys_metrics['disk_usage_percent']:.1f}%")
    print()


def demonstrate_metrics_export():
    """Demonstrate metrics export functionality."""
    print("=== Metrics Export ===")
    
    config = SystemConfig()
    collector = MetricsCollector(config)
    
    # Record some sample metrics
    collector.record_gauge("export.test_metric", 42.0)
    collector.record_counter("export.test_counter", 5)
    collector.record_timer("export.test_timer", 1.25)
    
    # Export metrics to file
    export_file = Path("metrics_export.json")
    collector.export_metrics(str(export_file))
    
    if export_file.exists():
        print(f"Metrics exported to: {export_file}")
        
        # Show export file size
        file_size = export_file.stat().st_size
        print(f"Export file size: {file_size} bytes")
        
        # Show sample of exported data
        with open(export_file, 'r') as f:
            data = json.load(f)
        
        print(f"Exported data contains:")
        print(f"  Metrics: {len(data.get('metrics', {}))}")
        print(f"  Alerts: {len(data.get('alerts', []))}")
        
        # Clean up
        export_file.unlink()
        print("Export file cleaned up")
    else:
        print("❌ Export failed")
    
    print()


def demonstrate_monitoring_lifecycle():
    """Demonstrate monitoring start/stop lifecycle."""
    print("=== Monitoring Lifecycle ===")
    
    config = SystemConfig()
    collector = MetricsCollector(config)
    
    print("Starting background monitoring...")
    collector.start_monitoring()
    
    print(f"Monitoring active: {collector.monitoring_active}")
    print(f"Monitoring thread alive: {collector.monitoring_thread.is_alive()}")
    
    # Let it run for a short time
    print("Letting monitoring run for 2 seconds...")
    time.sleep(2)
    
    print("Stopping monitoring...")
    collector.stop_monitoring()
    
    print(f"Monitoring active: {collector.monitoring_active}")
    print("Monitoring stopped successfully")
    print()


def main():
    """Run all monitoring demonstrations."""
    print("🔍 GeminiSDR Monitoring & Metrics Collection Demo")
    print("=" * 50)
    print()
    
    try:
        demonstrate_basic_metrics()
        demonstrate_ml_metrics()
        demonstrate_anomaly_detection()
        demonstrate_health_checks()
        demonstrate_dashboard_data()
        demonstrate_metrics_export()
        demonstrate_monitoring_lifecycle()
        
        print("✅ All monitoring demonstrations completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()