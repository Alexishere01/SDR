"""
GeminiSDR Core Module

This module contains core functionality including error handling, memory management,
model management, logging, metrics collection, and other system-level components.
"""

from .error_handling import (
    ErrorSeverity,
    GeminiSDRError,
    HardwareError,
    ConfigurationError,
    ModelError,
    MemoryError,
    ErrorHandler
)
from .recovery_strategies import (
    RecoveryStrategies,
    setup_default_recovery_strategies
)
from .model_metadata import (
    ModelMetadata,
    PerformanceMetrics,
    ModelFingerprinter,
    CompatibilityChecker,
    ModelComparator,
    ModelTracker
)
from .model_manager import (
    ModelManager,
    ModelRegistry
)
from .logging_manager import (
    StructuredLogger,
    LoggingManager,
    JSONFormatter,
    SimpleFormatter,
    LogEntry,
    PerformanceMetric,
    get_logger,
    shutdown_logging
)
from .metrics_collector import (
    MetricsCollector,
    Metric,
    Alert,
    SystemMetrics,
    MLMetrics,
    AnomalyDetector,
    MetricType,
    AlertSeverity
)

__all__ = [
    'ErrorSeverity',
    'GeminiSDRError',
    'HardwareError',
    'ConfigurationError',
    'ModelError',
    'MemoryError',
    'ErrorHandler',
    'RecoveryStrategies',
    'setup_default_recovery_strategies',
    'ModelMetadata',
    'PerformanceMetrics',
    'ModelFingerprinter',
    'CompatibilityChecker',
    'ModelComparator',
    'ModelTracker',
    'ModelManager',
    'ModelRegistry',
    'StructuredLogger',
    'LoggingManager',
    'JSONFormatter',
    'SimpleFormatter',
    'LogEntry',
    'PerformanceMetric',
    'get_logger',
    'shutdown_logging',
    'MetricsCollector',
    'Metric',
    'Alert',
    'SystemMetrics',
    'MLMetrics',
    'AnomalyDetector',
    'MetricType',
    'AlertSeverity'
]