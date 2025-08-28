"""
Tests for the structured logging system.

This module tests the logging infrastructure including structured logging,
context management, performance logging, and log rotation.
"""

import pytest
import json
import tempfile
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from geminisdr.config.config_models import LoggingConfig, LogLevel, LogFormat
from geminisdr.core.logging_manager import (
    StructuredLogger, LoggingManager, JSONFormatter, SimpleFormatter,
    ContextFilter, LogEntry, PerformanceMetric, get_logger, shutdown_logging
)


class TestLogEntry:
    """Test LogEntry dataclass."""
    
    def test_log_entry_creation(self):
        """Test creating a log entry."""
        entry = LogEntry(
            timestamp="2023-01-01T12:00:00",
            level="INFO",
            message="Test message",
            logger_name="test_logger",
            context={"key": "value"},
            extra={"extra_key": "extra_value"}
        )
        
        assert entry.timestamp == "2023-01-01T12:00:00"
        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert entry.logger_name == "test_logger"
        assert entry.context == {"key": "value"}
        assert entry.extra == {"extra_key": "extra_value"}
    
    def test_log_entry_to_dict(self):
        """Test converting log entry to dictionary."""
        entry = LogEntry(
            timestamp="2023-01-01T12:00:00",
            level="INFO",
            message="Test message",
            logger_name="test_logger",
            context={"key": "value"},
            extra={"extra_key": "extra_value"},
            thread_id=12345,
            process_id=67890
        )
        
        result = entry.to_dict()
        
        assert result["timestamp"] == "2023-01-01T12:00:00"
        assert result["level"] == "INFO"
        assert result["message"] == "Test message"
        assert result["logger_name"] == "test_logger"
        assert result["context"] == {"key": "value"}
        assert result["extra"] == {"extra_key": "extra_value"}
        assert result["thread_id"] == 12345
        assert result["process_id"] == 67890


class TestPerformanceMetric:
    """Test PerformanceMetric dataclass."""
    
    def test_performance_metric_creation(self):
        """Test creating a performance metric."""
        metric = PerformanceMetric(
            timestamp="2023-01-01T12:00:00",
            operation="test_operation",
            duration=1.5,
            success=True,
            context={"key": "value"},
            metrics={"count": 10}
        )
        
        assert metric.timestamp == "2023-01-01T12:00:00"
        assert metric.operation == "test_operation"
        assert metric.duration == 1.5
        assert metric.success is True
        assert metric.context == {"key": "value"}
        assert metric.metrics == {"count": 10}


class TestJSONFormatter:
    """Test JSON formatter."""
    
    def test_json_formatter_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        
        # Create a proper log record
        import logging
        logger = logging.getLogger("test_logger")
        record = logger.makeRecord(
            name="test_logger",
            level=logging.INFO,
            fn="test.py",
            lno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.context = {"key": "value"}
        record.extra_field = "extra_value"
        
        result = formatter.format(record)
        parsed = json.loads(result)
        
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["logger_name"] == "test_logger"
        assert parsed["context"] == {"key": "value"}
        assert parsed["thread_id"] == record.thread
        assert parsed["process_id"] == record.process
        assert "timestamp" in parsed
    
    def test_json_formatter_without_context(self):
        """Test JSON formatting without context."""
        formatter = JSONFormatter(include_context=False)
        
        import logging
        logger = logging.getLogger("test_logger")
        record = logger.makeRecord(
            name="test_logger",
            level=logging.INFO,
            fn="test.py",
            lno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.context = {"key": "value"}
        
        result = formatter.format(record)
        parsed = json.loads(result)
        
        assert parsed["context"] == {}


class TestSimpleFormatter:
    """Test simple text formatter."""
    
    def test_simple_formatter(self):
        """Test simple text formatting."""
        formatter = SimpleFormatter()
        
        import logging
        logger = logging.getLogger("test_logger")
        record = logger.makeRecord(
            name="test_logger",
            level=logging.INFO,
            fn="test.py",
            lno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        
        assert "test_logger" in result
        assert "INFO" in result
        assert "Test message" in result


class TestContextFilter:
    """Test context filter."""
    
    def test_context_filter(self):
        """Test adding context to log records."""
        context_manager = Mock()
        context_manager.get_context.return_value = {"key": "value"}
        
        filter_obj = ContextFilter(context_manager)
        record = Mock()
        
        result = filter_obj.filter(record)
        
        assert result is True
        assert record.context == {"key": "value"}
        context_manager.get_context.assert_called_once()


class TestStructuredLogger:
    """Test structured logger functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = LoggingConfig(
            level=LogLevel.DEBUG,
            format=LogFormat.STRUCTURED,
            output=["console"],
            log_dir=self.temp_dir
        )
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = StructuredLogger("test_logger", self.config)
        
        assert logger.name == "test_logger"
        assert logger.config == self.config
        assert logger.context == {}
        assert len(logger.performance_metrics) == 0
    
    def test_context_management(self):
        """Test context management."""
        logger = StructuredLogger("test_logger", self.config)
        
        # Add context
        logger.add_context(key1="value1", key2="value2")
        context = logger.get_context()
        
        assert context == {"key1": "value1", "key2": "value2"}
        
        # Remove context
        logger.remove_context("key1")
        context = logger.get_context()
        
        assert context == {"key2": "value2"}
        
        # Clear context
        logger.clear_context()
        context = logger.get_context()
        
        assert context == {}
    
    def test_context_manager(self):
        """Test temporary context manager."""
        logger = StructuredLogger("test_logger", self.config)
        
        # Set initial context
        logger.add_context(persistent="value")
        
        # Use temporary context
        with logger.context_manager(temp="temp_value"):
            context = logger.get_context()
            assert context == {"persistent": "value", "temp": "temp_value"}
        
        # Context should be restored
        context = logger.get_context()
        assert context == {"persistent": "value"}
    
    def test_structured_logging(self):
        """Test structured logging methods."""
        logger = StructuredLogger("test_logger", self.config)
        
        # Test different log levels
        logger.debug("Debug message", extra_field="debug_value")
        logger.info("Info message", extra_field="info_value")
        logger.warning("Warning message", extra_field="warning_value")
        logger.error("Error message", extra_field="error_value")
        logger.critical("Critical message", extra_field="critical_value")
        
        # Should not raise any exceptions
        assert True
    
    def test_performance_logging(self):
        """Test performance logging."""
        logger = StructuredLogger("test_logger", self.config)
        
        # Log performance metric
        logger.log_performance("test_operation", 1.5, success=True, count=10)
        
        # Check metrics were recorded
        assert len(logger.performance_metrics) == 1
        metric = logger.performance_metrics[0]
        
        assert metric.operation == "test_operation"
        assert metric.duration == 1.5
        assert metric.success is True
        assert metric.metrics == {"count": 10}
    
    def test_performance_timer(self):
        """Test performance timer context manager."""
        logger = StructuredLogger("test_logger", self.config)
        
        # Use performance timer
        with logger.performance_timer("test_operation", count=5):
            time.sleep(0.1)  # Simulate work
        
        # Check metrics were recorded
        assert len(logger.performance_metrics) == 1
        metric = logger.performance_metrics[0]
        
        assert metric.operation == "test_operation"
        assert metric.duration >= 0.1
        assert metric.success is True
        assert metric.metrics == {"count": 5}
    
    def test_performance_timer_with_exception(self):
        """Test performance timer with exception."""
        logger = StructuredLogger("test_logger", self.config)
        
        # Use performance timer with exception
        with pytest.raises(ValueError):
            with logger.performance_timer("test_operation"):
                raise ValueError("Test error")
        
        # Check metrics were recorded with success=False
        assert len(logger.performance_metrics) == 1
        metric = logger.performance_metrics[0]
        
        assert metric.operation == "test_operation"
        assert metric.success is False
    
    def test_error_logging_with_context(self):
        """Test error logging with context."""
        logger = StructuredLogger("test_logger", self.config)
        
        # Add context
        logger.add_context(operation="test_op")
        
        # Log error
        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.log_error_with_context(e, additional="context")
        
        # Should not raise any exceptions
        assert True
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        logger = StructuredLogger("test_logger", self.config)
        
        # Add some metrics
        logger.log_performance("op1", 1.0, success=True)
        logger.log_performance("op2", 2.0, success=False)
        logger.log_performance("op1", 1.5, success=True)
        
        # Get all metrics
        all_metrics = logger.get_performance_metrics()
        assert len(all_metrics) == 3
        
        # Get metrics for specific operation
        op1_metrics = logger.get_performance_metrics(operation="op1")
        assert len(op1_metrics) == 2
        assert all(m.operation == "op1" for m in op1_metrics)
        
        # Get metrics for time range
        time.sleep(0.1)
        recent_metrics = logger.get_performance_metrics(
            time_range=timedelta(seconds=1)
        )
        assert len(recent_metrics) == 3
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        logger = StructuredLogger("test_logger", self.config)
        
        # Add some metrics
        logger.log_performance("test_op", 1.0, success=True)
        logger.log_performance("test_op", 2.0, success=True)
        logger.log_performance("test_op", 1.5, success=False)
        
        # Get summary
        summary = logger.get_performance_summary(operation="test_op")
        
        assert summary["count"] == 3
        assert summary["success_rate"] == 2/3
        assert summary["avg_duration"] == 1.5
        assert summary["min_duration"] == 1.0
        assert summary["max_duration"] == 2.0
        assert summary["total_duration"] == 4.5
    
    def test_file_logging_configuration(self):
        """Test file logging configuration."""
        config = LoggingConfig(
            level=LogLevel.INFO,
            format=LogFormat.STRUCTURED,
            output=["file"],
            log_dir=self.temp_dir,
            rotation="daily"
        )
        
        logger = StructuredLogger("test_logger", config)
        logger.info("Test file logging")
        
        # Check log file was created
        log_file = Path(self.temp_dir) / "test_logger.log"
        assert log_file.exists()


class TestLoggingManager:
    """Test logging manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = LoggingConfig(
            level=LogLevel.DEBUG,
            format=LogFormat.STRUCTURED,
            output=["console"],
            log_dir=self.temp_dir
        )
        # Reset singleton
        LoggingManager._instance = None
    
    def test_singleton_pattern(self):
        """Test singleton pattern implementation."""
        manager1 = LoggingManager(self.config)
        manager2 = LoggingManager()
        
        assert manager1 is manager2
    
    def test_get_logger(self):
        """Test getting loggers."""
        manager = LoggingManager(self.config)
        
        logger1 = manager.get_logger("test_logger1")
        logger2 = manager.get_logger("test_logger2")
        logger1_again = manager.get_logger("test_logger1")
        
        assert isinstance(logger1, StructuredLogger)
        assert isinstance(logger2, StructuredLogger)
        assert logger1 is logger1_again
        assert logger1 is not logger2
    
    def test_update_config(self):
        """Test updating configuration."""
        manager = LoggingManager(self.config)
        logger = manager.get_logger("test_logger")
        
        # Update config
        new_config = LoggingConfig(
            level=LogLevel.ERROR,
            format=LogFormat.SIMPLE,
            output=["console"]
        )
        manager.update_config(new_config)
        
        # Check logger was updated
        assert logger.config == new_config
    
    def test_get_all_performance_metrics(self):
        """Test getting all performance metrics."""
        manager = LoggingManager(self.config)
        
        logger1 = manager.get_logger("logger1")
        logger2 = manager.get_logger("logger2")
        
        # Add metrics
        logger1.log_performance("op1", 1.0)
        logger2.log_performance("op2", 2.0)
        
        # Get all metrics
        all_metrics = manager.get_all_performance_metrics()
        
        assert "logger1" in all_metrics
        assert "logger2" in all_metrics
        assert len(all_metrics["logger1"]) == 1
        assert len(all_metrics["logger2"]) == 1
    
    def test_shutdown(self):
        """Test shutdown functionality."""
        manager = LoggingManager(self.config)
        logger = manager.get_logger("test_logger")
        
        # Shutdown
        manager.shutdown()
        
        # Check loggers were cleared
        assert len(manager.loggers) == 0


class TestGlobalFunctions:
    """Test global logging functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state
        shutdown_logging()
    
    def test_get_logger_function(self):
        """Test global get_logger function."""
        config = LoggingConfig(level=LogLevel.INFO)
        
        logger1 = get_logger("test_logger", config)
        logger2 = get_logger("test_logger")
        
        assert isinstance(logger1, StructuredLogger)
        assert logger1 is logger2
    
    def test_shutdown_logging_function(self):
        """Test global shutdown function."""
        logger = get_logger("test_logger")
        
        # Should not raise any exceptions
        shutdown_logging()
        
        # Getting logger again should create new manager
        new_logger = get_logger("test_logger")
        assert new_logger is not logger


class TestThreadSafety:
    """Test thread safety of logging components."""
    
    def test_concurrent_context_management(self):
        """Test concurrent context management."""
        config = LoggingConfig(level=LogLevel.DEBUG)
        logger = StructuredLogger("test_logger", config)
        
        results = []
        
        def worker(thread_id):
            logger.add_context(thread_id=thread_id)
            time.sleep(0.1)
            context = logger.get_context()
            results.append(context.get("thread_id"))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All threads should have added their context
        # (Note: due to shared context, the final result will be unpredictable,
        # but no exceptions should occur)
        assert len(results) == 5
    
    def test_concurrent_performance_logging(self):
        """Test concurrent performance logging."""
        config = LoggingConfig(level=LogLevel.DEBUG)
        logger = StructuredLogger("test_logger", config)
        
        def worker(thread_id):
            for i in range(10):
                logger.log_performance(f"op_{thread_id}", 0.1, count=i)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have 30 metrics total
        assert len(logger.performance_metrics) == 30


class TestConfigurationValidation:
    """Test configuration validation and error handling."""
    
    def test_invalid_log_level(self):
        """Test handling of invalid log level."""
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID")
    
    def test_invalid_log_format(self):
        """Test handling of invalid log format."""
        with pytest.raises(ValueError):
            LoggingConfig(format="INVALID")
    
    def test_invalid_output(self):
        """Test handling of invalid output."""
        with pytest.raises(ValueError):
            LoggingConfig(output=["invalid_output"])
    
    def test_invalid_rotation(self):
        """Test handling of invalid rotation."""
        with pytest.raises(ValueError):
            LoggingConfig(rotation="invalid_rotation")


if __name__ == "__main__":
    pytest.main([__file__])