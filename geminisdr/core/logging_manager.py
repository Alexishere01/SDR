"""
Structured logging and monitoring system for GeminiSDR.

This module provides comprehensive logging capabilities with JSON output,
context management, log rotation, and performance metrics collection.
"""

import logging
import logging.handlers
import json
import time
import threading
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import traceback
import sys
import os

from geminisdr.config.config_models import LoggingConfig, LogLevel, LogFormat


@dataclass
class LogEntry:
    """Structured log entry with context and metadata."""
    timestamp: str
    level: str
    message: str
    logger_name: str
    context: Dict[str, Any]
    extra: Dict[str, Any]
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PerformanceMetric:
    """Performance metric entry."""
    timestamp: str
    operation: str
    duration: float
    success: bool
    context: Dict[str, Any]
    metrics: Dict[str, Union[int, float]]


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract context from record if available
        context = getattr(record, 'context', {})
        extra = {k: v for k, v in record.__dict__.items() 
                if k not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                           'pathname', 'filename', 'module', 'lineno', 'funcName',
                           'created', 'msecs', 'relativeCreated', 'thread', 
                           'threadName', 'processName', 'process', 'context']}
        
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            context=context if self.include_context else {},
            extra=extra,
            thread_id=record.thread,
            process_id=record.process
        )
        
        return json.dumps(log_entry.to_dict(), default=str)


class SimpleFormatter(logging.Formatter):
    """Simple text formatter for human-readable logs."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def __init__(self, context_manager):
        super().__init__()
        self.context_manager = context_manager
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        record.context = self.context_manager.get_context()
        return True


class StructuredLogger:
    """
    Structured logging with JSON output and context management.
    
    Provides comprehensive logging capabilities with:
    - JSON structured output
    - Context management
    - Performance logging
    - Error tracking with full context
    """
    
    def __init__(self, name: str, config: LoggingConfig):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            config: Logging configuration
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.context = {}
        self.context_lock = threading.RLock()
        self.performance_metrics = deque(maxlen=1000)
        self.metrics_lock = threading.RLock()
        
        # Configure logger
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up logger with handlers and formatters."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        self.logger.setLevel(getattr(logging, self.config.level.value))
        
        # Create log directory if needed
        if "file" in self.config.output:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add context filter
        context_filter = ContextFilter(self)
        
        # Configure handlers based on output settings
        for output in self.config.output:
            handler = self._create_handler(output)
            if handler:
                handler.addFilter(context_filter)
                self.logger.addHandler(handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _create_handler(self, output: str) -> Optional[logging.Handler]:
        """Create appropriate handler for output type."""
        if output == "console":
            handler = logging.StreamHandler(sys.stdout)
        elif output == "file":
            log_file = Path(self.config.log_dir) / f"{self.name}.log"
            
            if self.config.rotation == "size":
                handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                    backupCount=self.config.backup_count
                )
            elif self.config.rotation == "daily":
                handler = logging.handlers.TimedRotatingFileHandler(
                    log_file,
                    when='midnight',
                    interval=1,
                    backupCount=self.config.backup_count
                )
            elif self.config.rotation == "weekly":
                handler = logging.handlers.TimedRotatingFileHandler(
                    log_file,
                    when='W0',  # Monday
                    interval=1,
                    backupCount=self.config.backup_count
                )
            elif self.config.rotation == "monthly":
                handler = logging.handlers.TimedRotatingFileHandler(
                    log_file,
                    when='midnight',
                    interval=30,
                    backupCount=self.config.backup_count
                )
            else:
                handler = logging.FileHandler(log_file)
        elif output == "syslog":
            try:
                handler = logging.handlers.SysLogHandler()
            except Exception:
                # Fallback to console if syslog not available
                handler = logging.StreamHandler(sys.stdout)
        else:
            return None
        
        # Set formatter
        if self.config.format == LogFormat.STRUCTURED:
            formatter = JSONFormatter()
        else:
            formatter = SimpleFormatter()
        
        handler.setFormatter(formatter)
        return handler
    
    def add_context(self, **kwargs) -> None:
        """
        Add persistent context to all log messages.
        
        Args:
            **kwargs: Context key-value pairs
        """
        with self.context_lock:
            self.context.update(kwargs)
    
    def remove_context(self, *keys) -> None:
        """
        Remove context keys.
        
        Args:
            *keys: Context keys to remove
        """
        with self.context_lock:
            for key in keys:
                self.context.pop(key, None)
    
    def clear_context(self) -> None:
        """Clear all context."""
        with self.context_lock:
            self.context.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context."""
        with self.context_lock:
            return self.context.copy()
    
    @contextmanager
    def context_manager(self, **kwargs):
        """
        Context manager for temporary context.
        
        Args:
            **kwargs: Temporary context key-value pairs
        """
        # Save current context
        original_context = self.get_context()
        
        try:
            # Add temporary context
            self.add_context(**kwargs)
            yield
        finally:
            # Restore original context
            with self.context_lock:
                self.context = original_context
    
    def log_structured(self, level: str, message: str, **extra) -> None:
        """
        Log structured message with context.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            **extra: Additional fields to include in log
        """
        log_level = getattr(logging, level.upper())
        
        # Create log record with extra fields
        extra_dict = dict(extra)
        self.logger.log(log_level, message, extra=extra_dict)
    
    def debug(self, message: str, **extra) -> None:
        """Log debug message."""
        self.log_structured("DEBUG", message, **extra)
    
    def info(self, message: str, **extra) -> None:
        """Log info message."""
        self.log_structured("INFO", message, **extra)
    
    def warning(self, message: str, **extra) -> None:
        """Log warning message."""
        self.log_structured("WARNING", message, **extra)
    
    def error(self, message: str, **extra) -> None:
        """Log error message."""
        self.log_structured("ERROR", message, **extra)
    
    def critical(self, message: str, **extra) -> None:
        """Log critical message."""
        self.log_structured("CRITICAL", message, **extra)
    
    def log_performance(self, operation: str, duration: float, 
                       success: bool = True, **metrics) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            success: Whether operation was successful
            **metrics: Additional performance metrics
        """
        if not self.config.enable_performance_logging:
            return
        
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            duration=duration,
            success=success,
            context=self.get_context(),
            metrics=metrics
        )
        
        with self.metrics_lock:
            self.performance_metrics.append(metric)
        
        # Log performance metric
        self.info(
            f"Performance: {operation}",
            operation=operation,
            duration=duration,
            success=success,
            **metrics
        )
    
    @contextmanager
    def performance_timer(self, operation: str, **metrics):
        """
        Context manager for timing operations.
        
        Args:
            operation: Operation name
            **metrics: Additional metrics to log
        """
        start_time = time.time()
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            self.log_error_with_context(e, operation=operation)
            raise
        finally:
            duration = time.time() - start_time
            self.log_performance(operation, duration, success, **metrics)
    
    def log_error_with_context(self, error: Exception, **context) -> None:
        """
        Log error with full context information.
        
        Args:
            error: Exception to log
            **context: Additional context
        """
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            **context
        }
        
        self.error(
            f"Error occurred: {str(error)}",
            **error_context
        )
    
    def get_performance_metrics(self, operation: Optional[str] = None,
                              time_range: Optional[timedelta] = None) -> List[PerformanceMetric]:
        """
        Get performance metrics.
        
        Args:
            operation: Filter by operation name
            time_range: Filter by time range from now
            
        Returns:
            List of performance metrics
        """
        with self.metrics_lock:
            metrics = list(self.performance_metrics)
        
        # Filter by operation
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        
        # Filter by time range
        if time_range:
            cutoff = datetime.now() - time_range
            metrics = [m for m in metrics 
                      if datetime.fromisoformat(m.timestamp) >= cutoff]
        
        return metrics
    
    def get_performance_summary(self, operation: Optional[str] = None,
                              time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get performance metrics summary.
        
        Args:
            operation: Filter by operation name
            time_range: Filter by time range from now
            
        Returns:
            Performance summary statistics
        """
        metrics = self.get_performance_metrics(operation, time_range)
        
        if not metrics:
            return {"count": 0}
        
        durations = [m.duration for m in metrics]
        success_count = sum(1 for m in metrics if m.success)
        
        return {
            "count": len(metrics),
            "success_rate": success_count / len(metrics),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations)
        }


class LoggingManager:
    """
    Central logging manager for the GeminiSDR system.
    
    Manages multiple structured loggers and provides system-wide
    logging configuration and coordination.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[LoggingConfig] = None):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        """Initialize logging manager."""
        if hasattr(self, '_initialized'):
            return
        
        self.config = config or LoggingConfig()
        self.loggers: Dict[str, StructuredLogger] = {}
        self.loggers_lock = threading.RLock()
        self._initialized = True
        
        # Set up root logger configuration
        self._setup_root_logger()
    
    def _setup_root_logger(self) -> None:
        """Configure root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.value))
        
        # Prevent duplicate logs from third-party libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> StructuredLogger:
        """
        Get or create a structured logger.
        
        Args:
            name: Logger name
            
        Returns:
            StructuredLogger instance
        """
        with self.loggers_lock:
            if name not in self.loggers:
                self.loggers[name] = StructuredLogger(name, self.config)
            return self.loggers[name]
    
    def update_config(self, config: LoggingConfig) -> None:
        """
        Update logging configuration for all loggers.
        
        Args:
            config: New logging configuration
        """
        self.config = config
        
        with self.loggers_lock:
            # Update existing loggers
            for logger in self.loggers.values():
                logger.config = config
                logger._setup_logger()
        
        # Update root logger
        self._setup_root_logger()
    
    def get_all_performance_metrics(self) -> Dict[str, List[PerformanceMetric]]:
        """Get performance metrics from all loggers."""
        metrics = {}
        
        with self.loggers_lock:
            for name, logger in self.loggers.items():
                metrics[name] = logger.get_performance_metrics()
        
        return metrics
    
    def shutdown(self) -> None:
        """Shutdown all loggers and handlers."""
        with self.loggers_lock:
            for logger in self.loggers.values():
                for handler in logger.logger.handlers:
                    handler.close()
                logger.logger.handlers.clear()
            self.loggers.clear()


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        config: Optional logging configuration (uses default if not provided)
        
    Returns:
        StructuredLogger instance
    """
    global _logging_manager
    
    if _logging_manager is None:
        _logging_manager = LoggingManager(config)
    elif config is not None:
        _logging_manager.update_config(config)
    
    return _logging_manager.get_logger(name)


def shutdown_logging() -> None:
    """Shutdown the logging system."""
    global _logging_manager
    
    if _logging_manager is not None:
        _logging_manager.shutdown()
        _logging_manager = None