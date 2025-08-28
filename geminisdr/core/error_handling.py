"""
GeminiSDR Error Handling and Recovery System

This module provides comprehensive error handling with severity levels, context tracking,
and recovery strategies for the GeminiSDR system.
"""

import logging
import functools
import time
import traceback
from typing import Type, Callable, Any, Optional, Dict, List, Union
from contextlib import contextmanager
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field


class ErrorSeverity(Enum):
    """Error severity levels for categorizing and handling different types of errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Structured error context information."""
    timestamp: datetime = field(default_factory=datetime.now)
    operation: Optional[str] = None
    component: Optional[str] = None
    user_data: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for logging."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'operation': self.operation,
            'component': self.component,
            'user_data': self.user_data,
            'system_state': self.system_state,
            'stack_trace': self.stack_trace
        }


class GeminiSDRError(Exception):
    """
    Base exception class for all GeminiSDR system errors.
    
    Provides structured error information with severity levels and context tracking.
    """
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Union[Dict[str, Any], ErrorContext]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.cause = cause
        
        # Handle context parameter - can be dict or ErrorContext
        if isinstance(context, ErrorContext):
            self.context = context
        elif isinstance(context, dict):
            self.context = ErrorContext(
                user_data=context,
                stack_trace=traceback.format_exc() if cause else None
            )
        else:
            self.context = ErrorContext(
                stack_trace=traceback.format_exc() if cause else None
            )
    
    def __str__(self) -> str:
        """String representation including severity and context."""
        base_msg = f"[{self.severity.value.upper()}] {self.message}"
        if self.context.operation:
            base_msg += f" (Operation: {self.context.operation})"
        if self.context.component:
            base_msg += f" (Component: {self.context.component})"
        return base_msg
    
    def get_structured_info(self) -> Dict[str, Any]:
        """Get structured error information for logging and debugging."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.value,
            'context': self.context.to_dict(),
            'cause': str(self.cause) if self.cause else None
        }


class HardwareError(GeminiSDRError):
    """
    Hardware-related errors including SDR connection failures, device unavailability,
    and hardware resource exhaustion.
    """
    
    def __init__(
        self, 
        message: str, 
        device_type: Optional[str] = None,
        device_id: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Union[Dict[str, Any], ErrorContext]] = None,
        cause: Optional[Exception] = None
    ):
        # Add hardware-specific context
        hw_context = context or {}
        if isinstance(hw_context, dict):
            hw_context.update({
                'device_type': device_type,
                'device_id': device_id
            })
        elif isinstance(hw_context, ErrorContext):
            hw_context.user_data.update({
                'device_type': device_type,
                'device_id': device_id
            })
        
        super().__init__(message, severity, hw_context, cause)
        self.device_type = device_type
        self.device_id = device_id


class ConfigurationError(GeminiSDRError):
    """
    Configuration-related errors including invalid config values, missing files,
    and environment mismatches.
    """
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Union[Dict[str, Any], ErrorContext]] = None,
        cause: Optional[Exception] = None
    ):
        # Add configuration-specific context
        config_context = context or {}
        if isinstance(config_context, dict):
            config_context.update({
                'config_key': config_key,
                'config_file': config_file
            })
        elif isinstance(config_context, ErrorContext):
            config_context.user_data.update({
                'config_key': config_key,
                'config_file': config_file
            })
        
        super().__init__(message, severity, config_context, cause)
        self.config_key = config_key
        self.config_file = config_file


class ModelError(GeminiSDRError):
    """
    ML model-related errors including loading failures, compatibility issues,
    and training problems.
    """
    
    def __init__(
        self, 
        message: str, 
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Union[Dict[str, Any], ErrorContext]] = None,
        cause: Optional[Exception] = None
    ):
        # Add model-specific context
        model_context = context or {}
        if isinstance(model_context, dict):
            model_context.update({
                'model_name': model_name,
                'model_version': model_version
            })
        elif isinstance(model_context, ErrorContext):
            model_context.user_data.update({
                'model_name': model_name,
                'model_version': model_version
            })
        
        super().__init__(message, severity, model_context, cause)
        self.model_name = model_name
        self.model_version = model_version


class MemoryError(GeminiSDRError):
    """
    Memory-related errors including out-of-memory conditions, allocation failures,
    and resource exhaustion.
    """
    
    def __init__(
        self, 
        message: str, 
        memory_type: Optional[str] = None,  # 'ram', 'gpu', 'swap'
        requested_mb: Optional[float] = None,
        available_mb: Optional[float] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Union[Dict[str, Any], ErrorContext]] = None,
        cause: Optional[Exception] = None
    ):
        # Add memory-specific context
        memory_context = context or {}
        if isinstance(memory_context, dict):
            memory_context.update({
                'memory_type': memory_type,
                'requested_mb': requested_mb,
                'available_mb': available_mb
            })
        elif isinstance(memory_context, ErrorContext):
            memory_context.user_data.update({
                'memory_type': memory_type,
                'requested_mb': requested_mb,
                'available_mb': available_mb
            })
        
        super().__init__(message, severity, memory_context, cause)
        self.memory_type = memory_type
        self.requested_mb = requested_mb
        self.available_mb = available_mb


class ErrorHandler:
    """
    Centralized error handling and recovery system with pluggable recovery strategies.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.recovery_strategies: Dict[Type[Exception], List[Callable]] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    def register_recovery_strategy(
        self, 
        error_type: Type[Exception], 
        strategy: Callable[[Exception, ErrorContext], bool]
    ) -> None:
        """
        Register a recovery strategy for a specific error type.
        
        Args:
            error_type: The exception type to handle
            strategy: Callable that takes (error, context) and returns success boolean
        """
        if error_type not in self.recovery_strategies:
            self.recovery_strategies[error_type] = []
        self.recovery_strategies[error_type].append(strategy)
        self.logger.debug(f"Registered recovery strategy for {error_type.__name__}")
    
    def handle_error(
        self, 
        error: Exception, 
        context: Optional[ErrorContext] = None
    ) -> bool:
        """
        Handle an error with appropriate recovery strategies.
        
        Args:
            error: The exception to handle
            context: Additional context information
            
        Returns:
            bool: True if error was successfully recovered, False otherwise
        """
        # Create context if not provided
        if context is None:
            context = ErrorContext(
                stack_trace=traceback.format_exc()
            )
        
        # Log the error
        self._log_error(error, context)
        
        # Add to error history
        self._add_to_history(error, context)
        
        # Try recovery strategies
        error_type = type(error)
        strategies = self.recovery_strategies.get(error_type, [])
        
        # Also try strategies for parent classes
        for base_type in error_type.__mro__[1:]:  # Skip the error type itself
            if base_type in self.recovery_strategies:
                strategies.extend(self.recovery_strategies[base_type])
        
        for strategy in strategies:
            try:
                self.logger.info(f"Attempting recovery strategy: {strategy.__name__}")
                if strategy(error, context):
                    self.logger.info(f"Recovery successful with strategy: {strategy.__name__}")
                    return True
            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery strategy {strategy.__name__} failed: {recovery_error}"
                )
        
        self.logger.error(f"No successful recovery strategy found for {error_type.__name__}")
        return False
    
    @contextmanager
    def error_context(self, operation: str, component: str = None, **context_data):
        """
        Context manager for error handling with automatic recovery.
        
        Args:
            operation: Name of the operation being performed
            component: Component name where operation is happening
            **context_data: Additional context data
        """
        error_context = ErrorContext(
            operation=operation,
            component=component,
            user_data=context_data
        )
        
        try:
            yield error_context
        except Exception as e:
            # Update context with current system state
            error_context.stack_trace = traceback.format_exc()
            
            # Try to handle the error
            if not self.handle_error(e, error_context):
                # Re-raise if no recovery was successful
                raise
    
    def _log_error(self, error: Exception, context: ErrorContext) -> None:
        """Log error with structured information."""
        if isinstance(error, GeminiSDRError):
            error_info = error.get_structured_info()
            log_level = self._severity_to_log_level(error.severity)
        else:
            error_info = {
                'error_type': type(error).__name__,
                'message': str(error),
                'severity': 'medium',
                'context': context.to_dict()
            }
            log_level = logging.ERROR
        
        self.logger.log(log_level, f"Error occurred: {error_info}")
    
    def _add_to_history(self, error: Exception, context: ErrorContext) -> None:
        """Add error to history for analysis."""
        error_record = {
            'timestamp': context.timestamp.isoformat(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context.to_dict()
        }
        
        self.error_history.append(error_record)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _severity_to_log_level(self, severity: ErrorSeverity) -> int:
        """Convert error severity to logging level."""
        severity_map = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return severity_map.get(severity, logging.ERROR)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent errors."""
        if not self.error_history:
            return {'total_errors': 0}
        
        error_types = {}
        for error_record in self.error_history:
            error_type = error_record['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'recent_errors': self.error_history[-10:]  # Last 10 errors
        }


def retry_with_backoff(
    max_retries: int = 3, 
    base_delay: float = 1.0, 
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to retry on
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Last attempt failed, raise the exception
                        raise
                    
                    # Log retry attempt
                    logger = logging.getLogger(func.__module__)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    # Wait before retry
                    time.sleep(delay)
                    
                    # Increase delay for next attempt
                    delay = min(delay * backoff_factor, max_delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


@contextmanager
def fallback_to_simulation(logger: Optional[logging.Logger] = None):
    """
    Context manager that falls back to simulation mode on hardware errors.
    
    Args:
        logger: Optional logger for recording fallback events
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        yield
    except HardwareError as e:
        logger.warning(f"Hardware error occurred, falling back to simulation: {e}")
        # Here you would implement the actual fallback logic
        # For now, we just log and re-raise with modified context
        fallback_context = e.context
        fallback_context.user_data['fallback_mode'] = 'simulation'
        
        # In a real implementation, you might:
        # 1. Switch configuration to simulation mode
        # 2. Initialize simulation hardware abstraction
        # 3. Continue with simulation instead of real hardware
        
        raise ConfigurationError(
            f"Switched to simulation mode due to hardware error: {e.message}",
            severity=ErrorSeverity.MEDIUM,
            context=fallback_context,
            cause=e
        )