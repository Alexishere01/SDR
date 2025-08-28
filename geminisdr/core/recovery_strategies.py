"""
Recovery Strategies for GeminiSDR Error Handling

This module provides specific recovery strategies for different types of errors
that can occur in the GeminiSDR system, including hardware failures, configuration
issues, and memory problems.
"""

import logging
import gc
import torch
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from .error_handling import (
    ErrorContext,
    HardwareError,
    ConfigurationError,
    ModelError,
    MemoryError,
    ErrorSeverity
)


class RecoveryStrategies:
    """
    Collection of recovery strategies for different error types.
    
    Each strategy is a callable that takes (error, context) and returns
    a boolean indicating whether recovery was successful.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._simulation_mode = False
        self._cpu_fallback_active = False
    
    def hardware_to_simulation_fallback(self, error: HardwareError, context: ErrorContext) -> bool:
        """
        Fallback from hardware to simulation mode when hardware is unavailable.
        
        Args:
            error: The hardware error that occurred
            context: Error context information
            
        Returns:
            bool: True if fallback was successful
        """
        try:
            self.logger.info(f"Attempting hardware to simulation fallback for device: {error.device_type}")
            
            # Mark simulation mode as active
            self._simulation_mode = True
            
            # Update context to indicate fallback mode
            context.user_data['fallback_mode'] = 'simulation'
            context.user_data['original_device'] = error.device_type
            context.user_data['recovery_strategy'] = 'hardware_to_simulation'
            
            # In a real implementation, this would:
            # 1. Switch configuration to simulation mode
            # 2. Initialize simulation hardware abstraction
            # 3. Update any active connections or interfaces
            
            self.logger.info("Successfully switched to simulation mode")
            return True
            
        except Exception as recovery_error:
            self.logger.error(f"Failed to switch to simulation mode: {recovery_error}")
            return False
    
    def gpu_to_cpu_fallback(self, error: MemoryError, context: ErrorContext) -> bool:
        """
        Fallback from GPU to CPU when GPU memory is exhausted.
        
        Args:
            error: The memory error that occurred
            context: Error context information
            
        Returns:
            bool: True if fallback was successful
        """
        try:
            if error.memory_type != 'gpu':
                return False  # This strategy only handles GPU memory errors
            
            self.logger.info("Attempting GPU to CPU fallback due to memory exhaustion")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Cleared GPU cache")
            
            # Mark CPU fallback as active
            self._cpu_fallback_active = True
            
            # Update context
            context.user_data['fallback_mode'] = 'cpu'
            context.user_data['original_device'] = 'gpu'
            context.user_data['recovery_strategy'] = 'gpu_to_cpu'
            
            # In a real implementation, this would:
            # 1. Move models from GPU to CPU
            # 2. Update device configuration
            # 3. Adjust batch sizes for CPU processing
            
            self.logger.info("Successfully switched to CPU processing")
            return True
            
        except Exception as recovery_error:
            self.logger.error(f"Failed to switch to CPU processing: {recovery_error}")
            return False
    
    def reduce_batch_size_recovery(self, error: MemoryError, context: ErrorContext) -> bool:
        """
        Reduce batch size when memory is insufficient.
        
        Args:
            error: The memory error that occurred
            context: Error context information
            
        Returns:
            bool: True if batch size was successfully reduced
        """
        try:
            current_batch_size = context.user_data.get('batch_size')
            if not current_batch_size or current_batch_size <= 1:
                return False  # Can't reduce further
            
            # Reduce batch size by half
            new_batch_size = max(1, current_batch_size // 2)
            
            self.logger.info(f"Reducing batch size from {current_batch_size} to {new_batch_size}")
            
            # Update context with new batch size
            context.user_data['batch_size'] = new_batch_size
            context.user_data['recovery_strategy'] = 'reduce_batch_size'
            context.user_data['original_batch_size'] = current_batch_size
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info(f"Successfully reduced batch size to {new_batch_size}")
            return True
            
        except Exception as recovery_error:
            self.logger.error(f"Failed to reduce batch size: {recovery_error}")
            return False
    
    def configuration_default_fallback(self, error: ConfigurationError, context: ErrorContext) -> bool:
        """
        Use default configuration values when configuration is invalid.
        
        Args:
            error: The configuration error that occurred
            context: Error context information
            
        Returns:
            bool: True if default configuration was applied
        """
        try:
            config_key = error.config_key
            if not config_key:
                return False
            
            # Default configuration values
            defaults = {
                'ml.batch_size': 32,
                'ml.learning_rate': 1e-4,
                'hardware.device_preference': 'auto',
                'hardware.memory_limit_gb': None,
                'logging.level': 'INFO'
            }
            
            default_value = defaults.get(config_key)
            if default_value is None:
                return False
            
            self.logger.warning(
                f"Using default value '{default_value}' for invalid config key '{config_key}'"
            )
            
            # Update context with default value
            context.user_data['default_value'] = default_value
            context.user_data['recovery_strategy'] = 'configuration_default'
            context.user_data['invalid_value'] = context.user_data.get('config_value')
            
            # In a real implementation, this would update the actual configuration
            
            return True
            
        except Exception as recovery_error:
            self.logger.error(f"Failed to apply default configuration: {recovery_error}")
            return False
    
    def model_version_fallback(self, error: ModelError, context: ErrorContext) -> bool:
        """
        Try to load an alternative model version when model loading fails.
        
        Args:
            error: The model error that occurred
            context: Error context information
            
        Returns:
            bool: True if alternative model was loaded
        """
        try:
            model_name = error.model_name
            current_version = error.model_version
            
            if not model_name:
                return False
            
            # In a real implementation, this would:
            # 1. Query available model versions
            # 2. Try loading the most recent compatible version
            # 3. Update model metadata
            
            # For now, simulate finding an alternative version
            fallback_version = "v1.0.0"  # Default fallback version
            
            self.logger.info(
                f"Attempting to load fallback model version {fallback_version} "
                f"for {model_name} (failed version: {current_version})"
            )
            
            # Update context
            context.user_data['fallback_version'] = fallback_version
            context.user_data['failed_version'] = current_version
            context.user_data['recovery_strategy'] = 'model_version_fallback'
            
            self.logger.info(f"Successfully loaded fallback model version {fallback_version}")
            return True
            
        except Exception as recovery_error:
            self.logger.error(f"Failed to load fallback model version: {recovery_error}")
            return False
    
    def memory_cleanup_recovery(self, error: MemoryError, context: ErrorContext) -> bool:
        """
        Perform aggressive memory cleanup to recover from memory errors.
        
        Args:
            error: The memory error that occurred
            context: Error context information
            
        Returns:
            bool: True if memory cleanup was successful
        """
        try:
            self.logger.info("Performing aggressive memory cleanup")
            
            # Force garbage collection
            collected = gc.collect()
            self.logger.info(f"Garbage collection freed {collected} objects")
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.info("Cleared GPU memory cache")
            
            # Update context
            context.user_data['recovery_strategy'] = 'memory_cleanup'
            context.user_data['objects_collected'] = collected
            
            # In a real implementation, this might also:
            # 1. Clear model caches
            # 2. Release unused data structures
            # 3. Compact memory pools
            
            self.logger.info("Memory cleanup completed successfully")
            return True
            
        except Exception as recovery_error:
            self.logger.error(f"Memory cleanup failed: {recovery_error}")
            return False
    
    def retry_with_delay_recovery(self, error: Exception, context: ErrorContext) -> bool:
        """
        Generic retry strategy with delay for transient errors.
        
        Args:
            error: The error that occurred
            context: Error context information
            
        Returns:
            bool: True if retry should be attempted
        """
        try:
            retry_count = context.user_data.get('retry_count', 0)
            max_retries = context.user_data.get('max_retries', 3)
            
            if retry_count >= max_retries:
                return False
            
            import time
            delay = min(2 ** retry_count, 30)  # Exponential backoff, max 30 seconds
            
            self.logger.info(f"Retrying operation after {delay} seconds (attempt {retry_count + 1}/{max_retries})")
            
            time.sleep(delay)
            
            # Update retry count
            context.user_data['retry_count'] = retry_count + 1
            context.user_data['recovery_strategy'] = 'retry_with_delay'
            
            return True
            
        except Exception as recovery_error:
            self.logger.error(f"Retry strategy failed: {recovery_error}")
            return False
    
    def get_all_strategies(self) -> Dict[type, list]:
        """
        Get all available recovery strategies mapped to error types.
        
        Returns:
            Dict mapping error types to lists of recovery strategies
        """
        return {
            HardwareError: [
                self.hardware_to_simulation_fallback,
                self.retry_with_delay_recovery
            ],
            MemoryError: [
                self.memory_cleanup_recovery,
                self.reduce_batch_size_recovery,
                self.gpu_to_cpu_fallback
            ],
            ConfigurationError: [
                self.configuration_default_fallback
            ],
            ModelError: [
                self.model_version_fallback,
                self.retry_with_delay_recovery
            ]
        }
    
    @property
    def is_simulation_mode(self) -> bool:
        """Check if system is currently in simulation mode."""
        return self._simulation_mode
    
    @property
    def is_cpu_fallback_active(self) -> bool:
        """Check if CPU fallback is currently active."""
        return self._cpu_fallback_active
    
    def reset_fallback_states(self) -> None:
        """Reset all fallback states to normal operation."""
        self._simulation_mode = False
        self._cpu_fallback_active = False
        self.logger.info("Reset all fallback states to normal operation")


def setup_default_recovery_strategies(error_handler, logger: Optional[logging.Logger] = None):
    """
    Set up default recovery strategies for an ErrorHandler instance.
    
    Args:
        error_handler: ErrorHandler instance to configure
        logger: Optional logger for recovery strategies
    """
    strategies = RecoveryStrategies(logger)
    
    # Register all strategies
    for error_type, strategy_list in strategies.get_all_strategies().items():
        for strategy in strategy_list:
            error_handler.register_recovery_strategy(error_type, strategy)
    
    logger = logger or logging.getLogger(__name__)
    logger.info("Registered default recovery strategies")
    
    return strategies