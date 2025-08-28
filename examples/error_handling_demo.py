#!/usr/bin/env python3
"""
GeminiSDR Error Handling System Demo

This script demonstrates the comprehensive error handling and recovery system
including different error types, severity levels, and recovery strategies.
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import geminisdr
sys.path.insert(0, str(Path(__file__).parent.parent))

from geminisdr.core import (
    ErrorHandler,
    HardwareError,
    ConfigurationError,
    ModelError,
    MemoryError,
    ErrorSeverity,
    setup_default_recovery_strategies
)


def setup_logging():
    """Set up logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def demo_hardware_error_recovery(error_handler, logger):
    """Demonstrate hardware error recovery with fallback to simulation."""
    logger.info("=== Hardware Error Recovery Demo ===")
    
    try:
        with error_handler.error_context("sdr_initialization", component="hardware") as ctx:
            # Simulate hardware error
            raise HardwareError(
                "Failed to connect to PlutoSDR",
                device_type="PlutoSDR",
                device_id="usb:1.2.3",
                severity=ErrorSeverity.HIGH
            )
    except Exception as e:
        logger.error(f"Hardware error was not recovered: {e}")
    
    logger.info("Hardware error recovery demo completed\n")


def demo_memory_error_recovery(error_handler, logger):
    """Demonstrate memory error recovery with batch size reduction."""
    logger.info("=== Memory Error Recovery Demo ===")
    
    try:
        with error_handler.error_context("model_training", batch_size=128) as ctx:
            # Simulate GPU memory error
            raise MemoryError(
                "CUDA out of memory",
                memory_type="gpu",
                requested_mb=4096.0,
                available_mb=2048.0,
                severity=ErrorSeverity.HIGH
            )
    except Exception as e:
        logger.error(f"Memory error was not recovered: {e}")
    
    logger.info("Memory error recovery demo completed\n")


def demo_configuration_error_recovery(error_handler, logger):
    """Demonstrate configuration error recovery with default values."""
    logger.info("=== Configuration Error Recovery Demo ===")
    
    try:
        with error_handler.error_context("config_loading") as ctx:
            # Simulate configuration error
            raise ConfigurationError(
                "Invalid batch size: must be positive integer",
                config_key="ml.batch_size",
                config_file="config.yaml",
                severity=ErrorSeverity.MEDIUM
            )
    except Exception as e:
        logger.error(f"Configuration error was not recovered: {e}")
    
    logger.info("Configuration error recovery demo completed\n")


def demo_model_error_recovery(error_handler, logger):
    """Demonstrate model error recovery with version fallback."""
    logger.info("=== Model Error Recovery Demo ===")
    
    try:
        with error_handler.error_context("model_loading") as ctx:
            # Simulate model loading error
            raise ModelError(
                "Model file corrupted or incompatible",
                model_name="neural_amr",
                model_version="v2.1.0",
                severity=ErrorSeverity.MEDIUM
            )
    except Exception as e:
        logger.error(f"Model error was not recovered: {e}")
    
    logger.info("Model error recovery demo completed\n")


def demo_error_statistics(error_handler, logger):
    """Demonstrate error statistics and monitoring."""
    logger.info("=== Error Statistics Demo ===")
    
    stats = error_handler.get_error_statistics()
    
    logger.info(f"Total errors handled: {stats['total_errors']}")
    if stats['total_errors'] > 0:
        logger.info("Error types breakdown:")
        for error_type, count in stats['error_types'].items():
            logger.info(f"  - {error_type}: {count}")
        
        logger.info("Recent errors:")
        for error in stats['recent_errors'][-3:]:  # Show last 3 errors
            logger.info(f"  - {error['timestamp']}: {error['error_type']} - {error['message']}")
    
    logger.info("Error statistics demo completed\n")


def main():
    """Main demo function."""
    logger = setup_logging()
    logger.info("Starting GeminiSDR Error Handling System Demo")
    
    # Create error handler and set up recovery strategies
    error_handler = ErrorHandler(logger)
    strategies = setup_default_recovery_strategies(error_handler, logger)
    
    logger.info(f"Registered recovery strategies for {len(error_handler.recovery_strategies)} error types")
    
    # Run demos
    demo_hardware_error_recovery(error_handler, logger)
    demo_memory_error_recovery(error_handler, logger)
    demo_configuration_error_recovery(error_handler, logger)
    demo_model_error_recovery(error_handler, logger)
    demo_error_statistics(error_handler, logger)
    
    # Show final system state
    logger.info("=== Final System State ===")
    logger.info(f"Simulation mode active: {strategies.is_simulation_mode}")
    logger.info(f"CPU fallback active: {strategies.is_cpu_fallback_active}")
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()