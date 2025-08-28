"""
Tests for GeminiSDR Error Handling System

This module tests the comprehensive error handling and recovery system,
including base error classes, severity levels, context tracking, and recovery strategies.
"""

import pytest
import logging
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, ANY

from geminisdr.core.error_handling import (
    ErrorSeverity,
    ErrorContext,
    GeminiSDRError,
    HardwareError,
    ConfigurationError,
    ModelError,
    MemoryError,
    ErrorHandler,
    retry_with_backoff,
    fallback_to_simulation
)


class TestErrorSeverity:
    """Test error severity enumeration."""
    
    def test_severity_values(self):
        """Test that all severity levels have correct values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestErrorContext:
    """Test error context data structure."""
    
    def test_error_context_creation(self):
        """Test creating error context with default values."""
        context = ErrorContext()
        
        assert isinstance(context.timestamp, datetime)
        assert context.operation is None
        assert context.component is None
        assert context.user_data == {}
        assert context.system_state == {}
        assert context.stack_trace is None
    
    def test_error_context_with_data(self):
        """Test creating error context with custom data."""
        user_data = {'key': 'value'}
        system_state = {'memory_usage': 80}
        
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            user_data=user_data,
            system_state=system_state,
            stack_trace="test_trace"
        )
        
        assert context.operation == "test_operation"
        assert context.component == "test_component"
        assert context.user_data == user_data
        assert context.system_state == system_state
        assert context.stack_trace == "test_trace"
    
    def test_to_dict(self):
        """Test converting error context to dictionary."""
        context = ErrorContext(
            operation="test_op",
            component="test_comp",
            user_data={'test': 'data'},
            system_state={'state': 'info'}
        )
        
        result = context.to_dict()
        
        assert result['operation'] == "test_op"
        assert result['component'] == "test_comp"
        assert result['user_data'] == {'test': 'data'}
        assert result['system_state'] == {'state': 'info'}
        assert 'timestamp' in result


class TestGeminiSDRError:
    """Test base GeminiSDR error class."""
    
    def test_basic_error_creation(self):
        """Test creating basic error with message only."""
        error = GeminiSDRError("Test error message")
        
        assert str(error) == "[MEDIUM] Test error message"
        assert error.message == "Test error message"
        assert error.severity == ErrorSeverity.MEDIUM
        assert isinstance(error.context, ErrorContext)
        assert error.cause is None
    
    def test_error_with_severity(self):
        """Test creating error with custom severity."""
        error = GeminiSDRError("Critical error", severity=ErrorSeverity.CRITICAL)
        
        assert error.severity == ErrorSeverity.CRITICAL
        assert "[CRITICAL]" in str(error)
    
    def test_error_with_dict_context(self):
        """Test creating error with dictionary context."""
        context_data = {'operation': 'test', 'value': 42}
        error = GeminiSDRError("Test error", context=context_data)
        
        assert error.context.user_data == context_data
    
    def test_error_with_error_context(self):
        """Test creating error with ErrorContext object."""
        context = ErrorContext(operation="test_op", component="test_comp")
        error = GeminiSDRError("Test error", context=context)
        
        assert error.context.operation == "test_op"
        assert error.context.component == "test_comp"
    
    def test_error_with_cause(self):
        """Test creating error with underlying cause."""
        cause = ValueError("Original error")
        error = GeminiSDRError("Wrapper error", cause=cause)
        
        assert error.cause == cause
    
    def test_get_structured_info(self):
        """Test getting structured error information."""
        context = ErrorContext(operation="test_op")
        error = GeminiSDRError(
            "Test error", 
            severity=ErrorSeverity.HIGH,
            context=context
        )
        
        info = error.get_structured_info()
        
        assert info['error_type'] == 'GeminiSDRError'
        assert info['message'] == 'Test error'
        assert info['severity'] == 'high'
        assert 'context' in info
        assert info['cause'] is None
    
    def test_string_representation_with_context(self):
        """Test string representation includes context information."""
        context = ErrorContext(operation="test_op", component="test_comp")
        error = GeminiSDRError("Test error", context=context)
        
        error_str = str(error)
        assert "test_op" in error_str
        assert "test_comp" in error_str


class TestSpecificErrorTypes:
    """Test specific error type implementations."""
    
    def test_hardware_error(self):
        """Test HardwareError with device information."""
        error = HardwareError(
            "SDR connection failed",
            device_type="PlutoSDR",
            device_id="usb:1.2.3",
            severity=ErrorSeverity.HIGH
        )
        
        assert error.device_type == "PlutoSDR"
        assert error.device_id == "usb:1.2.3"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.user_data['device_type'] == "PlutoSDR"
        assert error.context.user_data['device_id'] == "usb:1.2.3"
    
    def test_configuration_error(self):
        """Test ConfigurationError with config information."""
        error = ConfigurationError(
            "Invalid batch size",
            config_key="ml.batch_size",
            config_file="config.yaml"
        )
        
        assert error.config_key == "ml.batch_size"
        assert error.config_file == "config.yaml"
        assert error.context.user_data['config_key'] == "ml.batch_size"
        assert error.context.user_data['config_file'] == "config.yaml"
    
    def test_model_error(self):
        """Test ModelError with model information."""
        error = ModelError(
            "Model loading failed",
            model_name="neural_amr",
            model_version="v1.2.3"
        )
        
        assert error.model_name == "neural_amr"
        assert error.model_version == "v1.2.3"
        assert error.context.user_data['model_name'] == "neural_amr"
        assert error.context.user_data['model_version'] == "v1.2.3"
    
    def test_memory_error(self):
        """Test MemoryError with memory information."""
        error = MemoryError(
            "Out of GPU memory",
            memory_type="gpu",
            requested_mb=2048.0,
            available_mb=1024.0,
            severity=ErrorSeverity.HIGH
        )
        
        assert error.memory_type == "gpu"
        assert error.requested_mb == 2048.0
        assert error.available_mb == 1024.0
        assert error.severity == ErrorSeverity.HIGH


class TestErrorHandler:
    """Test centralized error handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.handler = ErrorHandler(logger=self.logger)
    
    def test_error_handler_creation(self):
        """Test creating error handler."""
        handler = ErrorHandler()
        assert handler.recovery_strategies == {}
        assert handler.error_history == []
        assert handler.max_history_size == 1000
    
    def test_register_recovery_strategy(self):
        """Test registering recovery strategies."""
        def mock_strategy(error, context):
            return True
        
        self.handler.register_recovery_strategy(HardwareError, mock_strategy)
        
        assert HardwareError in self.handler.recovery_strategies
        assert mock_strategy in self.handler.recovery_strategies[HardwareError]
        self.logger.debug.assert_called_once()
    
    def test_handle_error_with_successful_recovery(self):
        """Test handling error with successful recovery."""
        def successful_strategy(error, context):
            return True
        
        self.handler.register_recovery_strategy(HardwareError, successful_strategy)
        
        error = HardwareError("Test hardware error")
        result = self.handler.handle_error(error)
        
        assert result is True
        assert len(self.handler.error_history) == 1
    
    def test_handle_error_with_failed_recovery(self):
        """Test handling error with failed recovery."""
        def failed_strategy(error, context):
            return False
        
        self.handler.register_recovery_strategy(HardwareError, failed_strategy)
        
        error = HardwareError("Test hardware error")
        result = self.handler.handle_error(error)
        
        assert result is False
        assert len(self.handler.error_history) == 1
    
    def test_handle_error_with_strategy_exception(self):
        """Test handling error when recovery strategy raises exception."""
        def failing_strategy(error, context):
            raise RuntimeError("Strategy failed")
        
        self.handler.register_recovery_strategy(HardwareError, failing_strategy)
        
        error = HardwareError("Test hardware error")
        result = self.handler.handle_error(error)
        
        assert result is False
        self.logger.error.assert_called()
    
    def test_handle_error_inheritance(self):
        """Test that recovery strategies work with error inheritance."""
        def base_strategy(error, context):
            return True
        
        # Register strategy for base class
        self.handler.register_recovery_strategy(GeminiSDRError, base_strategy)
        
        # Test with derived class
        error = HardwareError("Test hardware error")
        result = self.handler.handle_error(error)
        
        assert result is True
    
    def test_error_context_manager(self):
        """Test error context manager."""
        with self.handler.error_context("test_operation", component="test_comp") as ctx:
            assert ctx.operation == "test_operation"
            assert ctx.component == "test_comp"
    
    def test_error_context_manager_with_exception(self):
        """Test error context manager handles exceptions."""
        def recovery_strategy(error, context):
            return True
        
        self.handler.register_recovery_strategy(ValueError, recovery_strategy)
        
        with self.handler.error_context("test_operation"):
            raise ValueError("Test error")
        
        # Should not raise because recovery was successful
        assert len(self.handler.error_history) == 1
    
    def test_error_context_manager_reraises_on_failed_recovery(self):
        """Test error context manager re-raises when recovery fails."""
        with pytest.raises(ValueError):
            with self.handler.error_context("test_operation"):
                raise ValueError("Test error")
    
    def test_get_error_statistics(self):
        """Test getting error statistics."""
        # Add some errors to history
        error1 = HardwareError("Error 1")
        error2 = ConfigurationError("Error 2")
        error3 = HardwareError("Error 3")
        
        self.handler.handle_error(error1)
        self.handler.handle_error(error2)
        self.handler.handle_error(error3)
        
        stats = self.handler.get_error_statistics()
        
        assert stats['total_errors'] == 3
        assert stats['error_types']['HardwareError'] == 2
        assert stats['error_types']['ConfigurationError'] == 1
        assert len(stats['recent_errors']) == 3
    
    def test_error_history_size_limit(self):
        """Test that error history respects size limit."""
        # Set small limit for testing
        self.handler.max_history_size = 2
        
        # Add more errors than the limit
        for i in range(5):
            error = GeminiSDRError(f"Error {i}")
            self.handler.handle_error(error)
        
        assert len(self.handler.error_history) == 2
        # Should keep the most recent errors
        assert "Error 3" in self.handler.error_history[-2]['message']
        assert "Error 4" in self.handler.error_history[-1]['message']


class TestRetryDecorator:
    """Test retry with backoff decorator."""
    
    def test_successful_function(self):
        """Test decorator with function that succeeds immediately."""
        @retry_with_backoff(max_retries=3)
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_function_succeeds_after_retries(self):
        """Test decorator with function that succeeds after some failures."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def eventually_successful_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = eventually_successful_function()
        assert result == "success"
        assert call_count == 3
    
    def test_function_fails_after_max_retries(self):
        """Test decorator with function that always fails."""
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_failing_function():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_failing_function()
    
    def test_specific_exceptions_only(self):
        """Test decorator only retries on specific exceptions."""
        @retry_with_backoff(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        def function_with_runtime_error():
            raise RuntimeError("Should not retry")
        
        with pytest.raises(RuntimeError, match="Should not retry"):
            function_with_runtime_error()
    
    @patch('time.sleep')
    def test_backoff_timing(self, mock_sleep):
        """Test that backoff timing works correctly."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=1.0, backoff_factor=2.0)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ValueError("Fail")
            return "success"
        
        result = failing_function()
        
        # Should have slept with increasing delays: 1.0, 2.0, 4.0
        expected_calls = [((1.0,),), ((2.0,),), ((4.0,),)]
        assert mock_sleep.call_args_list == expected_calls
        assert result == "success"


class TestFallbackToSimulation:
    """Test fallback to simulation context manager."""
    
    def test_no_error_case(self):
        """Test context manager when no error occurs."""
        with fallback_to_simulation():
            result = "success"
        
        assert result == "success"
    
    def test_hardware_error_fallback(self):
        """Test fallback when hardware error occurs."""
        logger = Mock(spec=logging.Logger)
        
        with pytest.raises(ConfigurationError) as exc_info:
            with fallback_to_simulation(logger=logger):
                raise HardwareError("SDR not found", device_type="PlutoSDR")
        
        # Should log the fallback
        logger.warning.assert_called_once()
        
        # Should raise ConfigurationError with fallback context
        config_error = exc_info.value
        assert "simulation mode" in config_error.message
        assert config_error.context.user_data['fallback_mode'] == 'simulation'
        assert isinstance(config_error.cause, HardwareError)
    
    def test_non_hardware_error_passthrough(self):
        """Test that non-hardware errors pass through unchanged."""
        with pytest.raises(ValueError):
            with fallback_to_simulation():
                raise ValueError("Not a hardware error")


class TestIntegration:
    """Integration tests for the complete error handling system."""
    
    def test_complete_error_handling_workflow(self):
        """Test complete workflow from error to recovery."""
        logger = Mock(spec=logging.Logger)
        handler = ErrorHandler(logger=logger)
        
        # Register recovery strategy
        def hardware_recovery_strategy(error, context):
            # Simulate successful hardware recovery
            context.user_data['recovery_attempted'] = True
            return True
        
        handler.register_recovery_strategy(HardwareError, hardware_recovery_strategy)
        
        # Test error handling workflow
        with handler.error_context("sdr_initialization", component="hardware") as ctx:
            # Simulate hardware error
            error = HardwareError(
                "Failed to connect to SDR",
                device_type="PlutoSDR",
                device_id="usb:1.2.3"
            )
            
            # Handle the error
            recovery_success = handler.handle_error(error, ctx)
            
            assert recovery_success is True
            assert ctx.user_data['recovery_attempted'] is True
            assert len(handler.error_history) == 1
            
            # Verify error was logged
            logger.log.assert_called()
            logger.info.assert_called()
    
    def test_error_severity_logging_levels(self):
        """Test that different error severities map to correct log levels."""
        logger = Mock(spec=logging.Logger)
        handler = ErrorHandler(logger=logger)
        
        # Test different severity levels
        severities_and_levels = [
            (ErrorSeverity.LOW, logging.INFO),
            (ErrorSeverity.MEDIUM, logging.WARNING),
            (ErrorSeverity.HIGH, logging.ERROR),
            (ErrorSeverity.CRITICAL, logging.CRITICAL)
        ]
        
        for severity, expected_level in severities_and_levels:
            error = GeminiSDRError("Test error", severity=severity)
            handler.handle_error(error)
            
            # Check that log was called with correct level
            logger.log.assert_called_with(expected_level, ANY)
            logger.reset_mock()


class TestRecoveryStrategies:
    """Test recovery strategies for error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from geminisdr.core.recovery_strategies import RecoveryStrategies
        self.logger = Mock(spec=logging.Logger)
        self.strategies = RecoveryStrategies(logger=self.logger)
    
    def test_hardware_to_simulation_fallback(self):
        """Test hardware to simulation fallback strategy."""
        error = HardwareError("SDR not found", device_type="PlutoSDR")
        context = ErrorContext(operation="sdr_init")
        
        result = self.strategies.hardware_to_simulation_fallback(error, context)
        
        assert result is True
        assert self.strategies.is_simulation_mode is True
        assert context.user_data['fallback_mode'] == 'simulation'
        assert context.user_data['original_device'] == 'PlutoSDR'
        self.logger.info.assert_called()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    def test_gpu_to_cpu_fallback(self, mock_empty_cache, mock_cuda_available):
        """Test GPU to CPU fallback strategy."""
        error = MemoryError("GPU out of memory", memory_type="gpu")
        context = ErrorContext(operation="model_training")
        
        result = self.strategies.gpu_to_cpu_fallback(error, context)
        
        assert result is True
        assert self.strategies.is_cpu_fallback_active is True
        assert context.user_data['fallback_mode'] == 'cpu'
        assert context.user_data['original_device'] == 'gpu'
        mock_empty_cache.assert_called_once()
    
    def test_gpu_to_cpu_fallback_wrong_memory_type(self):
        """Test GPU to CPU fallback with non-GPU memory error."""
        error = MemoryError("RAM exhausted", memory_type="ram")
        context = ErrorContext()
        
        result = self.strategies.gpu_to_cpu_fallback(error, context)
        
        assert result is False
        assert self.strategies.is_cpu_fallback_active is False
    
    def test_reduce_batch_size_recovery(self):
        """Test batch size reduction recovery strategy."""
        error = MemoryError("Insufficient memory")
        context = ErrorContext(user_data={'batch_size': 64})
        
        result = self.strategies.reduce_batch_size_recovery(error, context)
        
        assert result is True
        assert context.user_data['batch_size'] == 32
        assert context.user_data['original_batch_size'] == 64
        assert context.user_data['recovery_strategy'] == 'reduce_batch_size'
    
    def test_reduce_batch_size_minimum_limit(self):
        """Test batch size reduction when already at minimum."""
        error = MemoryError("Insufficient memory")
        context = ErrorContext(user_data={'batch_size': 1})
        
        result = self.strategies.reduce_batch_size_recovery(error, context)
        
        assert result is False
    
    def test_configuration_default_fallback(self):
        """Test configuration default fallback strategy."""
        error = ConfigurationError("Invalid batch size", config_key="ml.batch_size")
        context = ErrorContext()
        
        result = self.strategies.configuration_default_fallback(error, context)
        
        assert result is True
        assert context.user_data['default_value'] == 32
        assert context.user_data['recovery_strategy'] == 'configuration_default'
        self.logger.warning.assert_called()
    
    def test_configuration_default_fallback_unknown_key(self):
        """Test configuration default fallback with unknown key."""
        error = ConfigurationError("Invalid config", config_key="unknown.key")
        context = ErrorContext()
        
        result = self.strategies.configuration_default_fallback(error, context)
        
        assert result is False
    
    def test_model_version_fallback(self):
        """Test model version fallback strategy."""
        error = ModelError("Model load failed", model_name="neural_amr", model_version="v2.0.0")
        context = ErrorContext()
        
        result = self.strategies.model_version_fallback(error, context)
        
        assert result is True
        assert context.user_data['fallback_version'] == "v1.0.0"
        assert context.user_data['failed_version'] == "v2.0.0"
        assert context.user_data['recovery_strategy'] == 'model_version_fallback'
    
    @patch('gc.collect', return_value=42)
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    def test_memory_cleanup_recovery(self, mock_sync, mock_empty_cache, mock_cuda_available, mock_gc):
        """Test memory cleanup recovery strategy."""
        error = MemoryError("Memory exhausted")
        context = ErrorContext()
        
        result = self.strategies.memory_cleanup_recovery(error, context)
        
        assert result is True
        assert context.user_data['recovery_strategy'] == 'memory_cleanup'
        assert context.user_data['objects_collected'] == 42
        mock_gc.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_sync.assert_called_once()
    
    @patch('time.sleep')
    def test_retry_with_delay_recovery(self, mock_sleep):
        """Test retry with delay recovery strategy."""
        error = Exception("Transient error")
        context = ErrorContext(user_data={'retry_count': 1, 'max_retries': 3})
        
        result = self.strategies.retry_with_delay_recovery(error, context)
        
        assert result is True
        assert context.user_data['retry_count'] == 2
        assert context.user_data['recovery_strategy'] == 'retry_with_delay'
        mock_sleep.assert_called_once_with(2)  # 2^1 = 2 seconds delay
    
    @patch('time.sleep')
    def test_retry_with_delay_max_retries_exceeded(self, mock_sleep):
        """Test retry strategy when max retries exceeded."""
        error = Exception("Transient error")
        context = ErrorContext(user_data={'retry_count': 3, 'max_retries': 3})
        
        result = self.strategies.retry_with_delay_recovery(error, context)
        
        assert result is False
        mock_sleep.assert_not_called()
    
    def test_get_all_strategies(self):
        """Test getting all available strategies."""
        strategies = self.strategies.get_all_strategies()
        
        assert HardwareError in strategies
        assert MemoryError in strategies
        assert ConfigurationError in strategies
        assert ModelError in strategies
        
        # Check that each error type has strategies
        assert len(strategies[HardwareError]) > 0
        assert len(strategies[MemoryError]) > 0
        assert len(strategies[ConfigurationError]) > 0
        assert len(strategies[ModelError]) > 0
    
    def test_reset_fallback_states(self):
        """Test resetting fallback states."""
        # Set some fallback states
        self.strategies._simulation_mode = True
        self.strategies._cpu_fallback_active = True
        
        self.strategies.reset_fallback_states()
        
        assert self.strategies.is_simulation_mode is False
        assert self.strategies.is_cpu_fallback_active is False
        self.logger.info.assert_called_with("Reset all fallback states to normal operation")


class TestSetupDefaultRecoveryStrategies:
    """Test setup of default recovery strategies."""
    
    def test_setup_default_recovery_strategies(self):
        """Test setting up default recovery strategies."""
        from geminisdr.core.recovery_strategies import setup_default_recovery_strategies
        
        logger = Mock(spec=logging.Logger)
        handler = ErrorHandler(logger=logger)
        
        strategies = setup_default_recovery_strategies(handler, logger)
        
        # Verify strategies were registered
        assert len(handler.recovery_strategies) > 0
        assert HardwareError in handler.recovery_strategies
        assert MemoryError in handler.recovery_strategies
        assert ConfigurationError in handler.recovery_strategies
        assert ModelError in handler.recovery_strategies
        
        # Verify logger was called
        logger.info.assert_called_with("Registered default recovery strategies")
        
        # Verify returned strategies object
        assert strategies is not None
        assert hasattr(strategies, 'hardware_to_simulation_fallback')


class TestIntegratedRecoverySystem:
    """Integration tests for the complete recovery system."""
    
    def test_complete_recovery_workflow(self):
        """Test complete recovery workflow with real strategies."""
        from geminisdr.core.recovery_strategies import setup_default_recovery_strategies
        
        logger = Mock(spec=logging.Logger)
        handler = ErrorHandler(logger=logger)
        strategies = setup_default_recovery_strategies(handler, logger)
        
        # Test hardware error recovery
        with handler.error_context("sdr_initialization", component="hardware") as ctx:
            error = HardwareError("SDR connection failed", device_type="PlutoSDR")
            
            recovery_success = handler.handle_error(error, ctx)
            
            assert recovery_success is True
            assert ctx.user_data['fallback_mode'] == 'simulation'
            assert strategies.is_simulation_mode is True
    
    def test_memory_error_recovery_chain(self):
        """Test memory error recovery with multiple strategies."""
        from geminisdr.core.recovery_strategies import setup_default_recovery_strategies
        
        logger = Mock(spec=logging.Logger)
        handler = ErrorHandler(logger=logger)
        strategies = setup_default_recovery_strategies(handler, logger)
        
        # Test GPU memory error
        with handler.error_context("model_training", batch_size=64) as ctx:
            error = MemoryError("GPU out of memory", memory_type="gpu")
            
            recovery_success = handler.handle_error(error, ctx)
            
            assert recovery_success is True
            # Should have attempted memory cleanup first
            assert 'recovery_strategy' in ctx.user_data
    
    def test_configuration_error_recovery(self):
        """Test configuration error recovery."""
        from geminisdr.core.recovery_strategies import setup_default_recovery_strategies
        
        logger = Mock(spec=logging.Logger)
        handler = ErrorHandler(logger=logger)
        strategies = setup_default_recovery_strategies(handler, logger)
        
        # Test configuration error
        with handler.error_context("config_loading") as ctx:
            error = ConfigurationError("Invalid batch size", config_key="ml.batch_size")
            
            recovery_success = handler.handle_error(error, ctx)
            
            assert recovery_success is True
            assert ctx.user_data['default_value'] == 32
            assert ctx.user_data['recovery_strategy'] == 'configuration_default'