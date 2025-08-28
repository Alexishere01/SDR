"""
Unit tests for configuration data models and validation.
"""

import pytest
from geminisdr.config.config_models import (
    SystemConfig,
    HardwareConfig,
    MLConfig,
    LoggingConfig,
    PerformanceConfig,
    DeviceType,
    SDRMode,
    LogLevel,
    LogFormat,
    ConfigValidationError,
    validate_config,
    create_default_config
)


class TestHardwareConfig:
    """Test HardwareConfig validation and functionality."""
    
    def test_default_hardware_config(self):
        """Test default hardware configuration is valid."""
        config = HardwareConfig()
        assert config.device_preference == DeviceType.AUTO
        assert config.sdr_mode == SDRMode.AUTO
        assert config.memory_limit_gb is None
        assert config.gpu_memory_fraction == 0.8
        
    def test_hardware_config_with_string_enums(self):
        """Test hardware config accepts string values for enums."""
        config = HardwareConfig(
            device_preference="mps",
            sdr_mode="simulation"
        )
        assert config.device_preference == DeviceType.MPS
        assert config.sdr_mode == SDRMode.SIMULATION
        
    def test_invalid_memory_limit(self):
        """Test validation of memory limit."""
        with pytest.raises(ValueError, match="memory_limit_gb must be positive"):
            HardwareConfig(memory_limit_gb=-1.0)
            
    def test_invalid_max_threads(self):
        """Test validation of max threads."""
        with pytest.raises(ValueError, match="max_threads must be positive"):
            HardwareConfig(max_threads=0)
            
    def test_invalid_gpu_memory_fraction(self):
        """Test validation of GPU memory fraction."""
        with pytest.raises(ValueError, match="gpu_memory_fraction must be between"):
            HardwareConfig(gpu_memory_fraction=1.5)
            
        with pytest.raises(ValueError, match="gpu_memory_fraction must be between"):
            HardwareConfig(gpu_memory_fraction=0.05)


class TestMLConfig:
    """Test MLConfig validation and functionality."""
    
    def test_default_ml_config(self):
        """Test default ML configuration is valid."""
        config = MLConfig()
        assert config.batch_size is None
        assert config.learning_rate == 1e-4
        assert config.model_cache_size == 3
        assert config.checkpoint_frequency == 100
        
    def test_invalid_batch_size(self):
        """Test validation of batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            MLConfig(batch_size=0)
            
    def test_invalid_learning_rate(self):
        """Test validation of learning rate."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            MLConfig(learning_rate=0)
            
    def test_invalid_model_cache_size(self):
        """Test validation of model cache size."""
        with pytest.raises(ValueError, match="model_cache_size must be positive"):
            MLConfig(model_cache_size=-1)
            
    def test_invalid_checkpoint_frequency(self):
        """Test validation of checkpoint frequency."""
        with pytest.raises(ValueError, match="checkpoint_frequency must be positive"):
            MLConfig(checkpoint_frequency=0)


class TestLoggingConfig:
    """Test LoggingConfig validation and functionality."""
    
    def test_default_logging_config(self):
        """Test default logging configuration is valid."""
        config = LoggingConfig()
        assert config.level == LogLevel.INFO
        assert config.format == LogFormat.STRUCTURED
        assert config.output == ["console", "file"]
        
    def test_logging_config_with_string_enums(self):
        """Test logging config accepts string values for enums."""
        config = LoggingConfig(
            level="DEBUG",
            format="simple"
        )
        assert config.level == LogLevel.DEBUG
        assert config.format == LogFormat.SIMPLE
        
    def test_invalid_output(self):
        """Test validation of log output."""
        with pytest.raises(ValueError, match="Invalid log output"):
            LoggingConfig(output=["invalid_output"])
            
    def test_invalid_rotation(self):
        """Test validation of log rotation."""
        with pytest.raises(ValueError, match="Invalid rotation"):
            LoggingConfig(rotation="invalid_rotation")
            
    def test_invalid_max_file_size(self):
        """Test validation of max file size."""
        with pytest.raises(ValueError, match="max_file_size_mb must be positive"):
            LoggingConfig(max_file_size_mb=0)
            
    def test_invalid_backup_count(self):
        """Test validation of backup count."""
        with pytest.raises(ValueError, match="backup_count must be non-negative"):
            LoggingConfig(backup_count=-1)


class TestPerformanceConfig:
    """Test PerformanceConfig validation and functionality."""
    
    def test_default_performance_config(self):
        """Test default performance configuration is valid."""
        config = PerformanceConfig()
        assert config.memory_threshold == 0.8
        assert config.auto_optimize is True
        assert config.profiling_enabled is False
        
    def test_invalid_memory_threshold(self):
        """Test validation of memory threshold."""
        with pytest.raises(ValueError, match="memory_threshold must be between"):
            PerformanceConfig(memory_threshold=1.5)
            
    def test_invalid_memory_cleanup_threshold(self):
        """Test validation of memory cleanup threshold."""
        with pytest.raises(ValueError, match="memory_cleanup_threshold must be between"):
            PerformanceConfig(memory_cleanup_threshold=0.05)
            
    def test_invalid_gc_frequency(self):
        """Test validation of GC frequency."""
        with pytest.raises(ValueError, match="gc_frequency must be positive"):
            PerformanceConfig(gc_frequency=0)


class TestSystemConfig:
    """Test SystemConfig validation and functionality."""
    
    def test_default_system_config(self):
        """Test default system configuration is valid."""
        config = SystemConfig()
        assert config.environment == "development"
        assert config.debug_mode is False
        assert config.config_version == "1.0"
        assert isinstance(config.hardware, HardwareConfig)
        assert isinstance(config.ml, MLConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.performance, PerformanceConfig)
        
    def test_invalid_environment(self):
        """Test validation of environment."""
        with pytest.raises(ValueError, match="Invalid environment"):
            SystemConfig(environment="invalid_env")


class TestConfigValidation:
    """Test configuration validation functions."""
    
    def test_validate_valid_config(self):
        """Test validation of a valid configuration."""
        config = SystemConfig()
        errors = validate_config(config)
        assert len(errors) == 0
        
    def test_validate_config_with_threshold_error(self):
        """Test validation catches threshold configuration errors."""
        config = SystemConfig(
            performance=PerformanceConfig(
                memory_threshold=0.9,
                memory_cleanup_threshold=0.8
            )
        )
        errors = validate_config(config)
        assert len(errors) > 0
        assert any("memory_threshold should be less than memory_cleanup_threshold" in error for error in errors)
        
    def test_create_default_config_development(self):
        """Test creating default development configuration."""
        config = create_default_config("development")
        assert config.environment == "development"
        assert config.debug_mode is True
        assert config.logging.level == LogLevel.DEBUG
        assert config.performance.profiling_enabled is True
        
    def test_create_default_config_testing(self):
        """Test creating default testing configuration."""
        config = create_default_config("testing")
        assert config.environment == "testing"
        assert config.debug_mode is False
        assert config.hardware.device_preference == DeviceType.CPU
        assert config.ml.batch_size == 16
        
    def test_create_default_config_production(self):
        """Test creating default production configuration."""
        config = create_default_config("production")
        assert config.environment == "production"
        assert config.debug_mode is False
        assert config.hardware.enable_mixed_precision is True
        assert config.logging.output == ["file", "syslog"]
        
    def test_create_default_config_invalid_environment(self):
        """Test creating default config with invalid environment."""
        with pytest.raises(ValueError, match="Unknown environment"):
            create_default_config("invalid_env")


class TestConfigValidationError:
    """Test ConfigValidationError exception."""
    
    def test_config_validation_error_basic(self):
        """Test basic ConfigValidationError."""
        error = ConfigValidationError("Test error message")
        assert str(error) == "Configuration validation error: Test error message"
        
    def test_config_validation_error_with_field(self):
        """Test ConfigValidationError with field information."""
        error = ConfigValidationError("Invalid value", field="test_field", value=42)
        assert str(error) == "Configuration validation error in field 'test_field': Invalid value"
        assert error.field == "test_field"
        assert error.value == 42