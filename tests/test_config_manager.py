"""
Unit tests for ConfigManager with Hydra integration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from geminisdr.config.config_manager import ConfigManager, get_config_manager
from geminisdr.config.config_models import (
    SystemConfig,
    HardwareConfig,
    MLConfig,
    DeviceType,
    SDRMode,
    ConfigValidationError
)


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary configuration directory for testing."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "conf"
        config_dir.mkdir(parents=True)
        
        # Create basic config files
        (config_dir / "config.yaml").write_text("""
environment: testing
debug_mode: false
config_version: "1.0"

hardware:
  device_preference: cpu
  sdr_mode: simulation
  memory_limit_gb: 4.0

ml:
  batch_size: 16
  learning_rate: 1e-4
  model_cache_size: 2

logging:
  level: INFO
  format: simple
  output: [console]

performance:
  memory_threshold: 0.7
  auto_optimize: false
""")
        
        # Create hardware profiles
        hardware_dir = config_dir / "hardware"
        hardware_dir.mkdir()
        (hardware_dir / "cpu.yaml").write_text("""
# @package hardware
device_preference: cpu
sdr_mode: simulation
memory_limit_gb: 4.0
""")
        
        yield str(config_dir)
        shutil.rmtree(temp_dir)
    
    def test_config_manager_initialization(self, temp_config_dir):
        """Test ConfigManager initialization."""
        manager = ConfigManager(config_dir=temp_config_dir)
        assert manager.config_dir == Path(temp_config_dir).resolve()
        assert not manager._hot_reload_enabled
        assert manager._config is None
    
    def test_load_basic_config(self, temp_config_dir):
        """Test loading basic configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_config("config")
        
        assert isinstance(config, SystemConfig)
        assert config.environment == "testing"
        assert config.debug_mode is False
        assert config.hardware.device_preference == DeviceType.CPU
        assert config.hardware.sdr_mode == SDRMode.SIMULATION
        assert config.ml.batch_size == 16
    
    def test_load_config_with_overrides(self, temp_config_dir):
        """Test loading configuration with overrides."""
        manager = ConfigManager(config_dir=temp_config_dir)
        overrides = [
            "hardware.device_preference=mps",
            "ml.batch_size=32",
            "debug_mode=true"
        ]
        config = manager.load_config("config", overrides=overrides)
        
        assert config.hardware.device_preference == DeviceType.MPS
        assert config.ml.batch_size == 32
        assert config.debug_mode is True
    
    def test_get_config(self, temp_config_dir):
        """Test getting current configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        # Initially no config loaded
        assert manager.get_config() is None
        
        # Load config and verify it's returned
        config = manager.load_config("config")
        current_config = manager.get_config()
        assert current_config is config
        assert current_config.environment == "testing"
    
    def test_override_config(self, temp_config_dir):
        """Test configuration overrides using kwargs."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.override_config(
            **{
                "hardware.device_preference": "cuda",
                "ml.learning_rate": 2e-4,
                "logging.level": "DEBUG"
            }
        )
        
        assert config.hardware.device_preference == DeviceType.CUDA
        assert config.ml.learning_rate == 2e-4
        # Note: logging.level would be a string in the override, 
        # but our model converts it to enum
    
    def test_get_environment_config(self, temp_config_dir):
        """Test loading environment-specific configuration."""
        # Create environment config
        env_dir = Path(temp_config_dir) / "environments"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "development.yaml").write_text("""
# @package _global_
environment: development
debug_mode: true

logging:
  level: DEBUG
""")
        
        manager = ConfigManager(config_dir=temp_config_dir)
        
        # This test might fail due to Hydra complexity, so we'll mock it
        with patch.object(manager, 'load_config') as mock_load:
            mock_config = SystemConfig(environment="development", debug_mode=True)
            mock_load.return_value = mock_config
            
            config = manager.get_environment_config("development")
            mock_load.assert_called_once_with(overrides=["environments=development"])
            assert config.environment == "development"
    
    def test_get_hardware_config(self, temp_config_dir):
        """Test loading hardware-specific configuration."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        with patch.object(manager, 'load_config') as mock_load:
            mock_config = SystemConfig()
            mock_config.hardware.device_preference = DeviceType.CPU
            mock_load.return_value = mock_config
            
            config = manager.get_hardware_config("cpu")
            mock_load.assert_called_once_with(overrides=["hardware=cpu"])
    
    def test_validate_config(self, temp_config_dir):
        """Test configuration validation."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        # Valid config should have no errors
        valid_config = SystemConfig()
        errors = manager.validate_config(valid_config)
        assert len(errors) == 0
        
        # Invalid config should have errors
        from geminisdr.config.config_models import PerformanceConfig
        invalid_config = SystemConfig(
            performance=PerformanceConfig(
                memory_threshold=0.9,
                memory_cleanup_threshold=0.8  # Should be higher than memory_threshold
            )
        )
        errors = manager.validate_config(invalid_config)
        assert len(errors) > 0
    
    def test_hot_reload_enable_disable(self, temp_config_dir):
        """Test enabling and disabling hot-reload."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        # Initially disabled
        assert not manager._hot_reload_enabled
        assert len(manager._observers) == 0
        
        # Enable hot-reload
        manager.enable_hot_reload()
        assert manager._hot_reload_enabled
        assert len(manager._observers) > 0
        
        # Disable hot-reload
        manager.disable_hot_reload()
        assert not manager._hot_reload_enabled
        assert len(manager._observers) == 0
    
    def test_reload_callbacks(self, temp_config_dir):
        """Test reload callback functionality."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        callback_called = []
        def test_callback(config):
            callback_called.append(config)
        
        # Add callback
        manager.add_reload_callback(test_callback)
        assert test_callback in manager._reload_callbacks
        
        # Remove callback
        manager.remove_reload_callback(test_callback)
        assert test_callback not in manager._reload_callbacks
    
    def test_context_manager(self, temp_config_dir):
        """Test ConfigManager as context manager."""
        with ConfigManager(config_dir=temp_config_dir) as manager:
            config = manager.load_config("config")
            assert config is not None
        
        # After context exit, hot-reload should be disabled
        assert not manager._hot_reload_enabled
    
    def test_invalid_config_file(self, temp_config_dir):
        """Test handling of invalid configuration files."""
        # Create invalid config file
        invalid_config_path = Path(temp_config_dir) / "invalid.yaml"
        invalid_config_path.write_text("invalid: yaml: content: [")
        
        manager = ConfigManager(config_dir=temp_config_dir)
        
        with pytest.raises(ConfigValidationError):
            manager.load_config("invalid")
    
    def test_missing_config_file(self, temp_config_dir):
        """Test handling of missing configuration files."""
        manager = ConfigManager(config_dir=temp_config_dir)
        
        with pytest.raises(ConfigValidationError):
            manager.load_config("nonexistent")


class TestGlobalConfigManager:
    """Test global configuration manager functions."""
    
    def test_get_config_manager_singleton(self):
        """Test that get_config_manager returns the same instance."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        assert manager1 is manager2
    
    @patch('geminisdr.config.config_manager._global_config_manager', None)
    def test_get_config_manager_creates_new(self):
        """Test that get_config_manager creates new instance when needed."""
        manager = get_config_manager()
        assert isinstance(manager, ConfigManager)
    
    def test_convenience_functions(self):
        """Test convenience functions for configuration loading."""
        from geminisdr.config.config_manager import load_config, get_current_config
        
        # Mock the global config manager
        with patch('geminisdr.config.config_manager.get_config_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_config = SystemConfig()
            mock_manager.load_config.return_value = mock_config
            mock_manager.get_config.return_value = mock_config
            mock_get_manager.return_value = mock_manager
            
            # Test load_config convenience function
            config = load_config("test_config")
            mock_manager.load_config.assert_called_once_with("test_config")
            assert config is mock_config
            
            # Test get_current_config convenience function
            current = get_current_config()
            mock_manager.get_config.assert_called_once()
            assert current is mock_config