"""
Configuration Manager with Hydra integration for GeminiSDR system.

This module provides centralized configuration management with environment-specific
overrides, hot-reload capabilities, and Hydra integration.
"""

import os
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import asdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from hydra import compose, initialize_config_dir, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .config_models import (
    SystemConfig,
    HardwareConfig,
    MLConfig,
    LoggingConfig,
    PerformanceConfig,
    ConfigValidationError,
    validate_config
)

logger = logging.getLogger(__name__)


class ConfigFileWatcher(FileSystemEventHandler):
    """File system event handler for configuration file changes."""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.last_reload_time = 0
        self.reload_debounce_seconds = 1.0  # Prevent rapid reloads
        
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        # Only react to YAML files
        if not event.src_path.endswith(('.yaml', '.yml')):
            return
            
        current_time = time.time()
        if current_time - self.last_reload_time < self.reload_debounce_seconds:
            return
            
        logger.info(f"Configuration file changed: {event.src_path}")
        try:
            self.config_manager._reload_config()
            self.last_reload_time = current_time
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")


class ConfigManager:
    """
    Centralized configuration management with Hydra integration.
    
    Features:
    - Environment-specific configuration overrides
    - Hot-reload capability for supported configuration changes
    - Configuration validation with clear error messages
    - Thread-safe configuration access
    """
    
    def __init__(self, config_dir: str = "conf", version_key: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            version_key: Optional version key for configuration compatibility
        """
        self.config_dir = Path(config_dir).resolve()
        self.version_key = version_key
        self._config: Optional[SystemConfig] = None
        self._raw_config: Optional[DictConfig] = None
        self._config_lock = threading.RLock()
        self._observers: List[Observer] = []
        self._reload_callbacks: List[Callable[[SystemConfig], None]] = []
        self._hot_reload_enabled = False
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Hydra if not already initialized
        self._initialize_hydra()
        
    def _initialize_hydra(self):
        """Initialize Hydra configuration system."""
        try:
            # Clear any existing Hydra instance
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
                
            # Initialize with our config directory using absolute path method
            initialize_config_dir(config_dir=str(self.config_dir), version_base=None)
            logger.debug(f"Hydra initialized with config directory: {self.config_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hydra: {e}")
            raise ConfigValidationError(f"Hydra initialization failed: {e}")
    
    def load_config(self, config_name: str = "config", overrides: Optional[List[str]] = None) -> SystemConfig:
        """
        Load configuration with environment-specific overrides.
        
        Args:
            config_name: Name of the main configuration file (without .yaml extension)
            overrides: List of configuration overrides in Hydra format
            
        Returns:
            SystemConfig instance with loaded and validated configuration
            
        Raises:
            ConfigValidationError: If configuration is invalid or cannot be loaded
        """
        with self._config_lock:
            try:
                # Compose configuration using Hydra
                overrides = overrides or []
                cfg = compose(config_name=config_name, overrides=overrides)
                
                # Store raw config for hot-reload
                self._raw_config = cfg
                
                # Convert to our SystemConfig dataclass
                self._config = self._hydra_to_system_config(cfg)
                
                # Validate the configuration
                validation_errors = validate_config(self._config)
                if validation_errors:
                    error_msg = "Configuration validation failed:\n" + "\n".join(validation_errors)
                    raise ConfigValidationError(error_msg)
                
                logger.info(f"Configuration loaded successfully: {config_name}")
                logger.debug(f"Configuration environment: {self._config.environment}")
                
                return self._config
                
            except Exception as e:
                logger.error(f"Failed to load configuration '{config_name}': {e}")
                if isinstance(e, ConfigValidationError):
                    raise
                raise ConfigValidationError(f"Configuration loading failed: {e}")
    
    def _hydra_to_system_config(self, cfg: DictConfig) -> SystemConfig:
        """
        Convert Hydra DictConfig to SystemConfig dataclass.
        
        Args:
            cfg: Hydra configuration object
            
        Returns:
            SystemConfig instance
        """
        # Convert to regular dict for easier processing
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Extract subsystem configurations
        hardware_config = HardwareConfig(**config_dict.get('hardware', {}))
        ml_config = MLConfig(**config_dict.get('ml', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        performance_config = PerformanceConfig(**config_dict.get('performance', {}))
        
        # Create system configuration
        system_config = SystemConfig(
            hardware=hardware_config,
            ml=ml_config,
            logging=logging_config,
            performance=performance_config,
            environment=config_dict.get('environment', 'development'),
            debug_mode=config_dict.get('debug_mode', False),
            config_version=config_dict.get('config_version', '1.0')
        )
        
        return system_config
    
    def get_config(self) -> Optional[SystemConfig]:
        """
        Get the currently loaded configuration.
        
        Returns:
            Current SystemConfig instance or None if not loaded
        """
        with self._config_lock:
            return self._config
    
    def validate_config(self, config: SystemConfig) -> List[str]:
        """
        Validate a configuration and return any errors.
        
        Args:
            config: SystemConfig to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        return validate_config(config)
    
    def enable_hot_reload(self) -> None:
        """
        Enable hot-reload capability for configuration files.
        
        This will watch the configuration directory for changes and automatically
        reload the configuration when files are modified.
        """
        if self._hot_reload_enabled:
            logger.warning("Hot-reload is already enabled")
            return
            
        try:
            # Create file system observer
            observer = Observer()
            event_handler = ConfigFileWatcher(self)
            
            # Watch the config directory recursively
            observer.schedule(event_handler, str(self.config_dir), recursive=True)
            observer.start()
            
            self._observers.append(observer)
            self._hot_reload_enabled = True
            
            logger.info(f"Hot-reload enabled for configuration directory: {self.config_dir}")
            
        except Exception as e:
            logger.error(f"Failed to enable hot-reload: {e}")
            raise ConfigValidationError(f"Hot-reload setup failed: {e}")
    
    def disable_hot_reload(self) -> None:
        """Disable hot-reload capability."""
        if not self._hot_reload_enabled:
            return
            
        for observer in self._observers:
            observer.stop()
            observer.join()
            
        self._observers.clear()
        self._hot_reload_enabled = False
        logger.info("Hot-reload disabled")
    
    def _reload_config(self) -> None:
        """Internal method to reload configuration from files."""
        if not self._raw_config:
            logger.warning("No configuration loaded, cannot reload")
            return
            
        try:
            # Re-initialize Hydra to pick up file changes
            self._initialize_hydra()
            
            # Reload with the same configuration name
            # This is a simplified reload - in practice, we'd need to track
            # the original config name and overrides
            cfg = compose(config_name="config")
            
            # Convert and validate
            new_config = self._hydra_to_system_config(cfg)
            validation_errors = validate_config(new_config)
            
            if validation_errors:
                logger.error(f"Configuration reload failed validation: {validation_errors}")
                return
                
            # Update configuration atomically
            with self._config_lock:
                old_config = self._config
                self._config = new_config
                self._raw_config = cfg
                
            logger.info("Configuration reloaded successfully")
            
            # Notify callbacks
            for callback in self._reload_callbacks:
                try:
                    callback(new_config)
                except Exception as e:
                    logger.error(f"Reload callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
    
    def add_reload_callback(self, callback: Callable[[SystemConfig], None]) -> None:
        """
        Add a callback function to be called when configuration is reloaded.
        
        Args:
            callback: Function that takes a SystemConfig parameter
        """
        self._reload_callbacks.append(callback)
    
    def remove_reload_callback(self, callback: Callable[[SystemConfig], None]) -> None:
        """
        Remove a reload callback function.
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)
    
    def get_environment_config(self, environment: str) -> SystemConfig:
        """
        Load configuration for a specific environment.
        
        Args:
            environment: Environment name (development, testing, production)
            
        Returns:
            SystemConfig for the specified environment
        """
        overrides = [f"environments={environment}"]
        return self.load_config(overrides=overrides)
    
    def get_hardware_config(self, hardware_profile: str) -> SystemConfig:
        """
        Load configuration for a specific hardware profile.
        
        Args:
            hardware_profile: Hardware profile name (auto, m1_native, vm_ubuntu, cuda_cluster)
            
        Returns:
            SystemConfig with the specified hardware profile
        """
        overrides = [f"hardware={hardware_profile}"]
        return self.load_config(overrides=overrides)
    
    def get_ml_config(self, ml_profile: str) -> SystemConfig:
        """
        Load configuration for a specific ML profile.
        
        Args:
            ml_profile: ML profile name (default, training, inference)
            
        Returns:
            SystemConfig with the specified ML profile
        """
        overrides = [f"ml={ml_profile}"]
        return self.load_config(overrides=overrides)
    
    def override_config(self, **kwargs) -> SystemConfig:
        """
        Load configuration with runtime overrides.
        
        Args:
            **kwargs: Configuration overrides in dot notation
            
        Returns:
            SystemConfig with applied overrides
            
        Example:
            config = manager.override_config(
                hardware.device_preference="cuda",
                ml.batch_size=64,
                logging.level="DEBUG"
            )
        """
        overrides = [f"{key}={value}" for key, value in kwargs.items()]
        return self.load_config(overrides=overrides)
    
    def save_config(self, config: SystemConfig, filename: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: SystemConfig to save
            filename: Output filename (will be saved in config directory)
        """
        try:
            # Convert to dictionary
            config_dict = asdict(config)
            
            # Convert enums to strings for YAML serialization
            self._convert_enums_to_strings(config_dict)
            
            # Create OmegaConf object and save
            cfg = OmegaConf.create(config_dict)
            output_path = self.config_dir / filename
            
            with open(output_path, 'w') as f:
                OmegaConf.save(cfg, f)
                
            logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigValidationError(f"Configuration save failed: {e}")
    
    def _convert_enums_to_strings(self, config_dict: Dict[str, Any]) -> None:
        """Convert enum values to strings for YAML serialization."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._convert_enums_to_strings(value)
            elif hasattr(value, 'value'):  # Enum
                config_dict[key] = value.value
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.disable_hot_reload()
        
        # Clear Hydra instance
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()


# Global configuration manager instance
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        Global ConfigManager instance
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def load_config(config_name: str = "config", **kwargs) -> SystemConfig:
    """
    Convenience function to load configuration using the global manager.
    
    Args:
        config_name: Configuration file name
        **kwargs: Additional arguments passed to ConfigManager.load_config
        
    Returns:
        Loaded SystemConfig
    """
    return get_config_manager().load_config(config_name, **kwargs)


def get_current_config() -> Optional[SystemConfig]:
    """
    Convenience function to get current configuration from global manager.
    
    Returns:
        Current SystemConfig or None if not loaded
    """
    return get_config_manager().get_config()