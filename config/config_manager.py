"""
Configuration Management System for Dual Environment Setup

This module provides configuration loading, validation, and management
for both M1 native and Ubuntu VM environments.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Optimization settings for the platform"""
    threads: int = 4
    memory_fraction: float = 0.6
    mixed_precision: bool = False
    mps_fallback: bool = False
    batch_size_multiplier: float = 1.0


@dataclass
class PackageConfig:
    """Package configuration for the platform"""
    pytorch_variant: str = "cpu"
    numpy_version: str = ">=1.20.0"
    scipy_version: str = ">=1.9.0"
    accelerate_training: bool = False
    metal_performance_shaders: bool = False


@dataclass
class TrainingConfig:
    """Training-specific configuration"""
    learning_rate_multiplier: float = 1.0
    gradient_checkpointing: bool = False
    num_workers: int = 2
    pin_memory: bool = False
    use_amp: bool = False


@dataclass
class ModelConfig:
    """Model-specific configuration"""
    use_metal_optimizations: bool = False
    compile_model: bool = False
    memory_efficient_attention: bool = False


@dataclass
class HardwareConfig:
    """Hardware-specific settings"""
    max_memory_gb: int = 8
    unified_memory: bool = False
    use_neural_engine: bool = False
    mps_available: bool = False
    sdr_simulation_fallback: bool = True


@dataclass
class SDRConfig:
    """SDR hardware configuration"""
    simulation_mode: bool = True
    preferred_sample_rates: List[int] = field(default_factory=lambda: [1000000, 2000000])
    buffer_size: int = 4096


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    enable_mps_profiling: bool = False
    track_memory_usage: bool = True
    log_device_utilization: bool = False


@dataclass
class PlatformConfig:
    """Complete platform configuration"""
    platform: str = "unknown"
    architecture: str = "unknown"
    device: str = "cpu"
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    packages: PackageConfig = field(default_factory=PackageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    sdr: SDRConfig = field(default_factory=SDRConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    environment_vars: Dict[str, str] = field(default_factory=dict)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


class ConfigManager:
    """Manages platform-specific configurations"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, PlatformConfig] = {}
        self._current_platform: Optional[str] = None
    
    def load_config(self, platform: str) -> PlatformConfig:
        """Load configuration for specified platform"""
        if platform in self._configs:
            return self._configs[platform]
        
        # Map platform names to config file names
        platform_file_map = {
            'm1_native': 'm1_config.yaml',
            'vm_ubuntu': 'vm_config.yaml',
            'm1': 'm1_config.yaml',  # Allow short names too
            'vm': 'vm_config.yaml'
        }
        
        config_filename = platform_file_map.get(platform, f"{platform}_config.yaml")
        config_file = self.config_dir / config_filename
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            config = self._parse_config(config_data)
            self._validate_config(config, platform)
            
            self._configs[platform] = config
            logger.info(f"Loaded configuration for platform: {platform}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config for {platform}: {e}")
            raise ConfigValidationError(f"Invalid configuration for {platform}: {e}")
    
    def _parse_config(self, config_data: Dict[str, Any]) -> PlatformConfig:
        """Parse configuration data into PlatformConfig object"""
        
        # Parse optimization settings
        opt_data = config_data.get('optimization', {})
        optimization = OptimizationConfig(
            threads=opt_data.get('threads', 4),
            memory_fraction=opt_data.get('memory_fraction', 0.6),
            mixed_precision=opt_data.get('mixed_precision', False),
            mps_fallback=opt_data.get('mps_fallback', False),
            batch_size_multiplier=opt_data.get('batch_size_multiplier', 1.0)
        )
        
        # Parse package settings
        pkg_data = config_data.get('packages', {})
        packages = PackageConfig(
            pytorch_variant=pkg_data.get('pytorch_variant', 'cpu'),
            numpy_version=pkg_data.get('numpy_version', '>=1.20.0'),
            scipy_version=pkg_data.get('scipy_version', '>=1.9.0'),
            accelerate_training=pkg_data.get('accelerate_training', False),
            metal_performance_shaders=pkg_data.get('metal_performance_shaders', False)
        )
        
        # Parse training settings
        train_data = config_data.get('training', {})
        training = TrainingConfig(
            learning_rate_multiplier=train_data.get('learning_rate_multiplier', 1.0),
            gradient_checkpointing=train_data.get('gradient_checkpointing', False),
            num_workers=train_data.get('num_workers', 2),
            pin_memory=train_data.get('pin_memory', False),
            use_amp=train_data.get('use_amp', False)
        )
        
        # Parse model settings
        model_data = config_data.get('model', {})
        model = ModelConfig(
            use_metal_optimizations=model_data.get('use_metal_optimizations', False),
            compile_model=model_data.get('compile_model', False),
            memory_efficient_attention=model_data.get('memory_efficient_attention', False)
        )
        
        # Parse hardware settings
        hw_data = config_data.get('hardware', {})
        hardware = HardwareConfig(
            max_memory_gb=hw_data.get('max_memory_gb', 8),
            unified_memory=hw_data.get('unified_memory', False),
            use_neural_engine=hw_data.get('use_neural_engine', False),
            mps_available=hw_data.get('mps_available', False),
            sdr_simulation_fallback=hw_data.get('sdr_simulation_fallback', True)
        )
        
        # Parse SDR settings
        sdr_data = config_data.get('sdr', {})
        sdr = SDRConfig(
            simulation_mode=sdr_data.get('simulation_mode', True),
            preferred_sample_rates=sdr_data.get('preferred_sample_rates', [1000000, 2000000]),
            buffer_size=sdr_data.get('buffer_size', 4096)
        )
        
        # Parse logging settings
        log_data = config_data.get('logging', {})
        logging_config = LoggingConfig(
            level=log_data.get('level', 'INFO'),
            enable_mps_profiling=log_data.get('enable_mps_profiling', False),
            track_memory_usage=log_data.get('track_memory_usage', True),
            log_device_utilization=log_data.get('log_device_utilization', False)
        )
        
        return PlatformConfig(
            platform=config_data.get('platform', 'unknown'),
            architecture=config_data.get('architecture', 'unknown'),
            device=config_data.get('device', 'cpu'),
            optimization=optimization,
            packages=packages,
            training=training,
            model=model,
            hardware=hardware,
            sdr=sdr,
            logging=logging_config,
            environment_vars=config_data.get('environment_vars', {})
        )
    
    def _validate_config(self, config: PlatformConfig, platform: str) -> None:
        """Validate configuration for consistency and correctness"""
        
        # Validate platform consistency
        if config.platform != platform:
            raise ConfigValidationError(
                f"Platform mismatch: config says {config.platform}, expected {platform}"
            )
        
        # Validate device settings
        valid_devices = ['cpu', 'mps', 'cuda']
        if config.device not in valid_devices:
            raise ConfigValidationError(
                f"Invalid device '{config.device}'. Must be one of: {valid_devices}"
            )
        
        # Validate M1-specific settings
        if platform == 'm1' and config.device == 'mps':
            if not config.packages.metal_performance_shaders:
                logger.warning("MPS device specified but Metal Performance Shaders disabled")
        
        # Validate VM-specific settings
        if platform == 'vm' and config.device != 'cpu':
            raise ConfigValidationError(
                f"VM platform should use CPU device, got {config.device}"
            )
        
        # Validate memory settings
        if not 0.1 <= config.optimization.memory_fraction <= 1.0:
            raise ConfigValidationError(
                f"Memory fraction must be between 0.1 and 1.0, got {config.optimization.memory_fraction}"
            )
        
        # Validate thread count
        if config.optimization.threads < 1:
            raise ConfigValidationError(
                f"Thread count must be positive, got {config.optimization.threads}"
            )
        
        # Validate sample rates
        for rate in config.sdr.preferred_sample_rates:
            if rate < 100000 or rate > 50000000:
                logger.warning(f"Sample rate {rate} may be outside typical SDR range")
        
        logger.info(f"Configuration validation passed for platform: {platform}")
    
    def get_current_config(self) -> Optional[PlatformConfig]:
        """Get the currently active configuration"""
        if self._current_platform:
            return self._configs.get(self._current_platform)
        return None
    
    def set_current_platform(self, platform: str) -> PlatformConfig:
        """Set the current active platform and return its configuration"""
        config = self.load_config(platform)
        self._current_platform = platform
        return config
    
    def apply_environment_variables(self, config: PlatformConfig) -> None:
        """Apply environment variables from configuration"""
        for key, value in config.environment_vars.items():
            os.environ[key] = str(value)
            logger.debug(f"Set environment variable: {key}={value}")
    
    def get_optimization_settings(self, platform: str) -> Dict[str, Any]:
        """Get optimization settings for the specified platform"""
        config = self.load_config(platform)
        return {
            'device': config.device,
            'threads': config.optimization.threads,
            'memory_fraction': config.optimization.memory_fraction,
            'mixed_precision': config.optimization.mixed_precision,
            'batch_size_multiplier': config.optimization.batch_size_multiplier,
            'use_amp': config.training.use_amp,
            'num_workers': config.training.num_workers,
            'pin_memory': config.training.pin_memory
        }
    
    def get_package_requirements(self, platform: str) -> Dict[str, str]:
        """Get package requirements for the specified platform"""
        config = self.load_config(platform)
        return {
            'pytorch_variant': config.packages.pytorch_variant,
            'numpy_version': config.packages.numpy_version,
            'scipy_version': config.packages.scipy_version
        }
    
    def save_config(self, platform: str, config: PlatformConfig) -> None:
        """Save configuration to file"""
        config_file = self.config_dir / f"{platform}_config.yaml"
        
        # Convert config to dictionary
        config_dict = {
            'platform': config.platform,
            'architecture': config.architecture,
            'device': config.device,
            'optimization': {
                'threads': config.optimization.threads,
                'memory_fraction': config.optimization.memory_fraction,
                'mixed_precision': config.optimization.mixed_precision,
                'mps_fallback': config.optimization.mps_fallback,
                'batch_size_multiplier': config.optimization.batch_size_multiplier
            },
            'packages': {
                'pytorch_variant': config.packages.pytorch_variant,
                'numpy_version': config.packages.numpy_version,
                'scipy_version': config.packages.scipy_version,
                'accelerate_training': config.packages.accelerate_training,
                'metal_performance_shaders': config.packages.metal_performance_shaders
            },
            'training': {
                'learning_rate_multiplier': config.training.learning_rate_multiplier,
                'gradient_checkpointing': config.training.gradient_checkpointing,
                'num_workers': config.training.num_workers,
                'pin_memory': config.training.pin_memory,
                'use_amp': config.training.use_amp
            },
            'model': {
                'use_metal_optimizations': config.model.use_metal_optimizations,
                'compile_model': config.model.compile_model,
                'memory_efficient_attention': config.model.memory_efficient_attention
            },
            'hardware': {
                'max_memory_gb': config.hardware.max_memory_gb,
                'unified_memory': config.hardware.unified_memory,
                'use_neural_engine': config.hardware.use_neural_engine,
                'mps_available': config.hardware.mps_available,
                'sdr_simulation_fallback': config.hardware.sdr_simulation_fallback
            },
            'sdr': {
                'simulation_mode': config.sdr.simulation_mode,
                'preferred_sample_rates': config.sdr.preferred_sample_rates,
                'buffer_size': config.sdr.buffer_size
            },
            'logging': {
                'level': config.logging.level,
                'enable_mps_profiling': config.logging.enable_mps_profiling,
                'track_memory_usage': config.logging.track_memory_usage,
                'log_device_utilization': config.logging.log_device_utilization
            },
            'environment_vars': config.environment_vars
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration for platform: {platform}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config(platform: str) -> PlatformConfig:
    """Convenience function to get configuration for a platform"""
    return config_manager.load_config(platform)


def get_current_config() -> Optional[PlatformConfig]:
    """Convenience function to get current configuration"""
    return config_manager.get_current_config()


def set_platform(platform: str) -> PlatformConfig:
    """Convenience function to set current platform"""
    return config_manager.set_current_platform(platform)