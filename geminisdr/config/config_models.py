"""
Configuration data models and validation for GeminiSDR system.

This module defines the configuration data structures using dataclasses
with proper typing and validation logic.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types for computation."""
    AUTO = "auto"
    CPU = "cpu"
    MPS = "mps"  # Apple Metal Performance Shaders
    CUDA = "cuda"


class SDRMode(Enum):
    """SDR operation modes."""
    AUTO = "auto"
    HARDWARE = "hardware"
    SIMULATION = "simulation"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Log output formats."""
    STRUCTURED = "structured"
    SIMPLE = "simple"


@dataclass
class HardwareConfig:
    """Hardware-specific configuration."""
    device_preference: DeviceType = DeviceType.AUTO
    sdr_mode: SDRMode = SDRMode.AUTO
    memory_limit_gb: Optional[float] = None
    max_threads: Optional[int] = None
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = False
    
    def __post_init__(self):
        """Validate hardware configuration after initialization."""
        if isinstance(self.device_preference, str):
            self.device_preference = DeviceType(self.device_preference)
        if isinstance(self.sdr_mode, str):
            self.sdr_mode = SDRMode(self.sdr_mode)
            
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
        if self.max_threads is not None and self.max_threads <= 0:
            raise ValueError("max_threads must be positive")
        if not 0.1 <= self.gpu_memory_fraction <= 1.0:
            raise ValueError("gpu_memory_fraction must be between 0.1 and 1.0")


@dataclass
class MLConfig:
    """ML training and inference configuration."""
    batch_size: Optional[int] = None  # Auto-determined if None
    learning_rate: float = 1e-4
    model_cache_size: int = 3
    checkpoint_frequency: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    weight_decay: float = 0.01
    enable_gradient_checkpointing: bool = False
    
    def __post_init__(self):
        """Validate ML configuration after initialization."""
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.model_cache_size <= 0:
            raise ValueError("model_cache_size must be positive")
        if self.checkpoint_frequency <= 0:
            raise ValueError("checkpoint_frequency must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")


@dataclass
class LoggingConfig:
    """Logging system configuration."""
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.STRUCTURED
    output: List[str] = field(default_factory=lambda: ["console", "file"])
    rotation: str = "daily"
    max_file_size_mb: int = 100
    backup_count: int = 5
    log_dir: str = "logs"
    enable_performance_logging: bool = True
    
    def __post_init__(self):
        """Validate logging configuration after initialization."""
        if isinstance(self.level, str):
            self.level = LogLevel(self.level)
        if isinstance(self.format, str):
            self.format = LogFormat(self.format)
            
        valid_outputs = {"console", "file", "syslog"}
        for output in self.output:
            if output not in valid_outputs:
                raise ValueError(f"Invalid log output '{output}'. Must be one of: {valid_outputs}")
                
        valid_rotations = {"daily", "weekly", "monthly", "size"}
        if self.rotation not in valid_rotations:
            raise ValueError(f"Invalid rotation '{self.rotation}'. Must be one of: {valid_rotations}")
            
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        if self.backup_count < 0:
            raise ValueError("backup_count must be non-negative")


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    memory_threshold: float = 0.8
    auto_optimize: bool = True
    profiling_enabled: bool = False
    gc_frequency: int = 100
    memory_cleanup_threshold: float = 0.9
    batch_size_auto_tune: bool = True
    model_compilation: bool = False
    
    def __post_init__(self):
        """Validate performance configuration after initialization."""
        if not 0.1 <= self.memory_threshold <= 1.0:
            raise ValueError("memory_threshold must be between 0.1 and 1.0")
        if not 0.1 <= self.memory_cleanup_threshold <= 1.0:
            raise ValueError("memory_cleanup_threshold must be between 0.1 and 1.0")
        if self.gc_frequency <= 0:
            raise ValueError("gc_frequency must be positive")


@dataclass
class SystemConfig:
    """Main system configuration containing all subsystem configs."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    environment: str = "development"
    debug_mode: bool = False
    config_version: str = "1.0"
    
    def __post_init__(self):
        """Validate system configuration after initialization."""
        valid_environments = {"development", "testing", "production"}
        if self.environment not in valid_environments:
            raise ValueError(f"Invalid environment '{self.environment}'. Must be one of: {valid_environments}")


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.message = message
    
    def __str__(self):
        if self.field:
            return f"Configuration validation error in field '{self.field}': {self.message}"
        return f"Configuration validation error: {self.message}"


def validate_config(config: SystemConfig) -> List[str]:
    """
    Validate a complete system configuration and return list of validation errors.
    
    Args:
        config: SystemConfig instance to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    try:
        # Validate hardware config
        if config.hardware.device_preference == DeviceType.MPS:
            # Check if MPS is actually available (this would be done at runtime)
            logger.debug("MPS device preference set - runtime availability will be checked")
            
        if config.hardware.sdr_mode == SDRMode.HARDWARE:
            logger.debug("Hardware SDR mode set - hardware availability will be checked at runtime")
            
        # Validate ML config compatibility
        if config.ml.batch_size and config.performance.batch_size_auto_tune:
            logger.warning("Fixed batch_size set with auto_tune enabled - auto_tune will be ignored")
            
        # Validate performance settings
        if config.performance.memory_threshold >= config.performance.memory_cleanup_threshold:
            errors.append("memory_threshold should be less than memory_cleanup_threshold")
            
        # Cross-component validation
        if config.hardware.enable_mixed_precision and config.hardware.device_preference == DeviceType.CPU:
            logger.warning("Mixed precision enabled with CPU device - may not provide benefits")
            
        if config.debug_mode and config.logging.level != LogLevel.DEBUG:
            logger.info("Debug mode enabled but log level is not DEBUG - consider setting log level to DEBUG")
            
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
    
    return errors


def create_default_config(environment: str = "development") -> SystemConfig:
    """
    Create a default system configuration for the specified environment.
    
    Args:
        environment: Target environment (development, testing, production)
        
    Returns:
        SystemConfig with environment-appropriate defaults
    """
    if environment == "development":
        return SystemConfig(
            hardware=HardwareConfig(
                device_preference=DeviceType.AUTO,
                sdr_mode=SDRMode.SIMULATION,
                memory_limit_gb=None,
                enable_mixed_precision=False
            ),
            ml=MLConfig(
                batch_size=None,
                learning_rate=1e-4,
                model_cache_size=2,
                checkpoint_frequency=50
            ),
            logging=LoggingConfig(
                level=LogLevel.DEBUG,
                format=LogFormat.STRUCTURED,
                output=["console", "file"]
            ),
            performance=PerformanceConfig(
                memory_threshold=0.7,
                auto_optimize=True,
                profiling_enabled=True
            ),
            environment=environment,
            debug_mode=True
        )
    elif environment == "testing":
        return SystemConfig(
            hardware=HardwareConfig(
                device_preference=DeviceType.CPU,
                sdr_mode=SDRMode.SIMULATION,
                memory_limit_gb=4.0,
                enable_mixed_precision=False
            ),
            ml=MLConfig(
                batch_size=16,
                learning_rate=1e-4,
                model_cache_size=1,
                checkpoint_frequency=10
            ),
            logging=LoggingConfig(
                level=LogLevel.INFO,
                format=LogFormat.SIMPLE,
                output=["console"]
            ),
            performance=PerformanceConfig(
                memory_threshold=0.6,
                auto_optimize=False,
                profiling_enabled=False
            ),
            environment=environment,
            debug_mode=False
        )
    elif environment == "production":
        return SystemConfig(
            hardware=HardwareConfig(
                device_preference=DeviceType.AUTO,
                sdr_mode=SDRMode.AUTO,
                memory_limit_gb=None,
                enable_mixed_precision=True
            ),
            ml=MLConfig(
                batch_size=None,
                learning_rate=1e-4,
                model_cache_size=5,
                checkpoint_frequency=1000
            ),
            logging=LoggingConfig(
                level=LogLevel.INFO,
                format=LogFormat.STRUCTURED,
                output=["file", "syslog"]
            ),
            performance=PerformanceConfig(
                memory_threshold=0.8,
                auto_optimize=True,
                profiling_enabled=False
            ),
            environment=environment,
            debug_mode=False
        )
    else:
        raise ValueError(f"Unknown environment: {environment}")