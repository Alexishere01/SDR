"""
Hardware abstraction layer for dual environment setup.

This module provides automatic device detection and selection, MPS vs CPU backend
selection logic, and SDR hardware detection with simulation fallback.
"""

import os
import sys
import logging
import importlib.util
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from geminisdr.config.config_manager import get_config_manager, SystemConfig
from geminisdr.core.error_handling import (
    ErrorHandler, HardwareError, ConfigurationError, ErrorSeverity, ErrorContext
)
from geminisdr.core.logging_manager import StructuredLogger

try:
    from .platform_detector import PlatformDetector, PlatformInfo, PlatformCapabilities
except ImportError:
    # For standalone testing
    from platform_detector import PlatformDetector, PlatformInfo, PlatformCapabilities


class DeviceType(Enum):
    """Enumeration of supported compute device types."""
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


class SDRMode(Enum):
    """Enumeration of SDR operation modes."""
    HARDWARE = "hardware"
    SIMULATION = "simulation"


@dataclass
class DeviceInfo:
    """Information about a compute device."""
    device_type: DeviceType
    device_id: Optional[int] = None
    memory_mb: Optional[int] = None
    name: Optional[str] = None
    is_available: bool = True


@dataclass
class SDRInfo:
    """Information about SDR hardware configuration."""
    mode: SDRMode
    devices: List[str]
    sample_rates: List[int]
    max_bandwidth: Optional[int] = None
    simulation_params: Optional[Dict[str, Any]] = None


class HardwareAbstraction:
    """
    Hardware abstraction layer for cross-platform compatibility with configuration management.
    
    Provides automatic device selection, hardware detection, and simulation
    fallback for both compute devices and SDR hardware.
    """
    
    def __init__(self, force_device: Optional[str] = None, force_simulation: bool = False, config: SystemConfig = None):
        """
        Initialize hardware abstraction layer.
        
        Args:
            force_device: Force specific device type ('cpu', 'mps', 'cuda')
            force_simulation: Force simulation mode for SDR
            config: System configuration object
        """
        # Load configuration
        if config is None:
            try:
                config_manager = get_config_manager()
                self.config = config_manager.get_config()
                if self.config is None:
                    self.config = config_manager.load_config()
            except Exception as e:
                # Use fallback configuration
                from geminisdr.config.config_models import SystemConfig, HardwareConfig, LoggingConfig
                self.config = SystemConfig(
                    hardware=HardwareConfig(),
                    logging=LoggingConfig()
                )
        else:
            self.config = config
        
        # Initialize error handling and logging
        self.logger = StructuredLogger(__name__, self.config.logging)
        self.error_handler = ErrorHandler(self.logger.logger)
        
        self.platform_detector = PlatformDetector()
        
        # Use configuration for device preferences
        self.force_device = force_device or self.config.hardware.device_preference
        if self.force_device == "auto":
            self.force_device = None
            
        self.force_simulation = force_simulation or (self.config.hardware.sdr_mode == "simulation")
        
        # Cache for detected hardware
        self._compute_device = None
        self._sdr_config = None
        self._platform_info = None
        self._capabilities = None
        
        # Initialize hardware detection
        self._initialize()
    
    def _initialize(self):
        """Initialize hardware detection and caching with error handling."""
        try:
            with self.error_handler.error_context("hardware_initialization", component="HardwareAbstraction"):
                self._platform_info = self.platform_detector.get_platform_info()
                self._capabilities = self.platform_detector.get_hardware_capabilities()
                self.logger.logger.info(f"Initialized hardware abstraction for {self._platform_info.platform}")
        except Exception as e:
            self.logger.log_error_with_context(e, component="HardwareAbstraction", operation="initialization")
            # Continue with minimal functionality
    
    def get_compute_device(self) -> DeviceInfo:
        """
        Get the optimal compute device for the current platform.
        
        Returns:
            DeviceInfo: Information about the selected compute device
        """
        if self._compute_device is not None:
            return self._compute_device
        
        # Check for forced device selection
        if self.force_device:
            device_type = DeviceType(self.force_device.lower())
            if self._is_device_available(device_type):
                self._compute_device = self._create_device_info(device_type)
                self.logger.info(f"Using forced device: {device_type.value}")
                return self._compute_device
            else:
                self.logger.warning(f"Forced device {self.force_device} not available, falling back to auto-detection")
        
        # Auto-detect optimal device
        device_priority = self._get_device_priority()
        
        for device_type in device_priority:
            if self._is_device_available(device_type):
                self._compute_device = self._create_device_info(device_type)
                self.logger.info(f"Selected compute device: {device_type.value}")
                return self._compute_device
        
        # Fallback to CPU if nothing else works
        self._compute_device = self._create_device_info(DeviceType.CPU)
        self.logger.warning("Falling back to CPU device")
        return self._compute_device
    
    def _get_device_priority(self) -> List[DeviceType]:
        """
        Get device priority order based on platform.
        
        Returns:
            List[DeviceType]: Ordered list of device preferences
        """
        if not self._platform_info:
            return [DeviceType.CPU]
        
        if self._platform_info.platform == "m1_native":
            # M1 Mac: prefer MPS, fallback to CPU
            return [DeviceType.MPS, DeviceType.CPU]
        elif self._platform_info.platform == "vm_ubuntu":
            # Ubuntu VM: prefer CPU (CUDA unlikely in VM)
            return [DeviceType.CPU, DeviceType.CUDA]
        else:
            # Unknown platform: try all options
            return [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]
    
    def _is_device_available(self, device_type: DeviceType) -> bool:
        """
        Check if a specific device type is available.
        
        Args:
            device_type: The device type to check
            
        Returns:
            bool: True if device is available
        """
        try:
            if device_type == DeviceType.CPU:
                return True  # CPU always available
            
            elif device_type == DeviceType.MPS:
                # Check for MPS availability
                torch_spec = importlib.util.find_spec("torch")
                if torch_spec is None:
                    return False
                
                import torch
                return (hasattr(torch.backends, 'mps') and 
                       torch.backends.mps.is_available() and
                       torch.backends.mps.is_built())
            
            elif device_type == DeviceType.CUDA:
                # Check for CUDA availability
                torch_spec = importlib.util.find_spec("torch")
                if torch_spec is None:
                    return False
                
                import torch
                return torch.cuda.is_available()
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking device availability for {device_type.value}: {e}")
            return False
    
    def _create_device_info(self, device_type: DeviceType) -> DeviceInfo:
        """
        Create DeviceInfo object for the specified device type.
        
        Args:
            device_type: The device type
            
        Returns:
            DeviceInfo: Device information object
        """
        device_info = DeviceInfo(device_type=device_type)
        
        try:
            if device_type == DeviceType.CPU:
                device_info.name = "CPU"
                if self._capabilities:
                    device_info.memory_mb = self._capabilities.max_memory
            
            elif device_type == DeviceType.MPS:
                device_info.name = "Apple Silicon GPU (MPS)"
                if self._capabilities:
                    device_info.memory_mb = self._capabilities.gpu_memory
            
            elif device_type == DeviceType.CUDA:
                import torch
                if torch.cuda.is_available():
                    device_info.device_id = 0  # Use first GPU
                    device_info.name = torch.cuda.get_device_name(0)
                    device_info.memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        
        except Exception as e:
            self.logger.error(f"Error creating device info for {device_type.value}: {e}")
        
        return device_info
    
    def get_sdr_interface(self) -> SDRInfo:
        """
        Get SDR interface configuration with hardware detection and simulation fallback.
        
        Returns:
            SDRInfo: SDR configuration information
        """
        if self._sdr_config is not None:
            return self._sdr_config
        
        # Check for forced simulation mode
        if self.force_simulation:
            self._sdr_config = self._create_simulation_config()
            self.logger.info("Using forced SDR simulation mode")
            return self._sdr_config
        
        # Try to detect hardware SDR devices
        if self._platform_info and self._platform_info.has_sdr:
            hardware_config = self._detect_sdr_hardware()
            if hardware_config:
                self._sdr_config = hardware_config
                self.logger.info(f"Detected SDR hardware: {hardware_config.devices}")
                return self._sdr_config
        
        # Fallback to simulation
        self._sdr_config = self._create_simulation_config()
        self.logger.info("No SDR hardware detected, using simulation mode")
        return self._sdr_config
    
    def _detect_sdr_hardware(self) -> Optional[SDRInfo]:
        """
        Detect available SDR hardware.
        
        Returns:
            Optional[SDRInfo]: SDR hardware configuration if detected
        """
        try:
            devices = []
            sample_rates = []
            max_bandwidth = None
            
            # Try to detect PlutoSDR
            if self._detect_pluto_sdr():
                devices.append("PlutoSDR")
                sample_rates.extend([250000, 1000000, 2000000, 2400000])
                max_bandwidth = 20000000  # 20 MHz
            
            # Try to detect RTL-SDR
            if self._detect_rtl_sdr():
                devices.append("RTL-SDR")
                sample_rates.extend([250000, 1000000, 2000000, 2400000, 3200000])
                if not max_bandwidth:
                    max_bandwidth = 3200000  # 3.2 MHz
            
            # Try to detect HackRF
            if self._detect_hackrf():
                devices.append("HackRF")
                sample_rates.extend([2000000, 4000000, 8000000, 10000000, 20000000])
                max_bandwidth = 20000000  # 20 MHz
            
            if devices:
                return SDRInfo(
                    mode=SDRMode.HARDWARE,
                    devices=devices,
                    sample_rates=sorted(list(set(sample_rates))),
                    max_bandwidth=max_bandwidth
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting SDR hardware: {e}")
            return None
    
    def _detect_pluto_sdr(self) -> bool:
        """Detect PlutoSDR hardware."""
        try:
            # Check for PlutoSDR via iio library
            iio_spec = importlib.util.find_spec("iio")
            if iio_spec is not None:
                import iio
                contexts = iio.scan_contexts()
                for context_info in contexts:
                    if 'pluto' in context_info.lower() or 'ad9361' in context_info.lower():
                        return True
            
            # Check for USB device (basic check)
            import glob
            pluto_devices = glob.glob('/dev/ttyACM*')
            return len(pluto_devices) > 0
            
        except Exception:
            return False
    
    def _detect_rtl_sdr(self) -> bool:
        """Detect RTL-SDR hardware."""
        try:
            # Check for rtlsdr library
            rtlsdr_spec = importlib.util.find_spec("rtlsdr")
            if rtlsdr_spec is not None:
                from rtlsdr import RtlSdr
                try:
                    sdr = RtlSdr()
                    sdr.close()
                    return True
                except:
                    pass
            
            # Check via SoapySDR if available
            soapy_spec = importlib.util.find_spec("SoapySDR")
            if soapy_spec is not None:
                import SoapySDR
                devices = SoapySDR.Device.enumerate()
                for device in devices:
                    if device.get('driver', '').lower() == 'rtlsdr':
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_hackrf(self) -> bool:
        """Detect HackRF hardware."""
        try:
            # Check for hackrf library
            hackrf_spec = importlib.util.find_spec("hackrf")
            if hackrf_spec is not None:
                import hackrf
                try:
                    device_list = hackrf.device_list()
                    return len(device_list) > 0
                except:
                    pass
            
            # Check via SoapySDR if available
            soapy_spec = importlib.util.find_spec("SoapySDR")
            if soapy_spec is not None:
                import SoapySDR
                devices = SoapySDR.Device.enumerate()
                for device in devices:
                    if device.get('driver', '').lower() == 'hackrf':
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _create_simulation_config(self) -> SDRInfo:
        """
        Create simulation configuration for SDR.
        
        Returns:
            SDRInfo: Simulation configuration
        """
        simulation_params = {
            'noise_floor': -80,  # dBm
            'max_signal_power': -10,  # dBm
            'frequency_accuracy': 1e-6,  # 1 ppm
            'phase_noise': -120,  # dBc/Hz at 1kHz offset
            'spurious_free_dynamic_range': 80,  # dB
            'adc_bits': 12,
            'dac_bits': 12
        }
        
        return SDRInfo(
            mode=SDRMode.SIMULATION,
            devices=["Simulation"],
            sample_rates=[250000, 1000000, 2000000],
            max_bandwidth=2000000,  # 2 MHz for simulation
            simulation_params=simulation_params
        )
    
    def optimize_for_platform(self) -> Dict[str, Any]:
        """
        Get platform-specific optimization settings.
        
        Returns:
            Dict[str, Any]: Optimization configuration
        """
        device_info = self.get_compute_device()
        sdr_info = self.get_sdr_interface()
        
        optimization_config = {
            'device': device_info.device_type.value,
            'device_id': device_info.device_id,
            'mixed_precision': device_info.device_type == DeviceType.MPS,
            'dataloader_num_workers': self._get_optimal_workers(),
            'pin_memory': device_info.device_type in [DeviceType.CUDA, DeviceType.MPS],
            'sdr_mode': sdr_info.mode.value,
            'simulation_mode': sdr_info.mode == SDRMode.SIMULATION,
            'max_sample_rate': max(sdr_info.sample_rates) if sdr_info.sample_rates else 1000000,
            'recommended_batch_size': self._get_recommended_batch_size(device_info)
        }
        
        # Add device-specific optimizations
        if device_info.device_type == DeviceType.MPS:
            optimization_config.update({
                'mps_fallback': True,
                'use_deterministic_algorithms': False,  # Some ops not deterministic on MPS
                'benchmark_mode': False  # cuDNN benchmark not applicable
            })
        elif device_info.device_type == DeviceType.CUDA:
            optimization_config.update({
                'benchmark_mode': True,  # Enable cuDNN benchmark
                'use_deterministic_algorithms': False,
                'allow_tf32': True
            })
        else:  # CPU
            optimization_config.update({
                'use_deterministic_algorithms': True,
                'benchmark_mode': False,
                'thread_count': self._capabilities.cpu_cores if self._capabilities else 4
            })
        
        return optimization_config
    
    def _get_optimal_workers(self) -> int:
        """Get optimal number of dataloader workers."""
        if not self._capabilities:
            return 2
        
        # Use fewer workers on M1 due to memory constraints
        if self._platform_info and self._platform_info.platform == "m1_native":
            return min(4, self._capabilities.cpu_cores)
        else:
            return min(8, self._capabilities.cpu_cores)
    
    def _get_recommended_batch_size(self, device_info: DeviceInfo) -> int:
        """Get recommended batch size based on device capabilities."""
        if device_info.device_type == DeviceType.CPU:
            return 16  # Conservative for CPU
        elif device_info.device_type == DeviceType.MPS:
            # M1 has unified memory, be conservative
            if device_info.memory_mb and device_info.memory_mb > 16000:  # 16GB+
                return 64
            else:
                return 32
        elif device_info.device_type == DeviceType.CUDA:
            # Scale based on GPU memory
            if device_info.memory_mb and device_info.memory_mb > 8000:  # 8GB+
                return 128
            else:
                return 64
        else:
            return 32  # Default
    
    def get_device_string(self) -> str:
        """
        Get device string for PyTorch/ML frameworks.
        
        Returns:
            str: Device string (e.g., 'cpu', 'mps', 'cuda:0')
        """
        device_info = self.get_compute_device()
        
        if device_info.device_type == DeviceType.CPU:
            return "cpu"
        elif device_info.device_type == DeviceType.MPS:
            return "mps"
        elif device_info.device_type == DeviceType.CUDA:
            device_id = device_info.device_id or 0
            return f"cuda:{device_id}"
        else:
            return "cpu"  # Fallback
    
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate the current hardware configuration.
        
        Returns:
            Dict[str, bool]: Validation results
        """
        validation = {}
        
        try:
            # Validate compute device
            device_info = self.get_compute_device()
            validation['compute_device_available'] = device_info.is_available
            
            # Validate SDR configuration
            sdr_info = self.get_sdr_interface()
            validation['sdr_configured'] = len(sdr_info.devices) > 0
            validation['has_hardware_sdr'] = sdr_info.mode == SDRMode.HARDWARE
            
            # Validate platform compatibility
            platform_validation = self.platform_detector.validate_environment()
            validation.update(platform_validation)
            
            # Check for common issues
            if device_info.device_type == DeviceType.MPS:
                # Check MPS-specific issues
                validation['mps_fallback_enabled'] = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'
            
        except Exception as e:
            validation['validation_error'] = str(e)
            self.logger.error(f"Configuration validation failed: {e}")
        
        return validation
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get comprehensive status report of hardware configuration.
        
        Returns:
            Dict[str, Any]: Status report
        """
        device_info = self.get_compute_device()
        sdr_info = self.get_sdr_interface()
        optimization = self.optimize_for_platform()
        validation = self.validate_configuration()
        
        return {
            'platform': self._platform_info.platform if self._platform_info else "unknown",
            'compute_device': {
                'type': device_info.device_type.value,
                'name': device_info.name,
                'memory_mb': device_info.memory_mb,
                'device_string': self.get_device_string()
            },
            'sdr_configuration': {
                'mode': sdr_info.mode.value,
                'devices': sdr_info.devices,
                'sample_rates': sdr_info.sample_rates,
                'max_bandwidth': sdr_info.max_bandwidth
            },
            'optimization_settings': optimization,
            'validation_results': validation,
            'capabilities': {
                'cpu_cores': self._capabilities.cpu_cores if self._capabilities else None,
                'memory_mb': self._capabilities.max_memory if self._capabilities else None,
                'compute_devices': self._capabilities.compute_devices if self._capabilities else []
            }
        }


def main():
    """
    Main function for testing hardware abstraction.
    """
    print("=== Hardware Abstraction Layer Test ===")
    
    # Test with auto-detection
    hw = HardwareAbstraction()
    
    print("\n=== Compute Device Detection ===")
    device_info = hw.get_compute_device()
    print(f"Device Type: {device_info.device_type.value}")
    print(f"Device Name: {device_info.name}")
    print(f"Device String: {hw.get_device_string()}")
    print(f"Memory: {device_info.memory_mb} MB" if device_info.memory_mb else "Memory: Unknown")
    
    print("\n=== SDR Interface Detection ===")
    sdr_info = hw.get_sdr_interface()
    print(f"SDR Mode: {sdr_info.mode.value}")
    print(f"Devices: {sdr_info.devices}")
    print(f"Sample Rates: {sdr_info.sample_rates}")
    print(f"Max Bandwidth: {sdr_info.max_bandwidth} Hz" if sdr_info.max_bandwidth else "Max Bandwidth: Unknown")
    
    print("\n=== Platform Optimizations ===")
    optimizations = hw.optimize_for_platform()
    for key, value in optimizations.items():
        print(f"{key}: {value}")
    
    print("\n=== Configuration Validation ===")
    validation = hw.validate_configuration()
    for check, result in validation.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}: {result}")
    
    print("\n=== Status Report ===")
    status = hw.get_status_report()
    print(f"Platform: {status['platform']}")
    print(f"Compute: {status['compute_device']['type']} ({status['compute_device']['name']})")
    print(f"SDR: {status['sdr_configuration']['mode']} ({', '.join(status['sdr_configuration']['devices'])})")


if __name__ == "__main__":
    main()