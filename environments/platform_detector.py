"""
Platform detection system for dual environment setup.

This module provides automatic platform detection, hardware capability detection,
and platform validation functions for M1 Mac and Ubuntu VM environments.
"""

import platform
import sys
import os
import subprocess
import psutil
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import importlib.util


@dataclass
class PlatformInfo:
    """Information about the detected platform."""
    platform: str           # 'm1_native' | 'vm_ubuntu' | 'unknown'
    architecture: str       # 'arm64' | 'x86_64'
    has_mps: bool          # Metal Performance Shaders available
    has_sdr: bool          # Physical SDR hardware available
    python_version: str    # Python version string
    recommended_packages: List[str]  # Platform-specific packages


@dataclass
class PlatformCapabilities:
    """Hardware capabilities of the detected platform."""
    compute_devices: List[str]   # Available compute devices
    max_memory: int             # Maximum available memory in MB
    cpu_cores: int              # Number of CPU cores
    gpu_memory: Optional[int]   # GPU memory if available in MB
    sdr_hardware: List[str]     # Available SDR devices
    supported_sample_rates: List[int]  # Supported sample rates


class PlatformDetector:
    """
    Platform detection and validation system.
    
    Automatically detects M1 Mac vs Ubuntu VM environments and provides
    hardware capability information for optimal environment setup.
    """
    
    def __init__(self):
        self._platform_info = None
        self._capabilities = None
    
    def detect_platform(self) -> str:
        """
        Detect the current platform.
        
        Returns:
            str: 'm1_native', 'vm_ubuntu', or 'unknown'
        """
        system = platform.system()
        machine = platform.machine()
        
        if system == "Darwin" and machine == "arm64":
            return "m1_native"
        elif system == "Linux" and machine == "x86_64":
            # Additional check to distinguish VM from native Linux
            if self._is_vm_environment():
                return "vm_ubuntu"
            else:
                return "vm_ubuntu"  # Treat all Linux x86_64 as VM for this project
        else:
            return "unknown"
    
    def _is_vm_environment(self) -> bool:
        """
        Check if running in a virtual machine environment.
        
        Returns:
            bool: True if running in VM
        """
        try:
            # Check for common VM indicators
            vm_indicators = [
                "/proc/cpuinfo",
                "/sys/class/dmi/id/product_name",
                "/sys/class/dmi/id/sys_vendor"
            ]
            
            for indicator_file in vm_indicators:
                if os.path.exists(indicator_file):
                    try:
                        with open(indicator_file, 'r') as f:
                            content = f.read().lower()
                            if any(vm_name in content for vm_name in 
                                   ['vmware', 'virtualbox', 'qemu', 'kvm', 'xen', 'hyperv']):
                                return True
                    except (IOError, PermissionError):
                        continue
            
            # Check for VM-specific environment variables
            vm_env_vars = ['VIRTUAL_ENV', 'VM_NAME', 'VMWARE_TOOLS']
            for var in vm_env_vars:
                if os.environ.get(var):
                    return True
                    
            return False
            
        except Exception:
            # If we can't determine, assume VM for safety
            return True
    
    def _check_mps_availability(self) -> bool:
        """
        Check if Metal Performance Shaders (MPS) is available.
        
        Returns:
            bool: True if MPS is available
        """
        try:
            # Check if PyTorch is available and has MPS support
            torch_spec = importlib.util.find_spec("torch")
            if torch_spec is None:
                return False
                
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
        except (ImportError, AttributeError):
            # If PyTorch not installed or no MPS support, check system
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                # M1 Macs should support MPS, but we can't verify without PyTorch
                return True
            return False
    
    def _detect_sdr_hardware(self) -> Tuple[bool, List[str]]:
        """
        Detect available SDR hardware.
        
        Returns:
            Tuple[bool, List[str]]: (has_sdr, list_of_devices)
        """
        sdr_devices = []
        
        try:
            # Check for common SDR devices via USB
            if os.path.exists('/dev'):
                # Look for common SDR device patterns
                import glob
                
                # RTL-SDR devices
                rtl_devices = glob.glob('/dev/bus/usb/*/*')
                # This is a simplified check - in practice, you'd check USB vendor/product IDs
                
                # PlutoSDR devices (if connected via USB)
                pluto_devices = glob.glob('/dev/ttyACM*')
                
                # HackRF devices
                hackrf_devices = glob.glob('/dev/ttyUSB*')
                
                if rtl_devices:
                    sdr_devices.append('RTL-SDR')
                if pluto_devices:
                    sdr_devices.append('PlutoSDR')
                if hackrf_devices:
                    sdr_devices.append('HackRF')
            
            # Try to detect via software libraries if available
            try:
                # Check if SoapySDR is available
                soapy_spec = importlib.util.find_spec("SoapySDR")
                if soapy_spec is not None:
                    import SoapySDR
                    devices = SoapySDR.Device.enumerate()
                    for device in devices:
                        if 'driver' in device:
                            sdr_devices.append(device['driver'])
            except ImportError:
                pass
                
        except Exception:
            # If detection fails, assume no hardware
            pass
        
        return len(sdr_devices) > 0, sdr_devices
    
    def get_hardware_capabilities(self) -> PlatformCapabilities:
        """
        Get detailed hardware capabilities.
        
        Returns:
            PlatformCapabilities: Hardware capability information
        """
        if self._capabilities is not None:
            return self._capabilities
        
        # Get basic system info
        cpu_cores = psutil.cpu_count(logical=False) or 1
        memory_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        
        # Detect compute devices
        compute_devices = ['cpu']
        gpu_memory = None
        
        if self._check_mps_availability():
            compute_devices.append('mps')
            # M1 Macs have unified memory, so GPU memory is shared with system
            gpu_memory = memory_mb
        
        # Check for CUDA (unlikely in our target environments but good to check)
        try:
            cuda_spec = importlib.util.find_spec("torch")
            if cuda_spec is not None:
                import torch
                if torch.cuda.is_available():
                    compute_devices.append('cuda')
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except (ImportError, AttributeError):
            pass
        
        # Detect SDR hardware
        has_sdr, sdr_devices = self._detect_sdr_hardware()
        
        # Determine supported sample rates based on platform
        if has_sdr:
            # Real hardware - typical SDR sample rates
            sample_rates = [250000, 1000000, 2000000, 2400000, 3200000]
        else:
            # Simulation mode - flexible sample rates
            sample_rates = [250000, 1000000, 2000000]
        
        self._capabilities = PlatformCapabilities(
            compute_devices=compute_devices,
            max_memory=memory_mb,
            cpu_cores=cpu_cores,
            gpu_memory=gpu_memory,
            sdr_hardware=sdr_devices,
            supported_sample_rates=sample_rates
        )
        
        return self._capabilities
    
    def get_platform_info(self) -> PlatformInfo:
        """
        Get comprehensive platform information.
        
        Returns:
            PlatformInfo: Complete platform information
        """
        if self._platform_info is not None:
            return self._platform_info
        
        platform_type = self.detect_platform()
        architecture = platform.machine()
        has_mps = self._check_mps_availability()
        has_sdr, _ = self._detect_sdr_hardware()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Get platform-specific package recommendations
        recommended_packages = self._get_recommended_packages(platform_type, has_mps)
        
        self._platform_info = PlatformInfo(
            platform=platform_type,
            architecture=architecture,
            has_mps=has_mps,
            has_sdr=has_sdr,
            python_version=python_version,
            recommended_packages=recommended_packages
        )
        
        return self._platform_info
    
    def _get_recommended_packages(self, platform_type: str, has_mps: bool) -> List[str]:
        """
        Get recommended packages for the platform.
        
        Args:
            platform_type: The detected platform type
            has_mps: Whether MPS is available
            
        Returns:
            List[str]: Recommended package specifications
        """
        if platform_type == "m1_native":
            packages = [
                "torch>=2.0.0",  # MPS-enabled PyTorch
                "numpy>=1.24.0",  # Apple Silicon optimized
                "scipy>=1.10.0",
                "scikit-learn>=1.2.0",
                "matplotlib>=3.6.0",
                "psutil>=5.9.0"
            ]
            if has_mps:
                packages.append("accelerate>=0.20.0")  # Hugging Face accelerate for MPS
        elif platform_type == "vm_ubuntu":
            packages = [
                "torch>=2.0.0+cpu",  # CPU-only PyTorch
                "numpy<2.0",  # Stable NumPy for Ubuntu
                "scipy>=1.9.0",
                "scikit-learn>=1.1.0",
                "matplotlib>=3.5.0",
                "psutil>=5.8.0"
            ]
        else:
            # Unknown platform - conservative package list
            packages = [
                "torch>=2.0.0",
                "numpy>=1.21.0",
                "scipy>=1.8.0",
                "scikit-learn>=1.0.0",
                "matplotlib>=3.4.0",
                "psutil>=5.7.0"
            ]
        
        return packages
    
    def validate_environment(self) -> Dict[str, bool]:
        """
        Validate the current environment for compatibility.
        
        Returns:
            Dict[str, bool]: Validation results for different aspects
        """
        validation_results = {}
        
        try:
            # Check Python version compatibility
            python_version = sys.version_info
            validation_results['python_version'] = python_version >= (3, 8)
            
            # Check platform detection
            platform_type = self.detect_platform()
            validation_results['platform_detected'] = platform_type != "unknown"
            
            # Check memory availability (minimum 4GB)
            memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            validation_results['sufficient_memory'] = memory_mb >= 4096
            
            # Check CPU cores (minimum 2)
            cpu_cores = psutil.cpu_count(logical=False) or 1
            validation_results['sufficient_cpu'] = cpu_cores >= 2
            
            # Check disk space (minimum 2GB free)
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024 * 1024 * 1024)
            validation_results['sufficient_disk'] = free_gb >= 2.0
            
            # Platform-specific validations
            if platform_type == "m1_native":
                validation_results['mps_available'] = self._check_mps_availability()
            elif platform_type == "vm_ubuntu":
                validation_results['vm_environment'] = self._is_vm_environment()
            
        except Exception as e:
            # If validation fails, mark as invalid
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def get_optimal_config(self) -> Dict[str, any]:
        """
        Get optimal configuration for the detected platform.
        
        Returns:
            Dict[str, any]: Platform-specific configuration
        """
        platform_info = self.get_platform_info()
        capabilities = self.get_hardware_capabilities()
        
        config = {
            'platform': platform_info.platform,
            'device': 'mps' if platform_info.has_mps else 'cpu',
            'threads': min(capabilities.cpu_cores, 8),  # Cap at 8 threads
            'memory_fraction': 0.8 if platform_info.platform == "m1_native" else 0.6,
            'batch_size_multiplier': 2 if platform_info.has_mps else 1,
            'mixed_precision': platform_info.has_mps,
            'simulation_mode': not platform_info.has_sdr
        }
        
        return config


def main():
    """
    Main function for testing platform detection.
    """
    detector = PlatformDetector()
    
    print("=== Platform Detection Results ===")
    platform_info = detector.get_platform_info()
    print(f"Platform: {platform_info.platform}")
    print(f"Architecture: {platform_info.architecture}")
    print(f"MPS Available: {platform_info.has_mps}")
    print(f"SDR Hardware: {platform_info.has_sdr}")
    print(f"Python Version: {platform_info.python_version}")
    
    print("\n=== Hardware Capabilities ===")
    capabilities = detector.get_hardware_capabilities()
    print(f"Compute Devices: {capabilities.compute_devices}")
    print(f"CPU Cores: {capabilities.cpu_cores}")
    print(f"Memory: {capabilities.max_memory} MB")
    print(f"GPU Memory: {capabilities.gpu_memory} MB" if capabilities.gpu_memory else "GPU Memory: N/A")
    print(f"SDR Devices: {capabilities.sdr_hardware}")
    
    print("\n=== Environment Validation ===")
    validation = detector.validate_environment()
    for check, result in validation.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}: {result}")
    
    print("\n=== Optimal Configuration ===")
    config = detector.get_optimal_config()
    for key, value in config.items():
        print(f"{key}: {value}")
    
    print("\n=== Recommended Packages ===")
    for package in platform_info.recommended_packages:
        print(f"  - {package}")


if __name__ == "__main__":
    main()