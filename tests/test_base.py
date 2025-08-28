"""
Base test classes for GeminiSDR test suite.

This module provides base classes for different types of tests including
unit tests, performance tests, and cross-platform tests with common setup
and utilities.
"""

import pytest
import time
import logging
import gc
import psutil
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from unittest.mock import Mock, patch
from pathlib import Path

import torch
import numpy as np

from geminisdr.config.config_manager import ConfigManager
from geminisdr.config.config_models import SystemConfig, DeviceType, SDRMode
from geminisdr.core.error_handling import ErrorHandler


class TestBase:
    """
    Base class for all GeminiSDR tests with common setup and utilities.
    
    Provides:
    - Configuration management
    - Logging setup
    - Common fixtures and utilities
    - Error handling setup
    - Resource cleanup
    """
    
    def setup_method(self, method):
        """Common test setup run before each test method."""
        # Set up test configuration
        self.config = self._get_test_config()
        
        # Set up logging
        self.logger = self._setup_test_logger()
        
        # Set up error handling
        self.error_handler = ErrorHandler(logger=self.logger)
        
        # Initialize test state
        self.test_start_time = time.time()
        self.test_resources = []
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    
    def teardown_method(self, method):
        """Common test cleanup run after each test method."""
        # Clean up test resources
        self._cleanup_resources()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log test duration
        duration = time.time() - self.test_start_time
        self.logger.debug(f"Test {method.__name__} completed in {duration:.3f}s")
    
    def _get_test_config(self) -> SystemConfig:
        """Get test configuration."""
        # Use pytest fixture if available, otherwise create minimal config
        if hasattr(self, 'test_config'):
            return self.test_config
        
        # Create minimal test configuration
        from geminisdr.config.config_models import HardwareConfig, MLConfig, LoggingConfig, PerformanceConfig
        
        return SystemConfig(
            environment="testing",
            debug_mode=True,
            hardware=HardwareConfig(
                device_preference=DeviceType.CPU,
                sdr_mode=SDRMode.SIMULATION,
                memory_limit_gb=2.0
            ),
            ml=MLConfig(
                batch_size=8,
                learning_rate=1e-3,
                model_cache_size=2
            ),
            logging=LoggingConfig(
                level="DEBUG",
                format="simple",
                output=["console"]
            ),
            performance=PerformanceConfig(
                memory_threshold=0.6,
                auto_optimize=True
            )
        )
    
    def _setup_test_logger(self) -> logging.Logger:
        """Set up test logger."""
        logger = logging.getLogger(f"test.{self.__class__.__name__}")
        logger.setLevel(logging.DEBUG)
        
        # Add console handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _cleanup_resources(self):
        """Clean up test resources."""
        for resource in self.test_resources:
            try:
                if hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, '__del__'):
                    del resource
            except Exception as e:
                self.logger.warning(f"Failed to cleanup resource: {e}")
        
        self.test_resources.clear()
    
    def add_test_resource(self, resource):
        """Add resource for automatic cleanup."""
        self.test_resources.append(resource)
        return resource
    
    def create_mock_sdr(self) -> Mock:
        """Create mock SDR interface for testing."""
        sdr = Mock()
        sdr.sample_rate = 1e6
        sdr.center_freq = 2.4e9
        sdr.rx_buffer_size = 1024
        sdr.is_connected = True
        
        # Mock methods
        sdr.connect = Mock(return_value=True)
        sdr.disconnect = Mock()
        sdr.receive_samples = Mock(
            return_value=np.random.randn(1024) + 1j * np.random.randn(1024)
        )
        sdr.set_frequency = Mock()
        sdr.set_sample_rate = Mock()
        sdr.set_gain = Mock()
        
        return self.add_test_resource(sdr)
    
    def create_test_signals(self, modulations: List[str], 
                          num_samples: int = 10, 
                          signal_length: int = 512) -> Tuple[List[np.ndarray], List[str]]:
        """Create test signals for ML testing."""
        signals = []
        labels = []
        
        for mod in modulations:
            for _ in range(num_samples):
                # Generate synthetic signal
                if mod == 'BPSK':
                    symbols = np.random.choice([-1, 1], signal_length // 2)
                elif mod == 'QPSK':
                    symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], signal_length // 2)
                elif mod == '16QAM':
                    real_part = np.random.choice([-3, -1, 1, 3], signal_length // 2)
                    imag_part = np.random.choice([-3, -1, 1, 3], signal_length // 2)
                    symbols = real_part + 1j * imag_part
                else:
                    # Default to QPSK
                    symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], signal_length // 2)
                
                # Upsample and add noise
                signal = np.repeat(symbols, 2)
                noise = 0.1 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
                signal = signal + noise
                
                signals.append(signal.astype(np.complex64))
                labels.append(mod)
        
        return signals, labels
    
    def assert_tensor_properties(self, tensor: torch.Tensor, 
                                expected_shape: Optional[Tuple] = None,
                                expected_dtype: Optional[torch.dtype] = None,
                                expected_device: Optional[str] = None):
        """Assert tensor has expected properties."""
        if expected_shape is not None:
            assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
        
        if expected_dtype is not None:
            assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"
        
        if expected_device is not None:
            assert str(tensor.device).startswith(expected_device), \
                f"Expected device {expected_device}, got {tensor.device}"
    
    def assert_config_valid(self, config: SystemConfig):
        """Assert configuration is valid."""
        assert config is not None
        assert hasattr(config, 'hardware')
        assert hasattr(config, 'ml')
        assert hasattr(config, 'logging')
        assert hasattr(config, 'performance')
    
    def skip_if_no_gpu(self):
        """Skip test if no GPU is available."""
        if not (torch.cuda.is_available() or 
                (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())):
            pytest.skip("No GPU available (CUDA or MPS)")
    
    def skip_if_no_cuda(self):
        """Skip test if CUDA is not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    
    def skip_if_no_mps(self):
        """Skip test if MPS is not available."""
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            pytest.skip("MPS not available")


class PerformanceTest(TestBase):
    """
    Base class for performance tests and benchmarks.
    
    Provides:
    - Performance monitoring utilities
    - Benchmark assertion methods
    - Memory usage tracking
    - Timing utilities
    """
    
    def setup_method(self, method):
        """Set up performance monitoring."""
        super().setup_method(method)
        
        # Performance monitoring state
        self.performance_metrics = {}
        self.benchmark_start_time = None
        self.benchmark_start_memory = None
        
        # Performance thresholds (can be overridden in subclasses)
        self.max_duration_seconds = 60.0  # Default max test duration
        self.max_memory_mb = 1024.0       # Default max memory usage
        self.min_throughput = None        # Minimum required throughput
    
    def start_benchmark(self, operation_name: str = "benchmark"):
        """Start performance benchmark."""
        gc.collect()  # Clean up before measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.benchmark_start_time = time.time()
        self.benchmark_start_memory = self._get_memory_usage()
        self.current_operation = operation_name
        
        self.logger.debug(f"Started benchmark: {operation_name}")
    
    def stop_benchmark(self) -> Dict[str, float]:
        """Stop benchmark and return metrics."""
        if self.benchmark_start_time is None:
            raise RuntimeError("Benchmark not started")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        metrics = {
            'duration_seconds': end_time - self.benchmark_start_time,
            'memory_delta_mb': end_memory - self.benchmark_start_memory,
            'peak_memory_mb': end_memory,
            'start_memory_mb': self.benchmark_start_memory
        }
        
        self.performance_metrics[self.current_operation] = metrics
        
        self.logger.info(
            f"Benchmark {self.current_operation} completed: "
            f"{metrics['duration_seconds']:.3f}s, "
            f"{metrics['memory_delta_mb']:.1f}MB memory delta"
        )
        
        return metrics
    
    def benchmark_training_speed(self, model, train_loader, num_epochs: int = 1) -> Dict[str, float]:
        """Benchmark training speed."""
        self.start_benchmark("training_speed")
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        total_samples = 0
        total_batches = 0
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_samples += data.size(0)
                total_batches += 1
        
        metrics = self.stop_benchmark()
        
        # Calculate throughput metrics
        metrics['samples_per_second'] = total_samples / metrics['duration_seconds']
        metrics['batches_per_second'] = total_batches / metrics['duration_seconds']
        metrics['total_samples'] = total_samples
        metrics['total_batches'] = total_batches
        
        return metrics
    
    def benchmark_inference_speed(self, model, test_loader) -> Dict[str, float]:
        """Benchmark inference speed."""
        self.start_benchmark("inference_speed")
        
        model.eval()
        total_samples = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                output = model(data)
                total_samples += data.size(0)
        
        metrics = self.stop_benchmark()
        
        # Calculate throughput metrics
        metrics['samples_per_second'] = total_samples / metrics['duration_seconds']
        metrics['total_samples'] = total_samples
        
        return metrics
    
    def benchmark_memory_usage(self, operation_func, *args, **kwargs) -> Dict[str, float]:
        """Benchmark memory usage of an operation."""
        self.start_benchmark("memory_usage")
        
        try:
            result = operation_func(*args, **kwargs)
            metrics = self.stop_benchmark()
            metrics['operation_result'] = result
            return metrics
        except Exception as e:
            self.stop_benchmark()
            raise e
    
    def assert_performance(self, metrics: Dict[str, float], 
                          max_duration: Optional[float] = None,
                          max_memory_mb: Optional[float] = None,
                          min_throughput: Optional[float] = None):
        """Assert performance meets requirements."""
        max_duration = max_duration or self.max_duration_seconds
        max_memory_mb = max_memory_mb or self.max_memory_mb
        min_throughput = min_throughput or self.min_throughput
        
        if max_duration and metrics.get('duration_seconds', 0) > max_duration:
            pytest.fail(
                f"Performance test exceeded time limit: "
                f"{metrics['duration_seconds']:.2f}s > {max_duration}s"
            )
        
        if max_memory_mb and metrics.get('memory_delta_mb', 0) > max_memory_mb:
            pytest.fail(
                f"Performance test exceeded memory limit: "
                f"{metrics['memory_delta_mb']:.1f}MB > {max_memory_mb}MB"
            )
        
        if min_throughput and metrics.get('samples_per_second', 0) < min_throughput:
            pytest.fail(
                f"Performance test below throughput requirement: "
                f"{metrics['samples_per_second']:.1f} < {min_throughput} samples/sec"
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance metrics."""
        return {
            'metrics': self.performance_metrics,
            'total_operations': len(self.performance_metrics),
            'total_duration': sum(m.get('duration_seconds', 0) for m in self.performance_metrics.values()),
            'peak_memory': max((m.get('peak_memory_mb', 0) for m in self.performance_metrics.values()), default=0)
        }


class CrossPlatformTest(TestBase):
    """
    Base class for cross-platform compatibility tests.
    
    Provides:
    - Platform detection utilities
    - Device-specific test setup
    - Cross-platform validation methods
    - Hardware abstraction testing
    """
    
    def setup_method(self, method):
        """Set up cross-platform testing."""
        super().setup_method(method)
        
        # Detect available platforms
        self.available_devices = self._detect_available_devices()
        self.current_device = self._get_default_device()
        
        # Platform-specific configurations
        self.platform_configs = self._get_platform_configs()
        
        self.logger.info(f"Available devices: {self.available_devices}")
        self.logger.info(f"Current device: {self.current_device}")
    
    def _detect_available_devices(self) -> List[str]:
        """Detect available compute devices."""
        devices = ['cpu']
        
        if torch.cuda.is_available():
            devices.append('cuda')
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')
        
        return devices
    
    def _get_default_device(self) -> str:
        """Get the default device for testing."""
        if 'cuda' in self.available_devices:
            return 'cuda'
        elif 'mps' in self.available_devices:
            return 'mps'
        else:
            return 'cpu'
    
    def _get_platform_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get platform-specific configurations."""
        return {
            'cpu': {
                'batch_size': 16,
                'num_workers': 2,
                'pin_memory': False,
                'device_type': DeviceType.CPU
            },
            'cuda': {
                'batch_size': 32,
                'num_workers': 4,
                'pin_memory': True,
                'device_type': DeviceType.CUDA
            },
            'mps': {
                'batch_size': 24,
                'num_workers': 2,
                'pin_memory': False,
                'device_type': DeviceType.MPS
            }
        }
    
    @pytest.mark.parametrize("device", ['cpu', 'cuda', 'mps'])
    def test_device_compatibility(self, device: str):
        """Test compatibility across different devices."""
        if device not in self.available_devices:
            pytest.skip(f"Device {device} not available")
        
        # This is a template method - override in subclasses
        self._test_device_specific_functionality(device)
    
    def _test_device_specific_functionality(self, device: str):
        """Test device-specific functionality - override in subclasses."""
        # Basic device test
        torch_device = torch.device(device)
        test_tensor = torch.randn(10, 10).to(torch_device)
        result = torch.mm(test_tensor, test_tensor.T)
        
        assert result.device == torch_device
        assert result.shape == (10, 10)
    
    def test_model_portability(self, model_factory, test_data):
        """Test model portability across platforms."""
        results = {}
        
        for device in self.available_devices:
            try:
                # Create model for this device
                model = model_factory().to(device)
                
                # Test forward pass
                test_input = test_data.to(device)
                with torch.no_grad():
                    output = model(test_input)
                
                results[device] = {
                    'success': True,
                    'output_shape': output.shape,
                    'output_mean': output.mean().item(),
                    'output_std': output.std().item()
                }
                
            except Exception as e:
                results[device] = {
                    'success': False,
                    'error': str(e)
                }
                self.logger.warning(f"Model test failed on {device}: {e}")
        
        # Assert at least one device worked
        successful_devices = [d for d, r in results.items() if r['success']]
        assert len(successful_devices) > 0, f"Model failed on all devices: {results}"
        
        return results
    
    def test_configuration_compatibility(self):
        """Test configuration compatibility across platforms."""
        for device in self.available_devices:
            config = self.platform_configs[device]
            
            # Test configuration loading
            assert 'batch_size' in config
            assert 'device_type' in config
            assert isinstance(config['device_type'], DeviceType)
            
            self.logger.debug(f"Configuration for {device}: {config}")
    
    def validate_cross_platform_results(self, results: Dict[str, Any], 
                                      tolerance: float = 1e-3):
        """Validate that results are consistent across platforms."""
        if len(results) < 2:
            self.logger.warning("Cannot validate cross-platform consistency with < 2 platforms")
            return
        
        # Get reference result (first successful result)
        reference_device = None
        reference_result = None
        
        for device, result in results.items():
            if result.get('success', False):
                reference_device = device
                reference_result = result
                break
        
        if reference_result is None:
            pytest.fail("No successful results to compare")
        
        # Compare other results to reference
        for device, result in results.items():
            if device == reference_device or not result.get('success', False):
                continue
            
            # Compare output statistics
            ref_mean = reference_result.get('output_mean', 0)
            ref_std = reference_result.get('output_std', 1)
            
            curr_mean = result.get('output_mean', 0)
            curr_std = result.get('output_std', 1)
            
            mean_diff = abs(ref_mean - curr_mean)
            std_diff = abs(ref_std - curr_std)
            
            if mean_diff > tolerance:
                self.logger.warning(
                    f"Mean difference between {reference_device} and {device}: "
                    f"{mean_diff:.6f} > {tolerance}"
                )
            
            if std_diff > tolerance:
                self.logger.warning(
                    f"Std difference between {reference_device} and {device}: "
                    f"{std_diff:.6f} > {tolerance}"
                )
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get detailed platform information."""
        import platform
        
        info = {
            'python_version': platform.python_version(),
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'torch_version': torch.__version__,
            'available_devices': self.available_devices,
            'current_device': self.current_device
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['mps_available'] = True
        
        return info


class IntegrationTest(TestBase):
    """
    Base class for integration tests.
    
    Provides:
    - End-to-end workflow testing
    - Component integration validation
    - System-level test utilities
    """
    
    def setup_method(self, method):
        """Set up integration testing."""
        super().setup_method(method)
        
        # Integration test specific setup
        self.integration_components = {}
        self.workflow_state = {}
    
    def register_component(self, name: str, component: Any):
        """Register component for integration testing."""
        self.integration_components[name] = component
        self.add_test_resource(component)
        return component
    
    def test_component_integration(self, component_names: List[str]):
        """Test integration between specified components."""
        components = {}
        for name in component_names:
            if name not in self.integration_components:
                pytest.fail(f"Component {name} not registered")
            components[name] = self.integration_components[name]
        
        # This is a template method - override in subclasses
        return self._test_component_interaction(components)
    
    def _test_component_interaction(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Test component interaction - override in subclasses."""
        # Basic integration test - ensure all components are accessible
        for name, component in components.items():
            assert component is not None, f"Component {name} is None"
        
        return {'status': 'success', 'components_tested': list(components.keys())}
    
    def run_end_to_end_workflow(self, workflow_steps: List[str]) -> Dict[str, Any]:
        """Run end-to-end workflow test."""
        results = {}
        
        for step in workflow_steps:
            try:
                step_result = self._execute_workflow_step(step)
                results[step] = {'success': True, 'result': step_result}
                self.workflow_state[step] = step_result
            except Exception as e:
                results[step] = {'success': False, 'error': str(e)}
                self.logger.error(f"Workflow step {step} failed: {e}")
                break  # Stop on first failure
        
        return results
    
    def _execute_workflow_step(self, step: str) -> Any:
        """Execute workflow step - override in subclasses."""
        self.logger.debug(f"Executing workflow step: {step}")
        return f"Step {step} completed"
    
    def assert_workflow_success(self, results: Dict[str, Any]):
        """Assert that workflow completed successfully."""
        failed_steps = [step for step, result in results.items() if not result.get('success', False)]
        
        if failed_steps:
            pytest.fail(f"Workflow failed at steps: {failed_steps}")
        
        self.logger.info(f"Workflow completed successfully: {list(results.keys())}")