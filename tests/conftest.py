"""
pytest configuration and shared fixtures for GeminiSDR test suite.

This module provides common test fixtures, configuration, and utilities
for all GeminiSDR tests, including platform-specific setup and mock objects.
"""

import pytest
import tempfile
import shutil
import logging
import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional, Generator

import torch
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from geminisdr.config.config_manager import ConfigManager
from geminisdr.config.config_models import SystemConfig, HardwareConfig, MLConfig, DeviceType, SDRMode


# Test configuration constants
TEST_CONFIG_DIR = "test_configs"
TEST_DATA_DIR = "test_data"
TEST_MODELS_DIR = "test_models"


@pytest.fixture(scope="session")
def test_config_dir():
    """Create temporary configuration directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="geminisdr_test_")
    config_dir = Path(temp_dir) / "conf"
    config_dir.mkdir(parents=True)
    
    # Create basic test configuration
    (config_dir / "config.yaml").write_text("""
environment: testing
debug_mode: true
config_version: "1.0"

hardware:
  device_preference: cpu
  sdr_mode: simulation
  memory_limit_gb: 2.0

ml:
  batch_size: 8
  learning_rate: 1e-3
  model_cache_size: 2
  checkpoint_frequency: 10

logging:
  level: DEBUG
  format: simple
  output: [console]

performance:
  memory_threshold: 0.6
  auto_optimize: true
""")
    
    # Create hardware profiles
    hardware_dir = config_dir / "hardware"
    hardware_dir.mkdir()
    
    (hardware_dir / "cpu.yaml").write_text("""
# @package hardware
device_preference: cpu
sdr_mode: simulation
memory_limit_gb: 2.0
""")
    
    (hardware_dir / "cuda.yaml").write_text("""
# @package hardware
device_preference: cuda
sdr_mode: simulation
memory_limit_gb: 4.0
""")
    
    (hardware_dir / "mps.yaml").write_text("""
# @package hardware
device_preference: mps
sdr_mode: simulation
memory_limit_gb: 8.0
""")
    
    # Create environment configs
    env_dir = config_dir / "environments"
    env_dir.mkdir()
    
    (env_dir / "testing.yaml").write_text("""
# @package _global_
environment: testing
debug_mode: true

logging:
  level: DEBUG
""")
    
    yield str(config_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(test_config_dir):
    """Provide test configuration manager."""
    config_manager = ConfigManager(config_dir=test_config_dir)
    config = config_manager.load_config("config")
    return config


@pytest.fixture
def mock_logger():
    """Provide mock logger for testing."""
    logger = Mock(spec=logging.Logger)
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    logger.log = Mock()
    return logger


@pytest.fixture
def mock_sdr_interface():
    """Provide mock SDR interface for testing."""
    sdr = Mock()
    sdr.sample_rate = 1e6
    sdr.center_freq = 2.4e9
    sdr.rx_buffer_size = 1024
    sdr.is_connected = True
    
    # Mock methods
    sdr.connect = Mock(return_value=True)
    sdr.disconnect = Mock()
    sdr.receive_samples = Mock(return_value=np.random.randn(1024) + 1j * np.random.randn(1024))
    sdr.set_frequency = Mock()
    sdr.set_sample_rate = Mock()
    sdr.set_gain = Mock()
    
    return sdr


@pytest.fixture
def test_signal_data():
    """Provide test signal data for ML testing."""
    np.random.seed(42)  # Reproducible test data
    
    # Generate test signals for different modulations
    signals = {}
    modulations = ['BPSK', 'QPSK', '16QAM', '64QAM']
    
    for mod in modulations:
        # Generate 10 test signals per modulation
        mod_signals = []
        for _ in range(10):
            # Simple synthetic signal generation
            num_symbols = 256
            if mod == 'BPSK':
                symbols = np.random.choice([-1, 1], num_symbols)
            elif mod == 'QPSK':
                symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], num_symbols)
            elif mod == '16QAM':
                # Simplified 16QAM constellation
                real_part = np.random.choice([-3, -1, 1, 3], num_symbols)
                imag_part = np.random.choice([-3, -1, 1, 3], num_symbols)
                symbols = real_part + 1j * imag_part
            else:  # 64QAM
                # Simplified 64QAM constellation
                real_part = np.random.choice([-7, -5, -3, -1, 1, 3, 5, 7], num_symbols)
                imag_part = np.random.choice([-7, -5, -3, -1, 1, 3, 5, 7], num_symbols)
                symbols = real_part + 1j * imag_part
            
            # Add noise and convert to samples (2 samples per symbol)
            signal = np.repeat(symbols, 2)
            noise = 0.1 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
            signal = signal + noise
            
            mod_signals.append(signal.astype(np.complex64))
        
        signals[mod] = mod_signals
    
    return signals


@pytest.fixture
def test_model_metadata():
    """Provide test model metadata."""
    from datetime import datetime
    from geminisdr.core.model_metadata import ModelMetadata
    
    return ModelMetadata(
        name="test_model",
        version="1.0.0",
        timestamp=datetime.now(),
        hyperparameters={
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 10
        },
        performance_metrics={
            "accuracy": 0.95,
            "loss": 0.05,
            "f1_score": 0.94
        },
        training_data_hash="test_hash_123",
        code_version="test_version",
        platform="test_platform",
        device="cpu"
    )


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model testing."""
    temp_dir = tempfile.mkdtemp(prefix="geminisdr_models_")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_torch_model():
    """Provide mock PyTorch model for testing."""
    model = Mock()
    model.state_dict = Mock(return_value={'layer.weight': torch.randn(10, 5)})
    model.load_state_dict = Mock()
    model.eval = Mock()
    model.train = Mock()
    model.to = Mock(return_value=model)
    model.parameters = Mock(return_value=[torch.randn(10, 5, requires_grad=True)])
    
    # Mock forward pass
    def mock_forward(x):
        batch_size = x.shape[0] if hasattr(x, 'shape') else 1
        return torch.randn(batch_size, 4)  # 4 classes
    
    model.forward = Mock(side_effect=mock_forward)
    model.__call__ = Mock(side_effect=mock_forward)
    
    return model


@pytest.fixture
def mock_hardware_abstraction():
    """Provide mock hardware abstraction for testing."""
    try:
        from geminisdr.environments.hardware_abstraction import HardwareAbstraction, DeviceInfo, SDRInfo
        
        hw = Mock(spec=HardwareAbstraction)
        
        # Mock device info
        device_info = Mock(spec=DeviceInfo)
        device_info.device_type = DeviceType.CPU
        device_info.name = "Test CPU"
        device_info.memory_mb = 8192
        device_info.is_available = True
        
        # Mock SDR info
        sdr_info = Mock(spec=SDRInfo)
        sdr_info.mode = SDRMode.SIMULATION
        sdr_info.devices = ["simulation"]
        sdr_info.is_available = True
        
        # Mock methods
        hw.get_compute_device = Mock(return_value=device_info)
        hw.get_sdr_interface = Mock(return_value=sdr_info)
        hw.optimize_for_platform = Mock(return_value={
            'batch_size': 16,
            'num_workers': 2,
            'pin_memory': False
        })
        hw.validate_configuration = Mock(return_value={
            'device_available': True,
            'memory_sufficient': True,
            'sdr_accessible': True
        })
        
        return hw
        
    except ImportError:
        # Return basic mock if hardware abstraction not available
        hw = Mock()
        hw.get_compute_device = Mock(return_value=Mock(device_type=DeviceType.CPU))
        hw.get_sdr_interface = Mock(return_value=Mock(mode=SDRMode.SIMULATION))
        hw.optimize_for_platform = Mock(return_value={})
        hw.validate_configuration = Mock(return_value={})
        return hw


@pytest.fixture(params=['cpu', 'cuda', 'mps'])
def device_type(request):
    """Parametrized fixture for testing across different device types."""
    device = request.param
    
    # Skip CUDA tests if not available
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Skip MPS tests if not available
    if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    
    return device


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities for tests."""
    import time
    import psutil
    import gc
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.metrics = {}
        
        def start(self):
            """Start performance monitoring."""
            gc.collect()  # Clean up before measurement
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        def stop(self):
            """Stop monitoring and return metrics."""
            if self.start_time is None:
                return {}
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            self.metrics = {
                'duration_seconds': end_time - self.start_time,
                'memory_delta_mb': end_memory - self.start_memory,
                'peak_memory_mb': end_memory
            }
            
            return self.metrics
        
        def assert_performance(self, max_duration=None, max_memory_mb=None):
            """Assert performance constraints."""
            if max_duration and self.metrics.get('duration_seconds', 0) > max_duration:
                pytest.fail(f"Test took {self.metrics['duration_seconds']:.2f}s, expected < {max_duration}s")
            
            if max_memory_mb and self.metrics.get('memory_delta_mb', 0) > max_memory_mb:
                pytest.fail(f"Test used {self.metrics['memory_delta_mb']:.1f}MB, expected < {max_memory_mb}MB")
    
    return PerformanceMonitor()


# Platform detection utilities
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (CUDA or MPS)"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests that require real hardware"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to tests that might be slow
        if any(keyword in item.name.lower() for keyword in ['train', 'benchmark', 'performance', 'large']):
            item.add_marker(pytest.mark.slow)
        
        # Add GPU marker to tests that use GPU
        if any(keyword in item.name.lower() for keyword in ['cuda', 'gpu', 'mps']):
            item.add_marker(pytest.mark.gpu)
        
        # Add hardware marker to tests that need real hardware
        if any(keyword in item.name.lower() for keyword in ['sdr', 'pluto', 'hardware']):
            item.add_marker(pytest.mark.hardware)
        
        # Add integration marker to integration tests
        if 'integration' in item.name.lower() or 'test_integration' in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if 'performance' in item.name.lower() or 'benchmark' in item.name.lower():
            item.add_marker(pytest.mark.performance)


# Utility functions for tests
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    if not (torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())):
        pytest.skip("No GPU available (CUDA or MPS)")


def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def skip_if_no_mps():
    """Skip test if MPS is not available."""
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")


def get_available_device():
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


# Test data generators
def generate_test_signals(modulations, num_samples_per_mod=10, signal_length=512):
    """Generate test signals for ML testing."""
    signals = []
    labels = []
    
    for mod in modulations:
        for _ in range(num_samples_per_mod):
            # Generate synthetic signal based on modulation type
            if mod == 'BPSK':
                symbols = np.random.choice([-1, 1], signal_length // 2)
            elif mod == 'QPSK':
                symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], signal_length // 2)
            else:
                # Default to QPSK for unknown modulations
                symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], signal_length // 2)
            
            # Upsample and add noise
            signal = np.repeat(symbols, 2)
            noise = 0.1 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
            signal = signal + noise
            
            signals.append(signal.astype(np.complex64))
            labels.append(mod)
    
    return signals, labels