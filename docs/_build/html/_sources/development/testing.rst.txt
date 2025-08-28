Testing Guidelines
==================

This document outlines the testing strategy and guidelines for GeminiSDR development.

Testing Philosophy
------------------

GeminiSDR follows a comprehensive testing approach with multiple levels of validation:

* **Unit Tests**: Test individual components in isolation
* **Integration Tests**: Test component interactions and workflows
* **Performance Tests**: Validate performance requirements and detect regressions
* **Cross-Platform Tests**: Ensure compatibility across M1 Mac, Linux VM, and CUDA environments

Test Categories
---------------

Unit Tests
~~~~~~~~~~

Unit tests focus on testing individual functions and classes in isolation.

**Location**: ``tests/unit/``

**Characteristics**:
* Fast execution (< 1 second per test)
* No external dependencies
* Mock external services and hardware
* High code coverage (>90%)

**Example**:

.. code-block:: python

   # tests/unit/config/test_config_manager.py
   import pytest
   from unittest.mock import Mock, patch
   from geminisdr.config.config_manager import ConfigManager
   from geminisdr.core.error_handling import ConfigurationError
   
   class TestConfigManager:
       """Unit tests for ConfigManager class."""
       
       def test_load_config_with_valid_file_returns_system_config(self, tmp_path):
           """Test loading valid configuration file."""
           # Arrange
           config_file = tmp_path / "config.yaml"
           config_file.write_text("""
           hardware:
             device_preference: "auto"
           ml:
             batch_size: 32
           """)
           
           manager = ConfigManager(str(tmp_path))
           
           # Act
           config = manager.load_config()
           
           # Assert
           assert config.hardware.device_preference == "auto"
           assert config.ml.batch_size == 32
       
       def test_load_config_with_missing_file_raises_error(self, tmp_path):
           """Test that missing config file raises ConfigurationError."""
           # Arrange
           manager = ConfigManager(str(tmp_path))
           
           # Act & Assert
           with pytest.raises(ConfigurationError) as exc_info:
               manager.load_config("nonexistent.yaml")
           
           assert "Configuration file not found" in str(exc_info.value)

Integration Tests
~~~~~~~~~~~~~~~~

Integration tests validate interactions between components and end-to-end workflows.

**Location**: ``tests/integration/``

**Characteristics**:
* Test component interactions
* Use real configuration files
* May use test databases or file systems
* Moderate execution time (1-10 seconds per test)

**Example**:

.. code-block:: python

   # tests/integration/test_model_lifecycle.py
   import pytest
   from pathlib import Path
   from geminisdr.config.config_manager import ConfigManager
   from geminisdr.core.model_manager import ModelManager
   from geminisdr.core.model_metadata import ModelMetadata
   
   class TestModelLifecycle:
       """Integration tests for complete model lifecycle."""
       
       @pytest.fixture
       def setup_environment(self, tmp_path):
           """Set up test environment with config and directories."""
           config_manager = ConfigManager()
           config = config_manager.load_config("test_config")
           
           # Override paths for testing
           config.model_storage_path = str(tmp_path / "models")
           
           model_manager = ModelManager(config)
           return config, model_manager
       
       def test_save_and_load_model_preserves_metadata(self, setup_environment):
           """Test that saving and loading a model preserves all metadata."""
           # Arrange
           config, model_manager = setup_environment
           
           # Create a simple test model
           import torch
           model = torch.nn.Linear(10, 1)
           
           metadata = ModelMetadata(
               name="test_model",
               version="1.0.0",
               hyperparameters={"input_size": 10, "output_size": 1},
               performance_metrics={"accuracy": 0.95}
           )
           
           # Act
           saved_path = model_manager.save_model(model, metadata)
           loaded_model, loaded_metadata = model_manager.load_model(
               "test_model", "1.0.0"
           )
           
           # Assert
           assert loaded_metadata.name == metadata.name
           assert loaded_metadata.version == metadata.version
           assert loaded_metadata.hyperparameters == metadata.hyperparameters
           assert loaded_metadata.performance_metrics == metadata.performance_metrics
           
           # Verify model functionality
           test_input = torch.randn(1, 10)
           original_output = model(test_input)
           loaded_output = loaded_model(test_input)
           torch.testing.assert_close(original_output, loaded_output)

Performance Tests
~~~~~~~~~~~~~~~~

Performance tests validate system performance and detect regressions.

**Location**: ``tests/performance/``

**Characteristics**:
* Measure execution time and memory usage
* Compare against performance baselines
* Run on representative datasets
* Generate performance reports

**Example**:

.. code-block:: python

   # tests/performance/test_training_performance.py
   import pytest
   import time
   import torch
   from geminisdr.ml.neural_amr import NeuralAMR
   from geminisdr.core.memory_manager import MemoryManager
   
   class TestTrainingPerformance:
       """Performance tests for training operations."""
       
       @pytest.mark.performance
       def test_training_speed_meets_baseline(self, performance_config):
           """Test that training speed meets performance baseline."""
           # Arrange
           model = NeuralAMR(performance_config)
           memory_manager = MemoryManager(performance_config)
           
           # Generate test data
           batch_size = 64
           sequence_length = 1024
           test_data = torch.randn(batch_size, sequence_length, 2)
           test_labels = torch.randint(0, 8, (batch_size,))
           
           # Measure training time
           start_time = time.time()
           memory_stats_before = memory_manager.get_memory_stats()
           
           # Act
           for epoch in range(10):
               loss = model.train_step(test_data, test_labels)
           
           # Measure results
           end_time = time.time()
           memory_stats_after = memory_manager.get_memory_stats()
           
           training_time = end_time - start_time
           memory_used = memory_stats_after.used_ram_mb - memory_stats_before.used_ram_mb
           
           # Assert performance requirements
           assert training_time < 30.0, f"Training took {training_time:.2f}s, expected < 30s"
           assert memory_used < 1024, f"Memory usage {memory_used:.1f}MB, expected < 1GB"
           
           # Log performance metrics
           pytest.performance_data = {
               "training_time_seconds": training_time,
               "memory_usage_mb": memory_used,
               "samples_per_second": (batch_size * 10) / training_time
           }

Cross-Platform Tests
~~~~~~~~~~~~~~~~~~~

Cross-platform tests ensure compatibility across different hardware and operating systems.

**Location**: ``tests/cross_platform/``

**Characteristics**:
* Run on M1 Mac, Linux VM, and CUDA environments
* Test platform-specific optimizations
* Validate device detection and selection
* Ensure consistent behavior across platforms

**Example**:

.. code-block:: python

   # tests/cross_platform/test_device_compatibility.py
   import pytest
   import torch
   from geminisdr.environments.hardware_abstraction import HardwareAbstraction
   from geminisdr.config.config_manager import ConfigManager
   
   class TestDeviceCompatibility:
       """Cross-platform device compatibility tests."""
       
       @pytest.mark.parametrize("platform", ["m1_native", "vm_ubuntu", "cuda"])
       def test_device_detection_returns_valid_device(self, platform):
           """Test device detection on different platforms."""
           # Arrange
           config_manager = ConfigManager()
           config = config_manager.load_config(f"test_{platform}")
           hw_abstraction = HardwareAbstraction(config)
           
           # Act
           detected_device = hw_abstraction.detect_optimal_device()
           
           # Assert
           assert detected_device in ["cpu", "mps", "cuda"]
           
           # Platform-specific assertions
           if platform == "m1_native":
               if torch.backends.mps.is_available():
                   assert detected_device == "mps"
               else:
                   assert detected_device == "cpu"
           elif platform == "cuda":
               if torch.cuda.is_available():
                   assert detected_device == "cuda"
               else:
                   assert detected_device == "cpu"
           else:  # vm_ubuntu
               assert detected_device == "cpu"
       
       def test_model_inference_consistent_across_platforms(self):
           """Test that model inference produces consistent results across platforms."""
           # This test would be run on multiple platforms and results compared
           pass

Test Configuration
------------------

pytest Configuration
~~~~~~~~~~~~~~~~~~~

Configure pytest in ``pytest.ini``:

.. code-block:: ini

   [tool:pytest]
   testpaths = tests
   python_files = test_*.py
   python_classes = Test*
   python_functions = test_*
   addopts = 
       --strict-markers
       --strict-config
       --verbose
       --tb=short
       --cov=geminisdr
       --cov-report=html
       --cov-report=term-missing
       --cov-fail-under=90
   markers =
       unit: Unit tests
       integration: Integration tests
       performance: Performance tests
       cross_platform: Cross-platform tests
       slow: Slow tests (> 10 seconds)
       gpu: Tests requiring GPU
       hardware: Tests requiring hardware SDR

Test Fixtures
~~~~~~~~~~~~

Use fixtures for common test setup:

.. code-block:: python

   # tests/conftest.py
   import pytest
   import tempfile
   from pathlib import Path
   from geminisdr.config.config_manager import ConfigManager
   from geminisdr.config.config_models import SystemConfig
   
   @pytest.fixture
   def temp_dir():
       """Provide a temporary directory for tests."""
       with tempfile.TemporaryDirectory() as tmp_dir:
           yield Path(tmp_dir)
   
   @pytest.fixture
   def test_config():
       """Provide a test configuration."""
       config_manager = ConfigManager()
       return config_manager.load_config("test_config")
   
   @pytest.fixture
   def mock_hardware():
       """Provide mock hardware for testing."""
       from unittest.mock import Mock
       
       mock_sdr = Mock()
       mock_sdr.sample_rate = 2e6
       mock_sdr.center_freq = 100e6
       mock_sdr.gain = 20
       
       return mock_sdr
   
   @pytest.fixture(scope="session")
   def performance_baseline():
       """Load performance baseline data."""
       baseline_file = Path("tests/data/performance_baseline.json")
       if baseline_file.exists():
           import json
           with open(baseline_file) as f:
               return json.load(f)
       return {}

Running Tests
-------------

Local Testing
~~~~~~~~~~~~~

Run different test categories:

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run only unit tests
   pytest -m unit
   
   # Run integration tests
   pytest -m integration
   
   # Run performance tests
   pytest -m performance
   
   # Run tests for specific module
   pytest tests/unit/config/
   
   # Run with coverage
   pytest --cov=geminisdr --cov-report=html
   
   # Run tests in parallel
   pytest -n auto

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~

The CI pipeline runs tests across multiple platforms:

.. code-block:: yaml

   # .github/workflows/test.yml
   name: Test Suite
   
   on: [push, pull_request]
   
   jobs:
     test-matrix:
       strategy:
         matrix:
           platform: [macos-latest, ubuntu-latest]
           python-version: [3.9, 3.10, 3.11]
       
       runs-on: ${{ matrix.platform }}
       
       steps:
         - uses: actions/checkout@v3
         
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: ${{ matrix.python-version }}
         
         - name: Install dependencies
           run: |
             pip install -r requirements.txt
             pip install -r requirements-dev.txt
         
         - name: Run unit tests
           run: pytest -m unit
         
         - name: Run integration tests
           run: pytest -m integration
         
         - name: Upload coverage
           uses: codecov/codecov-action@v3

Performance Testing
~~~~~~~~~~~~~~~~~~

Performance tests run on a schedule to detect regressions:

.. code-block:: yaml

   # .github/workflows/performance.yml
   name: Performance Tests
   
   on:
     schedule:
       - cron: '0 2 * * *'  # Daily at 2 AM
     workflow_dispatch:
   
   jobs:
     performance:
       runs-on: ubuntu-latest
       
       steps:
         - uses: actions/checkout@v3
         
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: 3.9
         
         - name: Install dependencies
           run: |
             pip install -r requirements.txt
             pip install -r requirements-dev.txt
         
         - name: Run performance tests
           run: |
             pytest -m performance --benchmark-json=benchmark.json
         
         - name: Compare with baseline
           run: |
             python scripts/compare_benchmarks.py benchmark.json

Test Data Management
-------------------

Test Data Organization
~~~~~~~~~~~~~~~~~~~~~

Organize test data in a structured way:

.. code-block:: text

   tests/
   ├── data/
   │   ├── config/
   │   │   ├── valid_config.yaml
   │   │   ├── invalid_config.yaml
   │   │   └── test_environments/
   │   ├── models/
   │   │   ├── test_model_v1.pth
   │   │   └── metadata/
   │   ├── signals/
   │   │   ├── test_signal_bpsk.npy
   │   │   └── test_signal_qpsk.npy
   │   └── performance/
   │       └── baseline_metrics.json

Mock Data Generation
~~~~~~~~~~~~~~~~~~~

Generate realistic test data:

.. code-block:: python

   # tests/utils/data_generators.py
   import numpy as np
   import torch
   from typing import Tuple
   
   def generate_test_signal(
       modulation: str,
       num_samples: int = 1024,
       snr_db: float = 20.0
   ) -> np.ndarray:
       """Generate test signal with specified modulation and SNR."""
       if modulation == "bpsk":
           # Generate BPSK signal
           bits = np.random.randint(0, 2, num_samples // 2)
           symbols = 2 * bits - 1  # Map to {-1, 1}
           
           # Add noise
           noise_power = 10 ** (-snr_db / 10)
           noise = np.sqrt(noise_power / 2) * (
               np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
           )
           
           signal = symbols + noise
           return np.column_stack([signal.real, signal.imag])
       
       # Add other modulation types as needed
       raise ValueError(f"Unsupported modulation: {modulation}")
   
   def generate_test_dataset(
       size: int = 1000,
       modulations: list = None
   ) -> Tuple[torch.Tensor, torch.Tensor]:
       """Generate a test dataset for training/validation."""
       if modulations is None:
           modulations = ["bpsk", "qpsk", "8psk", "16qam"]
       
       signals = []
       labels = []
       
       for i in range(size):
           mod_idx = i % len(modulations)
           modulation = modulations[mod_idx]
           
           signal = generate_test_signal(modulation)
           signals.append(signal)
           labels.append(mod_idx)
       
       return torch.tensor(signals, dtype=torch.float32), torch.tensor(labels)

Debugging Tests
---------------

Test Debugging Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

Use these strategies for debugging failing tests:

.. code-block:: python

   # Add debug output
   def test_with_debug_output(capfd):
       """Test with captured output for debugging."""
       result = my_function()
       
       # Capture stdout/stderr
       captured = capfd.readouterr()
       print(f"Debug output: {captured.out}")
       
       assert result == expected
   
   # Use pytest fixtures for debugging
   @pytest.fixture
   def debug_config():
       """Configuration with debug logging enabled."""
       config = load_test_config()
       config.logging.level = "DEBUG"
       return config
   
   # Conditional debugging
   import os
   
   def test_with_conditional_debug():
       """Test with conditional debug information."""
       if os.getenv("DEBUG_TESTS"):
           import pdb; pdb.set_trace()
       
       result = complex_function()
       assert result is not None

Test Isolation
~~~~~~~~~~~~~

Ensure tests don't interfere with each other:

.. code-block:: python

   # Use fresh instances
   @pytest.fixture
   def fresh_manager():
       """Provide a fresh manager instance for each test."""
       return ModelManager(load_test_config())
   
   # Clean up after tests
   @pytest.fixture
   def cleanup_files():
       """Clean up test files after test completion."""
       test_files = []
       
       yield test_files
       
       # Cleanup
       for file_path in test_files:
           if Path(file_path).exists():
               Path(file_path).unlink()

Best Practices
--------------

Test Writing Guidelines
~~~~~~~~~~~~~~~~~~~~~~

1. **Test One Thing**: Each test should verify one specific behavior
2. **Clear Names**: Test names should describe what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Independent Tests**: Tests should not depend on each other
5. **Realistic Data**: Use realistic test data that represents actual usage

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fast Unit Tests**: Keep unit tests under 1 second each
2. **Parallel Execution**: Design tests to run in parallel safely
3. **Resource Cleanup**: Always clean up resources (files, memory, connections)
4. **Mock External Dependencies**: Mock slow external services and hardware

Maintenance
~~~~~~~~~~

1. **Regular Updates**: Update tests when code changes
2. **Remove Obsolete Tests**: Remove tests for deprecated functionality
3. **Performance Monitoring**: Monitor test execution time and optimize slow tests
4. **Documentation**: Keep test documentation up to date