"""
Example tests demonstrating the enhanced testing framework.

This module shows how to use the TestBase, PerformanceTest, and CrossPlatformTest
base classes for different types of testing scenarios.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from tests.test_base import TestBase, PerformanceTest, CrossPlatformTest, IntegrationTest


class TestFrameworkExamples(TestBase):
    """Example tests using TestBase functionality."""
    
    def test_basic_setup(self):
        """Test that basic setup works correctly."""
        # Configuration should be available
        self.assert_config_valid(self.config)
        
        # Logger should be set up
        assert self.logger is not None
        self.logger.info("Basic setup test passed")
        
        # Error handler should be available
        assert self.error_handler is not None
    
    def test_mock_sdr_creation(self):
        """Test creating mock SDR interface."""
        sdr = self.create_mock_sdr()
        
        # Test mock SDR properties
        assert sdr.sample_rate == 1e6
        assert sdr.center_freq == 2.4e9
        assert sdr.is_connected is True
        
        # Test mock methods
        assert sdr.connect() is True
        samples = sdr.receive_samples()
        assert len(samples) == 1024
        assert samples.dtype == np.complex128
    
    def test_signal_generation(self):
        """Test test signal generation."""
        modulations = ['BPSK', 'QPSK', '16QAM']
        signals, labels = self.create_test_signals(modulations, num_samples=5, signal_length=256)
        
        # Check signal properties
        assert len(signals) == 15  # 5 samples * 3 modulations
        assert len(labels) == 15
        assert all(len(sig) == 256 for sig in signals)
        assert all(sig.dtype == np.complex64 for sig in signals)
        
        # Check label distribution
        label_counts = {mod: labels.count(mod) for mod in modulations}
        assert all(count == 5 for count in label_counts.values())
    
    def test_tensor_assertions(self):
        """Test tensor assertion utilities."""
        test_tensor = torch.randn(10, 5)
        
        # Test shape assertion
        self.assert_tensor_properties(test_tensor, expected_shape=(10, 5))
        
        # Test dtype assertion
        self.assert_tensor_properties(test_tensor, expected_dtype=torch.float32)
        
        # Test device assertion
        self.assert_tensor_properties(test_tensor, expected_device='cpu')
        
        # Test combined assertions
        self.assert_tensor_properties(
            test_tensor,
            expected_shape=(10, 5),
            expected_dtype=torch.float32,
            expected_device='cpu'
        )
    
    def test_resource_cleanup(self):
        """Test automatic resource cleanup."""
        # Create a mock resource
        mock_resource = Mock()
        mock_resource.close = Mock()
        
        # Add to test resources
        self.add_test_resource(mock_resource)
        
        # Verify it's in the list
        assert mock_resource in self.test_resources
        
        # Cleanup will be called automatically in teardown_method


class TestPerformanceExamples(PerformanceTest):
    """Example performance tests using PerformanceTest functionality."""
    
    def setup_method(self, method):
        """Set up performance test with custom thresholds."""
        super().setup_method(method)
        
        # Set custom performance thresholds for these tests
        self.max_duration_seconds = 10.0
        self.max_memory_mb = 512.0
    
    def test_basic_benchmark(self):
        """Test basic benchmarking functionality."""
        self.start_benchmark("basic_operation")
        
        # Simulate some work
        result = torch.randn(1000, 1000)
        result = torch.mm(result, result.T)
        
        metrics = self.stop_benchmark()
        
        # Assert performance requirements
        self.assert_performance(metrics)
        
        # Check metrics structure
        assert 'duration_seconds' in metrics
        assert 'memory_delta_mb' in metrics
        assert metrics['duration_seconds'] > 0
    
    def test_training_speed_benchmark(self):
        """Test training speed benchmarking."""
        # Create simple model and data
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 4)
        )
        
        # Create test data loader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 4, (100,))
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        # Benchmark training
        metrics = self.benchmark_training_speed(model, train_loader, num_epochs=1)
        
        # Assert performance
        self.assert_performance(metrics, min_throughput=10.0)  # 10 samples/sec minimum
        
        # Check training-specific metrics
        assert 'samples_per_second' in metrics
        assert 'total_samples' in metrics
        assert metrics['total_samples'] == 100
    
    def test_inference_speed_benchmark(self):
        """Test inference speed benchmarking."""
        # Create simple model and data
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 4)
        )
        
        # Create test data loader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(50, 10),
            torch.randint(0, 4, (50,))
        )
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        # Benchmark inference
        metrics = self.benchmark_inference_speed(model, test_loader)
        
        # Assert performance
        self.assert_performance(metrics, min_throughput=50.0)  # 50 samples/sec minimum
        
        # Check inference-specific metrics
        assert 'samples_per_second' in metrics
        assert metrics['total_samples'] == 50
    
    def test_memory_usage_benchmark(self):
        """Test memory usage benchmarking."""
        def memory_intensive_operation():
            # Create large tensors
            tensors = []
            for _ in range(10):
                tensors.append(torch.randn(100, 100))
            return len(tensors)
        
        # Benchmark memory usage
        metrics = self.benchmark_memory_usage(memory_intensive_operation)
        
        # Assert memory usage is reasonable
        self.assert_performance(metrics, max_memory_mb=100.0)
        
        # Check operation result
        assert metrics['operation_result'] == 10
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Run multiple benchmarks
        self.start_benchmark("op1")
        torch.randn(100, 100)
        self.stop_benchmark()
        
        self.start_benchmark("op2")
        torch.randn(200, 200)
        self.stop_benchmark()
        
        # Get summary
        summary = self.get_performance_summary()
        
        assert summary['total_operations'] == 2
        assert 'op1' in summary['metrics']
        assert 'op2' in summary['metrics']
        assert summary['total_duration'] > 0


class TestCrossPlatformExamples(CrossPlatformTest):
    """Example cross-platform tests using CrossPlatformTest functionality."""
    
    def test_platform_detection(self):
        """Test platform detection functionality."""
        # Check that we detected at least CPU
        assert 'cpu' in self.available_devices
        
        # Check platform info
        info = self.get_platform_info()
        assert 'python_version' in info
        assert 'torch_version' in info
        assert 'available_devices' in info
        
        self.logger.info(f"Platform info: {info}")
    
    def test_device_specific_operations(self):
        """Test device-specific operations."""
        for device in self.available_devices:
            self._test_device_specific_functionality(device)
    
    def test_simple_model_portability(self):
        """Test simple model portability across devices."""
        def model_factory():
            return nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 4)
            )
        
        test_data = torch.randn(5, 10)
        
        results = self.test_model_portability(model_factory, test_data)
        
        # Validate results
        self.validate_cross_platform_results(results, tolerance=1e-2)
        
        # Check that at least one device worked
        successful_devices = [d for d, r in results.items() if r['success']]
        assert len(successful_devices) > 0
    
    def test_configuration_loading(self):
        """Test platform-specific configuration loading."""
        self.test_configuration_compatibility()
        
        # Test specific configurations
        for device in self.available_devices:
            config = self.platform_configs[device]
            
            # Verify required fields
            assert 'batch_size' in config
            assert 'device_type' in config
            assert config['batch_size'] > 0
    
    @pytest.mark.gpu
    def test_gpu_specific_functionality(self):
        """Test GPU-specific functionality."""
        self.skip_if_no_gpu()
        
        # Test CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.mm(test_tensor, test_tensor.T)
            assert result.device == device
        
        # Test MPS if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.mm(test_tensor, test_tensor.T)
            assert result.device == device


class TestIntegrationExamples(IntegrationTest):
    """Example integration tests using IntegrationTest functionality."""
    
    def setup_method(self, method):
        """Set up integration test components."""
        super().setup_method(method)
        
        # Register mock components for testing
        mock_config = Mock()
        mock_config.get_setting = Mock(return_value="test_value")
        self.register_component("config", mock_config)
        
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.1, 0.9]))
        self.register_component("model", mock_model)
        
        mock_data_loader = Mock()
        mock_data_loader.load_data = Mock(return_value=("test_data", "test_labels"))
        self.register_component("data_loader", mock_data_loader)
    
    def test_component_registration(self):
        """Test component registration functionality."""
        # Check that components were registered
        assert "config" in self.integration_components
        assert "model" in self.integration_components
        assert "data_loader" in self.integration_components
        
        # Test component access
        config = self.integration_components["config"]
        assert config.get_setting() == "test_value"
    
    def test_two_component_integration(self):
        """Test integration between two components."""
        result = self.test_component_integration(["config", "model"])
        
        assert result['status'] == 'success'
        assert 'config' in result['components_tested']
        assert 'model' in result['components_tested']
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow execution."""
        workflow_steps = ["load_config", "load_data", "train_model", "evaluate"]
        
        results = self.run_end_to_end_workflow(workflow_steps)
        
        # Assert workflow success
        self.assert_workflow_success(results)
        
        # Check that all steps completed
        assert len(results) == len(workflow_steps)
        for step in workflow_steps:
            assert step in results
            assert results[step]['success'] is True
    
    def _execute_workflow_step(self, step: str):
        """Override workflow step execution for testing."""
        if step == "load_config":
            config = self.integration_components["config"]
            return config.get_setting()
        elif step == "load_data":
            data_loader = self.integration_components["data_loader"]
            return data_loader.load_data()
        elif step == "train_model":
            # Simulate training
            return {"loss": 0.1, "accuracy": 0.95}
        elif step == "evaluate":
            model = self.integration_components["model"]
            return model.predict()
        else:
            return super()._execute_workflow_step(step)
    
    def test_workflow_failure_handling(self):
        """Test workflow failure handling."""
        # Create a workflow that will fail
        def failing_step():
            raise ValueError("Simulated failure")
        
        # Patch the step execution to fail
        with patch.object(self, '_execute_workflow_step', side_effect=failing_step):
            workflow_steps = ["failing_step"]
            results = self.run_end_to_end_workflow(workflow_steps)
            
            # Check that failure was recorded
            assert results["failing_step"]['success'] is False
            assert "error" in results["failing_step"]


class TestFrameworkIntegration(TestBase):
    """Test the testing framework itself."""
    
    def test_all_base_classes_work_together(self):
        """Test that all base classes can be used together."""
        # This test verifies that the framework components don't conflict
        
        # Test basic functionality
        assert self.config is not None
        assert self.logger is not None
        assert self.error_handler is not None
        
        # Test that we can create test data
        signals, labels = self.create_test_signals(['BPSK', 'QPSK'])
        assert len(signals) == 20  # 10 per modulation
        assert len(labels) == 20
        
        # Test that we can create mock objects
        sdr = self.create_mock_sdr()
        assert sdr is not None
        
        self.logger.info("Framework integration test passed")
    
    def test_pytest_markers_work(self):
        """Test that pytest markers are properly configured."""
        # This test should have markers applied automatically
        # based on the pytest_collection_modifyitems function
        pass
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        # This test is marked as slow
        import time
        time.sleep(0.1)  # Simulate slow operation
    
    @pytest.mark.gpu
    def test_gpu_marker(self):
        """Test that GPU marker works."""
        # This test is marked as requiring GPU
        self.skip_if_no_gpu()
        
        # If we get here, GPU is available
        assert torch.cuda.is_available() or (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        # This test is marked as integration test
        pass
    
    @pytest.mark.performance
    def test_performance_marker(self):
        """Test that performance marker works."""
        # This test is marked as performance test
        pass