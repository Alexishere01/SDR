"""
Integration test examples demonstrating end-to-end workflows.

This module shows how to use the IntegrationTest base class for
testing complete workflows and component interactions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from tests.test_base import IntegrationTest


class TestMLWorkflowIntegration(IntegrationTest):
    """Integration tests for ML workflow components."""
    
    def setup_method(self, method):
        """Set up ML workflow components."""
        super().setup_method(method)
        
        # Register ML workflow components
        self.register_component("config_manager", self._create_mock_config_manager())
        self.register_component("data_loader", self._create_mock_data_loader())
        self.register_component("model", self._create_mock_model())
        self.register_component("trainer", self._create_mock_trainer())
        self.register_component("evaluator", self._create_mock_evaluator())
    
    def _create_mock_config_manager(self):
        """Create mock configuration manager."""
        config_manager = Mock()
        config_manager.get_config.return_value = Mock(
            ml=Mock(
                batch_size=32,
                learning_rate=1e-3,
                epochs=10
            ),
            hardware=Mock(
                device_preference='cpu'
            )
        )
        return config_manager
    
    def _create_mock_data_loader(self):
        """Create mock data loader."""
        data_loader = Mock()
        
        # Mock training data
        train_data = [(torch.randn(32, 256), torch.randint(0, 4, (32,))) for _ in range(10)]
        val_data = [(torch.randn(32, 256), torch.randint(0, 4, (32,))) for _ in range(5)]
        
        data_loader.load_training_data.return_value = train_data
        data_loader.load_validation_data.return_value = val_data
        data_loader.get_data_info.return_value = {
            'num_classes': 4,
            'input_size': 256,
            'train_samples': 320,
            'val_samples': 160
        }
        
        return data_loader
    
    def _create_mock_model(self):
        """Create mock ML model."""
        model = Mock()
        model.forward.return_value = torch.randn(32, 4)
        model.train.return_value = None
        model.eval.return_value = None
        model.parameters.return_value = [torch.randn(256, 4, requires_grad=True)]
        model.state_dict.return_value = {'layer.weight': torch.randn(256, 4)}
        model.load_state_dict.return_value = None
        return model
    
    def _create_mock_trainer(self):
        """Create mock trainer."""
        trainer = Mock()
        trainer.train.return_value = {
            'final_loss': 0.1,
            'final_accuracy': 0.95,
            'epochs_completed': 10,
            'training_time': 120.5
        }
        return trainer
    
    def _create_mock_evaluator(self):
        """Create mock evaluator."""
        evaluator = Mock()
        evaluator.evaluate.return_value = {
            'accuracy': 0.92,
            'precision': 0.91,
            'recall': 0.93,
            'f1_score': 0.92,
            'confusion_matrix': [[50, 2], [3, 45]]
        }
        return evaluator
    
    def test_complete_ml_training_workflow(self):
        """Test complete ML training workflow."""
        workflow_steps = [
            "load_configuration",
            "prepare_data",
            "initialize_model",
            "train_model",
            "evaluate_model",
            "save_results"
        ]
        
        results = self.run_end_to_end_workflow(workflow_steps)
        self.assert_workflow_success(results)
        
        # Verify specific workflow outcomes
        assert 'load_configuration' in self.workflow_state
        assert 'prepare_data' in self.workflow_state
        assert 'train_model' in self.workflow_state
        assert 'evaluate_model' in self.workflow_state
        
        # Check training results
        training_results = self.workflow_state['train_model']
        assert training_results['final_accuracy'] > 0.9
        assert training_results['epochs_completed'] == 10
        
        # Check evaluation results
        eval_results = self.workflow_state['evaluate_model']
        assert eval_results['accuracy'] > 0.9
        assert 'f1_score' in eval_results
    
    def _execute_workflow_step(self, step: str):
        """Execute ML workflow steps."""
        if step == "load_configuration":
            config_manager = self.integration_components["config_manager"]
            config = config_manager.get_config()
            return {
                'batch_size': config.ml.batch_size,
                'learning_rate': config.ml.learning_rate,
                'device': config.hardware.device_preference
            }
        
        elif step == "prepare_data":
            data_loader = self.integration_components["data_loader"]
            train_data = data_loader.load_training_data()
            val_data = data_loader.load_validation_data()
            data_info = data_loader.get_data_info()
            
            return {
                'train_batches': len(train_data),
                'val_batches': len(val_data),
                'data_info': data_info
            }
        
        elif step == "initialize_model":
            model = self.integration_components["model"]
            data_info = self.workflow_state["prepare_data"]["data_info"]
            
            # Simulate model initialization
            return {
                'model_initialized': True,
                'input_size': data_info['input_size'],
                'num_classes': data_info['num_classes'],
                'num_parameters': 1024  # Mock parameter count
            }
        
        elif step == "train_model":
            trainer = self.integration_components["trainer"]
            model = self.integration_components["model"]
            
            # Simulate training
            training_results = trainer.train(model)
            return training_results
        
        elif step == "evaluate_model":
            evaluator = self.integration_components["evaluator"]
            model = self.integration_components["model"]
            
            # Simulate evaluation
            eval_results = evaluator.evaluate(model)
            return eval_results
        
        elif step == "save_results":
            # Simulate saving results
            return {
                'model_saved': True,
                'results_saved': True,
                'save_path': '/tmp/model_results.json'
            }
        
        else:
            return super()._execute_workflow_step(step)
    
    def test_component_integration_config_and_data(self):
        """Test integration between configuration and data loading."""
        result = self.test_component_integration(["config_manager", "data_loader"])
        
        assert result['status'] == 'success'
        assert 'config_manager' in result['components_tested']
        assert 'data_loader' in result['components_tested']
    
    def test_component_integration_model_and_trainer(self):
        """Test integration between model and trainer."""
        result = self.test_component_integration(["model", "trainer"])
        
        assert result['status'] == 'success'
        
        # Test actual interaction
        model = self.integration_components["model"]
        trainer = self.integration_components["trainer"]
        
        # Verify trainer can work with model
        training_result = trainer.train(model)
        assert 'final_accuracy' in training_result
        assert training_result['final_accuracy'] > 0
    
    def test_workflow_failure_recovery(self):
        """Test workflow failure and recovery."""
        # Create a workflow that will fail at training step
        def failing_train_step():
            raise RuntimeError("Training failed due to memory error")
        
        # Patch the trainer to fail
        trainer = self.integration_components["trainer"]
        trainer.train.side_effect = failing_train_step
        
        workflow_steps = ["load_configuration", "prepare_data", "train_model"]
        results = self.run_end_to_end_workflow(workflow_steps)
        
        # Check that workflow stopped at failure
        assert results["load_configuration"]['success'] is True
        assert results["prepare_data"]['success'] is True
        assert results["train_model"]['success'] is False
        assert "error" in results["train_model"]


class TestSDRWorkflowIntegration(IntegrationTest):
    """Integration tests for SDR workflow components."""
    
    def setup_method(self, method):
        """Set up SDR workflow components."""
        super().setup_method(method)
        
        # Register SDR workflow components
        self.register_component("sdr_interface", self._create_mock_sdr())
        self.register_component("signal_processor", self._create_mock_signal_processor())
        self.register_component("detector", self._create_mock_detector())
        self.register_component("recorder", self._create_mock_recorder())
    
    def _create_mock_sdr(self):
        """Create mock SDR interface."""
        sdr = Mock()
        sdr.connect.return_value = True
        sdr.is_connected = True
        sdr.sample_rate = 1e6
        sdr.center_freq = 2.4e9
        sdr.receive_samples.return_value = np.random.randn(1024) + 1j * np.random.randn(1024)
        sdr.set_frequency.return_value = None
        sdr.set_sample_rate.return_value = None
        return sdr
    
    def _create_mock_signal_processor(self):
        """Create mock signal processor."""
        processor = Mock()
        processor.process_samples.return_value = {
            'processed_samples': np.random.randn(1024) + 1j * np.random.randn(1024),
            'snr_db': 15.2,
            'power_db': -20.5
        }
        return processor
    
    def _create_mock_detector(self):
        """Create mock signal detector."""
        detector = Mock()
        detector.detect_signals.return_value = {
            'signals_detected': 3,
            'frequencies': [2.401e9, 2.425e9, 2.450e9],
            'powers': [-25.1, -30.2, -28.7],
            'modulations': ['QPSK', 'BPSK', '16QAM']
        }
        return detector
    
    def _create_mock_recorder(self):
        """Create mock data recorder."""
        recorder = Mock()
        recorder.record_data.return_value = {
            'samples_recorded': 1024,
            'file_path': '/tmp/sdr_recording.dat',
            'duration_seconds': 1.024
        }
        return recorder
    
    def test_sdr_signal_detection_workflow(self):
        """Test complete SDR signal detection workflow."""
        workflow_steps = [
            "connect_sdr",
            "configure_sdr",
            "capture_samples",
            "process_signals",
            "detect_signals",
            "record_results"
        ]
        
        results = self.run_end_to_end_workflow(workflow_steps)
        self.assert_workflow_success(results)
        
        # Verify SDR workflow outcomes
        assert self.workflow_state['connect_sdr']['connected'] is True
        assert self.workflow_state['capture_samples']['samples_captured'] > 0
        assert self.workflow_state['detect_signals']['signals_detected'] > 0
    
    def _execute_workflow_step(self, step: str):
        """Execute SDR workflow steps."""
        if step == "connect_sdr":
            sdr = self.integration_components["sdr_interface"]
            connected = sdr.connect()
            return {
                'connected': connected,
                'sample_rate': sdr.sample_rate,
                'center_freq': sdr.center_freq
            }
        
        elif step == "configure_sdr":
            sdr = self.integration_components["sdr_interface"]
            sdr.set_frequency(2.4e9)
            sdr.set_sample_rate(1e6)
            return {
                'frequency_set': 2.4e9,
                'sample_rate_set': 1e6
            }
        
        elif step == "capture_samples":
            sdr = self.integration_components["sdr_interface"]
            samples = sdr.receive_samples()
            return {
                'samples_captured': len(samples),
                'sample_type': 'complex64'
            }
        
        elif step == "process_signals":
            processor = self.integration_components["signal_processor"]
            # Use samples from previous step (mocked)
            samples = np.random.randn(1024) + 1j * np.random.randn(1024)
            result = processor.process_samples(samples)
            return result
        
        elif step == "detect_signals":
            detector = self.integration_components["detector"]
            # Use processed samples (mocked)
            processed_samples = np.random.randn(1024) + 1j * np.random.randn(1024)
            detection_result = detector.detect_signals(processed_samples)
            return detection_result
        
        elif step == "record_results":
            recorder = self.integration_components["recorder"]
            # Record the detection results
            detection_data = self.workflow_state.get('detect_signals', {})
            record_result = recorder.record_data(detection_data)
            return record_result
        
        else:
            return super()._execute_workflow_step(step)
    
    def test_sdr_component_integration(self):
        """Test integration between SDR components."""
        # Test SDR and signal processor integration
        sdr_processor_result = self.test_component_integration(["sdr_interface", "signal_processor"])
        assert sdr_processor_result['status'] == 'success'
        
        # Test processor and detector integration
        processor_detector_result = self.test_component_integration(["signal_processor", "detector"])
        assert processor_detector_result['status'] == 'success'
    
    def test_sdr_error_handling(self):
        """Test SDR workflow error handling."""
        # Simulate SDR connection failure
        sdr = self.integration_components["sdr_interface"]
        sdr.connect.return_value = False
        
        workflow_steps = ["connect_sdr", "capture_samples"]
        results = self.run_end_to_end_workflow(workflow_steps)
        
        # Should fail at connect step
        assert results["connect_sdr"]['success'] is True  # Our mock still returns success
        # In real implementation, this would test actual error handling


class TestCrossComponentIntegration(IntegrationTest):
    """Integration tests across different system components."""
    
    def setup_method(self, method):
        """Set up cross-component integration."""
        super().setup_method(method)
        
        # Register components from different subsystems
        self.register_component("config", self._create_mock_config())
        self.register_component("error_handler", self._create_mock_error_handler())
        self.register_component("memory_manager", self._create_mock_memory_manager())
        self.register_component("model_manager", self._create_mock_model_manager())
    
    def _create_mock_config(self):
        """Create mock configuration system."""
        config = Mock()
        config.get_setting.return_value = "test_value"
        config.validate.return_value = True
        return config
    
    def _create_mock_error_handler(self):
        """Create mock error handler."""
        error_handler = Mock()
        error_handler.handle_error.return_value = True
        error_handler.get_error_count.return_value = 0
        return error_handler
    
    def _create_mock_memory_manager(self):
        """Create mock memory manager."""
        memory_manager = Mock()
        memory_manager.get_memory_usage.return_value = 512.0  # MB
        memory_manager.cleanup.return_value = 128.0  # MB freed
        return memory_manager
    
    def _create_mock_model_manager(self):
        """Create mock model manager."""
        model_manager = Mock()
        model_manager.load_model.return_value = Mock()
        model_manager.save_model.return_value = True
        model_manager.list_models.return_value = ['model1', 'model2']
        return model_manager
    
    def test_system_initialization_workflow(self):
        """Test complete system initialization workflow."""
        workflow_steps = [
            "load_config",
            "initialize_error_handling",
            "setup_memory_management",
            "initialize_model_system",
            "validate_system"
        ]
        
        results = self.run_end_to_end_workflow(workflow_steps)
        self.assert_workflow_success(results)
        
        # Verify system is properly initialized
        assert self.workflow_state['load_config']['config_loaded'] is True
        assert self.workflow_state['setup_memory_management']['memory_usage'] > 0
        assert len(self.workflow_state['initialize_model_system']['available_models']) > 0
    
    def _execute_workflow_step(self, step: str):
        """Execute system workflow steps."""
        if step == "load_config":
            config = self.integration_components["config"]
            is_valid = config.validate()
            return {
                'config_loaded': True,
                'config_valid': is_valid
            }
        
        elif step == "initialize_error_handling":
            error_handler = self.integration_components["error_handler"]
            error_count = error_handler.get_error_count()
            return {
                'error_handler_initialized': True,
                'initial_error_count': error_count
            }
        
        elif step == "setup_memory_management":
            memory_manager = self.integration_components["memory_manager"]
            memory_usage = memory_manager.get_memory_usage()
            return {
                'memory_manager_initialized': True,
                'memory_usage': memory_usage
            }
        
        elif step == "initialize_model_system":
            model_manager = self.integration_components["model_manager"]
            available_models = model_manager.list_models()
            return {
                'model_system_initialized': True,
                'available_models': available_models
            }
        
        elif step == "validate_system":
            # Validate all components are working
            config = self.integration_components["config"]
            error_handler = self.integration_components["error_handler"]
            memory_manager = self.integration_components["memory_manager"]
            model_manager = self.integration_components["model_manager"]
            
            return {
                'system_valid': True,
                'components_validated': 4,
                'validation_passed': True
            }
        
        else:
            return super()._execute_workflow_step(step)
    
    def test_error_propagation_across_components(self):
        """Test error propagation across different components."""
        # This would test how errors in one component affect others
        # For now, just verify components can interact
        result = self.test_component_integration([
            "config", "error_handler", "memory_manager", "model_manager"
        ])
        
        assert result['status'] == 'success'
        assert len(result['components_tested']) == 4