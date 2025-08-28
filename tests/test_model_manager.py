"""
Tests for model management system with MLflow integration.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from geminisdr.core.model_manager import ModelManager, ModelRegistry
from geminisdr.core.model_metadata import ModelMetadata


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, hidden_size=5, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Mock configuration object."""
    config = Mock()
    config.ml = Mock()
    config.ml.model_cache_size = 3
    return config


@pytest.fixture
def model_manager(temp_dir, mock_config):
    """Create ModelManager instance for testing."""
    # Change to temp directory
    original_cwd = Path.cwd()
    import os
    os.chdir(temp_dir)
    
    try:
        # Mock MLflow to avoid actual tracking
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name', return_value=None), \
             patch('mlflow.create_experiment', return_value="test_exp_id"):
            
            manager = ModelManager(config=mock_config, tracking_uri="sqlite:///test.db")
            yield manager
    finally:
        os.chdir(original_cwd)


class TestModelManager:
    """Test ModelManager class."""
    
    def test_initialization(self, temp_dir, mock_config):
        """Test ModelManager initialization."""
        original_cwd = Path.cwd()
        import os
        os.chdir(temp_dir)
        
        try:
            with patch('mlflow.set_tracking_uri'), \
                 patch('mlflow.get_experiment_by_name', return_value=None), \
                 patch('mlflow.create_experiment', return_value="test_exp_id"):
                
                manager = ModelManager(config=mock_config)
                
                assert manager.config == mock_config
                assert manager.models_dir.exists()
                assert manager.experiment_name == "geminisdr_models"
        finally:
            os.chdir(original_cwd)
    
    @patch('mlflow.start_run')
    @patch('mlflow.pytorch.log_model')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.set_tags')
    @patch('mlflow.log_artifact')
    def test_save_model(self, mock_log_artifact, mock_set_tags, mock_log_metrics, 
                       mock_log_params, mock_log_model, mock_start_run, model_manager):
        """Test saving a model."""
        # Mock MLflow run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        # Create test model
        model = SimpleTestModel()
        
        # Test data
        hyperparameters = {"lr": 0.001, "batch_size": 32}
        performance_metrics = {"accuracy": 0.95, "loss": 0.05}
        
        # Save model
        run_id = model_manager.save_model(
            model=model,
            model_name="test_model",
            version="1.0.0",
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            description="Test model"
        )
        
        assert run_id == "test_run_id"
        
        # Check that MLflow methods were called
        mock_log_params.assert_called_once_with(hyperparameters)
        mock_log_metrics.assert_called_once_with(performance_metrics)
        mock_log_model.assert_called_once()
        mock_set_tags.assert_called_once()
        
        # Check local files were created
        model_dir = model_manager.models_dir / "test_model" / "1.0.0"
        assert model_dir.exists()
        assert (model_dir / "model.pth").exists()
        assert (model_dir / "complete_model.pth").exists()
        assert (model_dir / "metadata.json").exists()
        assert (model_dir / "model_hash.txt").exists()
    
    def test_save_and_load_model_local(self, model_manager):
        """Test saving and loading model locally (without MLflow)."""
        # Create test model
        model = SimpleTestModel(input_size=20, hidden_size=10, output_size=2)
        
        # Test data
        hyperparameters = {"lr": 0.001, "batch_size": 32}
        performance_metrics = {"accuracy": 0.95, "loss": 0.05}
        
        # Mock MLflow operations
        with patch('mlflow.start_run') as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            with patch('mlflow.pytorch.log_model'), \
                 patch('mlflow.log_params'), \
                 patch('mlflow.log_metrics'), \
                 patch('mlflow.set_tags'), \
                 patch('mlflow.log_artifact'):
                
                # Save model
                run_id = model_manager.save_model(
                    model=model,
                    model_name="test_model",
                    version="1.0.0",
                    hyperparameters=hyperparameters,
                    performance_metrics=performance_metrics
                )
        
        # Load model back
        loaded_model, metadata = model_manager.load_model("test_model", "1.0.0")
        
        # Verify model architecture
        assert isinstance(loaded_model, SimpleTestModel)
        assert loaded_model.linear1.in_features == 20
        assert loaded_model.linear1.out_features == 10
        assert loaded_model.linear2.out_features == 2
        
        # Verify metadata
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.hyperparameters == hyperparameters
        assert metadata.performance_metrics == performance_metrics
    
    def test_load_model_not_found(self, model_manager):
        """Test loading non-existent model."""
        with pytest.raises(FileNotFoundError):
            model_manager.load_model("nonexistent_model", "1.0.0")
    
    def test_list_models_empty(self, model_manager):
        """Test listing models when none exist."""
        models = model_manager.list_models()
        assert models == []
    
    def test_list_models_with_data(self, model_manager):
        """Test listing models with saved data."""
        # Save a test model first
        model = SimpleTestModel()
        hyperparameters = {"lr": 0.001}
        performance_metrics = {"accuracy": 0.95}
        
        with patch('mlflow.start_run') as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            with patch('mlflow.pytorch.log_model'), \
                 patch('mlflow.log_params'), \
                 patch('mlflow.log_metrics'), \
                 patch('mlflow.set_tags'), \
                 patch('mlflow.log_artifact'):
                
                model_manager.save_model(
                    model=model,
                    model_name="test_model",
                    version="1.0.0",
                    hyperparameters=hyperparameters,
                    performance_metrics=performance_metrics
                )
        
        # List models
        models = model_manager.list_models()
        
        assert len(models) == 1
        assert models[0]['name'] == "test_model"
        assert len(models[0]['versions']) == 1
        assert models[0]['versions'][0]['version'] == "1.0.0"
        assert models[0]['latest_version'] == "1.0.0"
    
    def test_compare_models(self, model_manager):
        """Test comparing two models."""
        # Save two test models
        model1 = SimpleTestModel()
        model2 = SimpleTestModel()
        
        hyperparameters1 = {"lr": 0.001, "batch_size": 32}
        performance_metrics1 = {"accuracy": 0.90, "loss": 0.10}
        
        hyperparameters2 = {"lr": 0.002, "batch_size": 64}
        performance_metrics2 = {"accuracy": 0.95, "loss": 0.08}
        
        with patch('mlflow.start_run') as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            with patch('mlflow.pytorch.log_model'), \
                 patch('mlflow.log_params'), \
                 patch('mlflow.log_metrics'), \
                 patch('mlflow.set_tags'), \
                 patch('mlflow.log_artifact'):
                
                model_manager.save_model(
                    model=model1,
                    model_name="model1",
                    version="1.0.0",
                    hyperparameters=hyperparameters1,
                    performance_metrics=performance_metrics1
                )
                
                model_manager.save_model(
                    model=model2,
                    model_name="model2",
                    version="1.0.0",
                    hyperparameters=hyperparameters2,
                    performance_metrics=performance_metrics2
                )
        
        # Compare models
        comparison = model_manager.compare_models(
            "model1", "1.0.0", "model2", "1.0.0"
        )
        
        assert "performance_comparison" in comparison
        assert "metadata_comparison" in comparison
        
        # Check performance comparison
        perf_comp = comparison["performance_comparison"]
        assert "accuracy" in perf_comp["metrics_comparison"]
        assert perf_comp["metrics_comparison"]["accuracy"]["model1"] == 0.90
        assert perf_comp["metrics_comparison"]["accuracy"]["model2"] == 0.95
    
    def test_migrate_model(self, model_manager):
        """Test model migration."""
        # Save original model
        model = SimpleTestModel()
        hyperparameters = {"lr": 0.001}
        performance_metrics = {"accuracy": 0.90}
        
        with patch('mlflow.start_run') as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            with patch('mlflow.pytorch.log_model'), \
                 patch('mlflow.log_params'), \
                 patch('mlflow.log_metrics'), \
                 patch('mlflow.set_tags'), \
                 patch('mlflow.log_artifact'):
                
                model_manager.save_model(
                    model=model,
                    model_name="test_model",
                    version="1.0.0",
                    hyperparameters=hyperparameters,
                    performance_metrics=performance_metrics
                )
                
                # Migrate model
                success = model_manager.migrate_model(
                    "test_model", "1.0.0", "2.0.0", "Migration test"
                )
        
        assert success
        
        # Check that new version exists
        models = model_manager.list_models()
        test_model = next(m for m in models if m['name'] == 'test_model')
        versions = [v['version'] for v in test_model['versions']]
        assert "2.0.0" in versions
        
        # Load migrated model and check metadata
        _, metadata = model_manager.load_model("test_model", "2.0.0")
        assert "migrated" in metadata.tags
        assert "Migration test" in metadata.description
    
    def test_delete_model_version(self, model_manager):
        """Test deleting a specific model version."""
        # Save test model
        model = SimpleTestModel()
        hyperparameters = {"lr": 0.001}
        performance_metrics = {"accuracy": 0.90}
        
        with patch('mlflow.start_run') as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            with patch('mlflow.pytorch.log_model'), \
                 patch('mlflow.log_params'), \
                 patch('mlflow.log_metrics'), \
                 patch('mlflow.set_tags'), \
                 patch('mlflow.log_artifact'):
                
                model_manager.save_model(
                    model=model,
                    model_name="test_model",
                    version="1.0.0",
                    hyperparameters=hyperparameters,
                    performance_metrics=performance_metrics
                )
        
        # Verify model exists
        models = model_manager.list_models()
        assert len(models) == 1
        
        # Delete model version
        success = model_manager.delete_model("test_model", "1.0.0")
        assert success
        
        # Verify model version is gone
        models = model_manager.list_models()
        assert len(models) == 0
    
    def test_delete_entire_model(self, model_manager):
        """Test deleting entire model."""
        # Save test model with multiple versions
        model = SimpleTestModel()
        hyperparameters = {"lr": 0.001}
        performance_metrics = {"accuracy": 0.90}
        
        with patch('mlflow.start_run') as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            with patch('mlflow.pytorch.log_model'), \
                 patch('mlflow.log_params'), \
                 patch('mlflow.log_metrics'), \
                 patch('mlflow.set_tags'), \
                 patch('mlflow.log_artifact'):
                
                model_manager.save_model(
                    model=model,
                    model_name="test_model",
                    version="1.0.0",
                    hyperparameters=hyperparameters,
                    performance_metrics=performance_metrics
                )
                
                model_manager.save_model(
                    model=model,
                    model_name="test_model",
                    version="2.0.0",
                    hyperparameters=hyperparameters,
                    performance_metrics=performance_metrics
                )
        
        # Verify model exists with multiple versions
        models = model_manager.list_models()
        assert len(models) == 1
        assert len(models[0]['versions']) == 2
        
        # Delete entire model
        success = model_manager.delete_model("test_model")
        assert success
        
        # Verify model is completely gone
        models = model_manager.list_models()
        assert len(models) == 0
    
    def test_validate_model_compatibility(self, model_manager):
        """Test model compatibility validation."""
        # Save test model
        model = SimpleTestModel()
        hyperparameters = {"lr": 0.001}
        performance_metrics = {"accuracy": 0.90}
        
        with patch('mlflow.start_run') as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            with patch('mlflow.pytorch.log_model'), \
                 patch('mlflow.log_params'), \
                 patch('mlflow.log_metrics'), \
                 patch('mlflow.set_tags'), \
                 patch('mlflow.log_artifact'):
                
                model_manager.save_model(
                    model=model,
                    model_name="test_model",
                    version="1.0.0",
                    hyperparameters=hyperparameters,
                    performance_metrics=performance_metrics
                )
        
        # Validate compatibility
        issues = model_manager.validate_model_compatibility("test_model", "1.0.0")
        
        # Should be a list (may be empty if compatible)
        assert isinstance(issues, list)
    
    def test_get_model_info(self, model_manager):
        """Test getting detailed model information."""
        # Save test model
        model = SimpleTestModel()
        hyperparameters = {"lr": 0.001}
        performance_metrics = {"accuracy": 0.90}
        
        with patch('mlflow.start_run') as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            with patch('mlflow.pytorch.log_model'), \
                 patch('mlflow.log_params'), \
                 patch('mlflow.log_metrics'), \
                 patch('mlflow.set_tags'), \
                 patch('mlflow.log_artifact'):
                
                model_manager.save_model(
                    model=model,
                    model_name="test_model",
                    version="1.0.0",
                    hyperparameters=hyperparameters,
                    performance_metrics=performance_metrics
                )
        
        # Get model info
        info = model_manager.get_model_info("test_model", "1.0.0")
        
        assert "metadata" in info
        assert "file_sizes_mb" in info
        assert "fingerprint" in info
        assert "compatibility_issues" in info
        
        # Check metadata
        assert info["metadata"]["name"] == "test_model"
        assert info["metadata"]["version"] == "1.0.0"
        
        # Check file sizes
        assert "model.pth" in info["file_sizes_mb"]
        assert "complete_model.pth" in info["file_sizes_mb"]
        assert "metadata.json" in info["file_sizes_mb"]


class TestModelRegistry:
    """Test ModelRegistry class."""
    
    def test_initialization(self, model_manager):
        """Test ModelRegistry initialization."""
        registry = ModelRegistry(model_manager)
        assert registry.manager == model_manager
    
    def test_search_models_empty(self, model_manager):
        """Test searching models when none exist."""
        registry = ModelRegistry(model_manager)
        results = registry.search_models()
        assert results == []
    
    def test_search_models_by_name(self, model_manager):
        """Test searching models by name."""
        # Save test models
        model = SimpleTestModel()
        hyperparameters = {"lr": 0.001}
        performance_metrics = {"accuracy": 0.90}
        
        with patch('mlflow.start_run') as mock_start_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            with patch('mlflow.pytorch.log_model'), \
                 patch('mlflow.log_params'), \
                 patch('mlflow.log_metrics'), \
                 patch('mlflow.set_tags'), \
                 patch('mlflow.log_artifact'):
                
                model_manager.save_model(
                    model=model,
                    model_name="neural_network",
                    version="1.0.0",
                    hyperparameters=hyperparameters,
                    performance_metrics=performance_metrics
                )
                
                model_manager.save_model(
                    model=model,
                    model_name="classifier",
                    version="1.0.0",
                    hyperparameters=hyperparameters,
                    performance_metrics=performance_metrics
                )
        
        registry = ModelRegistry(model_manager)
        
        # Search for "neural"
        results = registry.search_models(query="neural")
        assert len(results) == 1
        assert results[0]['name'] == "neural_network"
        
        # Search for "class"
        results = registry.search_models(query="class")
        assert len(results) == 1
        assert results[0]['name'] == "classifier"
        
        # Search for non-existent
        results = registry.search_models(query="nonexistent")
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__])