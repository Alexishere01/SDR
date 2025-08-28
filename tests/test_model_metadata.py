"""
Tests for model metadata and tracking system.
"""

import pytest
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from geminisdr.core.model_metadata import (
    ModelMetadata, PerformanceMetrics, ModelFingerprinter,
    CompatibilityChecker, ModelComparator, ModelTracker
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, hidden_size=5, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)


class TestModelMetadata:
    """Test ModelMetadata class."""
    
    def test_model_metadata_creation(self):
        """Test creating ModelMetadata with required fields."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            timestamp=datetime.now(),
            hyperparameters={"lr": 0.001, "batch_size": 32},
            performance_metrics={"accuracy": 0.95, "loss": 0.05},
            training_data_hash="abc123",
            code_version="1.0.0"
        )
        
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.hyperparameters["lr"] == 0.001
        assert metadata.performance_metrics["accuracy"] == 0.95
        assert metadata.training_data_hash == "abc123"
    
    def test_model_metadata_defaults(self):
        """Test ModelMetadata with default values."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            timestamp=datetime.now(),
            hyperparameters={},
            performance_metrics={},
            training_data_hash="abc123",
            code_version="1.0.0"
        )
        
        # Check that defaults are set
        assert metadata.platform is not None
        assert metadata.device in ["cpu", "cuda"]
        assert metadata.python_version is not None
        assert metadata.torch_version is not None
        assert metadata.tags == []
    
    def test_to_dict_conversion(self):
        """Test converting ModelMetadata to dictionary."""
        timestamp = datetime.now()
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            timestamp=timestamp,
            hyperparameters={"lr": 0.001},
            performance_metrics={"accuracy": 0.95},
            training_data_hash="abc123",
            code_version="1.0.0"
        )
        
        data = metadata.to_dict()
        
        assert data["name"] == "test_model"
        assert data["timestamp"] == timestamp.isoformat()
        assert data["hyperparameters"]["lr"] == 0.001
    
    def test_from_dict_conversion(self):
        """Test creating ModelMetadata from dictionary."""
        timestamp = datetime.now()
        data = {
            "name": "test_model",
            "version": "1.0.0",
            "timestamp": timestamp.isoformat(),
            "hyperparameters": {"lr": 0.001},
            "performance_metrics": {"accuracy": 0.95},
            "training_data_hash": "abc123",
            "code_version": "1.0.0",
            "platform": "test_platform",
            "device": "cpu",
            "python_version": "3.9.0",
            "torch_version": "1.12.0",
            "model_size_mb": None,
            "model_architecture": None,
            "input_shape": None,
            "output_shape": None,
            "description": None,
            "tags": [],
            "random_seed": None
        }
        
        metadata = ModelMetadata.from_dict(data)
        
        assert metadata.name == "test_model"
        assert metadata.timestamp == timestamp
        assert metadata.hyperparameters["lr"] == 0.001
    
    def test_fingerprint_generation(self):
        """Test model fingerprint generation."""
        metadata1 = ModelMetadata(
            name="test_model",
            version="1.0.0",
            timestamp=datetime.now(),
            hyperparameters={"lr": 0.001},
            performance_metrics={"accuracy": 0.95},
            training_data_hash="abc123",
            code_version="1.0.0"
        )
        
        metadata2 = ModelMetadata(
            name="test_model",
            version="1.0.0",
            timestamp=datetime.now() + timedelta(hours=1),  # Different timestamp
            hyperparameters={"lr": 0.001},
            performance_metrics={"accuracy": 0.96},  # Different performance
            training_data_hash="abc123",
            code_version="1.0.0"
        )
        
        # Same fingerprint despite different timestamp and performance
        assert metadata1.get_fingerprint() == metadata2.get_fingerprint()
        
        # Different fingerprint with different hyperparameters
        metadata3 = ModelMetadata(
            name="test_model",
            version="1.0.0",
            timestamp=datetime.now(),
            hyperparameters={"lr": 0.002},  # Different hyperparameter
            performance_metrics={"accuracy": 0.95},
            training_data_hash="abc123",
            code_version="1.0.0"
        )
        
        assert metadata1.get_fingerprint() != metadata3.get_fingerprint()


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""
    
    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics."""
        metrics = PerformanceMetrics(
            train_loss=0.1,
            train_accuracy=0.95,
            validation_loss=0.15,
            validation_accuracy=0.92
        )
        
        assert metrics.train_loss == 0.1
        assert metrics.train_accuracy == 0.95
        assert metrics.validation_loss == 0.15
        assert metrics.validation_accuracy == 0.92
    
    def test_to_dict_filters_none(self):
        """Test that to_dict filters out None values."""
        metrics = PerformanceMetrics(
            train_loss=0.1,
            train_accuracy=None,  # This should be filtered out
            validation_loss=0.15
        )
        
        data = metrics.to_dict()
        
        assert "train_loss" in data
        assert "train_accuracy" not in data
        assert "validation_loss" in data


class TestModelFingerprinter:
    """Test ModelFingerprinter class."""
    
    def test_hash_model_weights(self):
        """Test hashing model weights."""
        model1 = SimpleModel()
        model2 = SimpleModel()
        
        # Different random initialization should give different hashes
        hash1 = ModelFingerprinter.hash_model_weights(model1)
        hash2 = ModelFingerprinter.hash_model_weights(model2)
        
        assert hash1 != hash2
        assert len(hash1) == 64  # SHA256 hex digest length
        
        # Same model should give same hash
        hash1_again = ModelFingerprinter.hash_model_weights(model1)
        assert hash1 == hash1_again
    
    def test_hash_data_string(self):
        """Test hashing string data."""
        hash1 = ModelFingerprinter.hash_data("test_data")
        hash2 = ModelFingerprinter.hash_data("test_data")
        hash3 = ModelFingerprinter.hash_data("different_data")
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64
    
    def test_hash_data_bytes(self):
        """Test hashing bytes data."""
        data = b"test_data_bytes"
        hash1 = ModelFingerprinter.hash_data(data)
        hash2 = ModelFingerprinter.hash_data(data)
        
        assert hash1 == hash2
        assert len(hash1) == 64
    
    def test_hash_data_file(self):
        """Test hashing file data."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test file content")
            temp_path = Path(f.name)
        
        try:
            hash1 = ModelFingerprinter.hash_data(temp_path)
            hash2 = ModelFingerprinter.hash_data(temp_path)
            
            assert hash1 == hash2
            assert len(hash1) == 64
        finally:
            temp_path.unlink()
    
    def test_get_model_size_mb(self):
        """Test calculating model size."""
        model = SimpleModel(input_size=100, hidden_size=50, output_size=10)
        size_mb = ModelFingerprinter.get_model_size_mb(model)
        
        assert size_mb > 0
        assert isinstance(size_mb, float)
        
        # Larger model should have larger size
        large_model = SimpleModel(input_size=1000, hidden_size=500, output_size=100)
        large_size_mb = ModelFingerprinter.get_model_size_mb(large_model)
        
        assert large_size_mb > size_mb


class TestCompatibilityChecker:
    """Test CompatibilityChecker class."""
    
    def test_version_compatibility_matching(self):
        """Test compatibility checking with matching versions."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            timestamp=datetime.now(),
            hyperparameters={},
            performance_metrics={},
            training_data_hash="abc123",
            code_version="1.0.0",
            torch_version=torch.__version__
        )
        
        issues = CompatibilityChecker.check_version_compatibility(metadata, "1.0.0")
        
        # Should have no issues for matching versions
        platform_issues = [issue for issue in issues if "Platform" in issue]
        code_issues = [issue for issue in issues if "Code version" in issue]
        torch_issues = [issue for issue in issues if "PyTorch version" in issue]
        
        assert len(code_issues) == 0
        assert len(torch_issues) == 0
    
    def test_version_compatibility_mismatch(self):
        """Test compatibility checking with mismatched versions."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            timestamp=datetime.now(),
            hyperparameters={},
            performance_metrics={},
            training_data_hash="abc123",
            code_version="1.0.0",
            torch_version="1.10.0"  # Different from current
        )
        
        issues = CompatibilityChecker.check_version_compatibility(metadata, "2.0.0")
        
        # Should have issues for mismatched versions
        assert len(issues) >= 2  # At least code version and torch version
        assert any("Code version mismatch" in issue for issue in issues)
        assert any("PyTorch version mismatch" in issue for issue in issues)
    
    def test_device_compatibility(self):
        """Test device compatibility checking."""
        # Model trained on CUDA but current system is CPU
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            timestamp=datetime.now(),
            hyperparameters={},
            performance_metrics={},
            training_data_hash="abc123",
            code_version="1.0.0",
            device="cuda"
        )
        
        issues = CompatibilityChecker.check_version_compatibility(metadata, "1.0.0")
        
        if not torch.cuda.is_available():
            assert any("CUDA not available" in issue for issue in issues)


class TestModelComparator:
    """Test ModelComparator class."""
    
    def test_compare_performance(self):
        """Test comparing performance between models."""
        metadata1 = ModelMetadata(
            name="model1",
            version="1.0.0",
            timestamp=datetime.now(),
            hyperparameters={},
            performance_metrics={"accuracy": 0.90, "loss": 0.10},
            training_data_hash="abc123",
            code_version="1.0.0"
        )
        
        metadata2 = ModelMetadata(
            name="model2",
            version="1.0.0",
            timestamp=datetime.now(),
            hyperparameters={},
            performance_metrics={"accuracy": 0.95, "loss": 0.08},
            training_data_hash="abc123",
            code_version="1.0.0"
        )
        
        comparison = ModelComparator.compare_performance(metadata1, metadata2)
        
        assert comparison["model1"]["name"] == "model1"
        assert comparison["model2"]["name"] == "model2"
        assert "accuracy" in comparison["metrics_comparison"]
        assert "loss" in comparison["metrics_comparison"]
        
        # Check accuracy improvement
        acc_comparison = comparison["metrics_comparison"]["accuracy"]
        assert acc_comparison["model1"] == 0.90
        assert acc_comparison["model2"] == 0.95
        assert abs(acc_comparison["difference"] - 0.05) < 1e-10  # Handle floating point precision
        assert abs(acc_comparison["improvement_pct"] - 5.56) < 0.1  # ~5.56% improvement
    
    def test_compare_metadata(self):
        """Test comparing metadata between models."""
        metadata1 = ModelMetadata(
            name="model1",
            version="1.0.0",
            timestamp=datetime.now(),
            hyperparameters={"lr": 0.001},
            performance_metrics={},
            training_data_hash="abc123",
            code_version="1.0.0",
            model_size_mb=10.5
        )
        
        metadata2 = ModelMetadata(
            name="model2",
            version="2.0.0",
            timestamp=datetime.now(),
            hyperparameters={"lr": 0.002},  # Different
            performance_metrics={},
            training_data_hash="abc123",
            code_version="1.0.0",
            model_size_mb=15.2  # Different
        )
        
        comparison = ModelComparator.compare_metadata(metadata1, metadata2)
        
        assert "differences" in comparison
        assert "hyperparameters" in comparison["differences"]
        assert "model_size_mb" in comparison["differences"]
        
        # Check hyperparameters difference
        hp_diff = comparison["differences"]["hyperparameters"]
        assert hp_diff["model1"]["lr"] == 0.001
        assert hp_diff["model2"]["lr"] == 0.002


class TestModelTracker:
    """Test ModelTracker class."""
    
    def test_log_epoch(self):
        """Test logging epoch metrics."""
        tracker = ModelTracker()
        
        tracker.log_epoch(1, {"loss": 0.5, "accuracy": 0.8})
        tracker.log_epoch(2, {"loss": 0.4, "accuracy": 0.85})
        
        assert len(tracker.training_history) == 2
        assert tracker.current_epoch == 2
        assert tracker.training_history[0]["epoch"] == 1
        assert tracker.training_history[0]["metrics"]["loss"] == 0.5
        assert tracker.training_history[1]["metrics"]["accuracy"] == 0.85
    
    def test_get_best_epoch_maximize(self):
        """Test getting best epoch for maximizing metric."""
        tracker = ModelTracker()
        
        tracker.log_epoch(1, {"accuracy": 0.8, "loss": 0.5})
        tracker.log_epoch(2, {"accuracy": 0.85, "loss": 0.4})
        tracker.log_epoch(3, {"accuracy": 0.82, "loss": 0.45})
        
        best_epoch = tracker.get_best_epoch("accuracy", maximize=True)
        
        assert best_epoch is not None
        assert best_epoch["epoch"] == 2
        assert best_epoch["metrics"]["accuracy"] == 0.85
    
    def test_get_best_epoch_minimize(self):
        """Test getting best epoch for minimizing metric."""
        tracker = ModelTracker()
        
        tracker.log_epoch(1, {"accuracy": 0.8, "loss": 0.5})
        tracker.log_epoch(2, {"accuracy": 0.85, "loss": 0.4})
        tracker.log_epoch(3, {"accuracy": 0.82, "loss": 0.45})
        
        best_epoch = tracker.get_best_epoch("loss", maximize=False)
        
        assert best_epoch is not None
        assert best_epoch["epoch"] == 2
        assert best_epoch["metrics"]["loss"] == 0.4
    
    def test_get_best_epoch_no_data(self):
        """Test getting best epoch with no data."""
        tracker = ModelTracker()
        
        best_epoch = tracker.get_best_epoch("accuracy")
        assert best_epoch is None
    
    def test_get_training_summary(self):
        """Test getting training summary."""
        tracker = ModelTracker()
        
        # Log some epochs with different metrics
        tracker.log_epoch(1, {"loss": 0.5, "accuracy": 0.8})
        tracker.log_epoch(2, {"loss": 0.4, "accuracy": 0.85, "f1_score": 0.82})
        
        summary = tracker.get_training_summary()
        
        assert summary["total_epochs"] == 2
        assert "start_time" in summary
        assert "end_time" in summary
        assert "training_duration_hours" in summary
        assert set(summary["metrics_tracked"]) == {"loss", "accuracy", "f1_score"}
    
    def test_get_training_summary_empty(self):
        """Test getting training summary with no data."""
        tracker = ModelTracker()
        
        summary = tracker.get_training_summary()
        assert summary == {}


if __name__ == "__main__":
    pytest.main([__file__])