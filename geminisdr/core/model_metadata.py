"""
Model metadata and tracking system for GeminiSDR.

This module provides comprehensive metadata structures for model versioning,
fingerprinting, and compatibility checking.
"""

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch


@dataclass
class ModelMetadata:
    """Comprehensive model metadata for versioning and tracking."""
    
    # Basic identification
    name: str
    version: str
    timestamp: datetime
    
    # Training configuration
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    
    # Reproducibility information
    training_data_hash: str
    code_version: str
    random_seed: Optional[int] = None
    
    # System information
    platform: str = field(default_factory=lambda: platform.platform())
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    python_version: str = field(default_factory=lambda: sys.version)
    torch_version: str = field(default_factory=lambda: torch.__version__)
    
    # Model characteristics
    model_size_mb: Optional[float] = None
    model_architecture: Optional[str] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    
    # Additional metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        data = asdict(self)
        # Convert datetime to ISO format string
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create ModelMetadata from dictionary."""
        # Convert timestamp string back to datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def get_fingerprint(self) -> str:
        """Generate unique fingerprint for this model version."""
        # Create fingerprint from key identifying information
        fingerprint_data = {
            'name': self.name,
            'version': self.version,
            'hyperparameters': self.hyperparameters,
            'training_data_hash': self.training_data_hash,
            'code_version': self.code_version,
            'model_architecture': self.model_architecture
        }
        
        # Sort keys for consistent hashing
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()


@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    
    # Training metrics
    train_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    
    # Inference metrics
    inference_time_ms: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Domain-specific metrics
    signal_detection_accuracy: Optional[float] = None
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    
    # Training time metrics
    training_time_hours: Optional[float] = None
    epochs_trained: Optional[int] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary, filtering out None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class ModelFingerprinter:
    """Utility class for generating model fingerprints and checksums."""
    
    @staticmethod
    def hash_model_weights(model: torch.nn.Module) -> str:
        """Generate hash of model weights for integrity checking."""
        hasher = hashlib.sha256()
        
        for param in model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        
        return hasher.hexdigest()
    
    @staticmethod
    def hash_data(data: Union[str, bytes, Path]) -> str:
        """Generate hash of training data or file."""
        hasher = hashlib.sha256()
        
        if isinstance(data, (str, Path)):
            # Hash file contents
            path = Path(data)
            if path.exists():
                with open(path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
            else:
                # Hash the string itself
                hasher.update(str(data).encode())
        elif isinstance(data, bytes):
            hasher.update(data)
        else:
            # Convert to string and hash
            hasher.update(str(data).encode())
        
        return hasher.hexdigest()
    
    @staticmethod
    def get_model_size_mb(model: torch.nn.Module) -> float:
        """Calculate model size in megabytes."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return round(size_mb, 2)


class CompatibilityChecker:
    """Utility class for checking model compatibility."""
    
    @staticmethod
    def check_version_compatibility(metadata: ModelMetadata, 
                                  current_version: str) -> List[str]:
        """Check if model is compatible with current system version."""
        issues = []
        
        # Check code version compatibility
        if metadata.code_version != current_version:
            issues.append(f"Code version mismatch: model={metadata.code_version}, "
                         f"current={current_version}")
        
        # Check PyTorch version compatibility
        current_torch_version = torch.__version__
        if metadata.torch_version != current_torch_version:
            issues.append(f"PyTorch version mismatch: model={metadata.torch_version}, "
                         f"current={current_torch_version}")
        
        # Check platform compatibility
        current_platform = platform.platform()
        if metadata.platform != current_platform:
            issues.append(f"Platform difference: model={metadata.platform}, "
                         f"current={current_platform}")
        
        # Check device compatibility
        current_device = "cuda" if torch.cuda.is_available() else "cpu"
        if metadata.device == "cuda" and current_device == "cpu":
            issues.append("Model trained on CUDA but CUDA not available")
        
        return issues
    
    @staticmethod
    def check_model_integrity(model_path: Path, expected_hash: str) -> bool:
        """Check if model file integrity matches expected hash."""
        if not model_path.exists():
            return False
        
        actual_hash = ModelFingerprinter.hash_data(model_path)
        return actual_hash == expected_hash


class ModelComparator:
    """Utility class for comparing different model versions."""
    
    @staticmethod
    def compare_performance(metadata1: ModelMetadata, 
                          metadata2: ModelMetadata) -> Dict[str, Any]:
        """Compare performance metrics between two models."""
        comparison = {
            'model1': {'name': metadata1.name, 'version': metadata1.version},
            'model2': {'name': metadata2.name, 'version': metadata2.version},
            'metrics_comparison': {}
        }
        
        # Compare common metrics
        for metric in metadata1.performance_metrics:
            if metric in metadata2.performance_metrics:
                val1 = metadata1.performance_metrics[metric]
                val2 = metadata2.performance_metrics[metric]
                
                comparison['metrics_comparison'][metric] = {
                    'model1': val1,
                    'model2': val2,
                    'difference': val2 - val1,
                    'improvement_pct': ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                }
        
        return comparison
    
    @staticmethod
    def compare_metadata(metadata1: ModelMetadata, 
                        metadata2: ModelMetadata) -> Dict[str, Any]:
        """Compare all metadata between two models."""
        comparison = {
            'basic_info': {
                'model1': {'name': metadata1.name, 'version': metadata1.version},
                'model2': {'name': metadata2.name, 'version': metadata2.version}
            },
            'differences': {}
        }
        
        # Compare key fields
        fields_to_compare = [
            'hyperparameters', 'platform', 'device', 'model_architecture',
            'model_size_mb', 'input_shape', 'output_shape'
        ]
        
        for field in fields_to_compare:
            val1 = getattr(metadata1, field)
            val2 = getattr(metadata2, field)
            
            if val1 != val2:
                comparison['differences'][field] = {
                    'model1': val1,
                    'model2': val2
                }
        
        return comparison


class ModelTracker:
    """Track model training progress and metrics."""
    
    def __init__(self):
        self.training_history: List[Dict[str, Any]] = []
        self.current_epoch = 0
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a training epoch."""
        self.current_epoch = epoch
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now(),
            'metrics': metrics.copy()
        }
        self.training_history.append(epoch_data)
    
    def get_best_epoch(self, metric: str, maximize: bool = True) -> Optional[Dict[str, Any]]:
        """Get the epoch with the best value for a specific metric."""
        if not self.training_history:
            return None
        
        valid_epochs = [epoch for epoch in self.training_history 
                       if metric in epoch['metrics']]
        
        if not valid_epochs:
            return None
        
        if maximize:
            return max(valid_epochs, key=lambda x: x['metrics'][metric])
        else:
            return min(valid_epochs, key=lambda x: x['metrics'][metric])
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_history:
            return {}
        
        summary = {
            'total_epochs': len(self.training_history),
            'start_time': self.training_history[0]['timestamp'],
            'end_time': self.training_history[-1]['timestamp'],
            'metrics_tracked': set()
        }
        
        # Collect all metrics that were tracked
        for epoch in self.training_history:
            summary['metrics_tracked'].update(epoch['metrics'].keys())
        
        summary['metrics_tracked'] = list(summary['metrics_tracked'])
        
        # Calculate training duration
        duration = summary['end_time'] - summary['start_time']
        summary['training_duration_hours'] = duration.total_seconds() / 3600
        
        return summary