"""
Model management system with MLflow integration for GeminiSDR.

This module provides comprehensive model lifecycle management including
saving, loading, versioning, and migration with MLflow tracking.
"""

import os
import json
import shutil
import tempfile
import git
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from .model_metadata import (
    ModelMetadata, ModelFingerprinter, CompatibilityChecker, 
    ModelComparator, PerformanceMetrics
)
from ..config.config_manager import ConfigManager


class ModelManager:
    """Centralized model management with MLflow integration."""
    
    def __init__(self, config: Optional[Any] = None, tracking_uri: Optional[str] = None):
        """
        Initialize ModelManager.
        
        Args:
            config: System configuration object
            tracking_uri: MLflow tracking URI (defaults to local file store)
        """
        self.config = config or ConfigManager().load_config()
        
        # Set up MLflow tracking
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local file store
            mlflow_dir = Path("mlruns")
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
        
        self.client = MlflowClient()
        
        # Set up local models directory
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Set up experiment
        self.experiment_name = "geminisdr_models"
        try:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            print(f"Warning: Could not set up MLflow experiment: {e}")
            self.experiment_id = None
    
    def save_model(self, 
                   model: nn.Module, 
                   model_name: str,
                   version: str,
                   hyperparameters: Dict[str, Any],
                   performance_metrics: Dict[str, float],
                   training_data_info: Optional[Dict[str, Any]] = None,
                   description: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> str:
        """
        Save model with comprehensive metadata and MLflow tracking.
        
        Args:
            model: PyTorch model to save
            model_name: Name of the model
            version: Version string
            hyperparameters: Training hyperparameters
            performance_metrics: Model performance metrics
            training_data_info: Information about training data
            description: Optional model description
            tags: Optional tags for the model
            
        Returns:
            Model run ID from MLflow
        """
        # Generate training data hash
        training_data_hash = "unknown"
        if training_data_info:
            training_data_hash = ModelFingerprinter.hash_data(
                json.dumps(training_data_info, sort_keys=True)
            )
        
        # Get code version
        code_version = self._get_code_version()
        
        # Create metadata
        metadata = ModelMetadata(
            name=model_name,
            version=version,
            timestamp=datetime.now(),
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            training_data_hash=training_data_hash,
            code_version=code_version,
            model_size_mb=ModelFingerprinter.get_model_size_mb(model),
            model_architecture=str(type(model).__name__),
            description=description,
            tags=tags or []
        )
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_params(hyperparameters)
            
            # Log metrics
            mlflow.log_metrics(performance_metrics)
            
            # Log metadata as tags
            mlflow.set_tags({
                "model_name": model_name,
                "version": version,
                "code_version": code_version,
                "model_architecture": metadata.model_architecture,
                "model_size_mb": str(metadata.model_size_mb),
                "training_data_hash": training_data_hash
            })
            
            # Add custom tags
            if tags:
                for tag in tags:
                    mlflow.set_tag(f"custom_tag_{tag}", "true")
            
            # Log model with MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
            
            # Save metadata as artifact
            metadata_path = Path(tempfile.mkdtemp()) / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            mlflow.log_artifact(str(metadata_path), "metadata")
            
            # Save model locally as well
            local_model_dir = self.models_dir / model_name / version
            local_model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save PyTorch model
            model_path = local_model_dir / "model.pth"
            torch.save(model.state_dict(), model_path)
            
            # Save complete model (architecture + weights)
            complete_model_path = local_model_dir / "complete_model.pth"
            torch.save(model, complete_model_path)
            
            # Save metadata locally
            local_metadata_path = local_model_dir / "metadata.json"
            with open(local_metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Generate and save model hash for integrity checking
            model_hash = ModelFingerprinter.hash_model_weights(model)
            hash_path = local_model_dir / "model_hash.txt"
            with open(hash_path, 'w') as f:
                f.write(model_hash)
            
            print(f"Model saved successfully:")
            print(f"  MLflow run ID: {run_id}")
            print(f"  Local path: {local_model_dir}")
            print(f"  Model fingerprint: {metadata.get_fingerprint()}")
            
            return run_id
    
    def load_model(self, 
                   model_name: str, 
                   version: str = "latest",
                   run_id: Optional[str] = None) -> Tuple[nn.Module, ModelMetadata]:
        """
        Load model with metadata validation.
        
        Args:
            model_name: Name of the model to load
            version: Version to load ("latest" for most recent)
            run_id: Specific MLflow run ID to load from
            
        Returns:
            Tuple of (model, metadata)
        """
        if run_id:
            # Load from specific MLflow run
            return self._load_from_mlflow_run(run_id)
        
        # Load from local storage
        if version == "latest":
            version = self._get_latest_version(model_name)
        
        model_dir = self.models_dir / model_name / version
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name} version {version} not found")
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for {model_name} version {version}")
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)
        
        # Check compatibility
        current_version = self._get_code_version()
        compatibility_issues = CompatibilityChecker.check_version_compatibility(
            metadata, current_version
        )
        
        if compatibility_issues:
            print("Warning: Compatibility issues detected:")
            for issue in compatibility_issues:
                print(f"  - {issue}")
        
        # Verify model integrity
        hash_path = model_dir / "model_hash.txt"
        if hash_path.exists():
            with open(hash_path, 'r') as f:
                expected_hash = f.read().strip()
            
            # Load model to check hash
            complete_model_path = model_dir / "complete_model.pth"
            if complete_model_path.exists():
                model = torch.load(complete_model_path, map_location='cpu', weights_only=False)
                actual_hash = ModelFingerprinter.hash_model_weights(model)
                
                if actual_hash != expected_hash:
                    raise ValueError(f"Model integrity check failed for {model_name} v{version}")
            else:
                print("Warning: Complete model file not found, skipping integrity check")
                # Load state dict instead
                model_path = model_dir / "model.pth"
                if not model_path.exists():
                    raise FileNotFoundError(f"Model weights not found for {model_name} v{version}")
                
                # Note: Loading state dict requires model architecture to be provided separately
                raise NotImplementedError(
                    "Loading from state dict requires model architecture. "
                    "Use complete_model.pth or provide model architecture."
                )
        else:
            # Load complete model
            complete_model_path = model_dir / "complete_model.pth"
            if complete_model_path.exists():
                model = torch.load(complete_model_path, map_location='cpu', weights_only=False)
            else:
                raise FileNotFoundError(f"Model file not found for {model_name} v{version}")
        
        print(f"Model loaded successfully: {model_name} v{version}")
        return model, metadata
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with metadata."""
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            versions = []
            
            for version_dir in model_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                
                version = version_dir.name
                metadata_path = version_dir / "metadata.json"
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata_dict = json.load(f)
                        metadata = ModelMetadata.from_dict(metadata_dict)
                        
                        versions.append({
                            'version': version,
                            'timestamp': metadata.timestamp.isoformat(),
                            'performance_metrics': metadata.performance_metrics,
                            'model_size_mb': metadata.model_size_mb,
                            'description': metadata.description
                        })
                    except Exception as e:
                        print(f"Warning: Could not load metadata for {model_name} v{version}: {e}")
            
            if versions:
                # Sort versions by timestamp (newest first)
                versions.sort(key=lambda x: x['timestamp'], reverse=True)
                models.append({
                    'name': model_name,
                    'versions': versions,
                    'latest_version': versions[0]['version'] if versions else None
                })
        
        return models
    
    def compare_models(self, 
                      model1_name: str, model1_version: str,
                      model2_name: str, model2_version: str) -> Dict[str, Any]:
        """Compare two model versions."""
        # Load metadata for both models
        _, metadata1 = self.load_model(model1_name, model1_version)
        _, metadata2 = self.load_model(model2_name, model2_version)
        
        # Use ModelComparator for comparison
        performance_comparison = ModelComparator.compare_performance(metadata1, metadata2)
        metadata_comparison = ModelComparator.compare_metadata(metadata1, metadata2)
        
        return {
            'performance_comparison': performance_comparison,
            'metadata_comparison': metadata_comparison
        }
    
    def migrate_model(self, 
                     model_name: str, 
                     from_version: str, 
                     to_version: str,
                     migration_notes: Optional[str] = None) -> bool:
        """
        Migrate model between versions.
        
        This creates a copy of the model with updated metadata.
        """
        try:
            # Load the source model
            model, old_metadata = self.load_model(model_name, from_version)
            
            # Create new metadata with updated version
            new_metadata = ModelMetadata(
                name=model_name,
                version=to_version,
                timestamp=datetime.now(),
                hyperparameters=old_metadata.hyperparameters,
                performance_metrics=old_metadata.performance_metrics,
                training_data_hash=old_metadata.training_data_hash,
                code_version=self._get_code_version(),  # Update to current code version
                model_size_mb=old_metadata.model_size_mb,
                model_architecture=old_metadata.model_architecture,
                input_shape=old_metadata.input_shape,
                output_shape=old_metadata.output_shape,
                description=f"Migrated from v{from_version}. {migration_notes or ''}",
                tags=old_metadata.tags + ["migrated"]
            )
            
            # Save as new version
            self.save_model(
                model=model,
                model_name=model_name,
                version=to_version,
                hyperparameters=old_metadata.hyperparameters,
                performance_metrics=old_metadata.performance_metrics,
                description=new_metadata.description,
                tags=new_metadata.tags
            )
            
            print(f"Successfully migrated {model_name} from v{from_version} to v{to_version}")
            return True
            
        except Exception as e:
            print(f"Migration failed: {e}")
            return False
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """
        Delete a model version or entire model.
        
        Args:
            model_name: Name of the model
            version: Specific version to delete (if None, deletes entire model)
        """
        try:
            if version:
                # Delete specific version
                version_dir = self.models_dir / model_name / version
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                    print(f"Deleted {model_name} version {version}")
                else:
                    print(f"Version {version} of {model_name} not found")
                    return False
            else:
                # Delete entire model
                model_dir = self.models_dir / model_name
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    print(f"Deleted entire model {model_name}")
                else:
                    print(f"Model {model_name} not found")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Deletion failed: {e}")
            return False
    
    def validate_model_compatibility(self, model_name: str, version: str) -> List[str]:
        """Validate model compatibility with current system."""
        try:
            _, metadata = self.load_model(model_name, version)
            current_version = self._get_code_version()
            return CompatibilityChecker.check_version_compatibility(metadata, current_version)
        except Exception as e:
            return [f"Could not validate compatibility: {e}"]
    
    def get_model_info(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        try:
            _, metadata = self.load_model(model_name, version)
            
            # Get file sizes
            model_dir = self.models_dir / model_name / version
            file_sizes = {}
            for file_path in model_dir.iterdir():
                if file_path.is_file():
                    file_sizes[file_path.name] = file_path.stat().st_size / (1024 * 1024)  # MB
            
            return {
                'metadata': metadata.to_dict(),
                'file_sizes_mb': file_sizes,
                'fingerprint': metadata.get_fingerprint(),
                'compatibility_issues': self.validate_model_compatibility(model_name, version)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _load_from_mlflow_run(self, run_id: str) -> Tuple[nn.Module, ModelMetadata]:
        """Load model from specific MLflow run."""
        # Download model from MLflow
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)
        
        # Download metadata
        client = MlflowClient()
        metadata_path = client.download_artifacts(run_id, "metadata/metadata.json")
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)
        
        return model, metadata
    
    def _get_latest_version(self, model_name: str) -> str:
        """Get the latest version of a model."""
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        versions = []
        for version_dir in model_dir.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata_dict = json.load(f)
                        metadata = ModelMetadata.from_dict(metadata_dict)
                        versions.append((version_dir.name, metadata.timestamp))
                    except Exception:
                        continue
        
        if not versions:
            raise FileNotFoundError(f"No valid versions found for {model_name}")
        
        # Sort by timestamp and return latest
        versions.sort(key=lambda x: x[1], reverse=True)
        return versions[0][0]
    
    def _get_code_version(self) -> str:
        """Get current code version from git or fallback."""
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha[:8]  # Short commit hash
        except Exception:
            # Fallback to a default version
            return "unknown"


class ModelRegistry:
    """Registry for managing model metadata and discovery."""
    
    def __init__(self, manager: ModelManager):
        self.manager = manager
    
    def register_model_type(self, 
                           model_type: str, 
                           architecture_class: type,
                           default_hyperparameters: Dict[str, Any]) -> None:
        """Register a new model type with default configuration."""
        # This could be extended to maintain a registry of model types
        # For now, it's a placeholder for future functionality
        pass
    
    def search_models(self, 
                     query: str = "",
                     tags: Optional[List[str]] = None,
                     min_performance: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Search models based on criteria."""
        all_models = self.manager.list_models()
        filtered_models = []
        
        for model_info in all_models:
            model_name = model_info['name']
            
            # Filter by query (name matching)
            if query and query.lower() not in model_name.lower():
                continue
            
            # Check each version
            matching_versions = []
            for version_info in model_info['versions']:
                version = version_info['version']
                
                try:
                    # Load metadata for detailed filtering
                    _, metadata = self.manager.load_model(model_name, version)
                    
                    # Filter by tags
                    if tags:
                        if not any(tag in metadata.tags for tag in tags):
                            continue
                    
                    # Filter by minimum performance
                    if min_performance:
                        meets_criteria = True
                        for metric, min_value in min_performance.items():
                            if metric not in metadata.performance_metrics:
                                meets_criteria = False
                                break
                            if metadata.performance_metrics[metric] < min_value:
                                meets_criteria = False
                                break
                        
                        if not meets_criteria:
                            continue
                    
                    matching_versions.append(version_info)
                    
                except Exception as e:
                    print(f"Warning: Could not filter {model_name} v{version}: {e}")
                    continue
            
            if matching_versions:
                model_info_copy = model_info.copy()
                model_info_copy['versions'] = matching_versions
                filtered_models.append(model_info_copy)
        
        return filtered_models