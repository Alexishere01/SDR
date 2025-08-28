#!/usr/bin/env python3
"""
Model Management System Demo

This example demonstrates how to use the GeminiSDR model management system
to save, load, version, and track ML models with comprehensive metadata.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path

from geminisdr.core import ModelManager, ModelMetadata, ModelTracker


class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


def simulate_training(model, epochs=10):
    """Simulate model training and return training history."""
    tracker = ModelTracker()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Simulating training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Simulate training data
        x = torch.randn(32, 10)  # batch_size=32, input_size=10
        y = torch.randn(32, 1)   # target values
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Simulate validation
        with torch.no_grad():
            val_x = torch.randn(16, 10)
            val_y = torch.randn(16, 1)
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y)
            
            # Calculate dummy accuracy (for demonstration)
            accuracy = max(0.5, 1.0 - val_loss.item())
        
        # Log epoch metrics
        tracker.log_epoch(epoch + 1, {
            'train_loss': loss.item(),
            'val_loss': val_loss.item(),
            'accuracy': accuracy
        })
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}, Acc={accuracy:.4f}")
    
    return tracker


def main():
    """Main demonstration function."""
    print("=== GeminiSDR Model Management System Demo ===\n")
    
    # Initialize model manager
    print("1. Initializing Model Manager...")
    manager = ModelManager()
    print("   ✓ Model Manager initialized\n")
    
    # Create and train a model
    print("2. Creating and training model...")
    model = SimpleNeuralNetwork(input_size=10, hidden_size=20, output_size=1)
    tracker = simulate_training(model, epochs=10)
    
    # Get training summary
    training_summary = tracker.get_training_summary()
    best_epoch = tracker.get_best_epoch('accuracy', maximize=True)
    
    print(f"   ✓ Training completed in {training_summary['training_duration_hours']:.2f} hours")
    print(f"   ✓ Best accuracy: {best_epoch['metrics']['accuracy']:.4f} at epoch {best_epoch['epoch']}\n")
    
    # Save the model
    print("3. Saving model with metadata...")
    hyperparameters = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'hidden_size': 20,
        'dropout': 0.2
    }
    
    performance_metrics = {
        'final_accuracy': best_epoch['metrics']['accuracy'],
        'final_train_loss': tracker.training_history[-1]['metrics']['train_loss'],
        'final_val_loss': tracker.training_history[-1]['metrics']['val_loss'],
        'best_accuracy': best_epoch['metrics']['accuracy'],
        'training_time_hours': training_summary['training_duration_hours']
    }
    
    training_data_info = {
        'dataset': 'simulated_data',
        'input_features': 10,
        'samples': 1000,
        'validation_split': 0.2
    }
    
    run_id = manager.save_model(
        model=model,
        model_name="demo_neural_network",
        version="1.0.0",
        hyperparameters=hyperparameters,
        performance_metrics=performance_metrics,
        training_data_info=training_data_info,
        description="Demo neural network for model management system",
        tags=["demo", "neural_network", "regression"]
    )
    
    print(f"   ✓ Model saved with run ID: {run_id}\n")
    
    # List available models
    print("4. Listing available models...")
    models = manager.list_models()
    for model_info in models:
        print(f"   Model: {model_info['name']}")
        print(f"   Latest version: {model_info['latest_version']}")
        print(f"   Available versions: {len(model_info['versions'])}")
        for version in model_info['versions']:
            print(f"     - v{version['version']}: {version['description'] or 'No description'}")
    print()
    
    # Load the model back
    print("5. Loading model...")
    loaded_model, metadata = manager.load_model("demo_neural_network", "1.0.0")
    print(f"   ✓ Model loaded: {metadata.name} v{metadata.version}")
    print(f"   ✓ Model architecture: {metadata.model_architecture}")
    print(f"   ✓ Model size: {metadata.model_size_mb:.2f} MB")
    print(f"   ✓ Training accuracy: {metadata.performance_metrics['final_accuracy']:.4f}\n")
    
    # Validate model compatibility
    print("6. Checking model compatibility...")
    compatibility_issues = manager.validate_model_compatibility("demo_neural_network", "1.0.0")
    if compatibility_issues:
        print("   ⚠ Compatibility issues found:")
        for issue in compatibility_issues:
            print(f"     - {issue}")
    else:
        print("   ✓ Model is fully compatible with current system")
    print()
    
    # Get detailed model information
    print("7. Getting detailed model information...")
    model_info = manager.get_model_info("demo_neural_network", "1.0.0")
    print(f"   Model fingerprint: {model_info['fingerprint']}")
    print(f"   File sizes:")
    for filename, size_mb in model_info['file_sizes_mb'].items():
        print(f"     - {filename}: {size_mb:.2f} MB")
    print()
    
    # Create a second version with different hyperparameters
    print("8. Training and saving improved model version...")
    improved_model = SimpleNeuralNetwork(input_size=10, hidden_size=30, output_size=1)  # Larger hidden layer
    improved_tracker = simulate_training(improved_model, epochs=15)
    
    improved_summary = improved_tracker.get_training_summary()
    improved_best = improved_tracker.get_best_epoch('accuracy', maximize=True)
    
    improved_hyperparameters = hyperparameters.copy()
    improved_hyperparameters.update({
        'epochs': 15,
        'hidden_size': 30
    })
    
    improved_performance = {
        'final_accuracy': improved_best['metrics']['accuracy'],
        'final_train_loss': improved_tracker.training_history[-1]['metrics']['train_loss'],
        'final_val_loss': improved_tracker.training_history[-1]['metrics']['val_loss'],
        'best_accuracy': improved_best['metrics']['accuracy'],
        'training_time_hours': improved_summary['training_duration_hours']
    }
    
    manager.save_model(
        model=improved_model,
        model_name="demo_neural_network",
        version="1.1.0",
        hyperparameters=improved_hyperparameters,
        performance_metrics=improved_performance,
        training_data_info=training_data_info,
        description="Improved version with larger hidden layer",
        tags=["demo", "neural_network", "regression", "improved"]
    )
    
    print(f"   ✓ Improved model saved as v1.1.0\n")
    
    # Compare model versions
    print("9. Comparing model versions...")
    comparison = manager.compare_models(
        "demo_neural_network", "1.0.0",
        "demo_neural_network", "1.1.0"
    )
    
    print("   Performance comparison:")
    perf_comp = comparison['performance_comparison']['metrics_comparison']
    for metric, values in perf_comp.items():
        improvement = values['improvement_pct']
        print(f"     {metric}: {values['model1']:.4f} → {values['model2']:.4f} ({improvement:+.1f}%)")
    
    print("\n   Architecture differences:")
    meta_comp = comparison['metadata_comparison']['differences']
    for field, values in meta_comp.items():
        if field == 'hyperparameters':
            print(f"     {field}:")
            for param, val in values['model2'].items():
                if param not in values['model1'] or values['model1'][param] != val:
                    old_val = values['model1'].get(param, 'N/A')
                    print(f"       {param}: {old_val} → {val}")
        else:
            print(f"     {field}: {values['model1']} → {values['model2']}")
    print()
    
    # Demonstrate model migration
    print("10. Demonstrating model migration...")
    success = manager.migrate_model(
        "demo_neural_network", "1.0.0", "1.0.1",
        "Migrated to current codebase version"
    )
    
    if success:
        print("    ✓ Model successfully migrated from v1.0.0 to v1.0.1")
    else:
        print("    ✗ Model migration failed")
    print()
    
    # Final model listing
    print("11. Final model inventory...")
    final_models = manager.list_models()
    for model_info in final_models:
        print(f"   Model: {model_info['name']}")
        print(f"   Versions: {len(model_info['versions'])}")
        for version in model_info['versions']:
            metrics = version['performance_metrics']
            acc = metrics.get('final_accuracy', metrics.get('best_accuracy', 'N/A'))
            print(f"     - v{version['version']}: accuracy={acc}")
    
    print("\n=== Demo completed successfully! ===")
    print("\nThe model management system provides:")
    print("• Comprehensive metadata tracking")
    print("• Version control and comparison")
    print("• Compatibility checking")
    print("• MLflow integration for experiment tracking")
    print("• Model migration utilities")
    print("• Performance metrics tracking")


if __name__ == "__main__":
    main()