# ml/neural_amr.py
from geminisdr.core.error_handling import (
    ErrorHandler, GeminiSDRError, HardwareError, ModelError, MemoryError,
    ErrorSeverity, ErrorContext, retry_with_backoff
)
from geminisdr.core.logging_manager import StructuredLogger
from geminisdr.core.memory_manager import MemoryManager
from geminisdr.config.config_manager import get_config_manager, SystemConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import logging

class IQDataset(Dataset):
    """PyTorch dataset for I/Q samples."""
    
    def __init__(self, signals, labels, transform=None):
        self.signals = signals
        self.labels = labels
        self.transform = transform
        
        # Convert labels to numeric
        self.label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
        self.numeric_labels = [self.label_map[label] for label in labels]
        self.num_classes = len(self.label_map)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.numeric_labels[idx]
        
        # Convert complex to 2-channel real
        iq_data = np.stack([signal.real, signal.imag], axis=0).astype(np.float32)
        
        if self.transform:
            iq_data = self.transform(iq_data)
        
        return torch.from_numpy(iq_data), label

class CNNModulationClassifier(nn.Module):
    """CNN for modulation classification."""
    
    def __init__(self, num_classes=5, signal_length=1024):
        super(CNNModulationClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate size after convolutions
        conv_output_size = signal_length // 8  # After 3 max pools
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * conv_output_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class NeuralAMR:
    """Neural network-based AMR system with cross-platform compatibility and error handling."""
    
    def __init__(self, device=None, hardware_abstraction=None, config: SystemConfig = None):
        # Initialize error handling and logging
        if config is None:
            try:
                config_manager = get_config_manager()
                self.config = config_manager.get_config()
                if self.config is None:
                    self.config = config_manager.load_config()
            except Exception as e:
                # Use fallback configuration
                from geminisdr.config.config_models import SystemConfig, HardwareConfig, MLConfig, LoggingConfig, PerformanceConfig
                self.config = SystemConfig(
                    hardware=HardwareConfig(),
                    ml=MLConfig(),
                    logging=LoggingConfig(),
                    performance=PerformanceConfig()
                )
        else:
            self.config = config
        
        self.logger = StructuredLogger(__name__, self.config.logging)
        self.error_handler = ErrorHandler(self.logger.logger)
        self.memory_manager = MemoryManager(self.config)
        self._register_recovery_strategies()
        
        # Import hardware abstraction if available
        if hardware_abstraction is None:
            try:
                from environments.hardware_abstraction import HardwareAbstraction
                self.hw_abstraction = HardwareAbstraction()
            except ImportError:
                self.logger.logger.warning("Hardware abstraction not available, using fallback")
                self.hw_abstraction = None
        else:
            self.hw_abstraction = hardware_abstraction
        
        # Set device using hardware abstraction or fallback with error handling
        try:
            with self.error_handler.error_context("device_initialization", component="NeuralAMR"):
                if device is None:
                    if self.hw_abstraction:
                        self.device = torch.device(self.hw_abstraction.get_device_string())
                        self.optimization_config = self.hw_abstraction.optimize_for_platform()
                    else:
                        # Fallback device selection with improved compatibility
                        self.device = self._fallback_device_selection()
                        self.optimization_config = self._get_fallback_optimizations()
                else:
                    self.device = device
                    self.optimization_config = self._get_fallback_optimizations()
                
                self.logger.logger.info(f"Using device: {self.device}")
                if self.optimization_config:
                    self.logger.logger.info(f"Platform optimizations: {self.optimization_config}")
                
                # Apply platform-specific configurations
                self._configure_platform_settings()
                
                self.model = None
                self.label_map = None
                self.inverse_label_map = None
        except Exception as e:
            raise HardwareError(
                f"Failed to initialize device: {str(e)}",
                device_type=str(self.device) if hasattr(self, 'device') else 'unknown',
                severity=ErrorSeverity.HIGH,
                cause=e
            )
    
    def _fallback_device_selection(self):
        """Fallback device selection when hardware abstraction is unavailable."""
        try:
            # Check MPS availability (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device('mps')
            # Check CUDA availability
            elif torch.cuda.is_available():
                return torch.device('cuda')
            # Fallback to CPU
            else:
                return torch.device('cpu')
        except Exception as e:
            print(f"Warning: Error in device selection, falling back to CPU: {e}")
            return torch.device('cpu')
    
    def _get_fallback_optimizations(self):
        """Get basic optimizations when hardware abstraction is unavailable."""
        config = {
            'device': self.device.type,
            'recommended_batch_size': 32,
            'dataloader_num_workers': 2,
            'pin_memory': self.device.type in ['cuda', 'mps'],
            'thread_count': 4
        }
        
        if self.device.type == 'mps':
            config.update({
                'mixed_precision': False,  # Conservative for MPS
                'mps_fallback': True
            })
        elif self.device.type == 'cuda':
            config.update({
                'mixed_precision': True,
                'benchmark_mode': True
            })
        
        return config
    
    def _register_recovery_strategies(self):
        """Register error recovery strategies."""
        # Memory error recovery - reduce batch size and optimize memory
        def memory_recovery_strategy(error: Exception, context: ErrorContext) -> bool:
            try:
                self.logger.logger.warning(f"Memory error detected, attempting recovery: {error}")
                # Clean up memory
                self.memory_manager.cleanup_memory()
                
                # Reduce batch size if possible
                if hasattr(self, 'optimization_config'):
                    old_batch_size = self.optimization_config.get('recommended_batch_size', 32)
                    new_batch_size = max(8, old_batch_size // 2)
                    self.optimization_config['recommended_batch_size'] = new_batch_size
                    self.logger.logger.info(f"Reduced batch size from {old_batch_size} to {new_batch_size}")
                    return True
                return False
            except Exception as e:
                self.logger.logger.error(f"Memory recovery strategy failed: {e}")
                return False
        
        # Hardware error recovery - fallback to CPU
        def hardware_fallback_strategy(error: Exception, context: ErrorContext) -> bool:
            try:
                if self.device.type != 'cpu':
                    self.logger.logger.warning("Hardware error detected, falling back to CPU")
                    self.device = torch.device('cpu')
                    self._configure_platform_settings()
                    
                    # Move existing model to CPU if it exists
                    if self.model is not None:
                        self.model = self.model.to(self.device)
                    
                    self.logger.logger.info("Successfully switched to CPU device")
                    return True
                return False
            except Exception as e:
                self.logger.logger.error(f"Hardware fallback strategy failed: {e}")
                return False
        
        # Model error recovery - reinitialize model
        def model_recovery_strategy(error: Exception, context: ErrorContext) -> bool:
            try:
                self.logger.logger.warning("Model error detected, attempting to reinitialize")
                # Clear existing model
                if self.model is not None:
                    del self.model
                    self.memory_manager.cleanup_memory()
                
                self.model = None
                self.logger.logger.info("Model cleared for reinitialization")
                return True
            except Exception as e:
                self.logger.logger.error(f"Model recovery strategy failed: {e}")
                return False
        
        # Register strategies
        self.error_handler.register_recovery_strategy(MemoryError, memory_recovery_strategy)
        self.error_handler.register_recovery_strategy(HardwareError, hardware_fallback_strategy)
        self.error_handler.register_recovery_strategy(ModelError, model_recovery_strategy)
    
    def _configure_platform_settings(self):
        """Configure platform-specific PyTorch settings."""
        try:
            with self.error_handler.error_context("platform_configuration", component="NeuralAMR"):
                # Configure MPS fallback if needed
                if self.device.type == 'mps':
                    import os
                    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
                    self.logger.logger.info("Enabled MPS fallback for compatibility")
                
                # Configure CPU threading
                if self.device.type == 'cpu':
                    thread_count = self.optimization_config.get('thread_count', 4)
                    torch.set_num_threads(thread_count)
                    self.logger.logger.info(f"Set CPU threads to {thread_count}")
                
                # Configure CUDA optimizations
                elif self.device.type == 'cuda':
                    if self.optimization_config.get('benchmark_mode', False):
                        torch.backends.cudnn.benchmark = True
                        self.logger.logger.info("Enabled cuDNN benchmark mode")
                    
                    if self.optimization_config.get('use_deterministic_algorithms', False):
                        torch.use_deterministic_algorithms(True)
                        self.logger.logger.info("Enabled deterministic algorithms")
        except Exception as e:
            self.logger.log_error_with_context(e, component="NeuralAMR", operation="platform_configuration")
    
    def prepare_data(self, signals, labels, test_size=0.2, batch_size=None):
        """Prepare data for training with platform-specific optimizations."""
        # Use platform-optimized batch size if not specified
        if batch_size is None:
            batch_size = self.optimization_config.get('recommended_batch_size', 32)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            signals, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = IQDataset(X_train, y_train)
        val_dataset = IQDataset(X_val, y_val)
        
        # Store label mappings
        self.label_map = train_dataset.label_map
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Get platform-optimized dataloader settings
        num_workers = self.optimization_config.get('dataloader_num_workers', 2)
        pin_memory = self.optimization_config.get('pin_memory', False)
        
        # Platform-specific dataloader adjustments
        if self.device.type == 'mps':
            # MPS works better with fewer workers due to memory constraints
            num_workers = min(num_workers, 2)
            pin_memory = False  # Pin memory can cause issues on MPS
        elif self.device.type == 'cpu':
            # CPU can handle more workers for data loading
            num_workers = min(num_workers, 4)
            pin_memory = False
        
        print(f"DataLoader config: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
        
        # Create data loaders with platform optimizations
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0  # Improve performance with workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        return train_loader, val_loader, train_dataset.num_classes
    
    def train(self, signals, labels, epochs=50, learning_rate=0.001):
        """Train the neural network."""
        # Prepare data
        train_loader, val_loader, num_classes = self.prepare_data(signals, labels)
        
        # Initialize model
        signal_length = signals[0].shape[0]
        self.model = CNNModulationClassifier(num_classes, signal_length).to(self.device)
        
        # Apply platform-specific training optimizations
        self._apply_training_optimizations()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        print("Starting training...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    device_info = f"({self.device})" if self.device.type != 'cpu' else ""
                    print(f'\rEpoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] '
                          f'Loss: {loss.item():.4f} {device_info}', end='')
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100. * correct / len(val_loader.dataset)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_accuracy)
            
            print(f'\nEpoch {epoch+1}/{epochs} - '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Val Acc: {val_accuracy:.2f}%')
        
        # Plot training history
        self._plot_training_history(history)
        
        # Final evaluation
        self._evaluate_model(val_loader)
        
        return history
    
    def _apply_training_optimizations(self):
        """Apply platform-specific training optimizations."""
        if self.device.type == 'mps':
            # MPS-specific optimizations
            if self.optimization_config.get('mixed_precision', False):
                print("Note: Mixed precision training not fully supported on MPS, using float32")
            print("Applied MPS-specific optimizations")
        elif self.device.type == 'cuda':
            # CUDA-specific optimizations already applied in _configure_platform_settings
            print("Applied CUDA-specific optimizations")
        elif self.device.type == 'cpu':
            # CPU-specific optimizations already applied in _configure_platform_settings
            print("Applied CPU-specific optimizations")
    
    def _plot_training_history(self, history):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history['val_acc'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _evaluate_model(self, val_loader):
        """Evaluate model and print classification report."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                output = self.model(data)
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(target.numpy())
        
        # Convert back to string labels
        pred_labels = [self.inverse_label_map[p] for p in all_preds]
        true_labels = [self.inverse_label_map[l] for l in all_labels]
        
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels))
    
    def predict(self, signal):
        """Predict modulation type of a signal."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        
        self.model.eval()
        
        # Prepare signal
        iq_data = np.stack([signal.real, signal.imag], axis=0).astype(np.float32)
        iq_tensor = torch.from_numpy(iq_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(iq_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
        
        prediction = self.inverse_label_map[pred_idx]
        probs = probabilities[0].cpu().numpy()
        
        return prediction, probs
    
    def save_model(self, filepath):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_map': self.label_map,
            'inverse_label_map': self.inverse_label_map,
            'model_config': {
                'num_classes': len(self.label_map),
                'signal_length': 1024
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model."""
        try:
            # Try with weights_only=False for compatibility
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            checkpoint = torch.load(filepath, map_location=self.device)
        except Exception as e:
            print(f"Warning: Error loading model with weights_only=False: {e}")
            # Final fallback
            checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore label mappings
        self.label_map = checkpoint['label_map']
        self.inverse_label_map = checkpoint['inverse_label_map']
        
        # Initialize model
        config = checkpoint['model_config']
        self.model = CNNModulationClassifier(
            config['num_classes'], 
            config['signal_length']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {filepath} on device: {self.device}")