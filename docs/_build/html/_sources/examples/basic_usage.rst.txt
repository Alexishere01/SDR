Basic Usage Examples
===================

This section provides basic examples to get you started with GeminiSDR quickly.

Quick Start Example
------------------

Here's a minimal example to get GeminiSDR running:

.. code-block:: python

   from geminisdr.config.config_manager import ConfigManager
   from geminisdr.environments.hardware_abstraction import HardwareAbstraction
   from geminisdr.ml.neural_amr import NeuralAMR
   
   # Load configuration
   config_manager = ConfigManager()
   config = config_manager.load_config()
   
   # Initialize hardware abstraction
   hw_abstraction = HardwareAbstraction(config)
   device = hw_abstraction.detect_optimal_device()
   print(f"Using device: {device}")
   
   # Create and configure model
   model = NeuralAMR(config)
   model = model.to(device)
   
   # Generate sample data for testing
   import torch
   sample_data = torch.randn(32, 1024, 2).to(device)  # Batch of I/Q samples
   
   # Run inference
   with torch.no_grad():
       predictions = model(sample_data)
       print(f"Predictions shape: {predictions.shape}")

Configuration Examples
---------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~

Create a simple configuration file:

.. code-block:: yaml

   # conf/basic_config.yaml
   hardware:
     device_preference: "auto"  # Let system choose optimal device
     sdr_mode: "simulation"     # Use simulation for testing
   
   ml:
     batch_size: 32
     learning_rate: 0.001
     model_cache_size: 2
   
   logging:
     level: "INFO"
     format: "simple"
     output: ["console"]

Load and use the configuration:

.. code-block:: python

   from geminisdr.config.config_manager import ConfigManager
   
   # Load specific configuration
   config_manager = ConfigManager()
   config = config_manager.load_config("basic_config")
   
   # Access configuration values
   print(f"Device preference: {config.hardware.device_preference}")
   print(f"Batch size: {config.ml.batch_size}")
   print(f"Learning rate: {config.ml.learning_rate}")

Platform-Specific Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**M1 Mac Configuration**:

.. code-block:: yaml

   # conf/m1_config.yaml
   hardware:
     device_preference: "mps"
     memory_optimization: "unified"
   
   ml:
     batch_size: 48  # Larger batches for unified memory
     precision: "float16"
   
   performance:
     memory_threshold: 0.85

**Linux VM Configuration**:

.. code-block:: yaml

   # conf/vm_config.yaml
   hardware:
     device_preference: "cpu"
     memory_optimization: "conservative"
   
   ml:
     batch_size: 16  # Smaller batches for limited memory
     precision: "float32"
     gradient_checkpointing: true

**CUDA Configuration**:

.. code-block:: yaml

   # conf/cuda_config.yaml
   hardware:
     device_preference: "cuda"
     memory_optimization: "aggressive"
   
   ml:
     batch_size: 64  # Large batches for GPU
     precision: "mixed"
     distributed: true

Data Loading Examples
--------------------

Loading Signal Data
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import torch
   from torch.utils.data import Dataset, DataLoader
   
   class SignalDataset(Dataset):
       """Dataset for loading I/Q signal data."""
       
       def __init__(self, data_path, labels_path=None, transform=None):
           self.data = np.load(data_path)
           self.labels = np.load(labels_path) if labels_path else None
           self.transform = transform
       
       def __len__(self):
           return len(self.data)
       
       def __getitem__(self, idx):
           sample = self.data[idx]
           
           if self.transform:
               sample = self.transform(sample)
           
           if self.labels is not None:
               label = self.labels[idx]
               return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
           else:
               return torch.tensor(sample, dtype=torch.float32)
   
   # Create dataset and dataloader
   dataset = SignalDataset('signals.npy', 'labels.npy')
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
   
   # Iterate through data
   for batch_idx, (data, labels) in enumerate(dataloader):
       print(f"Batch {batch_idx}: data shape {data.shape}, labels shape {labels.shape}")
       if batch_idx >= 2:  # Just show first few batches
           break

Generating Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import torch
   
   def generate_bpsk_signal(num_samples=1024, snr_db=20):
       """Generate BPSK modulated signal with noise."""
       # Generate random bits
       bits = np.random.randint(0, 2, num_samples // 2)
       
       # BPSK modulation (map 0->-1, 1->1)
       symbols = 2 * bits - 1
       
       # Add noise
       noise_power = 10 ** (-snr_db / 10)
       noise = np.sqrt(noise_power / 2) * (
           np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
       )
       
       # Create complex signal
       signal = symbols + noise
       
       # Convert to I/Q format
       iq_signal = np.column_stack([signal.real, signal.imag])
       
       return iq_signal, bits
   
   def generate_dataset(num_signals=1000, modulations=['bpsk', 'qpsk', '8psk']):
       """Generate a dataset with multiple modulation types."""
       signals = []
       labels = []
       
       for i in range(num_signals):
           mod_type = np.random.choice(modulations)
           
           if mod_type == 'bpsk':
               signal, _ = generate_bpsk_signal()
               label = 0
           elif mod_type == 'qpsk':
               signal, _ = generate_qpsk_signal()
               label = 1
           elif mod_type == '8psk':
               signal, _ = generate_8psk_signal()
               label = 2
           
           signals.append(signal)
           labels.append(label)
       
       return np.array(signals), np.array(labels)
   
   # Generate and save dataset
   signals, labels = generate_dataset(1000)
   np.save('synthetic_signals.npy', signals)
   np.save('synthetic_labels.npy', labels)
   print(f"Generated dataset: {signals.shape} signals, {labels.shape} labels")

Model Usage Examples
-------------------

Neural AMR (Automatic Modulation Recognition)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.ml.neural_amr import NeuralAMR
   from geminisdr.config.config_manager import ConfigManager
   
   # Load configuration
   config = ConfigManager().load_config()
   
   # Create model
   model = NeuralAMR(config)
   
   # Load sample data
   sample_signals = torch.randn(10, 1024, 2)  # 10 signals, 1024 samples, I/Q
   
   # Run inference
   model.eval()
   with torch.no_grad():
       predictions = model(sample_signals)
       predicted_classes = torch.argmax(predictions, dim=1)
   
   print(f"Predicted modulation classes: {predicted_classes}")
   
   # Get class probabilities
   probabilities = torch.softmax(predictions, dim=1)
   print(f"Class probabilities shape: {probabilities.shape}")

Traditional AMR
~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.ml.traditional_amr import TraditionalAMR
   
   # Create traditional AMR classifier
   traditional_amr = TraditionalAMR(config)
   
   # Extract features from signals
   features = traditional_amr.extract_features(sample_signals)
   print(f"Extracted features shape: {features.shape}")
   
   # Classify using traditional methods
   predictions = traditional_amr.classify(features)
   print(f"Traditional AMR predictions: {predictions}")

Hardware Interface Examples
--------------------------

SDR Device Interface
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.core.sdr_interface import SDRInterface
   from geminisdr.environments.hardware_abstraction import HardwareAbstraction
   
   # Initialize hardware abstraction
   hw_abstraction = HardwareAbstraction(config)
   
   # Detect available SDR devices
   available_devices = hw_abstraction.detect_available_devices()
   print(f"Available SDR devices: {available_devices}")
   
   if available_devices:
       # Initialize SDR interface
       sdr = SDRInterface(config)
       
       # Configure SDR parameters
       sdr.set_sample_rate(2e6)  # 2 MHz
       sdr.set_center_frequency(100e6)  # 100 MHz
       sdr.set_gain(20)  # 20 dB
       
       # Capture samples
       samples = sdr.capture_samples(num_samples=1024)
       print(f"Captured {len(samples)} samples")
       
       # Close SDR
       sdr.close()
   else:
       print("No SDR devices found, using simulation mode")

Simulation Mode
~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.core.signal_generator import SignalGenerator
   
   # Create signal generator for simulation
   signal_gen = SignalGenerator(config)
   
   # Generate different types of signals
   bpsk_signal = signal_gen.generate_bpsk(
       num_samples=1024,
       symbol_rate=1e6,
       carrier_freq=0,  # Baseband
       snr_db=20
   )
   
   qpsk_signal = signal_gen.generate_qpsk(
       num_samples=1024,
       symbol_rate=1e6,
       carrier_freq=0,
       snr_db=15
   )
   
   print(f"Generated BPSK signal shape: {bpsk_signal.shape}")
   print(f"Generated QPSK signal shape: {qpsk_signal.shape}")

Error Handling Examples
----------------------

Basic Error Handling
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.core.error_handling import ErrorHandler, GeminiSDRError
   
   # Initialize error handler
   error_handler = ErrorHandler(config)
   
   # Use error context for automatic handling
   try:
       with error_handler.error_context("model_loading"):
           model = NeuralAMR(config)
           model.load_state_dict(torch.load("model.pth"))
   
   except GeminiSDRError as e:
       print(f"GeminiSDR error: {e}")
       print(f"Error severity: {e.severity}")
       print(f"Error context: {e.context}")
   
   except Exception as e:
       print(f"Unexpected error: {e}")

Automatic Recovery
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.core.error_handling import retry_with_backoff, fallback_to_simulation
   
   @retry_with_backoff(max_retries=3, base_delay=1.0)
   def load_model_with_retry(model_path):
       """Load model with automatic retry on failure."""
       return torch.load(model_path)
   
   # Use fallback to simulation on hardware errors
   def capture_samples_with_fallback():
       try:
           with fallback_to_simulation():
               # Try hardware first
               sdr = SDRInterface(config)
               samples = sdr.capture_samples(1024)
               return samples, "hardware"
       except Exception:
           # Fallback to simulation
           signal_gen = SignalGenerator(config)
           samples = signal_gen.generate_bpsk(1024)
           return samples, "simulation"
   
   samples, source = capture_samples_with_fallback()
   print(f"Captured samples from: {source}")

Memory Management Examples
-------------------------

Memory Monitoring
~~~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.core.memory_manager import MemoryManager
   
   # Initialize memory manager
   memory_manager = MemoryManager(config)
   
   # Monitor memory usage
   def monitor_memory_during_training():
       for epoch in range(10):
           # Get memory stats before training
           stats_before = memory_manager.get_memory_stats()
           
           # Simulate training
           large_tensor = torch.randn(1000, 1000, device=device)
           result = torch.mm(large_tensor, large_tensor)
           
           # Get memory stats after
           stats_after = memory_manager.get_memory_stats()
           
           print(f"Epoch {epoch}:")
           print(f"  Memory before: {stats_before.used_ram_mb:.1f} MB")
           print(f"  Memory after: {stats_after.used_ram_mb:.1f} MB")
           print(f"  Memory diff: {stats_after.used_ram_mb - stats_before.used_ram_mb:.1f} MB")
           
           # Cleanup
           del large_tensor, result
           memory_manager.cleanup_memory()
   
   monitor_memory_during_training()

Batch Size Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find optimal batch size for your hardware
   def find_optimal_batch_size(model, input_shape):
       memory_manager = MemoryManager(config)
       
       # Estimate model size
       model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
       
       # Get optimal batch size
       optimal_batch_size = memory_manager.optimize_batch_size(
           base_batch_size=32,
           model_size_mb=model_size_mb
       )
       
       print(f"Model size: {model_size_mb:.1f} MB")
       print(f"Optimal batch size: {optimal_batch_size}")
       
       return optimal_batch_size
   
   # Use optimal batch size
   model = NeuralAMR(config)
   optimal_batch_size = find_optimal_batch_size(model, (1024, 2))
   
   # Create dataloader with optimal batch size
   dataloader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=True)

Logging Examples
---------------

Structured Logging
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.core.logging_manager import StructuredLogger
   
   # Initialize structured logger
   logger = StructuredLogger("example_module", config.logging)
   
   # Add persistent context
   logger.add_context(
       experiment_id="exp_001",
       model_type="neural_amr",
       dataset="synthetic"
   )
   
   # Log with different levels
   logger.info("Starting training", extra={
       "epoch": 1,
       "batch_size": 32,
       "learning_rate": 0.001
   })
   
   logger.warning("Memory usage high", extra={
       "memory_usage_mb": 1500,
       "threshold_mb": 1024
   })
   
   # Log performance metrics
   logger.log_performance(
       operation="training_epoch",
       duration=45.2,
       samples_processed=1000,
       accuracy=0.95
   )

Complete Training Example
------------------------

Here's a complete example that puts everything together:

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader
   
   from geminisdr.config.config_manager import ConfigManager
   from geminisdr.ml.neural_amr import NeuralAMR
   from geminisdr.core.memory_manager import MemoryManager
   from geminisdr.core.logging_manager import StructuredLogger
   from geminisdr.core.error_handling import ErrorHandler
   
   def complete_training_example():
       # Load configuration
       config = ConfigManager().load_config()
       
       # Initialize components
       memory_manager = MemoryManager(config)
       logger = StructuredLogger("training", config.logging)
       error_handler = ErrorHandler(config)
       
       # Add logging context
       logger.add_context(experiment="basic_example", model="neural_amr")
       
       try:
           with error_handler.error_context("training_setup"):
               # Create model
               model = NeuralAMR(config)
               device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
               model = model.to(device)
               
               # Create synthetic dataset
               signals, labels = generate_dataset(1000)
               dataset = SignalDataset(signals, labels)
               
               # Optimize batch size
               optimal_batch_size = memory_manager.optimize_batch_size(
                   base_batch_size=32,
                   model_size_mb=50  # Estimated model size
               )
               
               dataloader = DataLoader(
                   dataset, 
                   batch_size=optimal_batch_size, 
                   shuffle=True
               )
               
               # Setup training
               criterion = nn.CrossEntropyLoss()
               optimizer = optim.Adam(model.parameters(), lr=config.ml.learning_rate)
               
               logger.info("Training started", extra={
                   "batch_size": optimal_batch_size,
                   "dataset_size": len(dataset),
                   "device": str(device)
               })
               
               # Training loop
               model.train()
               for epoch in range(5):  # Just 5 epochs for example
                   epoch_loss = 0.0
                   correct = 0
                   total = 0
                   
                   for batch_idx, (data, targets) in enumerate(dataloader):
                       data, targets = data.to(device), targets.to(device)
                       
                       optimizer.zero_grad()
                       outputs = model(data)
                       loss = criterion(outputs, targets)
                       loss.backward()
                       optimizer.step()
                       
                       # Statistics
                       epoch_loss += loss.item()
                       _, predicted = torch.max(outputs.data, 1)
                       total += targets.size(0)
                       correct += (predicted == targets).sum().item()
                       
                       # Memory cleanup
                       if batch_idx % 10 == 0:
                           memory_manager.cleanup_memory()
                   
                   # Log epoch results
                   accuracy = 100 * correct / total
                   avg_loss = epoch_loss / len(dataloader)
                   
                   logger.log_performance(
                       operation="training_epoch",
                       duration=0,  # Would measure actual time
                       epoch=epoch,
                       loss=avg_loss,
                       accuracy=accuracy
                   )
                   
                   print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
               
               logger.info("Training completed successfully")
               
       except Exception as e:
           logger.error("Training failed", extra={"error": str(e)})
           raise
   
   if __name__ == "__main__":
       complete_training_example()

Next Steps
----------

After trying these basic examples:

1. **Explore Advanced Examples**: :doc:`ml_training` for more complex training scenarios
2. **Learn About Configuration**: :doc:`../guides/configuration` for detailed configuration options
3. **Read API Documentation**: :doc:`../api/index` for complete API reference
4. **Check Troubleshooting**: :doc:`../guides/troubleshooting` if you encounter issues

For platform-specific optimizations, see:

* :doc:`../guides/installation/m1_mac` - M1 Mac specific examples
* :doc:`../guides/installation/linux_vm` - Linux VM specific examples
* :doc:`../guides/installation/cuda_cluster` - CUDA specific examples