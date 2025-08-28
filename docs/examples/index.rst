Examples and Tutorials
=====================

Comprehensive examples and tutorials for using GeminiSDR effectively.

.. toctree::
   :maxdepth: 1

   basic_usage

Getting Started
--------------

* :doc:`basic_usage` - Essential examples for getting started with GeminiSDR

**Coming Soon:**

* **ML Training Examples** - Advanced machine learning training scenarios
* **Adversarial Scenarios** - Anti-jamming and jamming detection examples  
* **Custom Implementations** - Building custom components and extensions

Quick Examples
--------------

**Basic Model Usage**:

.. code-block:: python

   from geminisdr.config.config_manager import ConfigManager
   from geminisdr.ml.neural_amr import NeuralAMR
   
   # Load configuration and create model
   config = ConfigManager().load_config()
   model = NeuralAMR(config)
   
   # Run inference on sample data
   import torch
   sample_data = torch.randn(32, 1024, 2)
   predictions = model(sample_data)

**Hardware Interface**:

.. code-block:: python

   from geminisdr.environments.hardware_abstraction import HardwareAbstraction
   
   # Detect and use optimal device
   hw = HardwareAbstraction(config)
   device = hw.detect_optimal_device()
   print(f"Using device: {device}")

**Memory Optimization**:

.. code-block:: python

   from geminisdr.core.memory_manager import MemoryManager
   
   # Optimize batch size for your hardware
   memory_manager = MemoryManager(config)
   optimal_batch_size = memory_manager.optimize_batch_size(32, model_size_mb=100)

For More Examples
----------------

* **Installation Guides**: :doc:`../guides/installation/index` - Platform-specific setup examples
* **Performance Optimization**: :doc:`../guides/performance_optimization` - Optimization examples
* **Troubleshooting**: :doc:`../guides/troubleshooting` - Common issue solutions
* **API Reference**: :doc:`../api/index` - Complete API documentation with examples