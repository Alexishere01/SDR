Troubleshooting Guide
====================

This guide provides solutions to common setup and runtime issues encountered with GeminiSDR.

Installation Issues
-------------------

Python Environment Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ``ModuleNotFoundError`` when importing GeminiSDR modules

**Symptoms**:
   .. code-block:: text
   
      ModuleNotFoundError: No module named 'geminisdr'

**Solutions**:

1. **Verify virtual environment activation**:
   
   .. code-block:: bash
   
      # Check if virtual environment is active
      which python
      # Should show path to your virtual environment
      
      # Activate if not active
      source geminisdr-env/bin/activate  # Linux/Mac
      # or
      geminisdr-env\Scripts\activate     # Windows

2. **Reinstall in development mode**:
   
   .. code-block:: bash
   
      cd /path/to/geminisdr
      pip install -e .

3. **Check Python path**:
   
   .. code-block:: python
   
      import sys
      print(sys.path)
      # Ensure GeminiSDR directory is in the path

**Issue**: Package dependency conflicts

**Symptoms**:
   .. code-block:: text
   
      ERROR: pip's dependency resolver does not currently consider all the ways that
      dependencies can conflict with each other.

**Solutions**:

1. **Create fresh virtual environment**:
   
   .. code-block:: bash
   
      # Remove old environment
      rm -rf geminisdr-env
      
      # Create new environment
      python3 -m venv geminisdr-env
      source geminisdr-env/bin/activate
      
      # Install with specific versions
      pip install -r requirements.txt

2. **Use pip-tools for dependency resolution**:
   
   .. code-block:: bash
   
      pip install pip-tools
      pip-compile requirements.in
      pip-sync requirements.txt

PyTorch Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: PyTorch not detecting GPU/MPS

**Symptoms**:
   .. code-block:: python
   
      import torch
      print(torch.cuda.is_available())  # False
      print(torch.backends.mps.is_available())  # False

**Solutions**:

1. **For CUDA systems**:
   
   .. code-block:: bash
   
      # Check NVIDIA driver
      nvidia-smi
      
      # Reinstall PyTorch with correct CUDA version
      pip uninstall torch torchvision torchaudio
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2. **For M1 Mac systems**:
   
   .. code-block:: bash
   
      # Check macOS version (need 12.3+)
      sw_vers
      
      # Reinstall PyTorch
      pip uninstall torch torchvision torchaudio
      pip install torch torchvision torchaudio

3. **Verify installation**:
   
   .. code-block:: python
   
      import torch
      print(f"PyTorch version: {torch.__version__}")
      print(f"CUDA available: {torch.cuda.is_available()}")
      print(f"MPS available: {torch.backends.mps.is_available()}")

Configuration Issues
--------------------

Configuration File Not Found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ``ConfigurationError: Configuration file not found``

**Solutions**:

1. **Check configuration file paths**:
   
   .. code-block:: bash
   
      # List configuration files
      find . -name "*.yaml" -path "*/conf/*"
      
      # Check current directory
      ls -la conf/

2. **Create missing configuration**:
   
   .. code-block:: bash
   
      # Copy from template
      cp conf/config.yaml conf/local_config.yaml
      
      # Or create minimal config
      mkdir -p conf
      cat > conf/config.yaml << EOF
      hardware:
        device_preference: "auto"
      ml:
        batch_size: 32
      logging:
        level: "INFO"
      EOF

3. **Set configuration path explicitly**:
   
   .. code-block:: python
   
      from geminisdr.config.config_manager import ConfigManager
      
      # Specify config path
      config_manager = ConfigManager(config_path="./conf")
      config = config_manager.load_config("config")

Invalid Configuration Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ``ValidationError: Invalid configuration value``

**Solutions**:

1. **Check configuration syntax**:
   
   .. code-block:: bash
   
      # Validate YAML syntax
      python -c "import yaml; yaml.safe_load(open('conf/config.yaml'))"

2. **Review configuration schema**:
   
   .. code-block:: python
   
      from geminisdr.config.config_models import SystemConfig
      
      # Check available fields
      print(SystemConfig.__annotations__)

3. **Use configuration validation**:
   
   .. code-block:: python
   
      from geminisdr.config.config_manager import ConfigManager
      
      config_manager = ConfigManager()
      try:
          config = config_manager.load_config()
      except Exception as e:
          print(f"Configuration error: {e}")
          if hasattr(e, 'context'):
              print(f"Context: {e.context}")

Hardware Issues
---------------

SDR Device Not Detected
~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: SDR hardware not recognized or accessible

**Symptoms**:
   .. code-block:: text
   
      HardwareError: SDR device not found
      No devices found

**Solutions**:

1. **Check device connection**:
   
   .. code-block:: bash
   
      # List USB devices
      lsusb  # Linux
      system_profiler SPUSBDataType  # Mac
      
      # Test RTL-SDR specifically
      rtl_test -t

2. **Fix permissions (Linux)**:
   
   .. code-block:: bash
   
      # Add user to plugdev group
      sudo usermod -a -G plugdev $USER
      
      # Create udev rules
      sudo tee /etc/udev/rules.d/20-rtlsdr.rules << EOF
      SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2832", GROUP="plugdev", MODE="0666"
      EOF
      
      # Reload rules
      sudo udevadm control --reload-rules
      sudo udevadm trigger
      
      # Logout and login again

3. **Install missing drivers**:
   
   .. code-block:: bash
   
      # Ubuntu/Debian
      sudo apt install rtl-sdr librtlsdr-dev
      
      # macOS
      brew install librtlsdr

4. **Test with SoapySDR**:
   
   .. code-block:: bash
   
      # List available devices
      SoapySDRUtil --find
      
      # Test specific device
      SoapySDRUtil --make="driver=rtlsdr"

GPU Memory Issues
~~~~~~~~~~~~~~~~

**Issue**: CUDA out of memory errors

**Symptoms**:
   .. code-block:: text
   
      RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB

**Solutions**:

1. **Reduce batch size**:
   
   .. code-block:: yaml
   
      # In configuration file
      ml:
        batch_size: 16  # Reduce from default

2. **Enable memory optimization**:
   
   .. code-block:: python
   
      from geminisdr.core.memory_manager import MemoryManager
      
      memory_manager = MemoryManager(config)
      
      # Use memory-efficient context
      with memory_manager.memory_efficient_context():
          # Training code here
          pass

3. **Clear GPU cache**:
   
   .. code-block:: python
   
      import torch
      torch.cuda.empty_cache()

4. **Enable gradient checkpointing**:
   
   .. code-block:: yaml
   
      # In configuration file
      ml:
        gradient_checkpointing: true

Runtime Issues
--------------

Training Performance Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Slow training or inference performance

**Solutions**:

1. **Check device utilization**:
   
   .. code-block:: bash
   
      # Monitor GPU usage
      nvidia-smi -l 1  # CUDA
      
      # Monitor CPU usage
      htop

2. **Optimize data loading**:
   
   .. code-block:: python
   
      # Increase data loader workers
      dataloader = DataLoader(
          dataset, 
          batch_size=batch_size,
          num_workers=4,  # Increase based on CPU cores
          pin_memory=True  # For GPU training
      )

3. **Enable performance optimizations**:
   
   .. code-block:: python
   
      import torch
      
      # Enable cuDNN benchmarking
      torch.backends.cudnn.benchmark = True
      
      # Enable TensorFloat-32 (A100 GPUs)
      torch.backends.cuda.matmul.allow_tf32 = True

4. **Profile performance**:
   
   .. code-block:: python
   
      import torch.profiler
      
      with torch.profiler.profile(
          activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
          record_shapes=True,
          profile_memory=True
      ) as prof:
          # Training code
          pass
      
      print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

Model Loading Failures
~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Models fail to load or produce incorrect results

**Solutions**:

1. **Check model file integrity**:
   
   .. code-block:: python
   
      import torch
      
      try:
          model = torch.load("model.pth", map_location="cpu")
          print("Model loaded successfully")
      except Exception as e:
          print(f"Model loading failed: {e}")

2. **Verify model compatibility**:
   
   .. code-block:: python
   
      from geminisdr.core.model_manager import ModelManager
      
      model_manager = ModelManager(config)
      
      # Check compatibility
      metadata = model_manager.load_model_metadata("model_name", "version")
      compatibility_issues = model_manager.validate_model_compatibility(metadata)
      
      if compatibility_issues:
          print(f"Compatibility issues: {compatibility_issues}")

3. **Try alternative model versions**:
   
   .. code-block:: python
   
      # List available versions
      versions = model_manager.list_model_versions("model_name")
      print(f"Available versions: {versions}")
      
      # Try loading different version
      model, metadata = model_manager.load_model("model_name", "1.0.0")

Memory Leaks
~~~~~~~~~~~

**Issue**: Memory usage continuously increases during training

**Solutions**:

1. **Enable memory monitoring**:
   
   .. code-block:: python
   
      from geminisdr.core.memory_manager import MemoryManager
      
      memory_manager = MemoryManager(config)
      
      # Monitor memory usage
      for epoch in range(num_epochs):
          stats = memory_manager.get_memory_stats()
          print(f"Epoch {epoch}: {stats.used_ram_mb} MB used")
          
          # Training code
          train_epoch()
          
          # Cleanup
          memory_manager.cleanup_memory()

2. **Fix common memory leak patterns**:
   
   .. code-block:: python
   
      # Avoid keeping references to tensors
      loss_values = []
      for batch in dataloader:
          loss = train_step(batch)
          loss_values.append(loss.item())  # Use .item(), not the tensor
      
      # Clear variables in loops
      for batch in dataloader:
          output = model(batch)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()
          
          # Clear references
          del output, loss

3. **Use memory profiling**:
   
   .. code-block:: python
   
      import tracemalloc
      
      tracemalloc.start()
      
      # Training code
      train_model()
      
      current, peak = tracemalloc.get_traced_memory()
      print(f"Current memory usage: {current / 1e6:.1f} MB")
      print(f"Peak memory usage: {peak / 1e6:.1f} MB")
      tracemalloc.stop()

Cross-Platform Issues
---------------------

Platform-Specific Errors
~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Code works on one platform but fails on another

**Solutions**:

1. **Check platform detection**:
   
   .. code-block:: python
   
      from geminisdr.environments.hardware_abstraction import HardwareAbstraction
      
      hw = HardwareAbstraction(config)
      platform_info = hw.detect_platform_info()
      print(f"Platform: {platform_info}")

2. **Use platform-specific configurations**:
   
   .. code-block:: bash
   
      # Copy appropriate config
      cp conf/hardware/m1_native.yaml conf/local_config.yaml      # M1 Mac
      cp conf/hardware/vm_ubuntu.yaml conf/local_config.yaml      # Linux VM
      cp conf/hardware/cuda_cluster.yaml conf/local_config.yaml  # CUDA

3. **Test cross-platform compatibility**:
   
   .. code-block:: bash
   
      # Run cross-platform tests
      pytest tests/cross_platform/ -v

Path and File System Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: File path problems across different operating systems

**Solutions**:

1. **Use pathlib for cross-platform paths**:
   
   .. code-block:: python
   
      from pathlib import Path
      
      # Instead of string concatenation
      config_path = Path("conf") / "config.yaml"
      model_path = Path("models") / "my_model.pth"

2. **Handle case sensitivity**:
   
   .. code-block:: python
   
      # Check file existence case-insensitively
      import os
      
      def find_file_case_insensitive(directory, filename):
          for file in os.listdir(directory):
              if file.lower() == filename.lower():
                  return os.path.join(directory, file)
          return None

Debugging Strategies
-------------------

Enable Debug Logging
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # In configuration file
   logging:
     level: "DEBUG"
     format: "structured"
     output: ["console", "file"]

.. code-block:: python

   # In Python code
   import logging
   logging.basicConfig(level=logging.DEBUG)

Use Error Context
~~~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.core.error_handling import ErrorHandler
   
   error_handler = ErrorHandler(config)
   
   with error_handler.error_context("model_training"):
       # Code that might fail
       train_model()

Generate Diagnostic Report
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.debug.diagnostics import generate_diagnostic_report
   
   report = generate_diagnostic_report()
   print(json.dumps(report, indent=2))

Getting Help
-----------

Community Resources
~~~~~~~~~~~~~~~~~~

* **GitHub Issues**: Report bugs and request features
* **Discussions**: Ask questions and share experiences
* **Documentation**: Comprehensive guides and API reference

When Reporting Issues
~~~~~~~~~~~~~~~~~~~~

Include the following information:

1. **System Information**:
   
   .. code-block:: bash
   
      python --version
      pip list | grep -E "(torch|geminisdr)"
      uname -a  # Linux/Mac
      
      # For GPU issues
      nvidia-smi  # CUDA systems

2. **Configuration**:
   
   .. code-block:: bash
   
      cat conf/config.yaml

3. **Error Messages**:
   
   .. code-block:: text
   
      Full error traceback and any relevant log messages

4. **Minimal Reproduction**:
   
   .. code-block:: python
   
      # Minimal code that reproduces the issue
      from geminisdr.config.config_manager import ConfigManager
      config = ConfigManager().load_config()
      # ... rest of minimal example

5. **Expected vs Actual Behavior**:
   
   Clear description of what you expected to happen vs what actually happened.

This troubleshooting guide covers the most common issues. For platform-specific problems, also refer to the relevant installation guides:

* :doc:`installation/m1_mac`
* :doc:`installation/linux_vm`
* :doc:`installation/cuda_cluster`