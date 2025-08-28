Debugging and Troubleshooting
=============================

This guide provides debugging strategies and troubleshooting information for GeminiSDR development and deployment.

Debugging Strategies
--------------------

Logging-Based Debugging
~~~~~~~~~~~~~~~~~~~~~~

GeminiSDR uses structured logging for comprehensive debugging information:

.. code-block:: python

   from geminisdr.core.logging_manager import StructuredLogger
   
   # Enable debug logging
   logger = StructuredLogger("my_module", config.logging)
   logger.add_context(operation="model_training", model_name="resnet50")
   
   # Debug-level logging
   logger.debug("Starting model initialization", extra={
       "model_params": model_params,
       "device": str(device),
       "memory_available": memory_stats.available_ram_mb
   })
   
   # Performance logging
   with logger.performance_context("model_forward_pass"):
       output = model(input_tensor)

Enable debug logging in configuration:

.. code-block:: yaml

   # conf/environments/debug.yaml
   logging:
     level: "DEBUG"
     format: "structured"
     output: ["console", "file"]
     console_format: "detailed"  # More verbose console output

Interactive Debugging
~~~~~~~~~~~~~~~~~~~~

Use Python debugger for interactive debugging:

.. code-block:: python

   import pdb
   
   def problematic_function(data):
       # Set breakpoint
       pdb.set_trace()
       
       # Debug interactively
       result = complex_computation(data)
       return result
   
   # Or use ipdb for enhanced debugging
   import ipdb
   ipdb.set_trace()

For remote debugging in containers or VMs:

.. code-block:: python

   import pdb
   import sys
   
   # Remote debugging
   pdb.Pdb(stdout=sys.__stdout__).set_trace()

Memory Debugging
~~~~~~~~~~~~~~~

Debug memory issues using the memory manager:

.. code-block:: python

   from geminisdr.core.memory_manager import MemoryManager
   
   memory_manager = MemoryManager(config)
   
   # Monitor memory usage
   def debug_memory_usage(operation_name):
       stats_before = memory_manager.get_memory_stats()
       
       # Your operation here
       yield
       
       stats_after = memory_manager.get_memory_stats()
       memory_diff = stats_after.used_ram_mb - stats_before.used_ram_mb
       
       logger.debug(f"Memory usage for {operation_name}", extra={
           "memory_before_mb": stats_before.used_ram_mb,
           "memory_after_mb": stats_after.used_ram_mb,
           "memory_diff_mb": memory_diff
       })
   
   # Usage
   with debug_memory_usage("model_training"):
       train_model(data)

GPU Memory Debugging
~~~~~~~~~~~~~~~~~~~

Debug GPU memory issues:

.. code-block:: python

   import torch
   
   def debug_gpu_memory():
       """Debug GPU memory usage."""
       if torch.cuda.is_available():
           print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
           print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
           print(f"GPU Memory Summary:")
           print(torch.cuda.memory_summary())
       elif torch.backends.mps.is_available():
           print(f"MPS Memory Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
   
   # Clear GPU memory
   def clear_gpu_memory():
       """Clear GPU memory cache."""
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
       elif torch.backends.mps.is_available():
           torch.mps.empty_cache()

Common Issues and Solutions
--------------------------

Configuration Issues
~~~~~~~~~~~~~~~~~~~

**Problem**: Configuration file not found or invalid

**Symptoms**:
* ``ConfigurationError: Configuration file not found``
* ``ValidationError: Invalid configuration value``

**Solutions**:

.. code-block:: python

   # Debug configuration loading
   from geminisdr.config.config_manager import ConfigManager
   
   config_manager = ConfigManager()
   
   # Check configuration file paths
   print(f"Config search paths: {config_manager.get_search_paths()}")
   
   # Validate configuration step by step
   try:
       raw_config = config_manager.load_raw_config()
       print(f"Raw config loaded: {raw_config}")
       
       validated_config = config_manager.validate_config(raw_config)
       print(f"Validation successful")
   except Exception as e:
       print(f"Configuration error: {e}")
       print(f"Error context: {getattr(e, 'context', {})}")

**Common fixes**:
* Verify configuration file exists and has correct permissions
* Check YAML syntax using online validators
* Ensure environment-specific overrides are properly formatted
* Validate configuration values against expected ranges

Memory Issues
~~~~~~~~~~~~

**Problem**: Out of memory errors during training or inference

**Symptoms**:
* ``RuntimeError: CUDA out of memory``
* ``RuntimeError: MPS backend out of memory``
* System becomes unresponsive

**Solutions**:

.. code-block:: python

   # Enable memory optimization
   from geminisdr.core.memory_manager import MemoryManager
   
   memory_manager = MemoryManager(config)
   
   # Use memory-efficient context
   with memory_manager.memory_efficient_context():
       # Reduce batch size automatically
       optimal_batch_size = memory_manager.optimize_batch_size(
           base_batch_size=64,
           model_size_mb=500
       )
       
       # Train with optimized batch size
       train_model(data, batch_size=optimal_batch_size)

**Prevention strategies**:
* Enable gradient checkpointing for large models
* Use mixed precision training when available
* Implement data streaming for large datasets
* Monitor memory usage and set appropriate limits

Model Loading Issues
~~~~~~~~~~~~~~~~~~~

**Problem**: Model fails to load or produces incorrect results

**Symptoms**:
* ``ModelError: Model file corrupted``
* ``ModelError: Incompatible model version``
* Unexpected model outputs

**Solutions**:

.. code-block:: python

   from geminisdr.core.model_manager import ModelManager
   from geminisdr.core.error_handling import ErrorHandler
   
   model_manager = ModelManager(config)
   error_handler = ErrorHandler(config)
   
   # Debug model loading
   try:
       with error_handler.error_context("model_loading"):
           model, metadata = model_manager.load_model("my_model", "1.0.0")
           
           # Validate model
           validation_errors = model_manager.validate_model_compatibility(metadata)
           if validation_errors:
               print(f"Compatibility issues: {validation_errors}")
           
   except Exception as e:
       print(f"Model loading failed: {e}")
       
       # Try alternative versions
       available_versions = model_manager.list_model_versions("my_model")
       print(f"Available versions: {available_versions}")

**Troubleshooting steps**:
* Verify model file integrity using checksums
* Check model metadata for compatibility information
* Try loading alternative model versions
* Validate model architecture matches expected format

Hardware Issues
~~~~~~~~~~~~~~

**Problem**: Hardware detection fails or SDR device not accessible

**Symptoms**:
* ``HardwareError: SDR device not found``
* ``HardwareError: Device initialization failed``
* Automatic fallback to simulation mode

**Solutions**:

.. code-block:: python

   from geminisdr.environments.hardware_abstraction import HardwareAbstraction
   
   hw_abstraction = HardwareAbstraction(config)
   
   # Debug hardware detection
   detected_devices = hw_abstraction.detect_available_devices()
   print(f"Detected devices: {detected_devices}")
   
   # Test device connectivity
   for device in detected_devices:
       try:
           hw_abstraction.test_device_connection(device)
           print(f"Device {device} is accessible")
       except Exception as e:
           print(f"Device {device} failed: {e}")
   
   # Check device permissions
   device_permissions = hw_abstraction.check_device_permissions()
   print(f"Device permissions: {device_permissions}")

**Common fixes**:
* Verify SDR device is properly connected
* Check USB permissions and udev rules on Linux
* Ensure device drivers are installed
* Test device with manufacturer software first

Performance Issues
~~~~~~~~~~~~~~~~~

**Problem**: Training or inference is slower than expected

**Symptoms**:
* Long training times
* High CPU/GPU utilization without progress
* Memory usage growing over time

**Debugging approach**:

.. code-block:: python

   import time
   import torch.profiler
   from geminisdr.core.metrics_collector import MetricsCollector
   
   metrics_collector = MetricsCollector(config)
   
   # Profile training performance
   def profile_training():
       with torch.profiler.profile(
           activities=[
               torch.profiler.ProfilerActivity.CPU,
               torch.profiler.ProfilerActivity.CUDA,
           ],
           record_shapes=True,
           profile_memory=True,
           with_stack=True
       ) as prof:
           # Training loop
           for batch in data_loader:
               output = model(batch)
               loss = criterion(output, targets)
               loss.backward()
               optimizer.step()
       
       # Export profiling results
       prof.export_chrome_trace("training_profile.json")
       print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
   
   # Monitor system metrics
   def monitor_performance():
       start_time = time.time()
       
       # Your operation
       train_model()
       
       end_time = time.time()
       
       metrics_collector.record_performance_metrics({
           "operation": "training",
           "duration_seconds": end_time - start_time,
           "samples_processed": len(dataset),
           "samples_per_second": len(dataset) / (end_time - start_time)
       })

**Optimization strategies**:
* Use appropriate data types (float16 vs float32)
* Optimize data loading with multiple workers
* Enable compiler optimizations (torch.compile)
* Use platform-specific optimizations

Cross-Platform Issues
~~~~~~~~~~~~~~~~~~~~

**Problem**: Code works on one platform but fails on another

**Symptoms**:
* Different results across platforms
* Platform-specific errors
* Performance variations

**Debugging approach**:

.. code-block:: python

   import platform
   import torch
   
   def debug_platform_differences():
       """Debug platform-specific issues."""
       platform_info = {
           "system": platform.system(),
           "machine": platform.machine(),
           "python_version": platform.python_version(),
           "pytorch_version": torch.__version__,
       }
       
       # Check device availability
       if torch.cuda.is_available():
           platform_info["cuda_version"] = torch.version.cuda
           platform_info["gpu_count"] = torch.cuda.device_count()
       
       if torch.backends.mps.is_available():
           platform_info["mps_available"] = True
       
       print(f"Platform info: {platform_info}")
       
       # Test basic operations
       test_tensor = torch.randn(100, 100)
       
       # Test on different devices
       for device in ["cpu", "cuda", "mps"]:
           if device == "cuda" and not torch.cuda.is_available():
               continue
           if device == "mps" and not torch.backends.mps.is_available():
               continue
           
           try:
               test_tensor_device = test_tensor.to(device)
               result = torch.mm(test_tensor_device, test_tensor_device)
               print(f"Device {device}: OK")
           except Exception as e:
               print(f"Device {device}: Failed - {e}")

**Solutions**:
* Use platform-specific configuration profiles
* Implement device-specific optimizations
* Add platform detection and fallback logic
* Test on all target platforms regularly

Debugging Tools and Utilities
-----------------------------

Log Analysis
~~~~~~~~~~~

Analyze structured logs for debugging:

.. code-block:: python

   import json
   from pathlib import Path
   
   def analyze_logs(log_file: Path):
       """Analyze structured log files for debugging."""
       errors = []
       performance_issues = []
       
       with open(log_file) as f:
           for line in f:
               try:
                   log_entry = json.loads(line)
                   
                   # Check for errors
                   if log_entry.get("level") == "ERROR":
                       errors.append(log_entry)
                   
                   # Check for performance issues
                   if "duration" in log_entry and log_entry["duration"] > 10.0:
                       performance_issues.append(log_entry)
                       
               except json.JSONDecodeError:
                   continue
       
       print(f"Found {len(errors)} errors")
       print(f"Found {len(performance_issues)} performance issues")
       
       return errors, performance_issues

System Diagnostics
~~~~~~~~~~~~~~~~~

Create diagnostic reports for troubleshooting:

.. code-block:: python

   def generate_diagnostic_report():
       """Generate comprehensive diagnostic report."""
       import psutil
       import torch
       from geminisdr.config.config_manager import ConfigManager
       from geminisdr.core.memory_manager import MemoryManager
       
       report = {
           "timestamp": datetime.now().isoformat(),
           "system": {
               "platform": platform.system(),
               "architecture": platform.machine(),
               "python_version": platform.python_version(),
               "cpu_count": psutil.cpu_count(),
               "memory_total_gb": psutil.virtual_memory().total / 1e9,
           },
           "pytorch": {
               "version": torch.__version__,
               "cuda_available": torch.cuda.is_available(),
               "mps_available": torch.backends.mps.is_available(),
           },
           "configuration": {},
           "memory": {},
           "errors": []
       }
       
       # Add configuration info
       try:
           config_manager = ConfigManager()
           config = config_manager.load_config()
           report["configuration"]["status"] = "loaded"
           report["configuration"]["device_preference"] = config.hardware.device_preference
       except Exception as e:
           report["configuration"]["status"] = "failed"
           report["configuration"]["error"] = str(e)
       
       # Add memory info
       try:
           memory_manager = MemoryManager(config)
           memory_stats = memory_manager.get_memory_stats()
           report["memory"] = {
               "total_ram_mb": memory_stats.total_ram_mb,
               "available_ram_mb": memory_stats.available_ram_mb,
               "used_ram_mb": memory_stats.used_ram_mb,
           }
       except Exception as e:
           report["errors"].append(f"Memory manager failed: {e}")
       
       return report

Testing and Validation
~~~~~~~~~~~~~~~~~~~~~~

Create debugging test cases:

.. code-block:: python

   # tests/debug/test_debugging_tools.py
   import pytest
   from geminisdr.debug.diagnostics import generate_diagnostic_report
   
   class TestDebuggingTools:
       """Test debugging and diagnostic tools."""
       
       def test_diagnostic_report_generation(self):
           """Test that diagnostic report can be generated."""
           report = generate_diagnostic_report()
           
           assert "timestamp" in report
           assert "system" in report
           assert "pytorch" in report
           
           # Validate system info
           assert report["system"]["platform"] in ["Darwin", "Linux", "Windows"]
           assert report["system"]["cpu_count"] > 0
           assert report["system"]["memory_total_gb"] > 0
       
       @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR"])
       def test_log_analysis_tools(self, log_level, tmp_path):
           """Test log analysis functionality."""
           # Create test log file
           log_file = tmp_path / "test.log"
           
           test_log_entry = {
               "timestamp": "2024-01-01T12:00:00",
               "level": log_level,
               "message": "Test message",
               "module": "test_module"
           }
           
           with open(log_file, "w") as f:
               f.write(json.dumps(test_log_entry) + "\n")
           
           # Analyze logs
           errors, performance_issues = analyze_logs(log_file)
           
           if log_level == "ERROR":
               assert len(errors) == 1
           else:
               assert len(errors) == 0

Best Practices for Debugging
----------------------------

Proactive Debugging
~~~~~~~~~~~~~~~~~~

1. **Add Comprehensive Logging**: Include debug information in all major operations
2. **Use Assertions**: Add assertions to validate assumptions
3. **Implement Health Checks**: Regular system health validation
4. **Monitor Resource Usage**: Track memory, CPU, and GPU usage

Debugging Workflow
~~~~~~~~~~~~~~~~~

1. **Reproduce the Issue**: Create minimal reproduction case
2. **Gather Information**: Collect logs, system info, and error context
3. **Form Hypothesis**: Based on available information
4. **Test Hypothesis**: Use debugging tools to validate
5. **Implement Fix**: Make targeted changes
6. **Verify Fix**: Ensure issue is resolved and no regressions

Documentation
~~~~~~~~~~~~

1. **Document Known Issues**: Maintain list of known issues and workarounds
2. **Update Troubleshooting Guide**: Add new solutions as they're discovered
3. **Share Debugging Tips**: Document effective debugging strategies
4. **Create Runbooks**: Step-by-step guides for common issues