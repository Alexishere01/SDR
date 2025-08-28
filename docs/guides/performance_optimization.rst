Performance Optimization Guide
==============================

This guide provides comprehensive strategies for optimizing GeminiSDR performance across different platforms and use cases.

General Optimization Principles
-------------------------------

Performance Hierarchy
~~~~~~~~~~~~~~~~~~~~~

Optimize in this order for maximum impact:

1. **Algorithm Efficiency**: Choose the right algorithms and data structures
2. **Memory Management**: Minimize memory allocation and copying
3. **Compute Optimization**: Leverage hardware acceleration (GPU, MPS)
4. **I/O Optimization**: Optimize data loading and storage access
5. **Parallelization**: Use multi-threading and distributed processing

Profiling First
~~~~~~~~~~~~~~~

Always profile before optimizing:

.. code-block:: python

   import torch.profiler
   import time
   
   # Profile training loop
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
       record_shapes=True,
       profile_memory=True,
       with_stack=True
   ) as prof:
       for batch in dataloader:
           output = model(batch)
           loss = criterion(output, target)
           loss.backward()
           optimizer.step()
   
   # Analyze results
   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
   prof.export_chrome_trace("trace.json")  # View in chrome://tracing

Platform-Specific Optimizations
-------------------------------

M1 Mac Optimization
~~~~~~~~~~~~~~~~~~

**Metal Performance Shaders (MPS)**:

.. code-block:: python

   import torch
   
   # Enable MPS optimizations
   if torch.backends.mps.is_available():
       device = torch.device("mps")
       
       # Enable MPS-specific optimizations
       torch.backends.mps.allow_tf32 = True
       
       # Set memory fraction for unified memory
       torch.mps.set_per_process_memory_fraction(0.8)
   
   # Move model and data to MPS
   model = model.to(device)
   data = data.to(device)

**Unified Memory Optimization**:

.. code-block:: python

   from geminisdr.core.memory_manager import MemoryManager
   
   # Configure for unified memory
   config.hardware.memory_optimization = "unified"
   config.ml.batch_size_multiplier = 1.5  # Larger batches
   
   memory_manager = MemoryManager(config)
   
   # Optimize batch size for unified memory
   optimal_batch_size = memory_manager.optimize_batch_size(
       base_batch_size=64,
       model_size_mb=500
   )

**ARM64 Optimizations**:

.. code-block:: python

   # Use ARM64-optimized libraries
   import torch
   
   # Enable ARM64 NEON optimizations
   torch.set_num_threads(8)  # M1 has 8 cores
   
   # Use half precision for memory efficiency
   model = model.half()  # Convert to float16
   
   # Enable automatic mixed precision
   from torch.cuda.amp import autocast
   with autocast(device_type='mps'):
       output = model(input)

Linux VM Optimization
~~~~~~~~~~~~~~~~~~~~

**CPU Optimization**:

.. code-block:: python

   import torch
   import os
   
   # Set optimal thread count for VM
   vm_cpu_count = 4  # Match VM allocation
   torch.set_num_threads(vm_cpu_count)
   os.environ['OMP_NUM_THREADS'] = str(vm_cpu_count)
   os.environ['MKL_NUM_THREADS'] = str(vm_cpu_count)
   
   # Enable CPU optimizations
   torch.backends.mkldnn.enabled = True

**Memory Conservation**:

.. code-block:: python

   # Use gradient checkpointing
   model.gradient_checkpointing_enable()
   
   # Smaller batch sizes
   config.ml.batch_size = 16
   
   # Enable memory cleanup
   import gc
   
   def cleanup_memory():
       gc.collect()
       if torch.cuda.is_available():
           torch.cuda.empty_cache()

**I/O Optimization for VMs**:

.. code-block:: python

   # Use memory mapping for large datasets
   import numpy as np
   
   # Memory-mapped arrays
   data = np.memmap('large_dataset.dat', dtype='float32', mode='r')
   
   # Efficient data loading
   dataloader = DataLoader(
       dataset,
       batch_size=16,
       num_workers=2,  # Limited for VMs
       pin_memory=False,  # Disable for CPU-only
       persistent_workers=True
   )

CUDA Cluster Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-GPU Setup**:

.. code-block:: python

   import torch
   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel as DDP
   
   # Initialize distributed training
   def setup_distributed():
       dist.init_process_group(backend='nccl')
       local_rank = int(os.environ['LOCAL_RANK'])
       torch.cuda.set_device(local_rank)
       return local_rank
   
   # Wrap model for distributed training
   local_rank = setup_distributed()
   model = model.to(local_rank)
   model = DDP(model, device_ids=[local_rank])

**CUDA Optimizations**:

.. code-block:: python

   # Enable TensorFloat-32 for A100 GPUs
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   
   # Enable cuDNN benchmarking
   torch.backends.cudnn.benchmark = True
   
   # Optimize memory allocation
   torch.cuda.empty_cache()
   torch.cuda.memory.set_per_process_memory_fraction(0.9)

**Mixed Precision Training**:

.. code-block:: python

   from torch.cuda.amp import GradScaler, autocast
   
   scaler = GradScaler()
   
   for batch in dataloader:
       optimizer.zero_grad()
       
       with autocast():
           output = model(batch)
           loss = criterion(output, target)
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()

Memory Optimization
------------------

Batch Size Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.core.memory_manager import MemoryManager
   
   def find_optimal_batch_size(model, input_shape, device):
       """Find the largest batch size that fits in memory."""
       memory_manager = MemoryManager(config)
       
       # Start with a reasonable batch size
       batch_size = 32
       max_batch_size = batch_size
       
       while batch_size <= 512:  # Maximum reasonable batch size
           try:
               # Test batch
               test_input = torch.randn(batch_size, *input_shape, device=device)
               
               with torch.no_grad():
                   output = model(test_input)
               
               max_batch_size = batch_size
               batch_size *= 2
               
               # Cleanup
               del test_input, output
               torch.cuda.empty_cache()
               
           except RuntimeError as e:
               if "out of memory" in str(e):
                   break
               else:
                   raise e
       
       return max_batch_size

Gradient Checkpointing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Enable gradient checkpointing for memory efficiency
   model.gradient_checkpointing_enable()
   
   # Or implement custom checkpointing
   from torch.utils.checkpoint import checkpoint
   
   class CheckpointedModel(nn.Module):
       def __init__(self, model):
           super().__init__()
           self.model = model
       
       def forward(self, x):
           # Checkpoint every few layers
           return checkpoint(self.model, x)

Model Caching
~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.core.memory_manager import ModelCache
   
   # Initialize model cache
   model_cache = ModelCache(max_size=3, max_memory_mb=2048)
   
   # Use cached models
   def get_model(model_name):
       model = model_cache.get_model(model_name)
       if model is None:
           model = load_model(model_name)
           model_cache.put_model(model_name, model)
       return model

Data Loading Optimization
-------------------------

Efficient Data Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader, Dataset
   
   class OptimizedDataset(Dataset):
       def __init__(self, data_path, transform=None):
           self.data_path = data_path
           self.transform = transform
           
           # Pre-load metadata
           self.metadata = self._load_metadata()
       
       def __getitem__(self, idx):
           # Lazy loading
           data = self._load_sample(idx)
           
           if self.transform:
               data = self.transform(data)
           
           return data
       
       def _load_sample(self, idx):
           # Efficient loading (memory mapping, etc.)
           pass
   
   # Optimized data loader
   def create_optimized_dataloader(dataset, batch_size, num_workers=None):
       if num_workers is None:
           num_workers = min(8, os.cpu_count())
       
       return DataLoader(
           dataset,
           batch_size=batch_size,
           num_workers=num_workers,
           pin_memory=torch.cuda.is_available(),
           persistent_workers=True,
           prefetch_factor=2
       )

Data Preprocessing
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Precompute expensive transformations
   def preprocess_dataset(raw_data_path, processed_data_path):
       """Preprocess and save dataset to avoid runtime computation."""
       
       raw_data = load_raw_data(raw_data_path)
       
       processed_data = []
       for sample in raw_data:
           # Apply expensive transformations
           processed_sample = expensive_transform(sample)
           processed_data.append(processed_sample)
       
       # Save preprocessed data
       torch.save(processed_data, processed_data_path)
   
   # Use preprocessed data
   class PreprocessedDataset(Dataset):
       def __init__(self, processed_data_path):
           self.data = torch.load(processed_data_path)
       
       def __getitem__(self, idx):
           return self.data[idx]  # No runtime preprocessing

Model Optimization
-----------------

Model Architecture Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use efficient model architectures
   import torch.nn as nn
   
   class EfficientBlock(nn.Module):
       def __init__(self, in_channels, out_channels):
           super().__init__()
           # Use depthwise separable convolutions
           self.depthwise = nn.Conv1d(in_channels, in_channels, 
                                    kernel_size=3, groups=in_channels)
           self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
           self.bn = nn.BatchNorm1d(out_channels)
           self.relu = nn.ReLU(inplace=True)
       
       def forward(self, x):
           x = self.depthwise(x)
           x = self.pointwise(x)
           x = self.bn(x)
           return self.relu(x)

Model Quantization
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Post-training quantization
   def quantize_model(model):
       """Apply post-training quantization."""
       model.eval()
       
       # Dynamic quantization
       quantized_model = torch.quantization.quantize_dynamic(
           model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
       )
       
       return quantized_model
   
   # Quantization-aware training
   def setup_qat(model):
       """Setup quantization-aware training."""
       model.train()
       
       # Prepare model for QAT
       model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
       torch.quantization.prepare_qat(model, inplace=True)
       
       return model

Model Pruning
~~~~~~~~~~~~

.. code-block:: python

   import torch.nn.utils.prune as prune
   
   def prune_model(model, pruning_ratio=0.2):
       """Apply structured pruning to model."""
       
       for name, module in model.named_modules():
           if isinstance(module, (nn.Conv1d, nn.Linear)):
               # Apply magnitude-based pruning
               prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
               
               # Make pruning permanent
               prune.remove(module, 'weight')
       
       return model

Training Optimization
--------------------

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Efficient learning rate schedules
   from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
   
   # One-cycle learning rate
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
   scheduler = OneCycleLR(
       optimizer,
       max_lr=1e-2,
       epochs=num_epochs,
       steps_per_epoch=len(dataloader)
   )
   
   # Cosine annealing with warm restarts
   scheduler = CosineAnnealingWarmRestarts(
       optimizer,
       T_0=10,  # Restart every 10 epochs
       T_mult=2  # Double restart period each time
   )

Optimizer Selection
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Choose efficient optimizers
   
   # AdamW for most cases
   optimizer = torch.optim.AdamW(
       model.parameters(),
       lr=1e-3,
       weight_decay=1e-4,
       betas=(0.9, 0.999)
   )
   
   # SGD with momentum for large-scale training
   optimizer = torch.optim.SGD(
       model.parameters(),
       lr=1e-2,
       momentum=0.9,
       weight_decay=1e-4,
       nesterov=True
   )

Early Stopping
~~~~~~~~~~~~~

.. code-block:: python

   class EarlyStopping:
       def __init__(self, patience=7, min_delta=0.001):
           self.patience = patience
           self.min_delta = min_delta
           self.counter = 0
           self.best_loss = float('inf')
       
       def __call__(self, val_loss):
           if val_loss < self.best_loss - self.min_delta:
               self.best_loss = val_loss
               self.counter = 0
           else:
               self.counter += 1
           
           return self.counter >= self.patience
   
   # Usage in training loop
   early_stopping = EarlyStopping(patience=10)
   
   for epoch in range(num_epochs):
       train_loss = train_epoch()
       val_loss = validate_epoch()
       
       if early_stopping(val_loss):
           print(f"Early stopping at epoch {epoch}")
           break

Inference Optimization
---------------------

Model Compilation
~~~~~~~~~~~~~~~~

.. code-block:: python

   # PyTorch 2.0 compilation
   if hasattr(torch, 'compile'):
       compiled_model = torch.compile(model, mode='max-autotune')
   else:
       compiled_model = model
   
   # TorchScript compilation
   def create_torchscript_model(model, example_input):
       """Create optimized TorchScript model."""
       model.eval()
       
       with torch.no_grad():
           traced_model = torch.jit.trace(model, example_input)
           
           # Optimize for inference
           traced_model = torch.jit.optimize_for_inference(traced_model)
       
       return traced_model

Batch Processing
~~~~~~~~~~~~~~~

.. code-block:: python

   def batch_inference(model, data, batch_size=32):
       """Efficient batch inference."""
       model.eval()
       results = []
       
       with torch.no_grad():
           for i in range(0, len(data), batch_size):
               batch = data[i:i+batch_size]
               
               # Move to device
               batch = batch.to(device)
               
               # Inference
               output = model(batch)
               
               # Move back to CPU and store
               results.append(output.cpu())
               
               # Cleanup
               del batch, output
       
       return torch.cat(results, dim=0)

Monitoring and Profiling
------------------------

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geminisdr.core.metrics_collector import MetricsCollector
   
   class PerformanceMonitor:
       def __init__(self, config):
           self.metrics_collector = MetricsCollector(config)
           self.start_time = None
       
       def start_timing(self, operation_name):
           self.operation_name = operation_name
           self.start_time = time.time()
       
       def end_timing(self, **extra_metrics):
           if self.start_time:
               duration = time.time() - self.start_time
               
               self.metrics_collector.record_metric(
                   f"{self.operation_name}_duration",
                   duration,
                   tags={"operation": self.operation_name}
               )
               
               for key, value in extra_metrics.items():
                   self.metrics_collector.record_metric(
                       f"{self.operation_name}_{key}",
                       value,
                       tags={"operation": self.operation_name}
                   )
   
   # Usage
   monitor = PerformanceMonitor(config)
   
   monitor.start_timing("training_epoch")
   train_loss = train_epoch()
   monitor.end_timing(loss=train_loss, samples_processed=len(dataset))

Memory Profiling
~~~~~~~~~~~~~~~

.. code-block:: python

   import tracemalloc
   import psutil
   
   class MemoryProfiler:
       def __init__(self):
           self.process = psutil.Process()
       
       def start_profiling(self):
           tracemalloc.start()
           self.initial_memory = self.process.memory_info().rss
       
       def get_memory_usage(self):
           current_memory = self.process.memory_info().rss
           memory_diff = current_memory - self.initial_memory
           
           current_trace, peak_trace = tracemalloc.get_traced_memory()
           
           return {
               'current_rss_mb': current_memory / 1e6,
               'memory_diff_mb': memory_diff / 1e6,
               'traced_current_mb': current_trace / 1e6,
               'traced_peak_mb': peak_trace / 1e6
           }
       
       def stop_profiling(self):
           tracemalloc.stop()

Benchmarking
-----------

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def benchmark_training_speed(model, dataloader, device, num_batches=100):
       """Benchmark training speed."""
       model.train()
       model = model.to(device)
       
       # Warmup
       for i, batch in enumerate(dataloader):
           if i >= 10:  # 10 warmup batches
               break
           
           batch = batch.to(device)
           output = model(batch)
           loss = F.mse_loss(output, torch.randn_like(output))
           loss.backward()
       
       # Benchmark
       torch.cuda.synchronize() if device.type == 'cuda' else None
       start_time = time.time()
       
       for i, batch in enumerate(dataloader):
           if i >= num_batches:
               break
           
           batch = batch.to(device)
           output = model(batch)
           loss = F.mse_loss(output, torch.randn_like(output))
           loss.backward()
       
       torch.cuda.synchronize() if device.type == 'cuda' else None
       end_time = time.time()
       
       total_time = end_time - start_time
       samples_per_second = (num_batches * batch.size(0)) / total_time
       
       return {
           'total_time': total_time,
           'samples_per_second': samples_per_second,
           'batches_per_second': num_batches / total_time
       }

Automated Optimization
---------------------

Auto-tuning
~~~~~~~~~~

.. code-block:: python

   def auto_tune_batch_size(model, dataloader, device):
       """Automatically find optimal batch size."""
       from geminisdr.core.memory_manager import MemoryManager
       
       memory_manager = MemoryManager(config)
       
       # Get sample input shape
       sample_batch = next(iter(dataloader))
       input_shape = sample_batch.shape[1:]  # Exclude batch dimension
       
       # Find optimal batch size
       optimal_batch_size = memory_manager.optimize_batch_size(
           base_batch_size=32,
           model_size_mb=get_model_size_mb(model)
       )
       
       return optimal_batch_size
   
   def get_model_size_mb(model):
       """Calculate model size in MB."""
       param_size = 0
       buffer_size = 0
       
       for param in model.parameters():
           param_size += param.nelement() * param.element_size()
       
       for buffer in model.buffers():
           buffer_size += buffer.nelement() * buffer.element_size()
       
       return (param_size + buffer_size) / 1e6

Best Practices Summary
---------------------

1. **Profile First**: Always profile before optimizing
2. **Start Simple**: Begin with basic optimizations before advanced techniques
3. **Measure Impact**: Quantify the impact of each optimization
4. **Platform-Specific**: Use platform-specific optimizations
5. **Memory Management**: Optimize memory usage for your hardware
6. **Batch Size**: Find optimal batch size for your setup
7. **Data Pipeline**: Optimize data loading and preprocessing
8. **Model Architecture**: Choose efficient model architectures
9. **Mixed Precision**: Use mixed precision training when available
10. **Monitor Continuously**: Continuously monitor performance in production

For platform-specific optimization details, see:

* :doc:`installation/m1_mac` - M1 Mac optimizations
* :doc:`installation/linux_vm` - Linux VM optimizations  
* :doc:`installation/cuda_cluster` - CUDA cluster optimizations