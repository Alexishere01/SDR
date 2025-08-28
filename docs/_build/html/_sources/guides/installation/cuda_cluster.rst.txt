CUDA Cluster Installation Guide
===============================

This guide provides detailed installation instructions for GeminiSDR on CUDA-enabled clusters and high-performance computing environments.

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~

* **Hardware**:
  
  * NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal architecture or newer)
  * 16GB+ system RAM (32GB+ recommended)
  * 8GB+ GPU memory per GPU (16GB+ recommended)
  * High-speed storage (NVMe SSD recommended)

* **Software**:
  
  * Linux distribution (Ubuntu 20.04 LTS, CentOS 8+, or RHEL 8+)
  * NVIDIA GPU drivers (470.x or later)
  * CUDA Toolkit 11.8 or later
  * cuDNN 8.6 or later
  * Python 3.9+

* **Network**:
  
  * High-bandwidth network for multi-node setups
  * InfiniBand or 10GbE recommended for distributed training

Cluster Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

**SLURM Integration** (if applicable):

.. code-block:: bash

   # Check SLURM availability
   sinfo
   squeue
   
   # Check available GPUs
   sinfo -o "%N %G"

**Module System** (if applicable):

.. code-block:: bash

   # Load required modules
   module load cuda/11.8
   module load python/3.9
   module load gcc/9.3.0

Installation Steps
------------------

1. Verify CUDA Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Check CUDA and GPU availability:

.. code-block:: bash

   # Check NVIDIA driver
   nvidia-smi
   
   # Check CUDA version
   nvcc --version
   
   # Check GPU compute capability
   nvidia-smi --query-gpu=compute_cap --format=csv
   
   # Test CUDA samples (if available)
   cd /usr/local/cuda/samples/1_Utilities/deviceQuery
   make && ./deviceQuery

2. Set Up Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create isolated Python environment:

.. code-block:: bash

   # Create virtual environment
   python3 -m venv geminisdr-cuda-env
   
   # Activate environment
   source geminisdr-cuda-env/bin/activate
   
   # Upgrade pip
   pip install --upgrade pip wheel setuptools

3. Install CUDA-Enabled PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install PyTorch with CUDA support:

.. code-block:: bash

   # Install PyTorch with CUDA 11.8 support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Verify CUDA availability
   python -c "
   import torch
   print(f'PyTorch version: {torch.__version__}')
   print(f'CUDA available: {torch.cuda.is_available()}')
   print(f'CUDA version: {torch.version.cuda}')
   print(f'GPU count: {torch.cuda.device_count()}')
   for i in range(torch.cuda.device_count()):
       print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
   "

4. Install System Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Ubuntu/Debian**:

.. code-block:: bash

   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install build tools
   sudo apt install -y build-essential cmake git
   sudo apt install -y python3-dev libfftw3-dev
   sudo apt install -y libusb-1.0-0-dev pkg-config
   
   # Install SDR libraries
   sudo apt install -y rtl-sdr librtlsdr-dev
   sudo apt install -y hackrf libhackrf-dev
   sudo apt install -y soapysdr-tools libsoapysdr-dev

**CentOS/RHEL**:

.. code-block:: bash

   # Update system
   sudo dnf update -y
   
   # Install development tools
   sudo dnf groupinstall -y "Development Tools"
   sudo dnf install -y cmake git python3-devel
   sudo dnf install -y fftw-devel libusb1-devel
   
   # Install EPEL for additional packages
   sudo dnf install -y epel-release
   sudo dnf install -y rtl-sdr rtl-sdr-devel

5. Install GeminiSDR
~~~~~~~~~~~~~~~~~~~

Clone and install with CUDA optimizations:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/your-org/geminisdr.git
   cd geminisdr
   
   # Install with CUDA support
   CUDA_HOME=/usr/local/cuda pip install -e .
   
   # Install additional dependencies
   pip install -r requirements-cuda.txt

6. Configure for CUDA Cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create CUDA-optimized configuration:

.. code-block:: bash

   # Copy CUDA configuration template
   cp conf/hardware/cuda_cluster.yaml conf/local_config.yaml
   
   # Edit for your cluster setup
   nano conf/local_config.yaml

Example CUDA cluster configuration:

.. code-block:: yaml

   # CUDA cluster optimized configuration
   hardware:
     device_preference: "cuda"
     memory_optimization: "aggressive"
     multi_gpu: true
     gpu_memory_fraction: 0.9
   
   ml:
     batch_size: null  # Auto-optimize for GPU memory
     precision: "mixed"  # Mixed precision training
     model_cache_size: 6  # Large cache for multiple GPUs
     distributed: true
     gradient_checkpointing: false  # Disable for performance
   
   performance:
     memory_threshold: 0.85
     auto_optimize: true
     profiling_enabled: true
     benchmark_mode: true
   
   distributed:
     backend: "nccl"  # NVIDIA Collective Communications Library
     init_method: "env://"
     world_size: 1  # Set based on number of nodes
     rank: 0  # Set based on node rank

Multi-GPU Configuration
----------------------

Single Node Multi-GPU
~~~~~~~~~~~~~~~~~~~~~

Configure for multiple GPUs on single node:

.. code-block:: python

   # Multi-GPU training setup
   import torch
   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel as DDP
   
   def setup_multi_gpu():
       """Setup multi-GPU training on single node."""
       if torch.cuda.device_count() > 1:
           # Initialize distributed training
           dist.init_process_group(
               backend='nccl',
               init_method='env://',
               world_size=torch.cuda.device_count(),
               rank=int(os.environ.get('LOCAL_RANK', 0))
           )
           
           # Set device
           device = torch.device(f'cuda:{dist.get_rank()}')
           torch.cuda.set_device(device)
           
           return device
       else:
           return torch.device('cuda:0')

Multi-Node Configuration
~~~~~~~~~~~~~~~~~~~~~~~

For distributed training across multiple nodes:

.. code-block:: bash

   # SLURM job script example
   #!/bin/bash
   #SBATCH --job-name=geminisdr-training
   #SBATCH --nodes=2
   #SBATCH --ntasks-per-node=4
   #SBATCH --gres=gpu:4
   #SBATCH --time=24:00:00
   
   # Load modules
   module load cuda/11.8
   module load python/3.9
   
   # Activate environment
   source geminisdr-cuda-env/bin/activate
   
   # Set distributed training environment
   export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
   export MASTER_PORT=29500
   export WORLD_SIZE=$SLURM_NTASKS
   export RANK=$SLURM_PROCID
   export LOCAL_RANK=$SLURM_LOCALID
   
   # Run distributed training
   srun python train_distributed.py

Performance Optimization
------------------------

CUDA Optimizations
~~~~~~~~~~~~~~~~~

Enable CUDA-specific optimizations:

.. code-block:: python

   import torch
   
   # Enable TensorFloat-32 for A100 GPUs
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   
   # Enable cuDNN benchmarking for consistent input sizes
   torch.backends.cudnn.benchmark = True
   
   # Enable cuDNN deterministic mode (if reproducibility needed)
   # torch.backends.cudnn.deterministic = True
   
   # Set memory allocation strategy
   torch.cuda.empty_cache()
   torch.cuda.memory.set_per_process_memory_fraction(0.9)

Mixed Precision Training
~~~~~~~~~~~~~~~~~~~~~~~

Use automatic mixed precision for better performance:

.. code-block:: python

   from torch.cuda.amp import GradScaler, autocast
   
   # Initialize scaler for mixed precision
   scaler = GradScaler()
   
   # Training loop with mixed precision
   for batch in dataloader:
       optimizer.zero_grad()
       
       with autocast():
           outputs = model(batch)
           loss = criterion(outputs, targets)
       
       # Scale loss and backward pass
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()

Memory Management
~~~~~~~~~~~~~~~~

Optimize GPU memory usage:

.. code-block:: python

   from geminisdr.core.memory_manager import MemoryManager
   
   # Initialize with CUDA-specific settings
   memory_manager = MemoryManager(config)
   
   # Monitor GPU memory
   def print_gpu_memory():
       for i in range(torch.cuda.device_count()):
           allocated = torch.cuda.memory_allocated(i) / 1e9
           reserved = torch.cuda.memory_reserved(i) / 1e9
           print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
   
   # Use memory-efficient training
   with memory_manager.memory_efficient_context():
       # Training code here
       pass

Monitoring and Profiling
------------------------

GPU Monitoring
~~~~~~~~~~~~~

Monitor GPU usage during training:

.. code-block:: bash

   # Monitor GPU usage
   nvidia-smi -l 1
   
   # Detailed GPU monitoring
   nvidia-smi dmon -s pucvmet -d 1
   
   # Monitor specific processes
   nvidia-smi pmon -d 1

Performance Profiling
~~~~~~~~~~~~~~~~~~~~

Profile CUDA kernels and memory usage:

.. code-block:: python

   import torch.profiler
   
   # Profile training with CUDA events
   with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
       on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
       record_shapes=True,
       profile_memory=True,
       with_stack=True
   ) as prof:
       for step, batch in enumerate(dataloader):
           # Training step
           outputs = model(batch)
           loss = criterion(outputs, targets)
           loss.backward()
           optimizer.step()
           
           prof.step()

Cluster Job Management
---------------------

SLURM Integration
~~~~~~~~~~~~~~~~

Example SLURM job scripts:

**Single GPU Job**:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=geminisdr-single
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --gres=gpu:1
   #SBATCH --mem=32G
   #SBATCH --time=12:00:00
   
   source geminisdr-cuda-env/bin/activate
   python train_single_gpu.py

**Multi-GPU Job**:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=geminisdr-multi
   #SBATCH --nodes=1
   #SBATCH --ntasks=4
   #SBATCH --gres=gpu:4
   #SBATCH --mem=128G
   #SBATCH --time=24:00:00
   
   source geminisdr-cuda-env/bin/activate
   
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export WORLD_SIZE=4
   
   srun --ntasks=4 python train_multi_gpu.py

Job Monitoring
~~~~~~~~~~~~~

Monitor running jobs:

.. code-block:: bash

   # Check job status
   squeue -u $USER
   
   # Monitor job output
   tail -f slurm-<jobid>.out
   
   # Check GPU usage for job
   srun --jobid=<jobid> --pty nvidia-smi

Testing and Validation
----------------------

CUDA Functionality Tests
~~~~~~~~~~~~~~~~~~~~~~~

Verify CUDA functionality:

.. code-block:: bash

   # Run CUDA-specific tests
   pytest tests/cuda/ -v
   
   # Test multi-GPU functionality
   pytest tests/distributed/ -v
   
   # Performance benchmarks
   python scripts/benchmark_cuda.py

Memory and Performance Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test memory usage and performance:

.. code-block:: python

   # Test GPU memory allocation
   python -c "
   import torch
   
   # Test memory allocation
   device = torch.device('cuda')
   x = torch.randn(10000, 10000, device=device)
   print(f'Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
   
   # Test computation
   y = torch.mm(x, x)
   print('Matrix multiplication successful')
   
   # Cleanup
   del x, y
   torch.cuda.empty_cache()
   "

Troubleshooting
--------------

Common CUDA Issues
~~~~~~~~~~~~~~~~~

**Issue**: CUDA out of memory

   **Solution**: Reduce batch size or enable memory optimizations:
   
   .. code-block:: python
   
      # Reduce batch size
      batch_size = 16  # Start small and increase
      
      # Enable gradient checkpointing
      model.gradient_checkpointing_enable()
      
      # Clear cache regularly
      torch.cuda.empty_cache()

**Issue**: cuDNN version mismatch

   **Solution**: Verify cuDNN installation:
   
   .. code-block:: bash
   
      # Check cuDNN version
      cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
      
      # Reinstall PyTorch with correct CUDA version
      pip uninstall torch torchvision torchaudio
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

**Issue**: Multi-GPU training hangs

   **Solution**: Check NCCL configuration:
   
   .. code-block:: bash
   
      # Set NCCL debug level
      export NCCL_DEBUG=INFO
      
      # Check network connectivity between nodes
      ping <other-node-ip>
      
      # Verify InfiniBand (if available)
      ibstat

**Issue**: Poor multi-GPU scaling

   **Solution**: Optimize data loading and communication:
   
   .. code-block:: python
   
      # Increase data loader workers
      dataloader = DataLoader(dataset, num_workers=8, pin_memory=True)
      
      # Use efficient communication backend
      dist.init_process_group(backend='nccl')
      
      # Optimize batch size per GPU
      batch_size_per_gpu = total_batch_size // world_size

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~

Expected performance on different GPU configurations:

.. list-table:: CUDA Performance Benchmarks
   :header-rows: 1
   :widths: 25 25 25 25

   * - GPU Configuration
     - Training Speed
     - Inference Latency
     - Memory Usage
   * - Single RTX 3080
     - 150 samples/sec
     - 5ms
     - 8GB
   * - Single A100 40GB
     - 300 samples/sec
     - 3ms
     - 16GB
   * - 4x A100 40GB
     - 1000 samples/sec
     - 3ms
     - 64GB
   * - 8x A100 80GB
     - 1800 samples/sec
     - 3ms
     - 128GB

Best Practices
--------------

Development Workflow
~~~~~~~~~~~~~~~~~~~

1. **Start Small**: Begin with single GPU, then scale up
2. **Profile Early**: Use profiling tools to identify bottlenecks
3. **Monitor Resources**: Keep track of GPU memory and utilization
4. **Version Control**: Track configuration changes and results

Production Deployment
~~~~~~~~~~~~~~~~~~~~

1. **Container Images**: Use Docker/Singularity for reproducible environments
2. **Resource Scheduling**: Use SLURM or Kubernetes for job management
3. **Monitoring**: Implement comprehensive monitoring and alerting
4. **Backup**: Regular backup of models and configurations

Next Steps
----------

After successful installation:

1. **Run Benchmarks**: Establish performance baselines
2. **Optimize Configuration**: Tune settings for your specific hardware
3. **Set Up Monitoring**: Implement comprehensive monitoring
4. **Scale Up**: Move to multi-node distributed training

For advanced CUDA optimization:

* :doc:`../advanced/performance_tuning`
* :doc:`../advanced/distributed_training`
* :doc:`../../development/profiling`