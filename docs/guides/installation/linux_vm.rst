Linux VM Installation Guide
===========================

This guide provides detailed installation instructions for GeminiSDR on Linux virtual machines, including Ubuntu, CentOS, and other distributions.

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~

* **VM Specifications**: 
  
  * 4+ CPU cores allocated to VM
  * 8GB+ RAM allocated (16GB recommended)
  * 20GB+ storage space
  * USB passthrough enabled (for SDR hardware)

* **Host System**: 
  
  * VMware Workstation/Fusion, VirtualBox, or KVM/QEMU
  * USB 3.0 support for SDR devices
  * Sufficient host resources

* **Guest OS**: 
  
  * Ubuntu 20.04 LTS or later (recommended)
  * CentOS 8+ / RHEL 8+
  * Debian 11+
  * Fedora 35+

VM Configuration
~~~~~~~~~~~~~~~

Recommended VM settings:

.. code-block:: text

   CPU: 4-8 cores
   RAM: 8-16GB
   Storage: 50GB+ (thin provisioned)
   Network: NAT or Bridged
   USB: USB 3.0 controller enabled
   Graphics: 3D acceleration enabled (if available)

Installation Steps
------------------

1. Prepare the VM Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Ubuntu/Debian**:

.. code-block:: bash

   # Update system packages
   sudo apt update && sudo apt upgrade -y
   
   # Install essential build tools
   sudo apt install -y build-essential cmake git curl wget
   sudo apt install -y python3 python3-pip python3-venv python3-dev
   sudo apt install -y pkg-config libfftw3-dev libusb-1.0-0-dev

**CentOS/RHEL/Fedora**:

.. code-block:: bash

   # Update system packages
   sudo dnf update -y  # or 'yum update -y' for older versions
   
   # Install development tools
   sudo dnf groupinstall -y "Development Tools"
   sudo dnf install -y cmake git curl wget
   sudo dnf install -y python3 python3-pip python3-devel
   sudo dnf install -y pkg-config fftw-devel libusb1-devel

2. Install SDR Libraries
~~~~~~~~~~~~~~~~~~~~~~~

**Ubuntu/Debian**:

.. code-block:: bash

   # Install SDR libraries from repositories
   sudo apt install -y rtl-sdr librtlsdr-dev
   sudo apt install -y hackrf libhackrf-dev
   sudo apt install -y airspy libairspy-dev
   sudo apt install -y soapysdr-tools libsoapysdr-dev
   
   # Install additional SoapySDR modules
   sudo apt install -y soapysdr-module-rtlsdr
   sudo apt install -y soapysdr-module-hackrf
   sudo apt install -y soapysdr-module-airspy

**CentOS/RHEL/Fedora**:

.. code-block:: bash

   # Enable EPEL repository (CentOS/RHEL)
   sudo dnf install -y epel-release  # or 'yum install epel-release'
   
   # Install SDR libraries
   sudo dnf install -y rtl-sdr rtl-sdr-devel
   sudo dnf install -y hackrf hackrf-devel
   sudo dnf install -y SoapySDR SoapySDR-devel
   
   # Build additional modules from source if needed
   # (Some distributions may not have all SDR packages)

3. Configure USB Access
~~~~~~~~~~~~~~~~~~~~~~

Set up USB permissions for SDR devices:

.. code-block:: bash

   # Create udev rules for SDR devices
   sudo tee /etc/udev/rules.d/20-rtlsdr.rules << EOF
   # RTL-SDR devices
   SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2832", GROUP="plugdev", MODE="0666", SYMLINK+="rtl_sdr"
   SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0666", SYMLINK+="rtl_sdr"
   
   # HackRF devices
   SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="6089", GROUP="plugdev", MODE="0666", SYMLINK+="hackrf"
   
   # Airspy devices
   SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="60a1", GROUP="plugdev", MODE="0666", SYMLINK+="airspy"
   EOF
   
   # Add user to plugdev group
   sudo usermod -a -G plugdev $USER
   
   # Reload udev rules
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   
   # Logout and login again for group changes to take effect

4. Install Python Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and configure Python environment:

.. code-block:: bash

   # Create virtual environment
   python3 -m venv geminisdr-env
   
   # Activate virtual environment
   source geminisdr-env/bin/activate
   
   # Upgrade pip and install wheel
   pip install --upgrade pip wheel setuptools

5. Install PyTorch (CPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install CPU-optimized PyTorch for VM environment:

.. code-block:: bash

   # Install PyTorch CPU version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # Verify installation
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torch; print(f'CPU device available: {torch.cuda.is_available() == False}')"

6. Install GeminiSDR
~~~~~~~~~~~~~~~~~~~

Clone and install GeminiSDR:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/your-org/geminisdr.git
   cd geminisdr
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install -r requirements-dev.txt

7. Configure for Linux VM
~~~~~~~~~~~~~~~~~~~~~~~~~

Create VM-specific configuration:

.. code-block:: bash

   # Copy VM configuration template
   cp conf/hardware/vm_ubuntu.yaml conf/local_config.yaml
   
   # Edit configuration for your VM
   nano conf/local_config.yaml

Example Linux VM configuration:

.. code-block:: yaml

   # Linux VM optimized configuration
   hardware:
     device_preference: "cpu"  # CPU-only processing
     memory_optimization: "conservative"  # Conservative memory usage
     sdr_mode: "auto"  # Auto-detect SDR hardware
     thread_count: 4  # Match VM CPU allocation
   
   ml:
     batch_size: 16  # Smaller batches for limited memory
     precision: "float32"  # Full precision for CPU
     model_cache_size: 2  # Limited cache due to memory constraints
     gradient_checkpointing: true  # Save memory during training
   
   performance:
     memory_threshold: 0.7  # Conservative memory threshold
     auto_optimize: true
     profiling_enabled: false
   
   logging:
     level: "INFO"
     format: "structured"
     output: ["console", "file"]

VM-Specific Optimizations
-------------------------

CPU Optimization
~~~~~~~~~~~~~~~

Optimize for CPU-only processing:

.. code-block:: python

   import torch
   import os
   
   # Set optimal thread count for VM
   vm_cpu_count = 4  # Match your VM CPU allocation
   torch.set_num_threads(vm_cpu_count)
   os.environ['OMP_NUM_THREADS'] = str(vm_cpu_count)
   os.environ['MKL_NUM_THREADS'] = str(vm_cpu_count)
   
   # Enable CPU optimizations
   torch.backends.mkldnn.enabled = True

Memory Management
~~~~~~~~~~~~~~~~

Configure conservative memory usage:

.. code-block:: python

   from geminisdr.core.memory_manager import MemoryManager
   
   # Initialize with conservative settings
   memory_manager = MemoryManager(config)
   
   # Use smaller batch sizes
   optimal_batch_size = memory_manager.optimize_batch_size(
       base_batch_size=32,  # Start smaller for VMs
       model_size_mb=300
   )
   
   # Enable frequent cleanup
   memory_manager.cleanup_memory()

Storage Optimization
~~~~~~~~~~~~~~~~~~~

Optimize for VM storage performance:

.. code-block:: bash

   # Use tmpfs for temporary files (if enough RAM)
   sudo mkdir -p /tmp/geminisdr
   sudo mount -t tmpfs -o size=2G tmpfs /tmp/geminisdr
   
   # Configure temporary directory
   export TMPDIR=/tmp/geminisdr

USB Passthrough Setup
---------------------

VMware Workstation/Fusion
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Enable USB 3.0**: VM Settings → USB Controller → USB 3.0
2. **Add USB Device**: VM Settings → USB Controller → Add → Select SDR device
3. **Auto-connect**: Check "Connect at power on"

VirtualBox
~~~~~~~~~

1. **Install Extension Pack**: Download and install VirtualBox Extension Pack
2. **Enable USB 3.0**: VM Settings → USB → USB 3.0 Controller
3. **Add USB Filter**: VM Settings → USB → Add filter for SDR device

KVM/QEMU
~~~~~~~

Add USB passthrough to VM configuration:

.. code-block:: xml

   <hostdev mode='subsystem' type='usb' managed='yes'>
     <source>
       <vendor id='0x0bda'/>  <!-- RTL-SDR vendor ID -->
       <product id='0x2832'/> <!-- RTL-SDR product ID -->
     </source>
   </hostdev>

Testing and Verification
------------------------

Test SDR Hardware
~~~~~~~~~~~~~~~~

Verify SDR devices are accessible:

.. code-block:: bash

   # Test RTL-SDR
   rtl_test -t
   
   # Test HackRF (if available)
   hackrf_info
   
   # List SoapySDR devices
   SoapySDRUtil --find
   
   # Test with GeminiSDR
   python -c "
   from geminisdr.environments.hardware_abstraction import HardwareAbstraction
   from geminisdr.config.config_manager import ConfigManager
   
   config = ConfigManager().load_config()
   hw = HardwareAbstraction(config)
   devices = hw.detect_available_devices()
   print(f'Available devices: {devices}')
   "

Run Test Suite
~~~~~~~~~~~~~

Execute comprehensive tests:

.. code-block:: bash

   # Run unit tests
   pytest tests/unit/ -v
   
   # Run Linux-specific tests
   pytest tests/cross_platform/ -k "linux" -v
   
   # Run hardware tests (with SDR connected)
   pytest tests/hardware/ -v --sdr-device=rtlsdr

Performance Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~

Benchmark VM performance:

.. code-block:: bash

   # Run performance tests
   python scripts/benchmark_training.py --platform=vm_ubuntu
   python scripts/benchmark_inference.py --platform=vm_ubuntu
   
   # Compare with baseline
   python scripts/compare_benchmarks.py

Troubleshooting
--------------

Common VM Issues
~~~~~~~~~~~~~~~

**Issue**: SDR device not detected in VM

   **Solution**: Check USB passthrough configuration:
   
   .. code-block:: bash
   
      # Check USB devices in VM
      lsusb
      
      # Check device permissions
      ls -la /dev/bus/usb/
      
      # Test device access
      rtl_test -t

**Issue**: Poor performance in VM

   **Solution**: Optimize VM resources and settings:
   
   .. code-block:: bash
   
      # Check CPU allocation
      nproc
      
      # Check memory allocation
      free -h
      
      # Monitor resource usage
      htop

**Issue**: Memory errors during training

   **Solution**: Reduce batch sizes and enable optimizations:
   
   .. code-block:: yaml
   
      # In configuration
      ml:
        batch_size: 8  # Very small for VMs
        gradient_checkpointing: true
      
      performance:
        memory_threshold: 0.6  # Very conservative

**Issue**: Network connectivity problems

   **Solution**: Check VM network configuration:
   
   .. code-block:: bash
   
      # Test network connectivity
      ping google.com
      
      # Check DNS resolution
      nslookup google.com
      
      # Test package installation
      pip install --upgrade pip

Performance Expectations
~~~~~~~~~~~~~~~~~~~~~~~

Typical performance on Linux VMs:

.. list-table:: VM Performance Benchmarks
   :header-rows: 1
   :widths: 25 25 25 25

   * - VM Type
     - 4 CPU/8GB RAM
     - 6 CPU/12GB RAM
     - 8 CPU/16GB RAM
   * - Training Speed
     - 15 samples/sec
     - 25 samples/sec
     - 35 samples/sec
   * - Inference Latency
     - 50ms
     - 35ms
     - 25ms
   * - Memory Usage
     - 4-6GB
     - 6-8GB
     - 8-10GB

Development in VMs
-----------------

IDE Setup
~~~~~~~~

**VS Code with Remote Development**:

.. code-block:: bash

   # Install VS Code Server in VM
   curl -fsSL https://code-server.dev/install.sh | sh
   
   # Start code-server
   code-server --bind-addr 0.0.0.0:8080

**SSH Development**:

.. code-block:: bash

   # Enable SSH in VM
   sudo systemctl enable ssh
   sudo systemctl start ssh
   
   # Connect from host
   ssh -L 8888:localhost:8888 user@vm-ip

Debugging in VMs
~~~~~~~~~~~~~~~

Enhanced debugging for VM environment:

.. code-block:: python

   # Enable verbose logging
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Monitor resource usage
   import psutil
   print(f"CPU usage: {psutil.cpu_percent()}%")
   print(f"Memory usage: {psutil.virtual_memory().percent}%")

Backup and Snapshots
~~~~~~~~~~~~~~~~~~~

Create VM snapshots before major changes:

.. code-block:: bash

   # VMware: Create snapshot via GUI or CLI
   # VirtualBox: VBoxManage snapshot "VM Name" take "snapshot_name"
   # KVM: virsh snapshot-create-as domain snapshot_name

Next Steps
----------

After successful installation:

1. **Configure SDR Hardware**: Test with your specific SDR devices
2. **Optimize Performance**: Tune VM settings for your use case
3. **Set Up Development Environment**: Configure IDE and debugging tools
4. **Explore Examples**: :doc:`../../examples/index`

For advanced VM optimization:

* :doc:`../advanced/performance_tuning`
* :doc:`../advanced/custom_environments`
* :doc:`../../development/debugging`