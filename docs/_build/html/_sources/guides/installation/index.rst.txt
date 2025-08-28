Installation Guides
===================

Platform-specific installation instructions for GeminiSDR.

.. toctree::
   :maxdepth: 1

   m1_mac
   linux_vm
   cuda_cluster

Quick Platform Selection
------------------------

**Apple Silicon Mac (M1/M2)**
   Best for: Development, prototyping, and moderate-scale training
   
   * Native ARM64 performance
   * Metal Performance Shaders acceleration
   * Unified memory architecture
   * :doc:`→ M1 Mac Guide <m1_mac>`

**Linux Virtual Machine**
   Best for: Development in constrained environments, CI/CD
   
   * CPU-optimized processing
   * Conservative memory usage
   * Cross-platform compatibility testing
   * :doc:`→ Linux VM Guide <linux_vm>`

**CUDA Cluster**
   Best for: Large-scale training, production workloads
   
   * Multi-GPU acceleration
   * Distributed training support
   * High-performance computing
   * :doc:`→ CUDA Cluster Guide <cuda_cluster>`

Detailed Installation Guides
----------------------------

* :doc:`m1_mac` - Installation on Apple Silicon Macs with Metal Performance Shaders
* :doc:`linux_vm` - Installation on Linux Virtual Machines with CPU optimization  
* :doc:`cuda_cluster` - Installation on CUDA-enabled clusters with multi-GPU support