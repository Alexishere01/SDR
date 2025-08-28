Memory Manager
==============

.. automodule:: geminisdr.core.memory_manager
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

MemoryStats
~~~~~~~~~~~

.. autoclass:: geminisdr.core.memory_manager.MemoryStats
   :members:
   :undoc-members:
   :show-inheritance:

   Data class containing system memory usage statistics.

MemoryManager
~~~~~~~~~~~~~

.. autoclass:: geminisdr.core.memory_manager.MemoryManager
   :members:
   :undoc-members:
   :show-inheritance:

   System memory management and optimization with automatic resource monitoring.

   Features:
   
   * Real-time memory usage monitoring
   * Dynamic batch size optimization based on available memory
   * Automatic memory cleanup and garbage collection
   * Memory-efficient context managers for large operations
   * GPU memory tracking and optimization

   Example usage::

       from geminisdr.core.memory_manager import MemoryManager
       
       memory_manager = MemoryManager(config)
       
       # Get current memory statistics
       stats = memory_manager.get_memory_stats()
       print(f"Available RAM: {stats.available_ram_mb} MB")
       
       # Optimize batch size for available memory
       optimal_batch = memory_manager.optimize_batch_size(32, model_size_mb=500)
       
       # Use memory-efficient context
       with memory_manager.memory_efficient_context():
           # Perform memory-intensive operations
           process_large_dataset()

ModelCache
~~~~~~~~~~

.. autoclass:: geminisdr.core.memory_manager.ModelCache
   :members:
   :undoc-members:
   :show-inheritance:

   LRU cache for ML models with memory-based eviction policies.

   Features:
   
   * Least Recently Used (LRU) eviction strategy
   * Memory-based cache size limits
   * Automatic model loading and caching
   * Memory usage tracking per cached model

   Example usage::

       from geminisdr.core.memory_manager import ModelCache
       
       cache = ModelCache(max_size=3, max_memory_mb=2048)
       
       # Models are automatically loaded and cached
       model = cache.get_model("resnet50_v1")
       
       # Cache handles eviction when limits are exceeded
       cache.put_model("new_model", my_model)