"""
Memory management and optimization system for GeminiSDR.

This module provides comprehensive memory monitoring, optimization, and cleanup
utilities for both system RAM and GPU memory across different platforms.
"""

import gc
import logging
import psutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List, Tuple
from collections import OrderedDict
import weakref

# Platform-specific imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from geminisdr.config.config_models import SystemConfig, PerformanceConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics for system and GPU."""
    total_ram_mb: float
    available_ram_mb: float
    used_ram_mb: float
    ram_usage_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_available_mb: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    swap_used_mb: Optional[float] = None
    process_memory_mb: Optional[float] = None


class MemoryMonitor:
    """Background memory monitoring with threshold alerts."""
    
    def __init__(self, memory_manager: 'MemoryManager', check_interval: float = 5.0):
        self.memory_manager = memory_manager
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable[[MemoryStats], None]] = []
        
    def add_callback(self, callback: Callable[[MemoryStats], None]) -> None:
        """Add callback to be called when memory stats are updated."""
        self.callbacks.append(callback)
        
    def start_monitoring(self) -> None:
        """Start background memory monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Memory monitoring stopped")
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                stats = self.memory_manager.get_memory_stats()
                
                # Check thresholds and trigger cleanup if needed
                if stats.ram_usage_percent > self.memory_manager.config.performance.memory_cleanup_threshold:
                    logger.warning(f"Memory usage high: {stats.ram_usage_percent:.1f}%")
                    self.memory_manager.cleanup_memory()
                    
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        logger.error(f"Error in memory monitor callback: {e}")
                        
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                
            time.sleep(self.check_interval)


class MemoryManager:
    """System memory management and optimization."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.performance_config = config.performance
        self.hardware_config = config.hardware
        self.memory_threshold = self.performance_config.memory_threshold
        self.cleanup_threshold = self.performance_config.memory_cleanup_threshold
        
        # Memory tracking
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._memory_history: List[MemoryStats] = []
        self._max_history = 100
        
        # Background monitoring
        self.monitor = MemoryMonitor(self)
        
        # Device detection
        self.device = self._detect_device()
        self.gpu_available = self._check_gpu_availability()
        
        logger.info(f"MemoryManager initialized for device: {self.device}")
        
    def _detect_device(self) -> str:
        """Detect the best available compute device."""
        if self.hardware_config.device_preference.value == "auto":
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            return "cpu"
        else:
            return self.hardware_config.device_preference.value
            
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for memory monitoring."""
        if not TORCH_AVAILABLE:
            return False
            
        if self.device == "cuda":
            return torch.cuda.is_available()
        elif self.device == "mps":
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        return False
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        # System memory stats
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        stats = MemoryStats(
            total_ram_mb=memory.total / (1024 * 1024),
            available_ram_mb=memory.available / (1024 * 1024),
            used_ram_mb=memory.used / (1024 * 1024),
            ram_usage_percent=memory.percent,
            swap_used_mb=swap.used / (1024 * 1024),
            process_memory_mb=process_memory.rss / (1024 * 1024)
        )
        
        # GPU memory stats
        if self.gpu_available and TORCH_AVAILABLE:
            try:
                if self.device == "cuda":
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_allocated = torch.cuda.memory_allocated(0)
                    gpu_reserved = torch.cuda.memory_reserved(0)
                    
                    stats.gpu_memory_mb = gpu_memory / (1024 * 1024)
                    stats.gpu_available_mb = (gpu_memory - gpu_reserved) / (1024 * 1024)
                    stats.gpu_usage_percent = (gpu_allocated / gpu_memory) * 100
                    
                elif self.device == "mps":
                    # MPS doesn't provide detailed memory stats, use approximation
                    stats.gpu_memory_mb = 8192.0  # Approximate for M1/M2
                    stats.gpu_available_mb = stats.gpu_memory_mb * 0.8  # Conservative estimate
                    stats.gpu_usage_percent = 20.0  # Placeholder
                    
            except Exception as e:
                logger.debug(f"Could not get GPU memory stats: {e}")
                
        # Store in history
        self._memory_history.append(stats)
        if len(self._memory_history) > self._max_history:
            self._memory_history.pop(0)
            
        return stats
        
    def optimize_batch_size(self, base_batch_size: int, model_size_mb: float = 0.0) -> int:
        """Dynamically optimize batch size based on available memory."""
        if not self.performance_config.batch_size_auto_tune:
            return base_batch_size
            
        stats = self.get_memory_stats()
        
        # Calculate available memory for processing
        if self.gpu_available and stats.gpu_available_mb:
            available_mb = stats.gpu_available_mb
            # Reserve some memory for model and operations
            usable_mb = available_mb - model_size_mb - 512  # 512MB buffer
        else:
            available_mb = stats.available_ram_mb
            # Reserve more memory for system operations
            usable_mb = available_mb - model_size_mb - 1024  # 1GB buffer
            
        if usable_mb <= 0:
            logger.warning("Very low memory available, using minimum batch size")
            return max(1, base_batch_size // 4)
            
        # Estimate memory per sample (rough heuristic)
        estimated_mb_per_sample = max(1.0, model_size_mb / 100)  # Very rough estimate
        max_batch_size = int(usable_mb / estimated_mb_per_sample)
        
        # Apply memory threshold
        target_batch_size = min(base_batch_size, max_batch_size)
        target_batch_size = max(1, int(target_batch_size * self.memory_threshold))
        
        if target_batch_size != base_batch_size:
            logger.info(f"Optimized batch size: {base_batch_size} -> {target_batch_size}")
            
        return target_batch_size
        
    def cleanup_memory(self) -> None:
        """Force memory cleanup and garbage collection."""
        logger.debug("Starting memory cleanup")
        
        # Call registered cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
                
        # Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        
        # GPU memory cleanup
        if self.gpu_available and TORCH_AVAILABLE:
            try:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.debug("CUDA cache cleared")
                elif self.device == "mps":
                    torch.mps.empty_cache()
                    logger.debug("MPS cache cleared")
            except Exception as e:
                logger.error(f"Error clearing GPU cache: {e}")
                
        # Force additional cleanup for numpy arrays if available
        if NUMPY_AVAILABLE:
            try:
                # This is a bit aggressive but helps with memory fragmentation
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass  # Not available on all systems
                
        logger.debug("Memory cleanup completed")
        
    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called during memory cleanup."""
        self._cleanup_callbacks.append(callback)
        
    def monitor_memory_usage(self) -> None:
        """Start monitoring memory usage and trigger cleanup if needed."""
        if not self.monitor.monitoring:
            self.monitor.start_monitoring()
            
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitor.stop_monitoring()
        
    def get_memory_history(self, last_n: Optional[int] = None) -> List[MemoryStats]:
        """Get memory usage history."""
        if last_n is None:
            return self._memory_history.copy()
        return self._memory_history[-last_n:]
        
    def estimate_model_memory(self, num_parameters: int, dtype_size: int = 4) -> float:
        """Estimate memory usage for a model in MB."""
        # Base model memory (parameters + gradients + optimizer states)
        base_memory = num_parameters * dtype_size * 3  # params + grads + optimizer
        
        # Add overhead for activations and intermediate computations
        overhead_factor = 1.5
        total_bytes = base_memory * overhead_factor
        
        return total_bytes / (1024 * 1024)  # Convert to MB
        
    @contextmanager
    def memory_efficient_context(self, cleanup_on_exit: bool = True):
        """Context manager for memory-efficient operations."""
        initial_stats = self.get_memory_stats()
        logger.debug(f"Entering memory-efficient context. RAM: {initial_stats.ram_usage_percent:.1f}%")
        
        try:
            # Pre-cleanup if memory usage is high
            if initial_stats.ram_usage_percent > self.memory_threshold * 100:
                self.cleanup_memory()
                
            yield self
            
        finally:
            if cleanup_on_exit:
                self.cleanup_memory()
                
            final_stats = self.get_memory_stats()
            logger.debug(f"Exiting memory-efficient context. RAM: {final_stats.ram_usage_percent:.1f}%")
            
    def get_optimal_worker_count(self) -> int:
        """Get optimal number of worker processes based on available memory."""
        stats = self.get_memory_stats()
        
        # Base on CPU count but limit by available memory
        cpu_count = psutil.cpu_count()
        
        # Estimate memory per worker (rough heuristic)
        memory_per_worker_mb = 512  # Conservative estimate
        max_workers_by_memory = int(stats.available_ram_mb / memory_per_worker_mb)
        
        # Don't exceed CPU count or memory constraints
        optimal_workers = min(cpu_count, max_workers_by_memory, 8)  # Cap at 8
        optimal_workers = max(1, optimal_workers)  # At least 1
        
        logger.debug(f"Optimal worker count: {optimal_workers} (CPU: {cpu_count}, Memory limit: {max_workers_by_memory})")
        return optimal_workers
        
    def __enter__(self):
        """Context manager entry."""
        self.monitor_memory_usage()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
        if self.performance_config.auto_optimize:
            self.cleanup_memory()


class ModelCache:
    """LRU cache for ML models with memory-based eviction."""
    
    def __init__(self, max_size: int = 3, max_memory_mb: float = 2048, 
                 memory_manager: Optional[MemoryManager] = None):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.memory_manager = memory_manager
        
        # Use OrderedDict for LRU behavior
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.model_sizes: Dict[str, float] = {}  # Track model sizes in MB
        self.access_count: Dict[str, int] = {}
        
        # Weak references to models to help with garbage collection
        self._model_refs: Dict[str, weakref.ref] = {}
        
        logger.info(f"ModelCache initialized: max_size={max_size}, max_memory_mb={max_memory_mb}")
        
    def get_model(self, model_key: str) -> Optional[Any]:
        """Get model from cache, loading if necessary."""
        if model_key in self.cache:
            # Move to end (most recently used)
            model_info = self.cache.pop(model_key)
            self.cache[model_key] = model_info
            self.access_count[model_key] = self.access_count.get(model_key, 0) + 1
            
            # Check if model is still alive via weak reference
            model_ref = self._model_refs.get(model_key)
            if model_ref:
                model = model_ref()
                if model is not None:
                    logger.debug(f"Cache hit for model: {model_key}")
                    return model
                else:
                    # Model was garbage collected, remove from cache
                    logger.debug(f"Model {model_key} was garbage collected, removing from cache")
                    self._remove_model(model_key)
                    
        logger.debug(f"Cache miss for model: {model_key}")
        return None
        
    def put_model(self, model_key: str, model: Any, size_mb: Optional[float] = None) -> None:
        """Add model to cache with LRU eviction."""
        # Estimate model size if not provided
        if size_mb is None:
            size_mb = self._estimate_model_size(model)
            
        # Check if we need to evict models to make space
        self._ensure_space(size_mb)
        
        # Remove existing entry if present
        if model_key in self.cache:
            self._remove_model(model_key)
            
        # Add new model
        model_info = {
            'size_mb': size_mb,
            'timestamp': time.time(),
            'access_count': 1
        }
        
        self.cache[model_key] = model_info
        self.model_sizes[model_key] = size_mb
        self.access_count[model_key] = 1
        
        # Store weak reference to help with memory management
        self._model_refs[model_key] = weakref.ref(model, lambda ref: self._on_model_deleted(model_key))
        
        logger.info(f"Added model to cache: {model_key} ({size_mb:.1f} MB)")
        
        # Final check to ensure we're within limits
        self._enforce_limits()
        
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        if TORCH_AVAILABLE and hasattr(model, 'parameters'):
            # PyTorch model
            total_params = sum(p.numel() for p in model.parameters())
            # Assume float32 (4 bytes per parameter)
            size_bytes = total_params * 4
            return size_bytes / (1024 * 1024)
        elif hasattr(model, '__sizeof__'):
            # Generic Python object
            return model.__sizeof__() / (1024 * 1024)
        else:
            # Fallback estimate
            return 100.0  # 100MB default estimate
            
    def _ensure_space(self, required_mb: float) -> None:
        """Ensure there's enough space for a new model."""
        current_memory = sum(self.model_sizes.values())
        
        # Check memory limit
        while (current_memory + required_mb > self.max_memory_mb and 
               len(self.cache) > 0):
            self._evict_lru()
            current_memory = sum(self.model_sizes.values())
            
        # Check size limit
        while len(self.cache) >= self.max_size and len(self.cache) > 0:
            self._evict_lru()
            
    def _evict_lru(self) -> None:
        """Evict least recently used model."""
        if not self.cache:
            return
            
        # Get least recently used (first item in OrderedDict)
        lru_key = next(iter(self.cache))
        self._remove_model(lru_key)
        logger.info(f"Evicted LRU model: {lru_key}")
        
    def _remove_model(self, model_key: str) -> None:
        """Remove model from cache and cleanup references."""
        if model_key in self.cache:
            del self.cache[model_key]
        if model_key in self.model_sizes:
            del self.model_sizes[model_key]
        if model_key in self.access_count:
            del self.access_count[model_key]
        if model_key in self._model_refs:
            del self._model_refs[model_key]
            
    def _on_model_deleted(self, model_key: str) -> None:
        """Callback when a model is garbage collected."""
        logger.debug(f"Model {model_key} was garbage collected")
        
    def _enforce_limits(self) -> None:
        """Enforce cache size and memory limits."""
        # Remove any dead references
        dead_keys = []
        for key, ref in self._model_refs.items():
            if ref() is None:
                dead_keys.append(key)
                
        for key in dead_keys:
            self._remove_model(key)
            
        # Enforce memory limit
        current_memory = sum(self.model_sizes.values())
        while current_memory > self.max_memory_mb and len(self.cache) > 0:
            self._evict_lru()
            current_memory = sum(self.model_sizes.values())
            
        # Enforce size limit
        while len(self.cache) > self.max_size:
            self._evict_lru()
            
    def clear(self) -> None:
        """Clear all models from cache."""
        keys = list(self.cache.keys())
        for key in keys:
            self._remove_model(key)
        logger.info("Model cache cleared")
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_memory = sum(self.model_sizes.values())
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'total_memory_mb': total_memory,
            'max_memory_mb': self.max_memory_mb,
            'memory_utilization': total_memory / self.max_memory_mb if self.max_memory_mb > 0 else 0,
            'models': list(self.cache.keys()),
            'access_counts': self.access_count.copy()
        }
        
    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a cached model."""
        if model_key not in self.cache:
            return None
            
        info = self.cache[model_key].copy()
        info['access_count'] = self.access_count.get(model_key, 0)
        info['is_alive'] = model_key in self._model_refs and self._model_refs[model_key]() is not None
        return info
        
    def __len__(self) -> int:
        """Return number of cached models."""
        return len(self.cache)
        
    def __contains__(self, model_key: str) -> bool:
        """Check if model is in cache."""
        return model_key in self.cache
        
    def __repr__(self) -> str:
        """String representation of cache."""
        total_memory = sum(self.model_sizes.values())
        return (f"ModelCache(size={len(self.cache)}/{self.max_size}, "
                f"memory={total_memory:.1f}/{self.max_memory_mb} MB)")


# Utility functions for memory management

def get_system_memory_info() -> Dict[str, float]:
    """Get comprehensive system memory information."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'total_ram_gb': memory.total / (1024**3),
        'available_ram_gb': memory.available / (1024**3),
        'used_ram_gb': memory.used / (1024**3),
        'ram_percent': memory.percent,
        'total_swap_gb': swap.total / (1024**3),
        'used_swap_gb': swap.used / (1024**3),
        'swap_percent': swap.percent
    }


def get_gpu_memory_info() -> Optional[Dict[str, float]]:
    """Get GPU memory information if available."""
    if not TORCH_AVAILABLE:
        return None
        
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            
            return {
                'total_gb': props.total_memory / (1024**3),
                'allocated_gb': allocated / (1024**3),
                'reserved_gb': reserved / (1024**3),
                'free_gb': (props.total_memory - reserved) / (1024**3),
                'utilization_percent': (allocated / props.total_memory) * 100
            }
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't provide detailed memory stats
            return {
                'total_gb': 8.0,  # Approximate for M1/M2
                'allocated_gb': 1.0,  # Placeholder
                'reserved_gb': 1.0,  # Placeholder
                'free_gb': 6.0,  # Placeholder
                'utilization_percent': 12.5  # Placeholder
            }
    except Exception as e:
        logger.debug(f"Could not get GPU memory info: {e}")
        
    return None


def optimize_torch_memory_settings(device: str) -> None:
    """Optimize PyTorch memory settings for the given device."""
    if not TORCH_AVAILABLE:
        return
        
    try:
        if device == "cuda" and torch.cuda.is_available():
            # Enable memory pool for better allocation
            torch.cuda.empty_cache()
            # Set memory fraction if configured
            # torch.cuda.set_per_process_memory_fraction(0.8)
            logger.info("Optimized CUDA memory settings")
            
        elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS-specific optimizations
            torch.mps.empty_cache()
            logger.info("Optimized MPS memory settings")
            
    except Exception as e:
        logger.warning(f"Could not optimize memory settings for {device}: {e}")