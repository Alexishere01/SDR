"""
Tests for memory management and optimization system.

This module tests the MemoryManager, ModelCache, and related utilities
for proper memory monitoring, optimization, and cleanup functionality.
"""

import gc
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from geminisdr.config.config_models import SystemConfig, PerformanceConfig, HardwareConfig
from geminisdr.core.memory_manager import (
    MemoryManager, ModelCache, MemoryStats, MemoryMonitor,
    get_system_memory_info, get_gpu_memory_info, optimize_torch_memory_settings
)


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, size_mb: float = 100.0):
        self.size_mb = size_mb
        # Create some data to simulate model parameters
        self.data = bytearray(int(size_mb * 1024 * 1024))
        
    def parameters(self):
        """Mock parameters method for PyTorch-like interface."""
        # Simulate parameters with the right number of elements
        num_params = int(self.size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
        return [Mock(numel=lambda: num_params)]
        
    def __sizeof__(self):
        """Return size in bytes."""
        return len(self.data)


@pytest.fixture
def test_config():
    """Create test configuration."""
    return SystemConfig(
        performance=PerformanceConfig(
            memory_threshold=0.7,
            memory_cleanup_threshold=0.9,
            auto_optimize=True,
            batch_size_auto_tune=True
        ),
        hardware=HardwareConfig(
            device_preference="cpu",
            memory_limit_gb=4.0
        )
    )


@pytest.fixture
def memory_manager(test_config):
    """Create MemoryManager instance for testing."""
    return MemoryManager(test_config)


@pytest.fixture
def model_cache():
    """Create ModelCache instance for testing."""
    return ModelCache(max_size=3, max_memory_mb=500.0)


class TestMemoryStats:
    """Test MemoryStats dataclass."""
    
    def test_memory_stats_creation(self):
        """Test MemoryStats creation and attributes."""
        stats = MemoryStats(
            total_ram_mb=8192.0,
            available_ram_mb=4096.0,
            used_ram_mb=4096.0,
            ram_usage_percent=50.0,
            gpu_memory_mb=2048.0,
            gpu_available_mb=1024.0,
            gpu_usage_percent=50.0
        )
        
        assert stats.total_ram_mb == 8192.0
        assert stats.available_ram_mb == 4096.0
        assert stats.used_ram_mb == 4096.0
        assert stats.ram_usage_percent == 50.0
        assert stats.gpu_memory_mb == 2048.0
        assert stats.gpu_available_mb == 1024.0
        assert stats.gpu_usage_percent == 50.0


class TestMemoryManager:
    """Test MemoryManager functionality."""
    
    def test_memory_manager_initialization(self, memory_manager):
        """Test MemoryManager initialization."""
        assert memory_manager.config is not None
        assert memory_manager.device in ["cpu", "cuda", "mps"]
        assert isinstance(memory_manager.gpu_available, bool)
        assert memory_manager.memory_threshold == 0.7
        assert memory_manager.cleanup_threshold == 0.9
        
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    @patch('psutil.Process')
    def test_get_memory_stats(self, mock_process, mock_swap, mock_memory, memory_manager):
        """Test memory statistics collection."""
        # Mock system memory
        mock_memory.return_value = Mock(
            total=8 * 1024**3,  # 8GB
            available=4 * 1024**3,  # 4GB
            used=4 * 1024**3,  # 4GB
            percent=50.0
        )
        
        mock_swap.return_value = Mock(used=0)
        mock_process.return_value.memory_info.return_value = Mock(rss=1024**3)  # 1GB
        
        stats = memory_manager.get_memory_stats()
        
        assert stats.total_ram_mb == 8192.0
        assert stats.available_ram_mb == 4096.0
        assert stats.used_ram_mb == 4096.0
        assert stats.ram_usage_percent == 50.0
        assert stats.process_memory_mb == 1024.0
        
    def test_optimize_batch_size(self, memory_manager):
        """Test batch size optimization."""
        # Test with auto-tune enabled
        with patch.object(memory_manager, 'get_memory_stats') as mock_stats:
            mock_stats.return_value = MemoryStats(
                total_ram_mb=8192.0,
                available_ram_mb=4096.0,
                used_ram_mb=4096.0,
                ram_usage_percent=50.0
            )
            
            optimized = memory_manager.optimize_batch_size(32, model_size_mb=100.0)
            assert isinstance(optimized, int)
            assert optimized > 0
            
    def test_optimize_batch_size_disabled(self, test_config):
        """Test batch size optimization when disabled."""
        test_config.performance.batch_size_auto_tune = False
        memory_manager = MemoryManager(test_config)
        
        optimized = memory_manager.optimize_batch_size(32, model_size_mb=100.0)
        assert optimized == 32
        
    def test_cleanup_memory(self, memory_manager):
        """Test memory cleanup functionality."""
        cleanup_called = False
        
        def cleanup_callback():
            nonlocal cleanup_called
            cleanup_called = True
            
        memory_manager.register_cleanup_callback(cleanup_callback)
        
        with patch('gc.collect', return_value=10) as mock_gc:
            memory_manager.cleanup_memory()
            
        assert cleanup_called
        mock_gc.assert_called_once()
        
    def test_memory_efficient_context(self, memory_manager):
        """Test memory-efficient context manager."""
        with patch.object(memory_manager, 'cleanup_memory') as mock_cleanup:
            with patch.object(memory_manager, 'get_memory_stats') as mock_stats:
                mock_stats.return_value = MemoryStats(
                    total_ram_mb=8192.0,
                    available_ram_mb=4096.0,
                    used_ram_mb=4096.0,
                    ram_usage_percent=50.0
                )
                
                with memory_manager.memory_efficient_context():
                    pass
                    
        # Should cleanup on exit
        mock_cleanup.assert_called()
        
    def test_estimate_model_memory(self, memory_manager):
        """Test model memory estimation."""
        # Test with number of parameters
        memory_mb = memory_manager.estimate_model_memory(1000000)  # 1M parameters
        assert memory_mb > 0
        assert isinstance(memory_mb, float)
        
    def test_get_optimal_worker_count(self, memory_manager):
        """Test optimal worker count calculation."""
        with patch.object(memory_manager, 'get_memory_stats') as mock_stats:
            mock_stats.return_value = MemoryStats(
                total_ram_mb=8192.0,
                available_ram_mb=4096.0,
                used_ram_mb=4096.0,
                ram_usage_percent=50.0
            )
            
            workers = memory_manager.get_optimal_worker_count()
            assert isinstance(workers, int)
            assert workers >= 1
            assert workers <= 8  # Should be capped
            
    def test_memory_history(self, memory_manager):
        """Test memory usage history tracking."""
        # Mock the underlying system calls but let get_memory_stats work normally
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.swap_memory') as mock_swap, \
             patch('psutil.Process') as mock_process:
            
            mock_memory.return_value = Mock(
                total=8 * 1024**3,
                available=4 * 1024**3,
                used=4 * 1024**3,
                percent=50.0
            )
            mock_swap.return_value = Mock(used=0)
            mock_process.return_value.memory_info.return_value = Mock(rss=1024**3)
            
            # Generate some history
            for _ in range(5):
                memory_manager.get_memory_stats()
                
            history = memory_manager.get_memory_history()
            assert len(history) == 5
            
            recent_history = memory_manager.get_memory_history(last_n=3)
            assert len(recent_history) == 3


class TestModelCache:
    """Test ModelCache functionality."""
    
    def test_model_cache_initialization(self, model_cache):
        """Test ModelCache initialization."""
        assert model_cache.max_size == 3
        assert model_cache.max_memory_mb == 500.0
        assert len(model_cache) == 0
        
    def test_put_and_get_model(self, model_cache):
        """Test putting and getting models from cache."""
        model = MockModel(size_mb=100.0)
        model_key = "test_model"
        
        # Put model in cache
        model_cache.put_model(model_key, model, size_mb=100.0)
        assert len(model_cache) == 1
        assert model_key in model_cache
        
        # Get model from cache
        cached_model = model_cache.get_model(model_key)
        assert cached_model is model
        
    def test_cache_miss(self, model_cache):
        """Test cache miss behavior."""
        result = model_cache.get_model("nonexistent_model")
        assert result is None
        
    def test_lru_eviction_by_size(self, model_cache):
        """Test LRU eviction when size limit is reached."""
        # Add models up to the limit
        models = []
        for i in range(4):  # One more than max_size
            model = MockModel(size_mb=50.0)
            models.append(model)
            model_cache.put_model(f"model_{i}", model, size_mb=50.0)
            
        # Should only have max_size models
        assert len(model_cache) == model_cache.max_size
        
        # First model should be evicted
        assert "model_0" not in model_cache
        assert "model_3" in model_cache
        
    def test_lru_eviction_by_memory(self):
        """Test LRU eviction when memory limit is reached."""
        cache = ModelCache(max_size=10, max_memory_mb=200.0)
        
        # Add models that exceed memory limit
        models = []
        for i in range(3):
            model = MockModel(size_mb=100.0)
            models.append(model)
            cache.put_model(f"model_{i}", model, size_mb=100.0)
            
        # Should evict models to stay under memory limit
        assert len(cache) <= 2  # 200MB / 100MB per model
        
    def test_model_size_estimation(self, model_cache):
        """Test automatic model size estimation."""
        model = MockModel(size_mb=150.0)
        model_cache.put_model("test_model", model)  # No size_mb provided
        
        stats = model_cache.get_cache_stats()
        assert stats['total_memory_mb'] > 0
        
    def test_cache_stats(self, model_cache):
        """Test cache statistics."""
        model = MockModel(size_mb=100.0)
        model_cache.put_model("test_model", model, size_mb=100.0)
        
        stats = model_cache.get_cache_stats()
        assert stats['size'] == 1
        assert stats['max_size'] == 3
        assert stats['total_memory_mb'] == 100.0
        assert stats['max_memory_mb'] == 500.0
        assert 'test_model' in stats['models']
        
    def test_model_info(self, model_cache):
        """Test getting model information."""
        model = MockModel(size_mb=100.0)
        model_cache.put_model("test_model", model, size_mb=100.0)
        
        info = model_cache.get_model_info("test_model")
        assert info is not None
        assert info['size_mb'] == 100.0
        assert info['access_count'] == 1
        assert info['is_alive'] is True
        
        # Test nonexistent model
        info = model_cache.get_model_info("nonexistent")
        assert info is None
        
    def test_cache_clear(self, model_cache):
        """Test clearing the cache."""
        model = MockModel(size_mb=100.0)
        model_cache.put_model("test_model", model, size_mb=100.0)
        assert len(model_cache) == 1
        
        model_cache.clear()
        assert len(model_cache) == 0
        
    def test_lru_access_pattern(self, model_cache):
        """Test LRU access pattern behavior."""
        # Fill cache to capacity
        models = []
        for i in range(3):
            model = MockModel(size_mb=50.0)
            models.append(model)
            model_cache.put_model(f"model_{i}", model, size_mb=50.0)
            
        # Access model_0 to make it most recently used
        model_cache.get_model("model_0")
        
        # Add another model, should evict model_1 (least recently used)
        new_model = MockModel(size_mb=50.0)
        model_cache.put_model("model_3", new_model, size_mb=50.0)
        
        assert "model_0" in model_cache  # Should still be there
        assert "model_1" not in model_cache  # Should be evicted
        assert "model_3" in model_cache  # Should be added


class TestMemoryMonitor:
    """Test MemoryMonitor functionality."""
    
    def test_memory_monitor_initialization(self, memory_manager):
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor(memory_manager, check_interval=1.0)
        assert monitor.memory_manager is memory_manager
        assert monitor.check_interval == 1.0
        assert not monitor.monitoring
        
    def test_monitor_start_stop(self, memory_manager):
        """Test starting and stopping memory monitoring."""
        monitor = MemoryMonitor(memory_manager, check_interval=0.1)
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring
        assert monitor.monitor_thread is not None
        
        # Give it a moment to run
        time.sleep(0.2)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring
        
    def test_monitor_callbacks(self, memory_manager):
        """Test memory monitor callbacks."""
        monitor = MemoryMonitor(memory_manager, check_interval=0.1)
        callback_called = False
        callback_stats = None
        
        def test_callback(stats):
            nonlocal callback_called, callback_stats
            callback_called = True
            callback_stats = stats
            
        monitor.add_callback(test_callback)
        
        # Mock get_memory_stats to return predictable data
        with patch.object(memory_manager, 'get_memory_stats') as mock_stats:
            mock_stats.return_value = MemoryStats(
                total_ram_mb=8192.0,
                available_ram_mb=4096.0,
                used_ram_mb=4096.0,
                ram_usage_percent=50.0
            )
            
            monitor.start_monitoring()
            time.sleep(0.2)  # Let it run briefly
            monitor.stop_monitoring()
            
        assert callback_called
        assert callback_stats is not None


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    def test_get_system_memory_info(self, mock_swap, mock_memory):
        """Test system memory information retrieval."""
        mock_memory.return_value = Mock(
            total=8 * 1024**3,
            available=4 * 1024**3,
            used=4 * 1024**3,
            percent=50.0
        )
        mock_swap.return_value = Mock(
            total=2 * 1024**3,
            used=1 * 1024**3,
            percent=50.0
        )
        
        info = get_system_memory_info()
        
        assert info['total_ram_gb'] == 8.0
        assert info['available_ram_gb'] == 4.0
        assert info['used_ram_gb'] == 4.0
        assert info['ram_percent'] == 50.0
        assert info['total_swap_gb'] == 2.0
        assert info['used_swap_gb'] == 1.0
        assert info['swap_percent'] == 50.0
        
    def test_get_gpu_memory_info_no_torch(self):
        """Test GPU memory info when torch is not available."""
        with patch('geminisdr.core.memory_manager.TORCH_AVAILABLE', False):
            info = get_gpu_memory_info()
            assert info is None
            
    def test_optimize_torch_memory_settings_no_torch(self):
        """Test torch memory optimization when torch is not available."""
        with patch('geminisdr.core.memory_manager.TORCH_AVAILABLE', False):
            # Should not raise an exception
            optimize_torch_memory_settings("cpu")


class TestPerformanceTests:
    """Performance tests for memory optimization features."""
    
    def test_memory_cleanup_performance(self, memory_manager):
        """Test memory cleanup performance."""
        start_time = time.time()
        memory_manager.cleanup_memory()
        cleanup_time = time.time() - start_time
        
        # Cleanup should be reasonably fast (< 1 second)
        assert cleanup_time < 1.0
        
    def test_cache_access_performance(self, model_cache):
        """Test cache access performance."""
        # Add some models to cache
        models = []
        for i in range(model_cache.max_size):
            model = MockModel(size_mb=50.0)
            models.append(model)
            model_cache.put_model(f"model_{i}", model, size_mb=50.0)
            
        # Time cache access
        start_time = time.time()
        for i in range(100):  # 100 accesses
            model_cache.get_model(f"model_{i % model_cache.max_size}")
        access_time = time.time() - start_time
        
        # Cache access should be very fast
        assert access_time < 0.1  # Less than 100ms for 100 accesses
        
    def test_memory_stats_collection_performance(self, memory_manager):
        """Test memory statistics collection performance."""
        start_time = time.time()
        for _ in range(10):  # Collect stats 10 times
            memory_manager.get_memory_stats()
        collection_time = time.time() - start_time
        
        # Stats collection should be reasonably fast
        assert collection_time < 1.0
        
    def test_batch_size_optimization_performance(self, memory_manager):
        """Test batch size optimization performance."""
        start_time = time.time()
        for _ in range(100):  # Optimize 100 times
            memory_manager.optimize_batch_size(32, model_size_mb=100.0)
        optimization_time = time.time() - start_time
        
        # Optimization should be fast
        assert optimization_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__])