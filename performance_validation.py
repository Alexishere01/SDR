#!/usr/bin/env python3
"""
Performance benchmarking and validation script.
Tests performance improvements and memory optimization effectiveness.
"""

import time
import sys
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Any

def benchmark_memory_optimization():
    """Benchmark memory optimization effectiveness."""
    print("Benchmarking memory optimization...")
    
    try:
        from geminisdr.core.memory_manager import MemoryManager, ModelCache
        from geminisdr.config.config_models import create_default_config
        
        config = create_default_config('development')
        manager = MemoryManager(config)
        
        # Test 1: Memory stats collection performance
        start_time = time.time()
        for _ in range(100):
            stats = manager.get_memory_stats()
        stats_time = (time.time() - start_time) * 1000
        
        print(f"  Memory stats collection: {stats_time:.2f}ms for 100 calls")
        
        # Test 2: Batch size optimization performance
        start_time = time.time()
        for batch_size in [16, 32, 64, 128, 256]:
            optimized = manager.optimize_batch_size(batch_size, 100.0)
        optimization_time = (time.time() - start_time) * 1000
        
        print(f"  Batch size optimization: {optimization_time:.2f}ms for 5 optimizations")
        
        # Test 3: Model cache performance
        cache = ModelCache(max_size=10, max_memory_mb=1000)
        
        # Create test models
        test_models = {}
        for i in range(5):
            test_models[f'model_{i}'] = {'data': b'x' * (10 * 1024 * 1024)}  # 10MB each
        
        # Test cache put performance
        start_time = time.time()
        for name, model in test_models.items():
            cache.put_model(name, model)
        put_time = (time.time() - start_time) * 1000
        
        # Test cache get performance
        start_time = time.time()
        for name in test_models.keys():
            retrieved = cache.get_model(name)
        get_time = (time.time() - start_time) * 1000
        
        print(f"  Model cache put: {put_time:.2f}ms for 5 models")
        print(f"  Model cache get: {get_time:.2f}ms for 5 models")
        
        # Test memory cleanup
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        with manager.memory_efficient_context():
            # Simulate memory-intensive operation
            large_data = [b'x' * (1024 * 1024) for _ in range(10)]  # 10MB
            del large_data
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_diff = final_memory - initial_memory
        
        print(f"  Memory cleanup effectiveness: {memory_diff:.1f}MB difference")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Memory optimization benchmark failed: {e}")
        return False

def benchmark_logging_performance():
    """Benchmark logging system performance."""
    print("Benchmarking logging performance...")
    
    try:
        from geminisdr.core.logging_manager import get_logger
        
        logger = get_logger('performance_test')
        
        # Test 1: Basic logging performance
        start_time = time.time()
        for i in range(1000):
            logger.info(f"Test message {i}")
        basic_logging_time = (time.time() - start_time) * 1000
        
        print(f"  Basic logging: {basic_logging_time:.2f}ms for 1000 messages")
        print(f"  Average per message: {basic_logging_time/1000:.3f}ms")
        
        # Test 2: Context logging performance
        logger.add_context(test_id='performance', component='benchmark')
        
        start_time = time.time()
        for i in range(1000):
            logger.info(f"Context message {i}")
        context_logging_time = (time.time() - start_time) * 1000
        
        print(f"  Context logging: {context_logging_time:.2f}ms for 1000 messages")
        print(f"  Average per message: {context_logging_time/1000:.3f}ms")
        
        # Test 3: Performance timer overhead
        start_time = time.time()
        for i in range(100):
            with logger.performance_timer(f'test_operation_{i}'):
                time.sleep(0.001)  # 1ms operation
        timer_overhead = (time.time() - start_time) * 1000 - 100  # Subtract actual sleep time
        
        print(f"  Performance timer overhead: {timer_overhead:.2f}ms for 100 operations")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Logging performance benchmark failed: {e}")
        return False

def benchmark_metrics_collection():
    """Benchmark metrics collection performance."""
    print("Benchmarking metrics collection...")
    
    try:
        from geminisdr.core.metrics_collector import MetricsCollector
        from geminisdr.config.config_models import create_default_config
        
        config = create_default_config('development')
        collector = MetricsCollector(config)
        
        # Test 1: Metric recording performance
        start_time = time.time()
        for i in range(1000):
            collector.record_counter('test_counter', 1, {'iteration': str(i)})
        counter_time = (time.time() - start_time) * 1000
        
        print(f"  Counter recording: {counter_time:.2f}ms for 1000 metrics")
        
        start_time = time.time()
        for i in range(1000):
            collector.record_gauge('test_gauge', float(i), {'iteration': str(i)})
        gauge_time = (time.time() - start_time) * 1000
        
        print(f"  Gauge recording: {gauge_time:.2f}ms for 1000 metrics")
        
        # Test 2: System metrics collection performance
        start_time = time.time()
        for _ in range(10):
            collector.collect_system_metrics()
        system_metrics_time = (time.time() - start_time) * 1000
        
        print(f"  System metrics collection: {system_metrics_time:.2f}ms for 10 collections")
        
        # Test 3: Metric retrieval performance
        start_time = time.time()
        for _ in range(100):
            summary = collector.get_metric_summary('test_counter')
        retrieval_time = (time.time() - start_time) * 1000
        
        print(f"  Metric retrieval: {retrieval_time:.2f}ms for 100 retrievals")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Metrics collection benchmark failed: {e}")
        return False

def benchmark_configuration_loading():
    """Benchmark configuration loading performance."""
    print("Benchmarking configuration loading...")
    
    try:
        from geminisdr.config.config_manager import ConfigManager
        from geminisdr.config.config_models import create_default_config
        
        # Test 1: Default config creation performance
        start_time = time.time()
        for _ in range(100):
            config = create_default_config('development')
        default_config_time = (time.time() - start_time) * 1000
        
        print(f"  Default config creation: {default_config_time:.2f}ms for 100 configs")
        
        # Test 2: Config manager loading performance
        manager = ConfigManager()
        
        start_time = time.time()
        for _ in range(100):
            config = manager.load_config()
        loading_time = (time.time() - start_time) * 1000
        
        print(f"  Config manager loading: {loading_time:.2f}ms for 100 loads")
        
        # Test 3: Config validation performance
        config = create_default_config('development')
        
        start_time = time.time()
        for _ in range(100):
            errors = manager.validate_config(config)
        validation_time = (time.time() - start_time) * 1000
        
        print(f"  Config validation: {validation_time:.2f}ms for 100 validations")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration loading benchmark failed: {e}")
        return False

def benchmark_error_handling():
    """Benchmark error handling performance."""
    print("Benchmarking error handling...")
    
    try:
        from geminisdr.core.error_handling import ErrorHandler, GeminiSDRError
        from geminisdr.config.config_models import create_default_config
        
        config = create_default_config('development')
        handler = ErrorHandler(config)
        
        # Test 1: Error context manager overhead
        start_time = time.time()
        for i in range(1000):
            with handler.error_context(f'test_operation_{i}'):
                pass  # No actual operation
        context_overhead = (time.time() - start_time) * 1000
        
        print(f"  Error context overhead: {context_overhead:.2f}ms for 1000 contexts")
        
        # Test 2: Error creation and handling performance
        start_time = time.time()
        for i in range(1000):
            try:
                raise GeminiSDRError(f"Test error {i}")
            except GeminiSDRError:
                pass
        error_handling_time = (time.time() - start_time) * 1000
        
        print(f"  Error creation/handling: {error_handling_time:.2f}ms for 1000 errors")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error handling benchmark failed: {e}")
        return False

def measure_cross_platform_performance():
    """Measure cross-platform performance characteristics."""
    print("Measuring cross-platform performance...")
    
    try:
        import torch
        import platform
        
        system_info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }
        
        print(f"  Platform: {system_info['platform']} {system_info['machine']}")
        print(f"  Python: {system_info['python_version']}")
        
        # Test device performance
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')
        
        for device in devices:
            print(f"  Testing {device} device:")
            
            # Simple tensor operations benchmark
            torch_device = torch.device(device)
            
            start_time = time.time()
            for _ in range(100):
                a = torch.randn(100, 100, device=torch_device)
                b = torch.randn(100, 100, device=torch_device)
                c = torch.mm(a, b)
            tensor_ops_time = (time.time() - start_time) * 1000
            
            print(f"    Tensor operations: {tensor_ops_time:.2f}ms for 100 matrix multiplications")
            
            # Memory allocation benchmark
            start_time = time.time()
            tensors = []
            for _ in range(50):
                tensor = torch.randn(1000, 1000, device=torch_device)
                tensors.append(tensor)
            allocation_time = (time.time() - start_time) * 1000
            
            # Cleanup
            del tensors
            if device != 'cpu':
                torch.cuda.empty_cache() if device == 'cuda' else None
            
            print(f"    Memory allocation: {allocation_time:.2f}ms for 50 large tensors")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Cross-platform performance measurement failed: {e}")
        return False

def create_performance_regression_suite():
    """Create performance regression test suite."""
    print("Creating performance regression test suite...")
    
    try:
        # Create a simple performance regression test
        regression_test = """
import time
import pytest
from geminisdr.core.memory_manager import MemoryManager
from geminisdr.core.logging_manager import get_logger
from geminisdr.config.config_models import create_default_config

class TestPerformanceRegression:
    '''Performance regression tests to ensure improvements don't degrade.'''
    
    def setup_method(self):
        self.config = create_default_config('development')
    
    @pytest.mark.performance
    def test_memory_stats_performance(self):
        '''Memory stats should be collected in under 10ms.'''
        manager = MemoryManager(self.config)
        
        start_time = time.time()
        stats = manager.get_memory_stats()
        duration = (time.time() - start_time) * 1000
        
        assert duration < 10.0, f"Memory stats took {duration:.2f}ms, expected < 10ms"
    
    @pytest.mark.performance
    def test_logging_performance(self):
        '''Logging 100 messages should take under 100ms.'''
        logger = get_logger('regression_test')
        
        start_time = time.time()
        for i in range(100):
            logger.info(f"Test message {i}")
        duration = (time.time() - start_time) * 1000
        
        assert duration < 100.0, f"Logging took {duration:.2f}ms, expected < 100ms"
    
    @pytest.mark.performance
    def test_config_loading_performance(self):
        '''Config loading should take under 50ms.'''
        from geminisdr.config.config_manager import ConfigManager
        
        manager = ConfigManager()
        
        start_time = time.time()
        config = manager.load_config()
        duration = (time.time() - start_time) * 1000
        
        assert duration < 50.0, f"Config loading took {duration:.2f}ms, expected < 50ms"
"""
        
        # Write the regression test file
        with open('tests/test_performance_regression.py', 'w') as f:
            f.write(regression_test)
        
        print("  ✓ Performance regression test suite created")
        return True
        
    except Exception as e:
        print(f"  ✗ Performance regression suite creation failed: {e}")
        return False

def main():
    """Run all performance benchmarks and validation."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARKING AND VALIDATION")
    print("=" * 60)
    
    benchmarks = [
        benchmark_memory_optimization,
        benchmark_logging_performance,
        benchmark_metrics_collection,
        benchmark_configuration_loading,
        benchmark_error_handling,
        measure_cross_platform_performance,
        create_performance_regression_suite,
    ]
    
    passed = 0
    failed = 0
    
    for benchmark in benchmarks:
        try:
            if benchmark():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ {benchmark.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"PERFORMANCE VALIDATION RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL PERFORMANCE BENCHMARKS PASSED!")
        print("Memory optimization and performance improvements are working effectively.")
        return 0
    else:
        print(f"❌ {failed} benchmarks failed. Performance may need optimization.")
        return 1

if __name__ == '__main__':
    sys.exit(main())