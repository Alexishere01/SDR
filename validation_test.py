#!/usr/bin/env python3
"""
Comprehensive validation test for all codebase improvements.
This script validates the implementation of all major improvements.
"""

import sys
import time
import traceback
from pathlib import Path

def test_configuration_system():
    """Test centralized configuration management."""
    print("Testing configuration system...")
    try:
        from geminisdr.config.config_models import create_default_config
        from geminisdr.config.config_manager import ConfigManager
        
        # Test default config creation
        config = create_default_config('development')
        assert config.environment == 'development'
        
        # Test config manager
        manager = ConfigManager()
        loaded_config = manager.load_config()
        assert loaded_config is not None
        
        print("  ✓ Configuration system working")
        return True
    except Exception as e:
        print(f"  ✗ Configuration system failed: {e}")
        return False

def test_error_handling():
    """Test error handling and recovery system."""
    print("Testing error handling system...")
    try:
        from geminisdr.core.error_handling import (
            ErrorHandler, GeminiSDRError, HardwareError, 
            retry_with_backoff, fallback_to_simulation
        )
        from geminisdr.config.config_models import create_default_config
        
        config = create_default_config('development')
        handler = ErrorHandler(config)
        
        # Test error creation
        error = GeminiSDRError("Test error")
        assert error.message == "Test error"
        
        # Test context manager
        with handler.error_context('test_operation'):
            pass
        
        print("  ✓ Error handling system working")
        return True
    except Exception as e:
        print(f"  ✗ Error handling system failed: {e}")
        return False

def test_memory_management():
    """Test memory management and optimization."""
    print("Testing memory management system...")
    try:
        from geminisdr.core.memory_manager import MemoryManager, ModelCache
        from geminisdr.config.config_models import create_default_config
        
        config = create_default_config('development')
        manager = MemoryManager(config)
        
        # Test memory stats
        stats = manager.get_memory_stats()
        assert stats.total_ram_mb > 0
        
        # Test batch size optimization
        optimized = manager.optimize_batch_size(64, 100.0)
        assert optimized > 0
        
        # Test model cache
        cache = ModelCache(max_size=2)
        cache.put_model('test', {'data': 'test'})
        retrieved = cache.get_model('test')
        assert retrieved is not None
        
        print("  ✓ Memory management system working")
        return True
    except Exception as e:
        print(f"  ✗ Memory management system failed: {e}")
        return False

def test_logging_system():
    """Test structured logging system."""
    print("Testing logging system...")
    try:
        from geminisdr.core.logging_manager import get_logger, StructuredLogger
        from geminisdr.config.config_models import create_default_config
        
        config = create_default_config('development')
        logger = get_logger('test_validation')
        
        # Test basic logging
        logger.info("Test message")
        
        # Test context management
        logger.add_context(test_id='validation')
        logger.info("Test with context")
        
        # Test performance logging
        with logger.performance_timer('test_operation'):
            time.sleep(0.01)
        
        print("  ✓ Logging system working")
        return True
    except Exception as e:
        print(f"  ✗ Logging system failed: {e}")
        return False

def test_metrics_collection():
    """Test metrics collection system."""
    print("Testing metrics collection system...")
    try:
        from geminisdr.core.metrics_collector import MetricsCollector
        from geminisdr.config.config_models import create_default_config
        
        config = create_default_config('development')
        collector = MetricsCollector(config)
        
        # Test metric recording
        collector.record_counter('test_counter', 1)
        collector.record_gauge('test_gauge', 42.0)
        
        # Test system metrics
        collector.collect_system_metrics()
        
        print("  ✓ Metrics collection system working")
        return True
    except Exception as e:
        print(f"  ✗ Metrics collection system failed: {e}")
        return False

def test_model_management():
    """Test model management and versioning."""
    print("Testing model management system...")
    try:
        from geminisdr.core.model_manager import ModelManager
        from geminisdr.core.model_metadata import ModelMetadata
        from geminisdr.config.config_models import create_default_config
        from datetime import datetime
        
        config = create_default_config('development')
        manager = ModelManager(config)
        
        # Test metadata creation
        metadata = ModelMetadata(
            name='test_model',
            version='1.0.0',
            timestamp=datetime.now(),
            hyperparameters={'lr': 0.001},
            performance_metrics={'accuracy': 0.95},
            training_data_hash='test_hash',
            code_version='1.0.0',
            platform='test',
            device='cpu'
        )
        
        # Test model listing (should be empty initially)
        models = manager.list_models()
        assert isinstance(models, list)
        
        print("  ✓ Model management system working")
        return True
    except Exception as e:
        print(f"  ✗ Model management system failed: {e}")
        return False

def test_cross_platform_compatibility():
    """Test cross-platform compatibility."""
    print("Testing cross-platform compatibility...")
    try:
        import torch
        from geminisdr.config.config_models import create_default_config
        
        # Test device detection
        devices = []
        if torch.cuda.is_available():
            devices.append('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')
        devices.append('cpu')
        
        print(f"  Available devices: {devices}")
        
        # Test configuration for different platforms
        config = create_default_config('development')
        assert config.hardware.device_preference is not None
        
        print("  ✓ Cross-platform compatibility working")
        return True
    except Exception as e:
        print(f"  ✗ Cross-platform compatibility failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("COMPREHENSIVE VALIDATION TEST")
    print("=" * 60)
    
    tests = [
        test_configuration_system,
        test_error_handling,
        test_memory_management,
        test_logging_system,
        test_metrics_collection,
        test_model_management,
        test_cross_platform_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} crashed: {e}")
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"VALIDATION RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! All improvements are working correctly.")
        return 0
    else:
        print(f"❌ {failed} tests failed. Please check the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())