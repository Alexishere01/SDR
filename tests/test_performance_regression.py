
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
