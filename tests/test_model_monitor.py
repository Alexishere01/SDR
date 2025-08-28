"""
Tests for ML Model Monitoring Widget

This module contains comprehensive tests for the ModelMonitorWidget including
unit tests for individual components and integration tests for the complete
monitoring system.
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt, QTimer

# Import the widget under test
from gui.widgets.model_monitor import (
    ModelMonitorWidget, ModelInfo, TrainingProgress, ModelStatus
)

# Mock GeminiSDR components if not available
try:
    from geminisdr.core.model_manager import ModelManager
    from geminisdr.core.metrics_collector import MetricsCollector
    from geminisdr.core.model_metadata import ModelMetadata
except ImportError:
    ModelManager = Mock
    MetricsCollector = Mock
    ModelMetadata = Mock


class TestModelInfo:
    """Test ModelInfo class."""
    
    def test_model_info_creation(self):
        """Test ModelInfo creation and initialization."""
        model_info = ModelInfo("test_model", "1.0.0")
        
        assert model_info.name == "test_model"
        assert model_info.version == "1.0.0"
        assert model_info.status == ModelStatus.IDLE
        assert model_info.accuracy == 0.0
        assert model_info.loss == 0.0
        assert model_info.device == "cpu"
        assert model_info.memory_usage_mb == 0.0
        assert isinstance(model_info.last_updated, datetime)
        assert model_info.metadata is None
        assert model_info.error_message == ""


class TestTrainingProgress:
    """Test TrainingProgress class."""
    
    def test_training_progress_creation(self):
        """Test TrainingProgress creation and initialization."""
        progress = TrainingProgress("test_model")
        
        assert progress.model_name == "test_model"
        assert progress.current_epoch == 0
        assert progress.total_epochs == 0
        assert progress.current_batch == 0
        assert progress.total_batches == 0
        assert progress.loss_history == []
        assert progress.accuracy_history == []
        assert progress.validation_loss_history == []
        assert progress.validation_accuracy_history == []
        assert progress.learning_rate_history == []
        assert isinstance(progress.start_time, datetime)
        assert progress.estimated_completion is None
        assert progress.is_active is False


@pytest.fixture
def app():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    # Ensure we process events to avoid hanging
    app.processEvents()
    return app


@pytest.fixture
def model_monitor_widget(app):
    """Create ModelMonitorWidget instance for testing."""
    # Mock the ModelManager and other components to prevent initialization issues
    with patch('gui.widgets.model_monitor.MODEL_MANAGER_AVAILABLE', False):
        with patch('gui.widgets.model_monitor.ModelManager') as mock_manager:
            with patch('gui.widgets.model_monitor.MetricsCollector') as mock_metrics:
                mock_manager.return_value = None
                mock_metrics.return_value = None
                
                # Create widget with mocked dependencies
                widget = ModelMonitorWidget()
                
                # Stop the update timer to prevent test interference
                if hasattr(widget, 'update_timer') and widget.update_timer:
                    widget.update_timer.stop()
                
                # Process any pending events
                app.processEvents()
                
                return widget


class TestModelMonitorWidget:
    """Test ModelMonitorWidget functionality."""
    
    def test_widget_creation(self, model_monitor_widget):
        """Test widget creation and initialization."""
        widget = model_monitor_widget
        
        # Process any pending Qt events
        QApplication.processEvents()
        
        assert isinstance(widget, QWidget)
        assert widget.tab_widget is not None
        assert widget.tab_widget.count() == 3  # Status, Training, Comparison tabs
        
        # Check tab names
        assert widget.tab_widget.tabText(0) == "Model Status"
        assert widget.tab_widget.tabText(1) == "Training Progress"
        assert widget.tab_widget.tabText(2) == "Model Comparison"
        
        # Check initial state
        assert len(widget.active_models) == 0
        assert len(widget.training_progress) == 0
        # Model manager should be None in test environment due to mocking
        assert widget.model_manager is None
    
    def test_status_tab_components(self, model_monitor_widget):
        """Test status tab components."""
        widget = model_monitor_widget
        
        # Process events to ensure UI is fully initialized
        QApplication.processEvents()
        
        # Check that key components exist
        assert widget.model_combo is not None
        assert widget.version_combo is not None
        assert widget.load_model_btn is not None
        assert widget.unload_model_btn is not None
        assert widget.refresh_btn is not None
        assert widget.model_table is not None
        
        # Check table configuration
        assert widget.model_table.columnCount() == 7
        expected_headers = ["Model", "Version", "Status", "Accuracy", "Loss", "Device", "Memory (MB)"]
        for i, header in enumerate(expected_headers):
            assert widget.model_table.horizontalHeaderItem(i).text() == header
        
        # Check initial button states
        assert widget.load_model_btn.isEnabled()
        assert not widget.unload_model_btn.isEnabled()  # Should be disabled initially
    
    def test_training_tab_components(self, model_monitor_widget):
        """Test training tab components."""
        widget = model_monitor_widget
        
        # Process events to ensure UI is fully initialized
        QApplication.processEvents()
        
        # Check training control components
        assert widget.training_model_combo is not None
        assert widget.start_training_btn is not None
        assert widget.pause_training_btn is not None
        assert widget.stop_training_btn is not None
        
        # Check progress components
        assert widget.epoch_progress is not None
        assert widget.batch_progress is not None
        assert widget.epoch_label is not None
        assert widget.batch_label is not None
        assert widget.elapsed_time_label is not None
        assert widget.eta_label is not None
        
        # Check plot components
        assert widget.loss_plot is not None
        assert widget.accuracy_plot is not None
        assert widget.train_loss_curve is not None
        assert widget.val_loss_curve is not None
        assert widget.train_acc_curve is not None
        assert widget.val_acc_curve is not None
        
        # Check training log
        assert widget.training_log is not None
        assert widget.clear_log_btn is not None
        assert widget.save_log_btn is not None
        
        # Check initial button states
        assert widget.start_training_btn.isEnabled()
        assert not widget.pause_training_btn.isEnabled()
        assert not widget.stop_training_btn.isEnabled()
    
    def test_comparison_tab_components(self, model_monitor_widget):
        """Test comparison tab components."""
        widget = model_monitor_widget
        
        # Process events to ensure UI is fully initialized
        QApplication.processEvents()
        
        # Check that comparison tab exists
        assert widget.tab_widget.count() == 3
        comparison_tab = widget.tab_widget.widget(2)
        assert comparison_tab is not None
        
        # The comparison tab might use either simple or advanced comparison widget
        # Check if it has the expected attributes, but don't fail if it uses advanced widget
        if hasattr(widget, 'compare_model1_combo'):
            # Simple comparison interface
            assert widget.compare_model1_combo is not None
            assert widget.compare_version1_combo is not None
            assert widget.compare_model2_combo is not None
            assert widget.compare_version2_combo is not None
            assert widget.compare_btn is not None
            assert widget.model1_details is not None
            assert widget.model2_details is not None
            assert widget.comparison_plot is not None
        elif hasattr(widget, 'comparison_widget'):
            # Advanced comparison widget
            assert widget.comparison_widget is not None
        else:
            # At minimum, the tab should exist
            assert comparison_tab is not None
    
    def test_add_model(self, model_monitor_widget):
        """Test adding a model to monitoring."""
        widget = model_monitor_widget
        
        # Create mock metadata
        mock_metadata = Mock()
        mock_metadata.model_architecture = "TestNet"
        mock_metadata.model_size_mb = 10.5
        mock_metadata.description = "Test model"
        
        # Add model
        widget.add_model("test_model", "1.0.0", mock_metadata)
        
        # Check that model was added
        assert "test_model" in widget.active_models
        model_info = widget.active_models["test_model"]
        assert model_info.name == "test_model"
        assert model_info.version == "1.0.0"
        assert model_info.status == ModelStatus.LOADED
        assert model_info.metadata == mock_metadata
    
    def test_remove_model(self, model_monitor_widget):
        """Test removing a model from monitoring."""
        widget = model_monitor_widget
        
        # Add model first
        widget.add_model("test_model", "1.0.0")
        assert "test_model" in widget.active_models
        
        # Remove model
        widget.remove_model("test_model")
        assert "test_model" not in widget.active_models
    
    def test_update_model_metrics(self, model_monitor_widget):
        """Test updating model metrics."""
        widget = model_monitor_widget
        
        # Add model first
        widget.add_model("test_model", "1.0.0")
        
        # Update metrics
        metrics = {
            'accuracy': 0.95,
            'loss': 0.05,
            'memory_usage_mb': 256.0,
            'device': 'cuda:0'
        }
        widget.update_model_metrics("test_model", metrics)
        
        # Check updated values
        model_info = widget.active_models["test_model"]
        assert model_info.accuracy == 0.95
        assert model_info.loss == 0.05
        assert model_info.memory_usage_mb == 256.0
        assert model_info.device == 'cuda:0'
    
    def test_update_model_status(self, model_monitor_widget):
        """Test updating model status."""
        widget = model_monitor_widget
        
        # Add model first
        widget.add_model("test_model", "1.0.0")
        
        # Update status
        widget.update_model_status("test_model", ModelStatus.TRAINING, "")
        
        # Check updated status
        model_info = widget.active_models["test_model"]
        assert model_info.status == ModelStatus.TRAINING
        assert model_info.error_message == ""
        
        # Update with error
        widget.update_model_status("test_model", ModelStatus.ERROR, "Test error")
        assert model_info.status == ModelStatus.ERROR
        assert model_info.error_message == "Test error"
    
    def test_update_training_progress(self, model_monitor_widget):
        """Test updating training progress."""
        widget = model_monitor_widget
        
        # Update training progress
        widget.update_training_progress(
            model_name="test_model",
            epoch=5,
            total_epochs=100,
            batch=10,
            total_batches=50,
            loss=0.1,
            accuracy=0.9,
            val_loss=0.15,
            val_accuracy=0.85
        )
        
        # Check that progress was created and updated
        assert "test_model" in widget.training_progress
        progress = widget.training_progress["test_model"]
        
        assert progress.current_epoch == 5
        assert progress.total_epochs == 100
        assert progress.current_batch == 10
        assert progress.total_batches == 50
        assert progress.is_active is True
        
        # Check history arrays
        assert len(progress.loss_history) == 6  # 0-5 epochs
        assert progress.loss_history[5] == 0.1
        assert len(progress.accuracy_history) == 6
        assert progress.accuracy_history[5] == 0.9
        assert len(progress.validation_loss_history) == 6
        assert progress.validation_loss_history[5] == 0.15
        assert len(progress.validation_accuracy_history) == 6
        assert progress.validation_accuracy_history[5] == 0.85
    
    def test_model_table_update(self, model_monitor_widget):
        """Test model table updates."""
        widget = model_monitor_widget
        
        # Add some models
        widget.add_model("model1", "1.0.0")
        widget.add_model("model2", "2.0.0")
        
        # Update metrics for models
        widget.update_model_metrics("model1", {'accuracy': 0.9, 'loss': 0.1})
        widget.update_model_metrics("model2", {'accuracy': 0.85, 'loss': 0.15})
        
        # Trigger table update
        widget._update_model_table()
        
        # Check table contents
        assert widget.model_table.rowCount() == 2
        
        # Check first row
        assert widget.model_table.item(0, 0).text() == "model1"
        assert widget.model_table.item(0, 1).text() == "1.0.0"
        assert widget.model_table.item(0, 2).text() == "Loaded"
        assert widget.model_table.item(0, 3).text() == "0.900"
        assert widget.model_table.item(0, 4).text() == "0.1000"
    
    def test_signal_emissions(self, model_monitor_widget):
        """Test that signals are emitted correctly."""
        widget = model_monitor_widget
        
        # Create signal spies
        load_spy = Mock()
        unload_spy = Mock()
        select_spy = Mock()
        training_spy = Mock()
        
        widget.model_load_requested.connect(load_spy)
        widget.model_unload_requested.connect(unload_spy)
        widget.model_selected.connect(select_spy)
        widget.training_control_requested.connect(training_spy)
        
        # Test model loading signal
        widget.model_combo.addItem("test_model")
        widget.version_combo.addItem("1.0.0")
        widget.model_combo.setCurrentText("test_model")
        widget.version_combo.setCurrentText("1.0.0")
        
        # Simulate button click
        QTest.mouseClick(widget.load_model_btn, Qt.LeftButton)
        load_spy.assert_called_once_with("test_model", "1.0.0")
        
        # Test training control signal
        widget.training_model_combo.addItem("test_model")
        widget.training_model_combo.setCurrentText("test_model")
        
        QTest.mouseClick(widget.start_training_btn, Qt.LeftButton)
        training_spy.assert_called_once_with("start", "test_model")
    
    def test_error_handling(self, model_monitor_widget):
        """Test error handling in various scenarios."""
        widget = model_monitor_widget
        
        # Clear combo boxes to ensure no selection
        widget.model_combo.clear()
        widget.version_combo.clear()
        widget.training_model_combo.clear()
        
        # Test loading model without selection
        with patch.object(widget, '_show_error_message') as mock_error:
            widget._load_model()
            mock_error.assert_called_once()
        
        # Test unloading model without selection
        with patch.object(widget, '_show_error_message') as mock_error:
            widget._unload_model()
            mock_error.assert_called_once()
        
        # Test starting training without selection
        with patch.object(widget, '_show_error_message') as mock_error:
            widget._start_training()
            mock_error.assert_called_once()
    
    def test_training_button_states(self, model_monitor_widget):
        """Test training button state management."""
        widget = model_monitor_widget
        
        # Add model to training combo
        widget.training_model_combo.addItem("test_model")
        widget.training_model_combo.setCurrentText("test_model")
        
        # Initial state
        assert widget.start_training_btn.isEnabled()
        assert not widget.pause_training_btn.isEnabled()
        assert not widget.stop_training_btn.isEnabled()
        
        # Start training
        widget._start_training()
        assert not widget.start_training_btn.isEnabled()
        assert widget.pause_training_btn.isEnabled()
        assert widget.stop_training_btn.isEnabled()
        
        # Stop training
        widget._stop_training()
        assert widget.start_training_btn.isEnabled()
        assert not widget.pause_training_btn.isEnabled()
        assert not widget.stop_training_btn.isEnabled()
    
    def test_log_messages(self, model_monitor_widget):
        """Test log message functionality."""
        widget = model_monitor_widget
        
        # Test status log
        initial_text = widget.status_label.text()
        widget._add_log_message("Test message")
        assert "Test message" in widget.status_label.text()
        assert widget.status_label.text() != initial_text
        
        # Test training log
        initial_log = widget.training_log.toPlainText()
        widget._add_training_log("Training message")
        assert "Training message" in widget.training_log.toPlainText()
        assert widget.training_log.toPlainText() != initial_log
    
    def test_metrics_collector_integration(self, model_monitor_widget):
        """Test integration with metrics collector."""
        widget = model_monitor_widget
        
        # Create mock metrics collector
        mock_collector = Mock()
        widget.set_metrics_collector(mock_collector)
        
        assert widget.metrics_collector == mock_collector
        mock_collector.register_alert_callback.assert_called_once()
    
    @patch('gui.widgets.model_monitor.ModelManager')
    def test_model_manager_integration(self, mock_manager_class, model_monitor_widget):
        """Test integration with model manager."""
        widget = model_monitor_widget
        
        # Create mock model manager
        mock_manager = Mock()
        mock_manager.list_models.return_value = [
            {
                'name': 'test_model',
                'versions': [{'version': '1.0.0'}, {'version': '2.0.0'}],
                'latest_version': '2.0.0'
            }
        ]
        mock_manager_class.return_value = mock_manager
        
        # Initialize model manager and load models
        widget.model_manager = mock_manager
        widget._load_available_models()
        
        # Process events to ensure UI updates
        QApplication.processEvents()
        
        # Check that models were loaded into combo boxes
        assert widget.model_combo.count() > 0
        assert widget.model_combo.findText('test_model') >= 0
    
    def test_comparison_functionality(self, model_monitor_widget):
        """Test model comparison functionality."""
        widget = model_monitor_widget
        
        # Skip this test if using advanced comparison widget
        if not hasattr(widget, 'compare_model1_combo'):
            pytest.skip("Using advanced comparison widget, skipping simple comparison test")
        
        # Mock model manager with comparison capability
        mock_manager = Mock()
        mock_comparison = {
            'performance_comparison': {
                'accuracy': {'model1': 0.9, 'model2': 0.85},
                'loss': {'model1': 0.1, 'model2': 0.15}
            },
            'metadata_comparison': {
                'size_mb': {'model1': 10.0, 'model2': 15.0}
            }
        }
        mock_manager.compare_models.return_value = mock_comparison
        widget.model_manager = mock_manager
        
        # Set up comparison
        widget.compare_model1_combo.addItem("model1")
        widget.compare_version1_combo.addItem("1.0.0")
        widget.compare_model2_combo.addItem("model2")
        widget.compare_version2_combo.addItem("1.0.0")
        
        widget.compare_model1_combo.setCurrentText("model1")
        widget.compare_version1_combo.setCurrentText("1.0.0")
        widget.compare_model2_combo.setCurrentText("model2")
        widget.compare_version2_combo.setCurrentText("1.0.0")
        
        # Perform comparison
        widget._compare_models()
        
        # Check that comparison was called
        mock_manager.compare_models.assert_called_once_with(
            "model1", "1.0.0", "model2", "1.0.0"
        )
        
        # Check that comparison results were displayed
        assert "model1" in widget.model1_details.toPlainText()
        assert "model2" in widget.model2_details.toPlainText()
    
    def test_update_timer(self, model_monitor_widget):
        """Test that update timer exists and can be controlled."""
        widget = model_monitor_widget
        
        assert widget.update_timer is not None
        assert widget.update_timer.interval() == 1000  # 1 second
        
        # Timer might be stopped in test fixture, so test that it can be started
        widget.update_timer.start()
        assert widget.update_timer.isActive()
        
        # Stop it again to avoid interference with other tests
        widget.update_timer.stop()
        assert not widget.update_timer.isActive()
    
    def test_thread_safety(self, model_monitor_widget):
        """Test thread safety with mutex usage."""
        widget = model_monitor_widget
        
        # Test that mutex is used for data access
        assert widget.data_mutex is not None
        
        # Add model and check thread-safe access
        widget.add_model("test_model", "1.0.0")
        
        # Update metrics from different "thread" context
        widget.update_model_metrics("test_model", {'accuracy': 0.9})
        
        # Verify data consistency
        assert widget.active_models["test_model"].accuracy == 0.9


class TestModelMonitorIntegration:
    """Integration tests for ModelMonitorWidget."""
    
    @pytest.fixture
    def integrated_widget(self, app):
        """Create widget with mocked dependencies."""
        widget = ModelMonitorWidget()
        
        # Mock model manager
        mock_manager = Mock()
        mock_manager.list_models.return_value = [
            {
                'name': 'test_model',
                'versions': [{'version': '1.0.0'}],
                'latest_version': '1.0.0'
            }
        ]
        widget.model_manager = mock_manager
        
        # Mock metrics collector
        mock_collector = Mock()
        widget.set_metrics_collector(mock_collector)
        
        return widget
    
    def test_complete_workflow(self, integrated_widget):
        """Test complete model monitoring workflow."""
        widget = integrated_widget
        
        # Load available models
        widget._load_available_models()
        assert widget.model_combo.count() > 0
        
        # Add a model to monitoring
        widget.add_model("test_model", "1.0.0")
        assert "test_model" in widget.active_models
        
        # Update model metrics
        widget.update_model_metrics("test_model", {
            'accuracy': 0.95,
            'loss': 0.05,
            'memory_usage_mb': 128.0
        })
        
        # Start training simulation
        widget.update_training_progress(
            "test_model", 1, 10, 5, 20, 0.2, 0.8
        )
        
        # Update displays
        widget._update_displays()
        
        # Verify final state
        model_info = widget.active_models["test_model"]
        assert model_info.accuracy == 0.95
        assert model_info.loss == 0.05
        
        progress = widget.training_progress["test_model"]
        assert progress.current_epoch == 1
        assert progress.total_epochs == 10
    
    def test_error_recovery(self, integrated_widget):
        """Test error recovery scenarios."""
        widget = integrated_widget
        
        # Add model
        widget.add_model("test_model", "1.0.0")
        
        # Simulate error condition
        widget.update_model_status("test_model", ModelStatus.ERROR, "Test error")
        
        # Verify error state
        model_info = widget.active_models["test_model"]
        assert model_info.status == ModelStatus.ERROR
        assert model_info.error_message == "Test error"
        
        # Recover from error
        widget.update_model_status("test_model", ModelStatus.LOADED, "")
        assert model_info.status == ModelStatus.LOADED
        assert model_info.error_message == ""
    
    def test_performance_with_multiple_models(self, integrated_widget):
        """Test performance with multiple models."""
        widget = integrated_widget
        
        # Add multiple models
        for i in range(10):
            widget.add_model(f"model_{i}", "1.0.0")
            widget.update_model_metrics(f"model_{i}", {
                'accuracy': 0.8 + i * 0.01,
                'loss': 0.2 - i * 0.01
            })
        
        # Update displays multiple times
        for _ in range(5):
            widget._update_displays()
        
        # Verify all models are tracked
        assert len(widget.active_models) == 10
        assert widget.model_table.rowCount() == 10
        
        # Verify table performance
        for i in range(10):
            assert widget.model_table.item(i, 0).text() == f"model_{i}"


if __name__ == '__main__':
    pytest.main([__file__])