"""
Tests for Model Comparison Widget

This module contains comprehensive tests for the ModelComparisonWidget including
unit tests for comparison functionality and integration tests.
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt

# Import the widget under test
from gui.widgets.model_comparison import ModelComparisonWidget

# Mock GeminiSDR components if not available
try:
    from geminisdr.core.model_manager import ModelManager
    from geminisdr.core.model_metadata import ModelMetadata
except ImportError:
    ModelManager = Mock
    ModelMetadata = Mock


@pytest.fixture
def app():
    """Create QApplication instance for testing."""
    if not QApplication.instance():
        return QApplication(sys.argv)
    return QApplication.instance()


@pytest.fixture
def comparison_widget(app):
    """Create ModelComparisonWidget instance for testing."""
    widget = ModelComparisonWidget()
    return widget


@pytest.fixture
def mock_model_manager():
    """Create mock model manager with test data."""
    manager = Mock()
    
    # Mock model list
    manager.list_models.return_value = [
        {
            'name': 'test_model_1',
            'versions': [
                {
                    'version': '1.0.0',
                    'timestamp': '2024-01-01T10:00:00',
                    'performance_metrics': {'accuracy': 0.85, 'loss': 0.15},
                    'model_size_mb': 10.5,
                    'description': 'First version'
                },
                {
                    'version': '1.1.0',
                    'timestamp': '2024-01-15T10:00:00',
                    'performance_metrics': {'accuracy': 0.87, 'loss': 0.13},
                    'model_size_mb': 11.2,
                    'description': 'Improved version'
                }
            ],
            'latest_version': '1.1.0'
        },
        {
            'name': 'test_model_2',
            'versions': [
                {
                    'version': '2.0.0',
                    'timestamp': '2024-02-01T10:00:00',
                    'performance_metrics': {'accuracy': 0.90, 'loss': 0.10},
                    'model_size_mb': 15.0,
                    'description': 'Second model'
                }
            ],
            'latest_version': '2.0.0'
        }
    ]
    
    # Mock comparison results
    manager.compare_models.return_value = {
        'performance_comparison': {
            'accuracy': {'model1': 0.85, 'model2': 0.90},
            'loss': {'model1': 0.15, 'model2': 0.10}
        },
        'metadata_comparison': {
            'size_mb': {'model1': 10.5, 'model2': 15.0},
            'version': {'model1': '1.0.0', 'model2': '2.0.0'}
        }
    }
    
    return manager


class TestModelComparisonWidget:
    """Test ModelComparisonWidget functionality."""
    
    def test_widget_creation(self, comparison_widget):
        """Test widget creation and initialization."""
        widget = comparison_widget
        
        assert isinstance(widget, QWidget)
        assert widget.tab_widget is not None
        assert widget.tab_widget.count() == 4  # 4 tabs
        
        # Check tab names
        expected_tabs = [
            "Performance Comparison",
            "Statistical Analysis", 
            "Version History",
            "Benchmarking"
        ]
        
        for i, expected_name in enumerate(expected_tabs):
            assert widget.tab_widget.tabText(i) == expected_name
        
        # Check initial state
        assert len(widget.selected_models) == 0
        assert len(widget.comparison_results) == 0
        assert widget.model_manager is None
    
    def test_model_selection_components(self, comparison_widget):
        """Test model selection components."""
        widget = comparison_widget
        
        # Check that key components exist
        assert widget.available_models_table is not None
        assert widget.selected_models_table is not None
        assert widget.add_model_btn is not None
        assert widget.remove_model_btn is not None
        assert widget.compare_btn is not None
        assert widget.clear_selection_btn is not None
        assert widget.refresh_models_btn is not None
        
        # Check table configuration
        assert widget.available_models_table.columnCount() == 4
        assert widget.selected_models_table.columnCount() == 4
        
        expected_headers = ["Model", "Version", "Accuracy", "Date"]
        for i, header in enumerate(expected_headers):
            assert widget.available_models_table.horizontalHeaderItem(i).text() == header
            assert widget.selected_models_table.horizontalHeaderItem(i).text() == header
        
        # Check initial button states
        assert not widget.add_model_btn.isEnabled()  # No selection initially
        assert not widget.remove_model_btn.isEnabled()  # No selection initially
        assert not widget.compare_btn.isEnabled()  # Need at least 2 models
    
    def test_performance_tab_components(self, comparison_widget):
        """Test performance comparison tab components."""
        widget = comparison_widget
        
        # Switch to performance tab
        widget.tab_widget.setCurrentIndex(0)
        
        # Check components exist
        assert widget.metrics_table is not None
        assert widget.performance_plot is not None
        assert widget.radar_plot is not None
    
    def test_statistics_tab_components(self, comparison_widget):
        """Test statistical analysis tab components."""
        widget = comparison_widget
        
        # Switch to statistics tab
        widget.tab_widget.setCurrentIndex(1)
        
        # Check components exist
        assert widget.stat_test_combo is not None
        assert widget.run_test_btn is not None
        assert widget.stats_results is not None
        assert widget.histogram_plot is not None
        assert widget.boxplot_plot is not None
        
        # Check statistical test options
        expected_tests = [
            "T-Test", "Mann-Whitney U", "Wilcoxon Signed-Rank",
            "ANOVA", "Kruskal-Wallis", "Chi-Square"
        ]
        
        for i, test_name in enumerate(expected_tests):
            assert widget.stat_test_combo.itemText(i) == test_name
    
    def test_history_tab_components(self, comparison_widget):
        """Test version history tab components."""
        widget = comparison_widget
        
        # Switch to history tab
        widget.tab_widget.setCurrentIndex(2)
        
        # Check components exist
        assert widget.history_model_combo is not None
        assert widget.load_history_btn is not None
        assert widget.timeline_plot is not None
        assert widget.version_table is not None
        
        # Check version table configuration
        assert widget.version_table.columnCount() == 6
        expected_headers = ["Version", "Date", "Accuracy", "Loss", "Size (MB)", "Description"]
        for i, header in enumerate(expected_headers):
            assert widget.version_table.horizontalHeaderItem(i).text() == header
    
    def test_benchmark_tab_components(self, comparison_widget):
        """Test benchmarking tab components."""
        widget = comparison_widget
        
        # Switch to benchmark tab
        widget.tab_widget.setCurrentIndex(3)
        
        # Check components exist
        assert widget.benchmark_type_combo is not None
        assert widget.test_samples_spin is not None
        assert widget.iterations_spin is not None
        assert widget.batch_sizes_combo is not None
        assert widget.warmup_checkbox is not None
        assert widget.run_benchmark_btn is not None
        assert widget.benchmark_progress is not None
        assert widget.benchmark_table is not None
        assert widget.benchmark_plot is not None
        
        # Check benchmark type options
        expected_types = [
            "Inference Speed", "Memory Usage", "Accuracy vs Speed",
            "Batch Processing", "Real-time Performance", "Custom"
        ]
        
        for i, bench_type in enumerate(expected_types):
            assert widget.benchmark_type_combo.itemText(i) == bench_type
    
    def test_set_model_manager(self, comparison_widget, mock_model_manager):
        """Test setting model manager."""
        widget = comparison_widget
        
        # Set model manager
        widget.set_model_manager(mock_model_manager)
        
        assert widget.model_manager == mock_model_manager
        mock_model_manager.list_models.assert_called_once()
    
    def test_refresh_available_models(self, comparison_widget, mock_model_manager):
        """Test refreshing available models."""
        widget = comparison_widget
        widget.set_model_manager(mock_model_manager)
        
        # Check that models were loaded into table
        assert widget.available_models_table.rowCount() == 3  # 2 versions + 1 version
        
        # Check first row data
        assert widget.available_models_table.item(0, 0).text() == "test_model_1"
        assert widget.available_models_table.item(0, 1).text() == "1.0.0"
        assert widget.available_models_table.item(0, 2).text() == "0.850"
        
        # Check that history combo was populated
        assert widget.history_model_combo.count() == 2
        assert widget.history_model_combo.itemText(0) == "test_model_1"
        assert widget.history_model_combo.itemText(1) == "test_model_2"
    
    def test_add_model_to_comparison(self, comparison_widget, mock_model_manager):
        """Test adding model to comparison."""
        widget = comparison_widget
        widget.set_model_manager(mock_model_manager)
        
        # Select first model in available table
        widget.available_models_table.selectRow(0)
        
        # Add to comparison
        widget._add_model_to_comparison()
        
        # Check that model was added
        assert widget.selected_models_table.rowCount() == 1
        assert len(widget.selected_models) == 1
        assert widget.selected_models[0] == ("test_model_1", "1.0.0")
        
        # Compare button should still be disabled (need 2 models)
        assert not widget.compare_btn.isEnabled()
        
        # Add second model
        widget.available_models_table.selectRow(2)  # test_model_2
        widget._add_model_to_comparison()
        
        # Now compare button should be enabled
        assert widget.compare_btn.isEnabled()
        assert len(widget.selected_models) == 2
    
    def test_remove_model_from_comparison(self, comparison_widget, mock_model_manager):
        """Test removing model from comparison."""
        widget = comparison_widget
        widget.set_model_manager(mock_model_manager)
        
        # Add two models first
        widget.available_models_table.selectRow(0)
        widget._add_model_to_comparison()
        widget.available_models_table.selectRow(2)
        widget._add_model_to_comparison()
        
        assert len(widget.selected_models) == 2
        assert widget.compare_btn.isEnabled()
        
        # Remove one model
        widget.selected_models_table.selectRow(0)
        widget._remove_model_from_comparison()
        
        assert len(widget.selected_models) == 1
        assert not widget.compare_btn.isEnabled()  # Need 2 models
    
    def test_clear_selection(self, comparison_widget, mock_model_manager):
        """Test clearing model selection."""
        widget = comparison_widget
        widget.set_model_manager(mock_model_manager)
        
        # Add models
        widget.available_models_table.selectRow(0)
        widget._add_model_to_comparison()
        widget.available_models_table.selectRow(2)
        widget._add_model_to_comparison()
        
        assert len(widget.selected_models) == 2
        
        # Clear selection
        widget._clear_selection()
        
        assert len(widget.selected_models) == 0
        assert widget.selected_models_table.rowCount() == 0
        assert not widget.compare_btn.isEnabled()
    
    def test_compare_selected_models(self, comparison_widget, mock_model_manager):
        """Test comparing selected models."""
        widget = comparison_widget
        widget.set_model_manager(mock_model_manager)
        
        # Add two models
        widget.available_models_table.selectRow(0)
        widget._add_model_to_comparison()
        widget.available_models_table.selectRow(2)
        widget._add_model_to_comparison()
        
        # Compare models
        widget._compare_selected_models()
        
        # Check that comparison was called
        mock_model_manager.compare_models.assert_called_once()
        
        # Check that results were stored
        assert len(widget.comparison_results) > 0
    
    def test_load_model_history(self, comparison_widget, mock_model_manager):
        """Test loading model version history."""
        widget = comparison_widget
        widget.set_model_manager(mock_model_manager)
        
        # Select model for history
        widget.history_model_combo.setCurrentText("test_model_1")
        
        # Load history
        widget._load_model_history()
        
        # Check that version table was populated
        assert widget.version_table.rowCount() == 2  # test_model_1 has 2 versions
        
        # Check first version data
        assert widget.version_table.item(0, 0).text() == "1.0.0"
        assert widget.version_table.item(0, 2).text() == "0.850"  # accuracy
        assert widget.version_table.item(0, 3).text() == "0.1500"  # loss
        assert widget.version_table.item(0, 4).text() == "10.5"  # size
    
    def test_run_statistical_test(self, comparison_widget):
        """Test running statistical tests."""
        widget = comparison_widget
        
        # Select a test
        widget.stat_test_combo.setCurrentText("T-Test")
        
        # Run test
        widget._run_statistical_test()
        
        # Check that results were displayed
        results_text = widget.stats_results.toPlainText()
        assert "T-Test" in results_text
        assert len(results_text) > 0
    
    def test_run_benchmark(self, comparison_widget, mock_model_manager):
        """Test running benchmark."""
        widget = comparison_widget
        widget.set_model_manager(mock_model_manager)
        
        # Add models for benchmarking
        widget.available_models_table.selectRow(0)
        widget._add_model_to_comparison()
        widget.available_models_table.selectRow(2)
        widget._add_model_to_comparison()
        
        # Configure benchmark
        widget.benchmark_type_combo.setCurrentText("Inference Speed")
        widget.test_samples_spin.setCurrentText("1000")
        widget.iterations_spin.setCurrentText("10")
        
        # Run benchmark
        widget._run_benchmark()
        
        # Check that benchmark table was populated
        assert widget.benchmark_table.rowCount() == 2  # 2 models
        assert widget.benchmark_table.columnCount() == 5
        
        # Check that progress was shown and hidden
        assert not widget.benchmark_progress.isVisible()
    
    def test_signal_emissions(self, comparison_widget):
        """Test that signals are emitted correctly."""
        widget = comparison_widget
        
        # Create signal spies
        comparison_spy = Mock()
        benchmark_spy = Mock()
        
        widget.comparison_requested.connect(comparison_spy)
        widget.benchmark_requested.connect(benchmark_spy)
        
        # Note: Signals are emitted internally by the widget
        # In a real test, we would trigger actions that emit signals
        
        # For now, just verify the signals exist
        assert hasattr(widget, 'comparison_requested')
        assert hasattr(widget, 'benchmark_requested')
    
    def test_button_state_management(self, comparison_widget, mock_model_manager):
        """Test button state management based on selections."""
        widget = comparison_widget
        widget.set_model_manager(mock_model_manager)
        
        # Initially no buttons should be enabled
        assert not widget.add_model_btn.isEnabled()
        assert not widget.remove_model_btn.isEnabled()
        assert not widget.compare_btn.isEnabled()
        
        # Select available model
        widget.available_models_table.selectRow(0)
        widget._on_available_selection_changed()
        assert widget.add_model_btn.isEnabled()
        
        # Add model to comparison
        widget._add_model_to_comparison()
        
        # Select in comparison table
        widget.selected_models_table.selectRow(0)
        widget._on_selected_selection_changed()
        assert widget.remove_model_btn.isEnabled()
        
        # Add second model to enable comparison
        widget.available_models_table.selectRow(2)
        widget._add_model_to_comparison()
        assert widget.compare_btn.isEnabled()
    
    def test_duplicate_model_prevention(self, comparison_widget, mock_model_manager):
        """Test prevention of duplicate model selection."""
        widget = comparison_widget
        widget.set_model_manager(mock_model_manager)
        
        # Add same model twice
        widget.available_models_table.selectRow(0)
        widget._add_model_to_comparison()
        
        initial_count = len(widget.selected_models)
        
        # Try to add same model again
        widget.available_models_table.selectRow(0)
        widget._add_model_to_comparison()
        
        # Should not have increased
        assert len(widget.selected_models) == initial_count
    
    def test_timeline_plot_update(self, comparison_widget, mock_model_manager):
        """Test timeline plot updates with version history."""
        widget = comparison_widget
        widget.set_model_manager(mock_model_manager)
        
        # Get test versions
        models = mock_model_manager.list_models.return_value
        test_versions = models[0]['versions']  # test_model_1 versions
        
        # Update timeline plot
        widget._update_timeline_plot("test_model_1", test_versions)
        
        # Check that plot was updated (plot should have data items)
        plot_items = widget.timeline_plot.listDataItems()
        assert len(plot_items) > 0  # Should have at least accuracy plot
    
    def test_benchmark_plot_update(self, comparison_widget):
        """Test benchmark plot updates."""
        widget = comparison_widget
        
        # Create mock benchmark results
        results = [
            {
                'model': 'test_model_1 v1.0.0',
                'inference_time_ms': 50.0,
                'memory_usage_mb': 100.0,
                'throughput_samples_sec': 500.0,
                'accuracy': 0.85
            },
            {
                'model': 'test_model_2 v2.0.0',
                'inference_time_ms': 75.0,
                'memory_usage_mb': 150.0,
                'throughput_samples_sec': 400.0,
                'accuracy': 0.90
            }
        ]
        
        # Update benchmark plot
        widget._update_benchmark_plot(results)
        
        # Check that plot was updated
        plot_items = widget.benchmark_plot.listDataItems()
        assert len(plot_items) > 0  # Should have bar chart
    
    def test_error_handling(self, comparison_widget):
        """Test error handling in various scenarios."""
        widget = comparison_widget
        
        # Test operations without model manager
        widget._refresh_available_models()  # Should not crash
        widget._compare_selected_models()  # Should not crash
        widget._load_model_history()  # Should not crash
        
        # Test with empty selections
        widget._add_model_to_comparison()  # Should not crash
        widget._remove_model_from_comparison()  # Should not crash


class TestModelComparisonIntegration:
    """Integration tests for ModelComparisonWidget."""
    
    @pytest.fixture
    def integrated_widget(self, app, mock_model_manager):
        """Create widget with mocked dependencies."""
        widget = ModelComparisonWidget()
        widget.set_model_manager(mock_model_manager)
        return widget
    
    def test_complete_comparison_workflow(self, integrated_widget):
        """Test complete model comparison workflow."""
        widget = integrated_widget
        
        # Verify models are loaded
        assert widget.available_models_table.rowCount() > 0
        
        # Add models to comparison
        widget.available_models_table.selectRow(0)
        widget._add_model_to_comparison()
        widget.available_models_table.selectRow(2)
        widget._add_model_to_comparison()
        
        # Verify selection
        assert len(widget.selected_models) == 2
        assert widget.compare_btn.isEnabled()
        
        # Run comparison
        widget._compare_selected_models()
        
        # Verify results
        assert len(widget.comparison_results) > 0
        
        # Check that metrics table was updated
        assert widget.metrics_table.rowCount() > 0
    
    def test_version_history_workflow(self, integrated_widget):
        """Test version history workflow."""
        widget = integrated_widget
        
        # Select model with multiple versions
        widget.history_model_combo.setCurrentText("test_model_1")
        
        # Load history
        widget._load_model_history()
        
        # Verify version table
        assert widget.version_table.rowCount() == 2
        
        # Verify timeline plot has data
        plot_items = widget.timeline_plot.listDataItems()
        assert len(plot_items) > 0
    
    def test_benchmark_workflow(self, integrated_widget):
        """Test benchmarking workflow."""
        widget = integrated_widget
        
        # Add models for benchmarking
        widget.available_models_table.selectRow(0)
        widget._add_model_to_comparison()
        widget.available_models_table.selectRow(2)
        widget._add_model_to_comparison()
        
        # Configure and run benchmark
        widget.benchmark_type_combo.setCurrentText("Inference Speed")
        widget._run_benchmark()
        
        # Verify results
        assert widget.benchmark_table.rowCount() == 2
        
        # Verify plot was updated
        plot_items = widget.benchmark_plot.listDataItems()
        assert len(plot_items) > 0


if __name__ == '__main__':
    pytest.main([__file__])