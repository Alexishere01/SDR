"""
Model Comparison and Analysis Widget

This module provides comprehensive model comparison capabilities including
performance benchmarking, statistical analysis, and model versioning visualization.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget,
    QTableWidgetItem, QLabel, QPushButton, QComboBox, QTextEdit,
    QSplitter, QFrame, QTabWidget, QScrollArea, QGridLayout,
    QHeaderView, QAbstractItemView, QProgressBar, QCheckBox
)
from PySide6.QtCore import Signal, Slot, Qt, QTimer, QThread, QObject
from PySide6.QtGui import QColor, QFont
import pyqtgraph as pg
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path
import statistics

# Import GeminiSDR components
try:
    from geminisdr.core.model_manager import ModelManager
    from geminisdr.core.model_metadata import ModelMetadata, ModelComparator
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    logging.warning("Model management components not available")


class ModelComparisonWidget(QWidget):
    """
    Advanced model comparison widget for side-by-side model evaluation.
    
    Provides performance benchmarking, statistical analysis, and model
    versioning with history tracking visualization.
    """
    
    # Signals
    comparison_requested = Signal(list)  # List of (model_name, version) tuples
    benchmark_requested = Signal(str, str)  # model_name, benchmark_type
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Model manager reference
        self.model_manager: Optional[ModelManager] = None
        
        # Comparison data storage
        self.comparison_results = {}
        self.benchmark_results = {}
        self.selected_models = []  # List of (name, version) tuples
        
        # Initialize UI
        self._setup_ui()
        self._setup_connections()
        
        self.logger.info("ModelComparisonWidget initialized")
    
    def _setup_ui(self) -> None:
        """Setup the model comparison UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Model selection panel
        selection_group = self._create_selection_panel()
        layout.addWidget(selection_group)
        
        # Create tab widget for different comparison views
        self.tab_widget = QTabWidget()
        
        # Performance comparison tab
        perf_tab = self._create_performance_tab()
        self.tab_widget.addTab(perf_tab, "Performance Comparison")
        
        # Statistical analysis tab
        stats_tab = self._create_statistics_tab()
        self.tab_widget.addTab(stats_tab, "Statistical Analysis")
        
        # Version history tab
        history_tab = self._create_history_tab()
        self.tab_widget.addTab(history_tab, "Version History")
        
        # Benchmarking tab
        benchmark_tab = self._create_benchmark_tab()
        self.tab_widget.addTab(benchmark_tab, "Benchmarking")
        
        layout.addWidget(self.tab_widget)
    
    def _create_selection_panel(self) -> QGroupBox:
        """Create model selection panel."""
        group = QGroupBox("Model Selection")
        layout = QVBoxLayout(group)
        
        # Available models list
        models_layout = QHBoxLayout()
        
        # Available models
        available_group = QGroupBox("Available Models")
        available_layout = QVBoxLayout(available_group)
        
        self.available_models_table = QTableWidget()
        self.available_models_table.setColumnCount(4)
        self.available_models_table.setHorizontalHeaderLabels([
            "Model", "Version", "Accuracy", "Date"
        ])
        self.available_models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.available_models_table.setAlternatingRowColors(True)
        
        # Configure table
        header = self.available_models_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        available_layout.addWidget(self.available_models_table)
        
        # Add/Remove buttons
        button_layout = QVBoxLayout()
        self.add_model_btn = QPushButton("Add →")
        self.add_model_btn.clicked.connect(self._add_model_to_comparison)
        self.add_model_btn.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.add_model_btn)
        
        self.remove_model_btn = QPushButton("← Remove")
        self.remove_model_btn.clicked.connect(self._remove_model_from_comparison)
        self.remove_model_btn.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.remove_model_btn)
        
        button_layout.addStretch()
        
        # Selected models for comparison
        selected_group = QGroupBox("Selected for Comparison")
        selected_layout = QVBoxLayout(selected_group)
        
        self.selected_models_table = QTableWidget()
        self.selected_models_table.setColumnCount(4)
        self.selected_models_table.setHorizontalHeaderLabels([
            "Model", "Version", "Accuracy", "Date"
        ])
        self.selected_models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.selected_models_table.setAlternatingRowColors(True)
        
        # Configure table
        header = self.selected_models_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        selected_layout.addWidget(self.selected_models_table)
        
        # Layout models selection
        models_layout.addWidget(available_group)
        models_layout.addLayout(button_layout)
        models_layout.addWidget(selected_group)
        
        layout.addLayout(models_layout)
        
        # Comparison controls
        controls_layout = QHBoxLayout()
        
        self.compare_btn = QPushButton("Compare Selected Models")
        self.compare_btn.clicked.connect(self._compare_selected_models)
        self.compare_btn.setEnabled(False)
        controls_layout.addWidget(self.compare_btn)
        
        self.clear_selection_btn = QPushButton("Clear Selection")
        self.clear_selection_btn.clicked.connect(self._clear_selection)
        controls_layout.addWidget(self.clear_selection_btn)
        
        self.refresh_models_btn = QPushButton("Refresh Models")
        self.refresh_models_btn.clicked.connect(self._refresh_available_models)
        controls_layout.addWidget(self.refresh_models_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        return group
    
    def _create_performance_tab(self) -> QWidget:
        """Create performance comparison tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Performance metrics comparison
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # Metrics comparison table
        self.metrics_table = QTableWidget()
        self.metrics_table.setAlternatingRowColors(True)
        metrics_layout.addWidget(self.metrics_table)
        
        layout.addWidget(metrics_group)
        
        # Performance visualization
        viz_group = QGroupBox("Performance Visualization")
        viz_layout = QHBoxLayout(viz_group)
        
        # Bar chart for metrics comparison
        self.performance_plot = pg.PlotWidget(title="Performance Comparison")
        self.performance_plot.setLabel('left', 'Score')
        self.performance_plot.setLabel('bottom', 'Metric')
        self.performance_plot.showGrid(True, True, alpha=0.3)
        viz_layout.addWidget(self.performance_plot)
        
        # Radar chart for multi-dimensional comparison
        self.radar_plot = pg.PlotWidget(title="Multi-Dimensional Comparison")
        self.radar_plot.setLabel('left', 'Normalized Score')
        self.radar_plot.setLabel('bottom', 'Dimension')
        viz_layout.addWidget(self.radar_plot)
        
        layout.addWidget(viz_group)
        
        return widget
    
    def _create_statistics_tab(self) -> QWidget:
        """Create statistical analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Statistical tests
        tests_group = QGroupBox("Statistical Analysis")
        tests_layout = QVBoxLayout(tests_group)
        
        # Test selection
        test_controls = QHBoxLayout()
        test_controls.addWidget(QLabel("Statistical Test:"))
        
        self.stat_test_combo = QComboBox()
        self.stat_test_combo.addItems([
            "T-Test", "Mann-Whitney U", "Wilcoxon Signed-Rank", 
            "ANOVA", "Kruskal-Wallis", "Chi-Square"
        ])
        test_controls.addWidget(self.stat_test_combo)
        
        self.run_test_btn = QPushButton("Run Test")
        self.run_test_btn.clicked.connect(self._run_statistical_test)
        test_controls.addWidget(self.run_test_btn)
        
        test_controls.addStretch()
        tests_layout.addLayout(test_controls)
        
        # Results display
        self.stats_results = QTextEdit()
        self.stats_results.setReadOnly(True)
        self.stats_results.setFont(QFont("Consolas", 9))
        tests_layout.addWidget(self.stats_results)
        
        layout.addWidget(tests_group)
        
        # Distribution plots
        dist_group = QGroupBox("Performance Distributions")
        dist_layout = QHBoxLayout(dist_group)
        
        # Histogram
        self.histogram_plot = pg.PlotWidget(title="Performance Distribution")
        self.histogram_plot.setLabel('left', 'Frequency')
        self.histogram_plot.setLabel('bottom', 'Performance Score')
        dist_layout.addWidget(self.histogram_plot)
        
        # Box plot
        self.boxplot_plot = pg.PlotWidget(title="Performance Box Plot")
        self.boxplot_plot.setLabel('left', 'Performance Score')
        self.boxplot_plot.setLabel('bottom', 'Model')
        dist_layout.addWidget(self.boxplot_plot)
        
        layout.addWidget(dist_group)
        
        return widget
    
    def _create_history_tab(self) -> QWidget:
        """Create version history tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Version timeline
        timeline_group = QGroupBox("Version Timeline")
        timeline_layout = QVBoxLayout(timeline_group)
        
        # Model selection for history
        history_controls = QHBoxLayout()
        history_controls.addWidget(QLabel("Model:"))
        
        self.history_model_combo = QComboBox()
        history_controls.addWidget(self.history_model_combo)
        
        self.load_history_btn = QPushButton("Load History")
        self.load_history_btn.clicked.connect(self._load_model_history)
        history_controls.addWidget(self.load_history_btn)
        
        history_controls.addStretch()
        timeline_layout.addLayout(history_controls)
        
        # Timeline plot
        self.timeline_plot = pg.PlotWidget(title="Model Performance Over Time")
        self.timeline_plot.setLabel('left', 'Performance Score')
        self.timeline_plot.setLabel('bottom', 'Version/Time')
        self.timeline_plot.showGrid(True, True, alpha=0.3)
        self.timeline_plot.addLegend()
        timeline_layout.addWidget(self.timeline_plot)
        
        layout.addWidget(timeline_group)
        
        # Version details
        details_group = QGroupBox("Version Details")
        details_layout = QVBoxLayout(details_group)
        
        # Version comparison table
        self.version_table = QTableWidget()
        self.version_table.setColumnCount(6)
        self.version_table.setHorizontalHeaderLabels([
            "Version", "Date", "Accuracy", "Loss", "Size (MB)", "Description"
        ])
        self.version_table.setAlternatingRowColors(True)
        
        header = self.version_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        details_layout.addWidget(self.version_table)
        layout.addWidget(details_group)
        
        return widget
    
    def _create_benchmark_tab(self) -> QWidget:
        """Create benchmarking tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Benchmark controls
        controls_group = QGroupBox("Benchmark Configuration")
        controls_layout = QVBoxLayout(controls_group)
        
        # Benchmark type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Benchmark Type:"))
        
        self.benchmark_type_combo = QComboBox()
        self.benchmark_type_combo.addItems([
            "Inference Speed", "Memory Usage", "Accuracy vs Speed",
            "Batch Processing", "Real-time Performance", "Custom"
        ])
        type_layout.addWidget(self.benchmark_type_combo)
        
        type_layout.addStretch()
        controls_layout.addLayout(type_layout)
        
        # Benchmark parameters
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Test Samples:"), 0, 0)
        self.test_samples_spin = QComboBox()
        self.test_samples_spin.addItems(["100", "500", "1000", "5000", "10000"])
        self.test_samples_spin.setCurrentText("1000")
        params_layout.addWidget(self.test_samples_spin, 0, 1)
        
        params_layout.addWidget(QLabel("Iterations:"), 0, 2)
        self.iterations_spin = QComboBox()
        self.iterations_spin.addItems(["1", "5", "10", "20", "50"])
        self.iterations_spin.setCurrentText("10")
        params_layout.addWidget(self.iterations_spin, 0, 3)
        
        params_layout.addWidget(QLabel("Batch Sizes:"), 1, 0)
        self.batch_sizes_combo = QComboBox()
        self.batch_sizes_combo.addItems(["1,8,32,64", "1,16,32", "Custom"])
        params_layout.addWidget(self.batch_sizes_combo, 1, 1)
        
        self.warmup_checkbox = QCheckBox("Include Warmup")
        self.warmup_checkbox.setChecked(True)
        params_layout.addWidget(self.warmup_checkbox, 1, 2)
        
        controls_layout.addLayout(params_layout)
        
        # Benchmark execution
        exec_layout = QHBoxLayout()
        self.run_benchmark_btn = QPushButton("Run Benchmark")
        self.run_benchmark_btn.clicked.connect(self._run_benchmark)
        exec_layout.addWidget(self.run_benchmark_btn)
        
        self.benchmark_progress = QProgressBar()
        self.benchmark_progress.setVisible(False)
        exec_layout.addWidget(self.benchmark_progress)
        
        exec_layout.addStretch()
        controls_layout.addLayout(exec_layout)
        
        layout.addWidget(controls_group)
        
        # Benchmark results
        results_group = QGroupBox("Benchmark Results")
        results_layout = QVBoxLayout(results_group)
        
        # Results table
        self.benchmark_table = QTableWidget()
        self.benchmark_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.benchmark_table)
        
        # Results visualization
        self.benchmark_plot = pg.PlotWidget(title="Benchmark Results")
        self.benchmark_plot.setLabel('left', 'Performance Metric')
        self.benchmark_plot.setLabel('bottom', 'Model')
        self.benchmark_plot.showGrid(True, True, alpha=0.3)
        results_layout.addWidget(self.benchmark_plot)
        
        layout.addWidget(results_group)
        
        return widget
    
    def _setup_connections(self) -> None:
        """Setup signal-slot connections."""
        # Table selection changes
        self.available_models_table.itemSelectionChanged.connect(
            self._on_available_selection_changed
        )
        self.selected_models_table.itemSelectionChanged.connect(
            self._on_selected_selection_changed
        )
    
    def set_model_manager(self, model_manager: ModelManager) -> None:
        """Set model manager reference."""
        self.model_manager = model_manager
        self._refresh_available_models()
    
    @Slot()
    def _refresh_available_models(self) -> None:
        """Refresh available models list."""
        if not self.model_manager:
            return
        
        try:
            models = self.model_manager.list_models()
            
            # Clear and populate available models table
            self.available_models_table.setRowCount(0)
            
            for model_info in models:
                model_name = model_info['name']
                for version_info in model_info['versions']:
                    row = self.available_models_table.rowCount()
                    self.available_models_table.insertRow(row)
                    
                    # Model name
                    self.available_models_table.setItem(
                        row, 0, QTableWidgetItem(model_name)
                    )
                    
                    # Version
                    self.available_models_table.setItem(
                        row, 1, QTableWidgetItem(version_info['version'])
                    )
                    
                    # Accuracy
                    accuracy = version_info.get('performance_metrics', {}).get('accuracy', 'N/A')
                    if isinstance(accuracy, (int, float)):
                        accuracy_text = f"{accuracy:.3f}"
                    else:
                        accuracy_text = str(accuracy)
                    self.available_models_table.setItem(
                        row, 2, QTableWidgetItem(accuracy_text)
                    )
                    
                    # Date
                    timestamp = version_info.get('timestamp', 'Unknown')
                    if isinstance(timestamp, str):
                        try:
                            date_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            date_text = date_obj.strftime('%Y-%m-%d')
                        except:
                            date_text = timestamp
                    else:
                        date_text = str(timestamp)
                    self.available_models_table.setItem(
                        row, 3, QTableWidgetItem(date_text)
                    )
            
            # Update history model combo
            self.history_model_combo.clear()
            for model_info in models:
                self.history_model_combo.addItem(model_info['name'])
            
            self.logger.info(f"Refreshed {len(models)} available models")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh available models: {e}")
    
    @Slot()
    def _on_available_selection_changed(self) -> None:
        """Handle available models selection change."""
        self.add_model_btn.setEnabled(
            len(self.available_models_table.selectedItems()) > 0
        )
    
    @Slot()
    def _on_selected_selection_changed(self) -> None:
        """Handle selected models selection change."""
        self.remove_model_btn.setEnabled(
            len(self.selected_models_table.selectedItems()) > 0
        )
    
    @Slot()
    def _add_model_to_comparison(self) -> None:
        """Add selected model to comparison."""
        current_row = self.available_models_table.currentRow()
        if current_row < 0:
            return
        
        model_name = self.available_models_table.item(current_row, 0).text()
        version = self.available_models_table.item(current_row, 1).text()
        
        # Check if already selected
        for i in range(self.selected_models_table.rowCount()):
            if (self.selected_models_table.item(i, 0).text() == model_name and
                self.selected_models_table.item(i, 1).text() == version):
                return  # Already selected
        
        # Add to selected models table
        row = self.selected_models_table.rowCount()
        self.selected_models_table.insertRow(row)
        
        for col in range(4):
            item_text = self.available_models_table.item(current_row, col).text()
            self.selected_models_table.setItem(row, col, QTableWidgetItem(item_text))
        
        # Update selected models list
        self.selected_models.append((model_name, version))
        
        # Enable compare button if we have at least 2 models
        self.compare_btn.setEnabled(len(self.selected_models) >= 2)
    
    @Slot()
    def _remove_model_from_comparison(self) -> None:
        """Remove selected model from comparison."""
        current_row = self.selected_models_table.currentRow()
        if current_row < 0:
            return
        
        model_name = self.selected_models_table.item(current_row, 0).text()
        version = self.selected_models_table.item(current_row, 1).text()
        
        # Remove from table
        self.selected_models_table.removeRow(current_row)
        
        # Remove from selected models list
        self.selected_models = [
            (name, ver) for name, ver in self.selected_models
            if not (name == model_name and ver == version)
        ]
        
        # Update compare button state
        self.compare_btn.setEnabled(len(self.selected_models) >= 2)
    
    @Slot()
    def _clear_selection(self) -> None:
        """Clear all selected models."""
        self.selected_models_table.setRowCount(0)
        self.selected_models.clear()
        self.compare_btn.setEnabled(False)
    
    @Slot()
    def _compare_selected_models(self) -> None:
        """Compare selected models."""
        if len(self.selected_models) < 2:
            return
        
        if not self.model_manager:
            self.logger.error("Model manager not available")
            return
        
        try:
            # Perform pairwise comparisons
            comparisons = {}
            
            for i, (model1_name, model1_version) in enumerate(self.selected_models):
                for j, (model2_name, model2_version) in enumerate(self.selected_models[i+1:], i+1):
                    comparison_key = f"{model1_name}_v{model1_version}_vs_{model2_name}_v{model2_version}"
                    
                    comparison = self.model_manager.compare_models(
                        model1_name, model1_version,
                        model2_name, model2_version
                    )
                    comparisons[comparison_key] = comparison
            
            self.comparison_results = comparisons
            self._update_comparison_displays()
            
            self.logger.info(f"Compared {len(self.selected_models)} models")
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
    
    def _update_comparison_displays(self) -> None:
        """Update comparison result displays."""
        if not self.comparison_results:
            return
        
        # Update performance metrics table
        self._update_metrics_table()
        
        # Update performance plots
        self._update_performance_plots()
        
        # Update statistical analysis
        self._update_statistical_analysis()
    
    def _update_metrics_table(self) -> None:
        """Update performance metrics comparison table."""
        if not self.comparison_results:
            return
        
        # Collect all metrics from all comparisons
        all_metrics = set()
        model_metrics = {}
        
        for comparison in self.comparison_results.values():
            perf_comp = comparison.get('performance_comparison', {})
            for metric, values in perf_comp.items():
                all_metrics.add(metric)
                if isinstance(values, dict):
                    if 'model1' in values:
                        model_key = f"Model1"  # This would need model name in real implementation
                        if model_key not in model_metrics:
                            model_metrics[model_key] = {}
                        model_metrics[model_key][metric] = values['model1']
                    if 'model2' in values:
                        model_key = f"Model2"  # This would need model name in real implementation
                        if model_key not in model_metrics:
                            model_metrics[model_key] = {}
                        model_metrics[model_key][metric] = values['model2']
        
        # Setup table
        metrics_list = sorted(all_metrics)
        models_list = sorted(model_metrics.keys())
        
        self.metrics_table.setRowCount(len(metrics_list))
        self.metrics_table.setColumnCount(len(models_list) + 1)
        
        # Set headers
        headers = ["Metric"] + models_list
        self.metrics_table.setHorizontalHeaderLabels(headers)
        
        # Populate table
        for row, metric in enumerate(metrics_list):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(metric))
            
            for col, model in enumerate(models_list, 1):
                value = model_metrics.get(model, {}).get(metric, 'N/A')
                if isinstance(value, (int, float)):
                    value_text = f"{value:.4f}"
                else:
                    value_text = str(value)
                self.metrics_table.setItem(row, col, QTableWidgetItem(value_text))
    
    def _update_performance_plots(self) -> None:
        """Update performance visualization plots."""
        # This would implement the actual plotting logic
        # For now, just clear the plots
        self.performance_plot.clear()
        self.radar_plot.clear()
    
    def _update_statistical_analysis(self) -> None:
        """Update statistical analysis display."""
        if not self.comparison_results:
            return
        
        # Collect performance data for statistical analysis
        analysis_text = "Statistical Analysis Results:\n\n"
        
        for comparison_key, comparison in self.comparison_results.items():
            analysis_text += f"Comparison: {comparison_key}\n"
            
            perf_comp = comparison.get('performance_comparison', {})
            for metric, values in perf_comp.items():
                if isinstance(values, dict) and 'model1' in values and 'model2' in values:
                    val1, val2 = values['model1'], values['model2']
                    diff = abs(val1 - val2)
                    better = "Model1" if val1 > val2 else "Model2"
                    analysis_text += f"  {metric}: {better} is better by {diff:.4f}\n"
            
            analysis_text += "\n"
        
        self.stats_results.setText(analysis_text)
    
    @Slot()
    def _run_statistical_test(self) -> None:
        """Run selected statistical test."""
        test_type = self.stat_test_combo.currentText()
        
        # This would implement actual statistical testing
        result_text = f"Running {test_type}...\n\n"
        result_text += "Statistical test results would appear here.\n"
        result_text += "This requires actual performance data from multiple runs.\n"
        
        self.stats_results.setText(result_text)
    
    @Slot()
    def _load_model_history(self) -> None:
        """Load version history for selected model."""
        model_name = self.history_model_combo.currentText()
        if not model_name or not self.model_manager:
            return
        
        try:
            models = self.model_manager.list_models()
            model_info = next((m for m in models if m['name'] == model_name), None)
            
            if not model_info:
                return
            
            # Update version table
            versions = model_info['versions']
            self.version_table.setRowCount(len(versions))
            
            for row, version_info in enumerate(versions):
                # Version
                self.version_table.setItem(
                    row, 0, QTableWidgetItem(version_info['version'])
                )
                
                # Date
                timestamp = version_info.get('timestamp', 'Unknown')
                if isinstance(timestamp, str):
                    try:
                        date_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        date_text = date_obj.strftime('%Y-%m-%d %H:%M')
                    except:
                        date_text = timestamp
                else:
                    date_text = str(timestamp)
                self.version_table.setItem(row, 1, QTableWidgetItem(date_text))
                
                # Performance metrics
                perf_metrics = version_info.get('performance_metrics', {})
                accuracy = perf_metrics.get('accuracy', 'N/A')
                if isinstance(accuracy, (int, float)):
                    accuracy_text = f"{accuracy:.3f}"
                else:
                    accuracy_text = str(accuracy)
                self.version_table.setItem(row, 2, QTableWidgetItem(accuracy_text))
                
                # Loss (if available)
                loss = perf_metrics.get('loss', 'N/A')
                if isinstance(loss, (int, float)):
                    loss_text = f"{loss:.4f}"
                else:
                    loss_text = str(loss)
                self.version_table.setItem(row, 3, QTableWidgetItem(loss_text))
                
                # Size
                size_mb = version_info.get('model_size_mb', 'N/A')
                if isinstance(size_mb, (int, float)):
                    size_text = f"{size_mb:.1f}"
                else:
                    size_text = str(size_mb)
                self.version_table.setItem(row, 4, QTableWidgetItem(size_text))
                
                # Description
                description = version_info.get('description', '')
                self.version_table.setItem(row, 5, QTableWidgetItem(description))
            
            # Update timeline plot
            self._update_timeline_plot(model_name, versions)
            
        except Exception as e:
            self.logger.error(f"Failed to load model history: {e}")
    
    def _update_timeline_plot(self, model_name: str, versions: List[Dict]) -> None:
        """Update timeline plot with version history."""
        self.timeline_plot.clear()
        
        if not versions:
            return
        
        # Extract data for plotting
        x_data = []  # Version indices
        accuracy_data = []
        loss_data = []
        
        for i, version_info in enumerate(versions):
            x_data.append(i)
            
            perf_metrics = version_info.get('performance_metrics', {})
            accuracy = perf_metrics.get('accuracy')
            loss = perf_metrics.get('loss')
            
            accuracy_data.append(accuracy if isinstance(accuracy, (int, float)) else None)
            loss_data.append(loss if isinstance(loss, (int, float)) else None)
        
        # Plot accuracy
        valid_acc = [(x, y) for x, y in zip(x_data, accuracy_data) if y is not None]
        if valid_acc:
            x_acc, y_acc = zip(*valid_acc)
            self.timeline_plot.plot(
                x_acc, y_acc, pen=pg.mkPen(color='#4ecdc4', width=2),
                symbol='o', symbolBrush='#4ecdc4', name='Accuracy'
            )
        
        # Plot loss (on secondary axis if needed)
        valid_loss = [(x, y) for x, y in zip(x_data, loss_data) if y is not None]
        if valid_loss:
            x_loss, y_loss = zip(*valid_loss)
            # Normalize loss to similar scale as accuracy for visualization
            if y_loss:
                max_loss = max(y_loss)
                normalized_loss = [l / max_loss for l in y_loss]
                self.timeline_plot.plot(
                    x_loss, normalized_loss, pen=pg.mkPen(color='#ff6b6b', width=2),
                    symbol='s', symbolBrush='#ff6b6b', name='Loss (normalized)'
                )
        
        # Set x-axis labels to version names
        version_names = [v['version'] for v in versions]
        ax = self.timeline_plot.getAxis('bottom')
        ax.setTicks([[(i, name) for i, name in enumerate(version_names)]])
    
    @Slot()
    def _run_benchmark(self) -> None:
        """Run benchmark on selected models."""
        if not self.selected_models:
            return
        
        benchmark_type = self.benchmark_type_combo.currentText()
        test_samples = int(self.test_samples_spin.currentText())
        iterations = int(self.iterations_spin.currentText())
        
        # Show progress
        self.benchmark_progress.setVisible(True)
        self.benchmark_progress.setRange(0, len(self.selected_models) * iterations)
        self.benchmark_progress.setValue(0)
        
        # Simulate benchmark execution
        self.logger.info(f"Running {benchmark_type} benchmark on {len(self.selected_models)} models")
        
        # This would implement actual benchmarking
        # For now, just simulate with timer
        self._simulate_benchmark()
    
    def _simulate_benchmark(self) -> None:
        """Simulate benchmark execution."""
        # This would be replaced with actual benchmarking logic
        import random
        
        # Generate mock benchmark results
        results = []
        for model_name, version in self.selected_models:
            result = {
                'model': f"{model_name} v{version}",
                'inference_time_ms': random.uniform(10, 100),
                'memory_usage_mb': random.uniform(50, 500),
                'throughput_samples_sec': random.uniform(100, 1000),
                'accuracy': random.uniform(0.8, 0.95)
            }
            results.append(result)
        
        # Update benchmark table
        self.benchmark_table.setRowCount(len(results))
        self.benchmark_table.setColumnCount(5)
        self.benchmark_table.setHorizontalHeaderLabels([
            "Model", "Inference Time (ms)", "Memory (MB)", "Throughput (samples/s)", "Accuracy"
        ])
        
        for row, result in enumerate(results):
            self.benchmark_table.setItem(row, 0, QTableWidgetItem(result['model']))
            self.benchmark_table.setItem(row, 1, QTableWidgetItem(f"{result['inference_time_ms']:.2f}"))
            self.benchmark_table.setItem(row, 2, QTableWidgetItem(f"{result['memory_usage_mb']:.1f}"))
            self.benchmark_table.setItem(row, 3, QTableWidgetItem(f"{result['throughput_samples_sec']:.0f}"))
            self.benchmark_table.setItem(row, 4, QTableWidgetItem(f"{result['accuracy']:.3f}"))
        
        # Update benchmark plot
        self._update_benchmark_plot(results)
        
        # Hide progress
        self.benchmark_progress.setVisible(False)
        
        self.logger.info("Benchmark simulation completed")
    
    def _update_benchmark_plot(self, results: List[Dict]) -> None:
        """Update benchmark visualization."""
        self.benchmark_plot.clear()
        
        if not results:
            return
        
        # Create bar chart for inference time comparison
        models = [r['model'] for r in results]
        inference_times = [r['inference_time_ms'] for r in results]
        
        x = np.arange(len(models))
        bars = pg.BarGraphItem(
            x=x, height=inference_times, width=0.6,
            brush='#4ecdc4', name='Inference Time (ms)'
        )
        self.benchmark_plot.addItem(bars)
        
        # Set x-axis labels
        ax = self.benchmark_plot.getAxis('bottom')
        ax.setTicks([[(i, model) for i, model in enumerate(models)]])