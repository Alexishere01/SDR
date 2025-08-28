"""
ML Model Monitoring and Control Widget

This module provides comprehensive ML model monitoring capabilities including
real-time performance metrics, training progress visualization, and model
comparison tools.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget, 
    QTableWidgetItem, QProgressBar, QLabel, QPushButton, QTextEdit,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QSplitter, QFrame, QScrollArea, QGridLayout, QMessageBox,
    QFileDialog, QHeaderView, QAbstractItemView, QLineEdit
)
from PySide6.QtCore import (
    Signal, Slot, QTimer, Qt, QThread, QObject, QMutex, QMutexLocker
)
from PySide6.QtGui import QColor, QPalette, QFont, QPixmap, QIcon, QTextCursor
import pyqtgraph as pg
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

# Import GeminiSDR components
try:
    from geminisdr.core.model_manager import ModelManager
    from geminisdr.core.metrics_collector import MetricsCollector, MLMetrics
    from geminisdr.core.model_metadata import ModelMetadata
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    logging.warning("Model management components not available")


class ModelStatus:
    """Model status enumeration."""
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    TRAINING = "training"
    INFERENCING = "inferencing"
    ERROR = "error"
    UNLOADED = "unloaded"


class ModelInfo:
    """Model information container."""
    def __init__(self, name: str, version: str = "latest"):
        self.name = name
        self.version = version
        self.status = ModelStatus.IDLE
        self.accuracy = 0.0
        self.loss = 0.0
        self.device = "cpu"
        self.memory_usage_mb = 0.0
        self.last_updated = datetime.now()
        self.metadata: Optional[ModelMetadata] = None
        self.error_message = ""


class TrainingProgress:
    """Training progress tracking."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches = 0
        self.loss_history = []
        self.accuracy_history = []
        self.validation_loss_history = []
        self.validation_accuracy_history = []
        self.learning_rate_history = []
        self.start_time = datetime.now()
        self.estimated_completion = None
        self.is_active = False


class ModelMonitorWidget(QWidget):
    """
    Main ML model monitoring widget with tabbed interface.
    
    Provides real-time model performance metrics display, model loading/unloading
    capabilities, and visual indicators for model status and resource usage.
    """
    
    # Signals
    model_load_requested = Signal(str, str)  # model_name, version
    model_unload_requested = Signal(str)     # model_name
    model_selected = Signal(str)             # model_name
    training_control_requested = Signal(str, str)  # action, model_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Model tracking
        self.active_models: Dict[str, ModelInfo] = {}
        self.training_progress: Dict[str, TrainingProgress] = {}
        self.model_manager: Optional[ModelManager] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # Thread safety
        self.data_mutex = QMutex()
        
        # Training log storage for filtering
        self.full_training_log = []  # Store all log entries
        self.filtered_log_entries = []  # Store filtered entries
        
        # Update timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_displays)
        self.update_timer.start(1000)  # Update every second
        
        # Initialize UI
        self._setup_ui()
        self._setup_connections()
        
        # Initialize model manager if available
        if MODEL_MANAGER_AVAILABLE:
            try:
                self.model_manager = ModelManager()
                self._load_available_models()
            except Exception as e:
                self.logger.warning(f"Could not initialize model manager: {e}")
        
        self.logger.info("ModelMonitorWidget initialized")
    
    def _setup_ui(self) -> None:
        """Setup the model monitoring UI with tabbed interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        
        # Model status tab
        self.status_tab = self._create_status_tab()
        self.tab_widget.addTab(self.status_tab, "Model Status")
        
        # Training progress tab
        self.training_tab = self._create_training_tab()
        self.tab_widget.addTab(self.training_tab, "Training Progress")
        
        # Model comparison tab
        self.comparison_tab = self._create_comparison_tab()
        self.tab_widget.addTab(self.comparison_tab, "Model Comparison")
        
        layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 5px; background-color: #404040; border-radius: 3px;")
        layout.addWidget(self.status_label)
    
    def _create_status_tab(self) -> QWidget:
        """Create model status monitoring tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Control panel
        control_group = QGroupBox("Model Control")
        control_layout = QHBoxLayout(control_group)
        
        # Model selection
        control_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.model_combo.currentTextChanged.connect(self._on_model_selected)
        control_layout.addWidget(self.model_combo)
        
        # Version selection
        control_layout.addWidget(QLabel("Version:"))
        self.version_combo = QComboBox()
        self.version_combo.setMinimumWidth(100)
        control_layout.addWidget(self.version_combo)
        
        control_layout.addStretch()
        
        # Control buttons
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self._load_model)
        control_layout.addWidget(self.load_model_btn)
        
        self.unload_model_btn = QPushButton("Unload Model")
        self.unload_model_btn.clicked.connect(self._unload_model)
        self.unload_model_btn.setEnabled(False)
        control_layout.addWidget(self.unload_model_btn)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_models)
        control_layout.addWidget(self.refresh_btn)
        
        layout.addWidget(control_group)
        
        # Model status table
        status_group = QGroupBox("Active Models")
        status_layout = QVBoxLayout(status_group)
        
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(7)
        self.model_table.setHorizontalHeaderLabels([
            "Model", "Version", "Status", "Accuracy", "Loss", "Device", "Memory (MB)"
        ])
        
        # Configure table
        header = self.model_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        
        status_layout.addWidget(self.model_table)
        layout.addWidget(status_group)
        
        # Model details panel
        details_group = QGroupBox("Model Details")
        details_layout = QVBoxLayout(details_group)
        
        # Create scrollable area for details
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        
        # Model information labels
        self.detail_labels = {}
        detail_fields = [
            ("Name", "name"), ("Version", "version"), ("Status", "status"),
            ("Architecture", "architecture"), ("Parameters", "parameters"),
            ("Model Size", "size"), ("Device", "device"), ("Memory Usage", "memory"),
            ("Last Updated", "updated"), ("Description", "description")
        ]
        
        for i, (label, key) in enumerate(detail_fields):
            scroll_layout.addWidget(QLabel(f"{label}:"), i, 0)
            value_label = QLabel("N/A")
            value_label.setWordWrap(True)
            scroll_layout.addWidget(value_label, i, 1)
            self.detail_labels[key] = value_label
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)
        details_layout.addWidget(scroll_area)
        
        layout.addWidget(details_group)
        
        return widget
    
    def _create_training_tab(self) -> QWidget:
        """Create training progress monitoring tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Training control panel
        control_group = QGroupBox("Training Control")
        control_layout = QVBoxLayout(control_group)
        
        # Model and parameter selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.training_model_combo = QComboBox()
        self.training_model_combo.setMinimumWidth(150)
        model_layout.addWidget(self.training_model_combo)
        
        model_layout.addStretch()
        control_layout.addLayout(model_layout)
        
        # Training parameters
        params_layout = QGridLayout()
        
        # Epochs
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(100)
        params_layout.addWidget(self.epochs_spinbox, 0, 1)
        
        # Batch size
        params_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 1024)
        self.batch_size_spinbox.setValue(32)
        params_layout.addWidget(self.batch_size_spinbox, 0, 3)
        
        # Learning rate
        params_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.learning_rate_spinbox = QDoubleSpinBox()
        self.learning_rate_spinbox.setRange(0.0001, 1.0)
        self.learning_rate_spinbox.setValue(0.001)
        self.learning_rate_spinbox.setDecimals(4)
        self.learning_rate_spinbox.setSingleStep(0.0001)
        params_layout.addWidget(self.learning_rate_spinbox, 1, 1)
        
        # Validation split
        params_layout.addWidget(QLabel("Validation Split:"), 1, 2)
        self.validation_split_spinbox = QDoubleSpinBox()
        self.validation_split_spinbox.setRange(0.0, 0.5)
        self.validation_split_spinbox.setValue(0.2)
        self.validation_split_spinbox.setDecimals(2)
        self.validation_split_spinbox.setSingleStep(0.05)
        params_layout.addWidget(self.validation_split_spinbox, 1, 3)
        
        control_layout.addLayout(params_layout)
        
        # Training control buttons
        buttons_layout = QHBoxLayout()
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self._start_training)
        buttons_layout.addWidget(self.start_training_btn)
        
        self.pause_training_btn = QPushButton("Pause Training")
        self.pause_training_btn.clicked.connect(self._pause_training)
        self.pause_training_btn.setEnabled(False)
        buttons_layout.addWidget(self.pause_training_btn)
        
        self.stop_training_btn = QPushButton("Stop Training")
        self.stop_training_btn.clicked.connect(self._stop_training)
        self.stop_training_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_training_btn)
        
        buttons_layout.addStretch()
        control_layout.addLayout(buttons_layout)
        
        layout.addWidget(control_group)
        
        # Training progress display
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Progress bars and info
        info_layout = QGridLayout()
        
        # Epoch progress
        info_layout.addWidget(QLabel("Epoch Progress:"), 0, 0)
        self.epoch_progress = QProgressBar()
        info_layout.addWidget(self.epoch_progress, 0, 1)
        self.epoch_label = QLabel("0/0")
        info_layout.addWidget(self.epoch_label, 0, 2)
        
        # Batch progress
        info_layout.addWidget(QLabel("Batch Progress:"), 1, 0)
        self.batch_progress = QProgressBar()
        info_layout.addWidget(self.batch_progress, 1, 1)
        self.batch_label = QLabel("0/0")
        info_layout.addWidget(self.batch_label, 1, 2)
        
        # Time information
        info_layout.addWidget(QLabel("Elapsed Time:"), 2, 0)
        self.elapsed_time_label = QLabel("00:00:00")
        info_layout.addWidget(self.elapsed_time_label, 2, 1)
        
        info_layout.addWidget(QLabel("Estimated Completion:"), 3, 0)
        self.eta_label = QLabel("N/A")
        info_layout.addWidget(self.eta_label, 3, 1)
        
        progress_layout.addLayout(info_layout)
        layout.addWidget(progress_group)
        
        # Training plots
        plots_group = QGroupBox("Training Metrics")
        plots_layout = QHBoxLayout(plots_group)
        
        # Loss plot
        self.loss_plot = pg.PlotWidget(title="Training Loss")
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.setLabel('bottom', 'Epoch')
        self.loss_plot.showGrid(True, True, alpha=0.3)
        self.loss_plot.addLegend()
        
        # Add curves for training and validation loss
        self.train_loss_curve = self.loss_plot.plot(
            pen=pg.mkPen(color='#ff6b6b', width=2), name='Training Loss'
        )
        self.val_loss_curve = self.loss_plot.plot(
            pen=pg.mkPen(color='#4ecdc4', width=2), name='Validation Loss'
        )
        
        plots_layout.addWidget(self.loss_plot)
        
        # Accuracy plot
        self.accuracy_plot = pg.PlotWidget(title="Training Accuracy")
        self.accuracy_plot.setLabel('left', 'Accuracy (%)')
        self.accuracy_plot.setLabel('bottom', 'Epoch')
        self.accuracy_plot.showGrid(True, True, alpha=0.3)
        self.accuracy_plot.addLegend()
        
        # Add curves for training and validation accuracy
        self.train_acc_curve = self.accuracy_plot.plot(
            pen=pg.mkPen(color='#ff6b6b', width=2), name='Training Accuracy'
        )
        self.val_acc_curve = self.accuracy_plot.plot(
            pen=pg.mkPen(color='#4ecdc4', width=2), name='Validation Accuracy'
        )
        
        plots_layout.addWidget(self.accuracy_plot)
        layout.addWidget(plots_group)
        
        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        
        # Log search and filter controls
        log_filter_layout = QHBoxLayout()
        log_filter_layout.addWidget(QLabel("Filter:"))
        
        self.log_filter_combo = QComboBox()
        self.log_filter_combo.addItems(["All", "Info", "Warning", "Error", "Training", "Validation"])
        self.log_filter_combo.currentTextChanged.connect(self._filter_training_log)
        log_filter_layout.addWidget(self.log_filter_combo)
        
        log_filter_layout.addWidget(QLabel("Search:"))
        self.log_search_box = QLineEdit()
        self.log_search_box.setPlaceholderText("Search log entries...")
        self.log_search_box.textChanged.connect(self._search_training_log)
        log_filter_layout.addWidget(self.log_search_box)
        
        log_filter_layout.addStretch()
        log_layout.addLayout(log_filter_layout)
        
        self.training_log = QTextEdit()
        self.training_log.setMaximumHeight(150)
        self.training_log.setReadOnly(True)
        self.training_log.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.training_log)
        
        # Log controls
        log_controls = QHBoxLayout()
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self._clear_training_log)
        log_controls.addWidget(self.clear_log_btn)
        
        self.save_log_btn = QPushButton("Save Log")
        self.save_log_btn.clicked.connect(self._save_training_log)
        log_controls.addWidget(self.save_log_btn)
        
        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        log_controls.addWidget(self.auto_scroll_checkbox)
        
        log_controls.addStretch()
        log_layout.addLayout(log_controls)
        
        layout.addWidget(log_group)
        
        return widget
    
    def _create_comparison_tab(self) -> QWidget:
        """Create model comparison tab."""
        try:
            from gui.widgets.model_comparison import ModelComparisonWidget
            
            # Create the advanced comparison widget
            comparison_widget = ModelComparisonWidget()
            
            # Set model manager if available
            if self.model_manager:
                comparison_widget.set_model_manager(self.model_manager)
            
            # Connect signals
            comparison_widget.comparison_requested.connect(self._on_comparison_requested)
            comparison_widget.benchmark_requested.connect(self._on_benchmark_requested)
            
            # Store reference for later use
            self.comparison_widget = comparison_widget
            
            return comparison_widget
            
        except ImportError as e:
            self.logger.warning(f"Advanced comparison widget not available: {e}")
            
            # Fallback to simple comparison interface
            return self._create_simple_comparison_tab()
    
    def _create_simple_comparison_tab(self) -> QWidget:
        """Create simple model comparison tab as fallback."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model selection for comparison
        selection_group = QGroupBox("Model Selection")
        selection_layout = QHBoxLayout(selection_group)
        
        # Model 1 selection
        selection_layout.addWidget(QLabel("Model 1:"))
        self.compare_model1_combo = QComboBox()
        self.compare_model1_combo.setMinimumWidth(150)
        selection_layout.addWidget(self.compare_model1_combo)
        
        self.compare_version1_combo = QComboBox()
        selection_layout.addWidget(self.compare_version1_combo)
        
        selection_layout.addWidget(QLabel("vs"))
        
        # Model 2 selection
        selection_layout.addWidget(QLabel("Model 2:"))
        self.compare_model2_combo = QComboBox()
        self.compare_model2_combo.setMinimumWidth(150)
        selection_layout.addWidget(self.compare_model2_combo)
        
        self.compare_version2_combo = QComboBox()
        selection_layout.addWidget(self.compare_version2_combo)
        
        selection_layout.addStretch()
        
        self.compare_btn = QPushButton("Compare Models")
        self.compare_btn.clicked.connect(self._compare_models)
        selection_layout.addWidget(self.compare_btn)
        
        layout.addWidget(selection_group)
        
        # Comparison results
        results_group = QGroupBox("Comparison Results")
        results_layout = QVBoxLayout(results_group)
        
        # Create splitter for side-by-side comparison
        splitter = QSplitter(Qt.Horizontal)
        
        # Model 1 details
        model1_frame = QFrame()
        model1_frame.setFrameStyle(QFrame.StyledPanel)
        model1_layout = QVBoxLayout(model1_frame)
        model1_layout.addWidget(QLabel("Model 1 Details"))
        
        self.model1_details = QTextEdit()
        self.model1_details.setReadOnly(True)
        model1_layout.addWidget(self.model1_details)
        
        splitter.addWidget(model1_frame)
        
        # Model 2 details
        model2_frame = QFrame()
        model2_frame.setFrameStyle(QFrame.StyledPanel)
        model2_layout = QVBoxLayout(model2_frame)
        model2_layout.addWidget(QLabel("Model 2 Details"))
        
        self.model2_details = QTextEdit()
        self.model2_details.setReadOnly(True)
        model2_layout.addWidget(self.model2_details)
        
        splitter.addWidget(model2_frame)
        
        results_layout.addWidget(splitter)
        
        # Performance comparison chart
        self.comparison_plot = pg.PlotWidget(title="Performance Comparison")
        self.comparison_plot.setLabel('left', 'Score')
        self.comparison_plot.setLabel('bottom', 'Metric')
        results_layout.addWidget(self.comparison_plot)
        
        layout.addWidget(results_group)
        
        return widget
    
    def _setup_connections(self) -> None:
        """Setup signal-slot connections."""
        # Connect model selection changes
        self.model_combo.currentTextChanged.connect(self._update_version_combo)
        
        # Only connect comparison combo boxes if they exist (simple comparison tab)
        if hasattr(self, 'compare_model1_combo'):
            self.compare_model1_combo.currentTextChanged.connect(
                lambda: self._update_comparison_version_combo(1)
            )
        if hasattr(self, 'compare_model2_combo'):
            self.compare_model2_combo.currentTextChanged.connect(
                lambda: self._update_comparison_version_combo(2)
            )
    
    def _load_available_models(self) -> None:
        """Load available models from model manager."""
        if not self.model_manager:
            return
        
        try:
            models = self.model_manager.list_models()
            
            # Update model combo boxes
            self.model_combo.clear()
            self.training_model_combo.clear()
            
            # Only clear comparison combos if they exist (simple comparison interface)
            if hasattr(self, 'compare_model1_combo'):
                self.compare_model1_combo.clear()
            if hasattr(self, 'compare_model2_combo'):
                self.compare_model2_combo.clear()
            
            for model_info in models:
                model_name = model_info['name']
                self.model_combo.addItem(model_name)
                self.training_model_combo.addItem(model_name)
                
                # Only add to comparison combos if they exist
                if hasattr(self, 'compare_model1_combo'):
                    self.compare_model1_combo.addItem(model_name)
                if hasattr(self, 'compare_model2_combo'):
                    self.compare_model2_combo.addItem(model_name)
            
            self.logger.info(f"Loaded {len(models)} available models")
            
        except Exception as e:
            self.logger.error(f"Failed to load available models: {e}")
            self._show_error_message("Model Loading Error", str(e))
    
    @Slot(str)
    def _update_version_combo(self, model_name: str) -> None:
        """Update version combo box based on selected model."""
        if not self.model_manager or not model_name:
            return
        
        try:
            models = self.model_manager.list_models()
            model_info = next((m for m in models if m['name'] == model_name), None)
            
            if model_info:
                self.version_combo.clear()
                for version_info in model_info['versions']:
                    self.version_combo.addItem(version_info['version'])
                
                # Select latest version by default
                if model_info['latest_version']:
                    index = self.version_combo.findText(model_info['latest_version'])
                    if index >= 0:
                        self.version_combo.setCurrentIndex(index)
                        
        except Exception as e:
            self.logger.error(f"Failed to update version combo: {e}")
    
    def _update_comparison_version_combo(self, model_num: int) -> None:
        """Update comparison version combo boxes."""
        # Only proceed if comparison combos exist (simple comparison interface)
        if not hasattr(self, 'compare_model1_combo') or not hasattr(self, 'compare_model2_combo'):
            return
            
        if model_num == 1:
            model_name = self.compare_model1_combo.currentText()
            version_combo = self.compare_version1_combo
        else:
            model_name = self.compare_model2_combo.currentText()
            version_combo = self.compare_version2_combo
        
        if not self.model_manager or not model_name:
            return
        
        try:
            models = self.model_manager.list_models()
            model_info = next((m for m in models if m['name'] == model_name), None)
            
            if model_info:
                version_combo.clear()
                for version_info in model_info['versions']:
                    version_combo.addItem(version_info['version'])
                    
        except Exception as e:
            self.logger.error(f"Failed to update comparison version combo: {e}")
    
    @Slot()
    def _load_model(self) -> None:
        """Load selected model."""
        model_name = self.model_combo.currentText()
        version = self.version_combo.currentText()
        
        if not model_name or not version:
            self._show_error_message("Selection Error", "Please select a model and version")
            return
        
        self.model_load_requested.emit(model_name, version)
        self._add_log_message(f"Loading model: {model_name} v{version}")
    
    @Slot()
    def _unload_model(self) -> None:
        """Unload selected model."""
        current_row = self.model_table.currentRow()
        if current_row < 0:
            self._show_error_message("Selection Error", "Please select a model to unload")
            return
        
        model_name = self.model_table.item(current_row, 0).text()
        self.model_unload_requested.emit(model_name)
        self._add_log_message(f"Unloading model: {model_name}")
    
    @Slot()
    def _refresh_models(self) -> None:
        """Refresh available models list."""
        self._load_available_models()
        self._add_log_message("Model list refreshed")
    
    @Slot()
    def _on_table_selection_changed(self) -> None:
        """Handle model table selection changes."""
        current_row = self.model_table.currentRow()
        self.unload_model_btn.setEnabled(current_row >= 0)
        
        if current_row >= 0:
            model_name = self.model_table.item(current_row, 0).text()
            self.model_selected.emit(model_name)
            self._update_model_details(model_name)
    
    @Slot(str)
    def _on_model_selected(self, model_name: str) -> None:
        """Handle model selection from combo box."""
        if model_name:
            self._update_model_details(model_name)
    
    def _update_model_details(self, model_name: str) -> None:
        """Update model details panel."""
        with QMutexLocker(self.data_mutex):
            model_info = self.active_models.get(model_name)
        
        if not model_info:
            # Clear details
            for label in self.detail_labels.values():
                label.setText("N/A")
            return
        
        # Update detail labels
        self.detail_labels["name"].setText(model_info.name)
        self.detail_labels["version"].setText(model_info.version)
        self.detail_labels["status"].setText(model_info.status)
        self.detail_labels["device"].setText(model_info.device)
        self.detail_labels["memory"].setText(f"{model_info.memory_usage_mb:.1f} MB")
        self.detail_labels["updated"].setText(model_info.last_updated.strftime("%Y-%m-%d %H:%M:%S"))
        
        if model_info.metadata:
            self.detail_labels["architecture"].setText(model_info.metadata.model_architecture)
            self.detail_labels["size"].setText(f"{model_info.metadata.model_size_mb:.1f} MB")
            self.detail_labels["description"].setText(model_info.metadata.description or "N/A")
            
            # Calculate parameter count if available
            if hasattr(model_info.metadata, 'parameters'):
                self.detail_labels["parameters"].setText(f"{model_info.metadata.parameters:,}")
            else:
                self.detail_labels["parameters"].setText("N/A")
        else:
            self.detail_labels["architecture"].setText("N/A")
            self.detail_labels["size"].setText("N/A")
            self.detail_labels["description"].setText("N/A")
            self.detail_labels["parameters"].setText("N/A")
    
    # Training control methods
    @Slot()
    def _start_training(self) -> None:
        """Start training for selected model."""
        model_name = self.training_model_combo.currentText()
        if not model_name:
            self._show_error_message("Selection Error", "Please select a model for training")
            return
        
        # Get training parameters
        epochs = self.epochs_spinbox.value()
        batch_size = self.batch_size_spinbox.value()
        learning_rate = self.learning_rate_spinbox.value()
        validation_split = self.validation_split_spinbox.value()
        
        # Log training start with parameters
        self._add_training_log(
            f"Starting training for {model_name} - "
            f"Epochs: {epochs}, Batch Size: {batch_size}, "
            f"LR: {learning_rate}, Val Split: {validation_split}",
            "training"
        )
        
        # Emit signal with parameters
        training_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'validation_split': validation_split
        }
        
        self.training_control_requested.emit("start", model_name)
        
        # Update button states
        self.start_training_btn.setEnabled(False)
        self.pause_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(True)
    
    @Slot()
    def _pause_training(self) -> None:
        """Pause training for selected model."""
        model_name = self.training_model_combo.currentText()
        if model_name:
            self.training_control_requested.emit("pause", model_name)
            self._add_training_log(f"Pausing training for {model_name}", "training")
    
    @Slot()
    def _stop_training(self) -> None:
        """Stop training for selected model."""
        model_name = self.training_model_combo.currentText()
        if model_name:
            self.training_control_requested.emit("stop", model_name)
            self._add_training_log(f"Stopping training for {model_name}", "training")
            
            # Update button states
            self.start_training_btn.setEnabled(True)
            self.pause_training_btn.setEnabled(False)
            self.stop_training_btn.setEnabled(False)
    
    @Slot()
    def _compare_models(self) -> None:
        """Compare selected models."""
        model1_name = self.compare_model1_combo.currentText()
        model1_version = self.compare_version1_combo.currentText()
        model2_name = self.compare_model2_combo.currentText()
        model2_version = self.compare_version2_combo.currentText()
        
        if not all([model1_name, model1_version, model2_name, model2_version]):
            self._show_error_message("Selection Error", "Please select both models and versions")
            return
        
        if not self.model_manager:
            self._show_error_message("Error", "Model manager not available")
            return
        
        try:
            # Get comparison data
            comparison = self.model_manager.compare_models(
                model1_name, model1_version,
                model2_name, model2_version
            )
            
            # Update comparison display
            self._update_comparison_display(comparison, model1_name, model2_name)
            self._add_log_message(f"Compared {model1_name} v{model1_version} vs {model2_name} v{model2_version}")
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            self._show_error_message("Comparison Error", str(e))
    
    def _update_comparison_display(self, comparison: Dict[str, Any], 
                                 model1_name: str, model2_name: str) -> None:
        """Update comparison results display."""
        # Update model details text
        perf_comp = comparison.get('performance_comparison', {})
        meta_comp = comparison.get('metadata_comparison', {})
        
        # Model 1 details
        model1_text = f"Model: {model1_name}\n\n"
        model1_text += "Performance Metrics:\n"
        for metric, values in perf_comp.items():
            if isinstance(values, dict) and 'model1' in values:
                model1_text += f"  {metric}: {values['model1']:.4f}\n"
        
        model1_text += "\nMetadata:\n"
        for key, values in meta_comp.items():
            if isinstance(values, dict) and 'model1' in values:
                model1_text += f"  {key}: {values['model1']}\n"
        
        self.model1_details.setText(model1_text)
        
        # Model 2 details
        model2_text = f"Model: {model2_name}\n\n"
        model2_text += "Performance Metrics:\n"
        for metric, values in perf_comp.items():
            if isinstance(values, dict) and 'model2' in values:
                model2_text += f"  {metric}: {values['model2']:.4f}\n"
        
        model2_text += "\nMetadata:\n"
        for key, values in meta_comp.items():
            if isinstance(values, dict) and 'model2' in values:
                model2_text += f"  {key}: {values['model2']}\n"
        
        self.model2_details.setText(model2_text)
        
        # Update comparison plot
        self._update_comparison_plot(perf_comp, model1_name, model2_name)
    
    def _update_comparison_plot(self, performance_comparison: Dict[str, Any],
                              model1_name: str, model2_name: str) -> None:
        """Update performance comparison plot."""
        self.comparison_plot.clear()
        
        if not performance_comparison:
            return
        
        metrics = []
        model1_values = []
        model2_values = []
        
        for metric, values in performance_comparison.items():
            if isinstance(values, dict) and 'model1' in values and 'model2' in values:
                metrics.append(metric)
                model1_values.append(values['model1'])
                model2_values.append(values['model2'])
        
        if not metrics:
            return
        
        # Create bar chart
        x = np.arange(len(metrics))
        width = 0.35
        
        # Model 1 bars
        bar1 = pg.BarGraphItem(
            x=x - width/2, height=model1_values, width=width,
            brush='#ff6b6b', name=model1_name
        )
        self.comparison_plot.addItem(bar1)
        
        # Model 2 bars
        bar2 = pg.BarGraphItem(
            x=x + width/2, height=model2_values, width=width,
            brush='#4ecdc4', name=model2_name
        )
        self.comparison_plot.addItem(bar2)
        
        # Set x-axis labels
        ax = self.comparison_plot.getAxis('bottom')
        ax.setTicks([[(i, metric) for i, metric in enumerate(metrics)]])
    
    @Slot()
    def _save_training_log(self) -> None:
        """Save training log to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Training Log", "training_log.txt", "Text Files (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.training_log.toPlainText())
                self._add_log_message(f"Training log saved to {filename}")
            except Exception as e:
                self._show_error_message("Save Error", f"Failed to save log: {e}")
    
    def _update_displays(self) -> None:
        """Update all displays with current data."""
        self._update_model_table()
        self._update_training_displays()
    
    def _update_model_table(self) -> None:
        """Update the model status table."""
        with QMutexLocker(self.data_mutex):
            models = list(self.active_models.values())
        
        self.model_table.setRowCount(len(models))
        
        for row, model_info in enumerate(models):
            # Model name
            self.model_table.setItem(row, 0, QTableWidgetItem(model_info.name))
            
            # Version
            self.model_table.setItem(row, 1, QTableWidgetItem(model_info.version))
            
            # Status with color coding
            status_item = QTableWidgetItem(model_info.status.title())
            if model_info.status == ModelStatus.LOADED:
                status_item.setBackground(QColor("#4ecdc4"))
            elif model_info.status == ModelStatus.TRAINING:
                status_item.setBackground(QColor("#ffa500"))
            elif model_info.status == ModelStatus.ERROR:
                status_item.setBackground(QColor("#ff6b6b"))
            else:
                status_item.setBackground(QColor("#666666"))
            self.model_table.setItem(row, 2, status_item)
            
            # Accuracy
            accuracy_item = QTableWidgetItem(f"{model_info.accuracy:.3f}")
            self.model_table.setItem(row, 3, accuracy_item)
            
            # Loss
            loss_item = QTableWidgetItem(f"{model_info.loss:.4f}")
            self.model_table.setItem(row, 4, loss_item)
            
            # Device
            self.model_table.setItem(row, 5, QTableWidgetItem(model_info.device))
            
            # Memory usage
            memory_item = QTableWidgetItem(f"{model_info.memory_usage_mb:.1f}")
            self.model_table.setItem(row, 6, memory_item)
    
    def _update_training_displays(self) -> None:
        """Update training progress displays."""
        current_model = self.training_model_combo.currentText()
        if not current_model:
            return
        
        with QMutexLocker(self.data_mutex):
            progress = self.training_progress.get(current_model)
        
        if not progress:
            return
        
        # Update progress bars
        if progress.total_epochs > 0:
            epoch_percent = int((progress.current_epoch / progress.total_epochs) * 100)
            self.epoch_progress.setValue(epoch_percent)
            self.epoch_label.setText(f"{progress.current_epoch}/{progress.total_epochs}")
        
        if progress.total_batches > 0:
            batch_percent = int((progress.current_batch / progress.total_batches) * 100)
            self.batch_progress.setValue(batch_percent)
            self.batch_label.setText(f"{progress.current_batch}/{progress.total_batches}")
        
        # Update time information
        elapsed = datetime.now() - progress.start_time
        self.elapsed_time_label.setText(str(elapsed).split('.')[0])  # Remove microseconds
        
        if progress.estimated_completion:
            self.eta_label.setText(progress.estimated_completion.strftime("%H:%M:%S"))
        
        # Update plots
        if progress.loss_history:
            epochs = list(range(len(progress.loss_history)))
            self.train_loss_curve.setData(epochs, progress.loss_history)
        
        if progress.validation_loss_history:
            epochs = list(range(len(progress.validation_loss_history)))
            self.val_loss_curve.setData(epochs, progress.validation_loss_history)
        
        if progress.accuracy_history:
            epochs = list(range(len(progress.accuracy_history)))
            self.train_acc_curve.setData(epochs, progress.accuracy_history)
        
        if progress.validation_accuracy_history:
            epochs = list(range(len(progress.validation_accuracy_history)))
            self.val_acc_curve.setData(epochs, progress.validation_accuracy_history)
    
    # Public interface methods for external updates
    def add_model(self, name: str, version: str = "latest", 
                  metadata: Optional[ModelMetadata] = None) -> None:
        """Add a model to the monitoring system."""
        with QMutexLocker(self.data_mutex):
            model_info = ModelInfo(name, version)
            model_info.metadata = metadata
            model_info.status = ModelStatus.LOADED
            self.active_models[name] = model_info
        
        self._add_log_message(f"Added model: {name} v{version}")
    
    def remove_model(self, name: str) -> None:
        """Remove a model from monitoring."""
        with QMutexLocker(self.data_mutex):
            if name in self.active_models:
                del self.active_models[name]
            if name in self.training_progress:
                del self.training_progress[name]
        
        self._add_log_message(f"Removed model: {name}")
    
    def update_model_metrics(self, name: str, metrics: Dict[str, Any]) -> None:
        """Update model performance metrics."""
        with QMutexLocker(self.data_mutex):
            if name in self.active_models:
                model_info = self.active_models[name]
                model_info.accuracy = metrics.get('accuracy', model_info.accuracy)
                model_info.loss = metrics.get('loss', model_info.loss)
                model_info.memory_usage_mb = metrics.get('memory_usage_mb', model_info.memory_usage_mb)
                model_info.device = metrics.get('device', model_info.device)
                model_info.last_updated = datetime.now()
    
    def update_model_status(self, name: str, status: str, error_message: str = "") -> None:
        """Update model status."""
        with QMutexLocker(self.data_mutex):
            if name in self.active_models:
                self.active_models[name].status = status
                self.active_models[name].error_message = error_message
                self.active_models[name].last_updated = datetime.now()
    
    def update_training_progress(self, model_name: str, epoch: int, total_epochs: int,
                               batch: int, total_batches: int, loss: float, 
                               accuracy: float, val_loss: Optional[float] = None,
                               val_accuracy: Optional[float] = None) -> None:
        """Update training progress for a model."""
        with QMutexLocker(self.data_mutex):
            if model_name not in self.training_progress:
                self.training_progress[model_name] = TrainingProgress(model_name)
            
            progress = self.training_progress[model_name]
            progress.current_epoch = epoch
            progress.total_epochs = total_epochs
            progress.current_batch = batch
            progress.total_batches = total_batches
            progress.is_active = True
            
            # Update history
            if len(progress.loss_history) <= epoch:
                progress.loss_history.extend([0] * (epoch + 1 - len(progress.loss_history)))
            progress.loss_history[epoch] = loss
            
            if len(progress.accuracy_history) <= epoch:
                progress.accuracy_history.extend([0] * (epoch + 1 - len(progress.accuracy_history)))
            progress.accuracy_history[epoch] = accuracy
            
            if val_loss is not None:
                if len(progress.validation_loss_history) <= epoch:
                    progress.validation_loss_history.extend([0] * (epoch + 1 - len(progress.validation_loss_history)))
                progress.validation_loss_history[epoch] = val_loss
            
            if val_accuracy is not None:
                if len(progress.validation_accuracy_history) <= epoch:
                    progress.validation_accuracy_history.extend([0] * (epoch + 1 - len(progress.validation_accuracy_history)))
                progress.validation_accuracy_history[epoch] = val_accuracy
            
            # Estimate completion time and performance predictions
            if epoch > 0:
                elapsed = datetime.now() - progress.start_time
                time_per_epoch = elapsed / epoch
                remaining_epochs = total_epochs - epoch
                progress.estimated_completion = datetime.now() + (time_per_epoch * remaining_epochs)
                
                # Performance prediction based on trend
                if len(progress.accuracy_history) >= 3:
                    recent_acc = progress.accuracy_history[-3:]
                    acc_trend = (recent_acc[-1] - recent_acc[0]) / 2  # Average improvement per epoch
                    predicted_final_acc = accuracy + (acc_trend * remaining_epochs)
                    
                    # Log performance prediction
                    if epoch % 5 == 0:  # Log every 5 epochs
                        self._add_training_log(
                            f"Performance prediction for {model_name}: "
                            f"Final accuracy ~{predicted_final_acc:.3f}, "
                            f"ETA: {progress.estimated_completion.strftime('%H:%M:%S')}",
                            "training"
                        )
    
    def _add_log_message(self, message: str) -> None:
        """Add message to status log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(f"[{timestamp}] {message}")
    
    def _add_training_log(self, message: str, level: str = "info") -> None:
        """Add message to training log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'full_text': f"[{timestamp}] [{level.upper()}] {message}"
        }
        
        # Store in full log
        self.full_training_log.append(log_entry)
        
        # Apply current filter
        self._apply_log_filter()
        
        # Auto-scroll to bottom if enabled
        if self.auto_scroll_checkbox.isChecked():
            cursor = self.training_log.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.training_log.setTextCursor(cursor)
    
    @Slot(str)
    def _filter_training_log(self, filter_type: str) -> None:
        """Filter training log by type."""
        self._apply_log_filter()
    
    @Slot(str)
    def _search_training_log(self, search_text: str) -> None:
        """Search training log entries."""
        self._apply_log_filter()
    
    def _apply_log_filter(self) -> None:
        """Apply current filter and search to training log."""
        filter_type = self.log_filter_combo.currentText().lower()
        search_text = self.log_search_box.text().lower()
        
        # Filter entries
        filtered_entries = []
        for entry in self.full_training_log:
            # Apply level filter
            if filter_type != "all" and filter_type != entry['level'].lower():
                continue
            
            # Apply search filter
            if search_text and search_text not in entry['message'].lower():
                continue
            
            filtered_entries.append(entry)
        
        # Update display
        self.training_log.clear()
        for entry in filtered_entries:
            self.training_log.append(entry['full_text'])
        
        self.filtered_log_entries = filtered_entries
    
    @Slot()
    def _clear_training_log(self) -> None:
        """Clear training log."""
        self.full_training_log.clear()
        self.filtered_log_entries.clear()
        self.training_log.clear()
    
    def _show_error_message(self, title: str, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, title, message)
        self.logger.error(f"{title}: {message}")
    
    def set_metrics_collector(self, metrics_collector: 'MetricsCollector') -> None:
        """Set metrics collector for integration."""
        self.metrics_collector = metrics_collector
        if metrics_collector:
            # Register for ML metrics updates
            metrics_collector.register_alert_callback(self._handle_metrics_alert)
    
    def _handle_metrics_alert(self, alert) -> None:
        """Handle metrics alerts."""
        if alert.metric_name.startswith('ml.'):
            self._add_training_log(f"Alert: {alert.message}")
    
    # Comparison widget signal handlers
    @Slot(list)
    def _on_comparison_requested(self, model_list: List[Tuple[str, str]]) -> None:
        """Handle comparison request from comparison widget."""
        self.logger.info(f"Comparison requested for {len(model_list)} models")
        # The comparison widget handles the actual comparison
    
    @Slot(str, str)
    def _on_benchmark_requested(self, model_name: str, benchmark_type: str) -> None:
        """Handle benchmark request from comparison widget."""
        self.logger.info(f"Benchmark requested: {benchmark_type} for {model_name}")
        # This would trigger actual benchmarking in a real implementation