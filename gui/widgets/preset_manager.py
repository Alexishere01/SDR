"""
Preset Management System

This module provides comprehensive preset management for SDR configurations,
including saving, loading, organizing, and sharing capabilities.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QListWidget, QListWidgetItem, QPushButton, QLineEdit,
                               QLabel, QComboBox, QTextEdit, QSplitter, QFrame,
                               QMessageBox, QFileDialog, QInputDialog, QMenu,
                               QTreeWidget, QTreeWidgetItem, QHeaderView,
                               QTabWidget, QFormLayout, QCheckBox, QSpinBox,
                               QProgressBar, QDialog, QDialogButtonBox)
from PySide6.QtCore import Signal, Slot, Qt, QTimer, QStandardPaths, QDir
from PySide6.QtGui import QIcon, QFont, QColor, QPalette, QAction, QPixmap
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
import hashlib
import time
from datetime import datetime
from .sdr_control_panel import SDRConfig, SDRMode


@dataclass
class PresetMetadata:
    """Metadata for SDR presets."""
    name: str
    description: str = ""
    category: str = "General"
    tags: List[str] = None
    created_date: str = ""
    modified_date: str = ""
    author: str = ""
    version: str = "1.0"
    compatibility: List[str] = None  # Compatible hardware types
    usage_count: int = 0
    rating: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.compatibility is None:
            self.compatibility = ["PlutoSDR", "Simulation"]
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        if not self.modified_date:
            self.modified_date = self.created_date


@dataclass
class Preset:
    """Complete preset with configuration and metadata."""
    config: SDRConfig
    metadata: PresetMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "metadata": asdict(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Preset':
        """Create from dictionary."""
        config = SDRConfig.from_dict(data["config"])
        metadata = PresetMetadata(**data["metadata"])
        return cls(config=config, metadata=metadata)
    
    def get_hash(self) -> str:
        """Get hash of preset for duplicate detection."""
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class PresetCategories:
    """Predefined preset categories."""
    GENERAL = "General"
    AMATEUR_RADIO = "Amateur Radio"
    AVIATION = "Aviation"
    MARINE = "Marine"
    SATELLITE = "Satellite"
    CELLULAR = "Cellular"
    WIFI_BLUETOOTH = "WiFi/Bluetooth"
    ISM_BANDS = "ISM Bands"
    BROADCAST = "Broadcast"
    EMERGENCY = "Emergency Services"
    CUSTOM = "Custom"
    
    @classmethod
    def get_all(cls) -> List[str]:
        """Get all category names."""
        return [
            cls.GENERAL, cls.AMATEUR_RADIO, cls.AVIATION, cls.MARINE,
            cls.SATELLITE, cls.CELLULAR, cls.WIFI_BLUETOOTH, cls.ISM_BANDS,
            cls.BROADCAST, cls.EMERGENCY, cls.CUSTOM
        ]


class PresetEditDialog(QDialog):
    """Dialog for editing preset metadata."""
    
    def __init__(self, preset: Optional[Preset] = None, parent=None):
        super().__init__(parent)
        self.preset = preset
        self.setWindowTitle("Edit Preset" if preset else "New Preset")
        self.setModal(True)
        self.resize(400, 500)
        
        self._setup_ui()
        
        if preset:
            self._load_preset_data()
            
    def _setup_ui(self) -> None:
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        
        # Form layout for metadata
        form_layout = QFormLayout()
        
        # Name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter preset name")
        form_layout.addRow("Name:", self.name_edit)
        
        # Category
        self.category_combo = QComboBox()
        self.category_combo.addItems(PresetCategories.get_all())
        self.category_combo.setEditable(True)
        form_layout.addRow("Category:", self.category_combo)
        
        # Description
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        self.description_edit.setPlaceholderText("Enter description (optional)")
        form_layout.addRow("Description:", self.description_edit)
        
        # Tags
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("Enter tags separated by commas")
        form_layout.addRow("Tags:", self.tags_edit)
        
        # Author
        self.author_edit = QLineEdit()
        self.author_edit.setPlaceholderText("Enter author name")
        form_layout.addRow("Author:", self.author_edit)
        
        # Compatibility
        compat_layout = QVBoxLayout()
        self.pluto_check = QCheckBox("PlutoSDR")
        self.pluto_check.setChecked(True)
        compat_layout.addWidget(self.pluto_check)
        
        self.sim_check = QCheckBox("Simulation")
        self.sim_check.setChecked(True)
        compat_layout.addWidget(self.sim_check)
        
        form_layout.addRow("Compatible with:", compat_layout)
        
        layout.addLayout(form_layout)
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def _load_preset_data(self) -> None:
        """Load preset data into form."""
        if not self.preset:
            return
            
        metadata = self.preset.metadata
        
        self.name_edit.setText(metadata.name)
        self.category_combo.setCurrentText(metadata.category)
        self.description_edit.setPlainText(metadata.description)
        self.tags_edit.setText(", ".join(metadata.tags))
        self.author_edit.setText(metadata.author)
        
        self.pluto_check.setChecked("PlutoSDR" in metadata.compatibility)
        self.sim_check.setChecked("Simulation" in metadata.compatibility)
        
    def get_metadata(self) -> PresetMetadata:
        """Get metadata from form."""
        tags = [tag.strip() for tag in self.tags_edit.text().split(",") if tag.strip()]
        
        compatibility = []
        if self.pluto_check.isChecked():
            compatibility.append("PlutoSDR")
        if self.sim_check.isChecked():
            compatibility.append("Simulation")
            
        metadata = PresetMetadata(
            name=self.name_edit.text().strip(),
            category=self.category_combo.currentText(),
            description=self.description_edit.toPlainText().strip(),
            tags=tags,
            author=self.author_edit.text().strip(),
            compatibility=compatibility
        )
        
        if self.preset:
            # Preserve existing metadata
            metadata.created_date = self.preset.metadata.created_date
            metadata.usage_count = self.preset.metadata.usage_count
            metadata.rating = self.preset.metadata.rating
            metadata.modified_date = datetime.now().isoformat()
        
        return metadata


class PresetListWidget(QTreeWidget):
    """Custom tree widget for displaying presets."""
    
    preset_selected = Signal(Preset)
    preset_double_clicked = Signal(Preset)
    preset_context_menu = Signal(Preset, object)  # preset, position
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup tree widget
        self.setHeaderLabels(["Name", "Category", "Frequency", "Sample Rate", "Modified"])
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.sortByColumn(0, Qt.AscendingOrder)
        
        # Resize columns
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        
        # Connect signals
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        
        # Category items
        self.category_items = {}
        
    def add_preset(self, preset: Preset) -> None:
        """Add preset to the tree."""
        # Get or create category item
        category = preset.metadata.category
        if category not in self.category_items:
            category_item = QTreeWidgetItem(self, [category])
            category_item.setFont(0, QFont("", -1, QFont.Bold))
            category_item.setExpanded(True)
            self.category_items[category] = category_item
        
        category_item = self.category_items[category]
        
        # Create preset item
        freq_str = self._format_frequency(preset.config.center_freq_hz)
        rate_str = self._format_sample_rate(preset.config.sample_rate_hz)
        modified_str = self._format_date(preset.metadata.modified_date)
        
        preset_item = QTreeWidgetItem(category_item, [
            preset.metadata.name,
            preset.metadata.category,
            freq_str,
            rate_str,
            modified_str
        ])
        
        # Store preset data
        preset_item.setData(0, Qt.UserRole, preset)
        
        # Set tooltip
        tooltip = self._create_tooltip(preset)
        for col in range(self.columnCount()):
            preset_item.setToolTip(col, tooltip)
            
    def remove_preset(self, preset: Preset) -> None:
        """Remove preset from the tree."""
        for category_item in self.category_items.values():
            for i in range(category_item.childCount()):
                child = category_item.child(i)
                stored_preset = child.data(0, Qt.UserRole)
                if stored_preset and stored_preset.metadata.name == preset.metadata.name:
                    category_item.removeChild(child)
                    break
                    
    def update_preset(self, preset: Preset) -> None:
        """Update existing preset in the tree."""
        self.remove_preset(preset)
        self.add_preset(preset)
        
    def clear_presets(self) -> None:
        """Clear all presets."""
        self.clear()
        self.category_items.clear()
        
    def get_selected_preset(self) -> Optional[Preset]:
        """Get currently selected preset."""
        current_item = self.currentItem()
        if current_item and current_item.parent():  # Not a category item
            return current_item.data(0, Qt.UserRole)
        return None
        
    def _format_frequency(self, freq_hz: float) -> str:
        """Format frequency for display."""
        if freq_hz >= 1e9:
            return f"{freq_hz/1e9:.3f} GHz"
        elif freq_hz >= 1e6:
            return f"{freq_hz/1e6:.3f} MHz"
        elif freq_hz >= 1e3:
            return f"{freq_hz/1e3:.3f} kHz"
        else:
            return f"{freq_hz:.0f} Hz"
            
    def _format_sample_rate(self, rate_hz: float) -> str:
        """Format sample rate for display."""
        if rate_hz >= 1e6:
            return f"{rate_hz/1e6:.1f} MSps"
        elif rate_hz >= 1e3:
            return f"{rate_hz/1e3:.1f} kSps"
        else:
            return f"{rate_hz:.0f} Sps"
            
    def _format_date(self, date_str: str) -> str:
        """Format date for display."""
        try:
            dt = datetime.fromisoformat(date_str)
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return date_str
            
    def _create_tooltip(self, preset: Preset) -> str:
        """Create tooltip text for preset."""
        config = preset.config
        metadata = preset.metadata
        
        tooltip = f"""<b>{metadata.name}</b><br/>
<b>Category:</b> {metadata.category}<br/>
<b>Frequency:</b> {self._format_frequency(config.center_freq_hz)}<br/>
<b>Sample Rate:</b> {self._format_sample_rate(config.sample_rate_hz)}<br/>
<b>Gain:</b> {config.gain_db:.1f} dB {'(AGC)' if config.agc_enabled else ''}<br/>
<b>Mode:</b> {config.mode.value.title()}<br/>"""

        if metadata.description:
            tooltip += f"<b>Description:</b> {metadata.description}<br/>"
            
        if metadata.tags:
            tooltip += f"<b>Tags:</b> {', '.join(metadata.tags)}<br/>"
            
        if metadata.author:
            tooltip += f"<b>Author:</b> {metadata.author}<br/>"
            
        tooltip += f"<b>Usage Count:</b> {metadata.usage_count}<br/>"
        tooltip += f"<b>Modified:</b> {self._format_date(metadata.modified_date)}"
        
        return tooltip
        
    @Slot()
    def _on_selection_changed(self) -> None:
        """Handle selection change."""
        preset = self.get_selected_preset()
        if preset:
            self.preset_selected.emit(preset)
            
    @Slot(QTreeWidgetItem, int)
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle item double click."""
        if item.parent():  # Not a category item
            preset = item.data(0, Qt.UserRole)
            if preset:
                self.preset_double_clicked.emit(preset)
                
    @Slot(object)
    def _on_context_menu(self, position) -> None:
        """Handle context menu request."""
        item = self.itemAt(position)
        if item and item.parent():  # Not a category item
            preset = item.data(0, Qt.UserRole)
            if preset:
                self.preset_context_menu.emit(preset, self.mapToGlobal(position))


class PresetManager(QWidget):
    """Comprehensive preset management interface."""
    
    # Signals
    preset_loaded = Signal(Preset)
    preset_saved = Signal(Preset)
    preset_deleted = Signal(str)  # preset name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("GeminiSDR.PresetManager")
        
        # Storage
        self.presets = {}  # name -> Preset
        self.preset_file = self._get_preset_file_path()
        self.current_config = None
        
        # Search and filter state
        self.search_text = ""
        self.filter_category = "All"
        
        self._setup_ui()
        self._setup_connections()
        self._load_presets()
        
        self.logger.info("Preset Manager initialized")
        
    def _get_preset_file_path(self) -> Path:
        """Get path to preset storage file."""
        config_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "sdr_presets.json"
        
    def _setup_ui(self) -> None:
        """Setup preset manager UI."""
        layout = QVBoxLayout(self)
        
        # Search and filter controls
        search_layout = QHBoxLayout()
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search presets...")
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_edit)
        
        self.category_filter = QComboBox()
        self.category_filter.addItem("All")
        self.category_filter.addItems(PresetCategories.get_all())
        search_layout.addWidget(QLabel("Category:"))
        search_layout.addWidget(self.category_filter)
        
        layout.addLayout(search_layout)
        
        # Main content area
        splitter = QSplitter(Qt.Horizontal)
        
        # Preset list
        list_frame = QFrame()
        list_layout = QVBoxLayout(list_frame)
        
        list_layout.addWidget(QLabel("Presets:"))
        self.preset_list = PresetListWidget()
        list_layout.addWidget(self.preset_list)
        
        # List controls
        list_controls = QHBoxLayout()
        
        self.load_btn = QPushButton("Load")
        self.load_btn.setEnabled(False)
        list_controls.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Save Current")
        list_controls.addWidget(self.save_btn)
        
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setEnabled(False)
        list_controls.addWidget(self.edit_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setEnabled(False)
        list_controls.addWidget(self.delete_btn)
        
        list_layout.addLayout(list_controls)
        
        splitter.addWidget(list_frame)
        
        # Preset details
        details_frame = QFrame()
        details_layout = QVBoxLayout(details_frame)
        
        details_layout.addWidget(QLabel("Preset Details:"))
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        details_layout.addWidget(self.details_text)
        
        # Import/Export controls
        io_group = QGroupBox("Import/Export")
        io_layout = QHBoxLayout(io_group)
        
        self.import_btn = QPushButton("Import...")
        io_layout.addWidget(self.import_btn)
        
        self.export_btn = QPushButton("Export...")
        self.export_btn.setEnabled(False)
        io_layout.addWidget(self.export_btn)
        
        self.export_all_btn = QPushButton("Export All...")
        io_layout.addWidget(self.export_all_btn)
        
        details_layout.addWidget(io_group)
        
        details_layout.addStretch()
        
        splitter.addWidget(details_frame)
        
        # Set splitter proportions
        splitter.setSizes([400, 300])
        
        layout.addWidget(splitter)
        
    def _setup_connections(self) -> None:
        """Setup signal connections."""
        # Search and filter
        self.search_edit.textChanged.connect(self._on_search_changed)
        self.category_filter.currentTextChanged.connect(self._on_category_filter_changed)
        
        # Preset list
        self.preset_list.preset_selected.connect(self._on_preset_selected)
        self.preset_list.preset_double_clicked.connect(self._on_preset_double_clicked)
        self.preset_list.preset_context_menu.connect(self._on_preset_context_menu)
        
        # Controls
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.edit_btn.clicked.connect(self._on_edit_clicked)
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        
        # Import/Export
        self.import_btn.clicked.connect(self._on_import_clicked)
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.export_all_btn.clicked.connect(self._on_export_all_clicked)
        
    def _load_presets(self) -> None:
        """Load presets from storage."""
        try:
            if self.preset_file.exists():
                with open(self.preset_file, 'r') as f:
                    data = json.load(f)
                    
                for preset_data in data.get("presets", []):
                    try:
                        preset = Preset.from_dict(preset_data)
                        self.presets[preset.metadata.name] = preset
                    except Exception as e:
                        self.logger.warning(f"Failed to load preset: {e}")
                        
                self._refresh_preset_list()
                self.logger.info(f"Loaded {len(self.presets)} presets")
                
        except Exception as e:
            self.logger.error(f"Failed to load presets: {e}")
            
    def _save_presets(self) -> None:
        """Save presets to storage."""
        try:
            data = {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "presets": [preset.to_dict() for preset in self.presets.values()]
            }
            
            with open(self.preset_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved {len(self.presets)} presets")
            
        except Exception as e:
            self.logger.error(f"Failed to save presets: {e}")
            
    def _refresh_preset_list(self) -> None:
        """Refresh the preset list display."""
        self.preset_list.clear_presets()
        
        for preset in self.presets.values():
            # Apply search filter
            if self.search_text and self.search_text.lower() not in preset.metadata.name.lower():
                if not any(self.search_text.lower() in tag.lower() for tag in preset.metadata.tags):
                    continue
                    
            # Apply category filter
            if self.filter_category != "All" and preset.metadata.category != self.filter_category:
                continue
                
            self.preset_list.add_preset(preset)
            
    @Slot(str)
    def _on_search_changed(self, text: str) -> None:
        """Handle search text change."""
        self.search_text = text
        self._refresh_preset_list()
        
    @Slot(str)
    def _on_category_filter_changed(self, category: str) -> None:
        """Handle category filter change."""
        self.filter_category = category
        self._refresh_preset_list()
        
    @Slot(Preset)
    def _on_preset_selected(self, preset: Preset) -> None:
        """Handle preset selection."""
        self.load_btn.setEnabled(True)
        self.edit_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        # Update details display
        self._update_details_display(preset)
        
    @Slot(Preset)
    def _on_preset_double_clicked(self, preset: Preset) -> None:
        """Handle preset double click - load preset."""
        self._load_preset(preset)
        
    @Slot(Preset, object)
    def _on_preset_context_menu(self, preset: Preset, position) -> None:
        """Handle preset context menu."""
        menu = QMenu(self)
        
        load_action = menu.addAction("Load Preset")
        load_action.triggered.connect(lambda: self._load_preset(preset))
        
        edit_action = menu.addAction("Edit Preset")
        edit_action.triggered.connect(lambda: self._edit_preset(preset))
        
        duplicate_action = menu.addAction("Duplicate Preset")
        duplicate_action.triggered.connect(lambda: self._duplicate_preset(preset))
        
        menu.addSeparator()
        
        export_action = menu.addAction("Export Preset")
        export_action.triggered.connect(lambda: self._export_preset(preset))
        
        menu.addSeparator()
        
        delete_action = menu.addAction("Delete Preset")
        delete_action.triggered.connect(lambda: self._delete_preset(preset))
        
        menu.exec(position)
        
    @Slot()
    def _on_load_clicked(self) -> None:
        """Handle load button click."""
        preset = self.preset_list.get_selected_preset()
        if preset:
            self._load_preset(preset)
            
    @Slot()
    def _on_save_clicked(self) -> None:
        """Handle save button click."""
        if not self.current_config:
            QMessageBox.warning(self, "No Configuration", 
                              "No current configuration to save as preset.")
            return
            
        dialog = PresetEditDialog(parent=self)
        if dialog.exec() == QDialog.Accepted:
            metadata = dialog.get_metadata()
            
            if not metadata.name:
                QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty.")
                return
                
            # Check for duplicate names
            if metadata.name in self.presets:
                reply = QMessageBox.question(
                    self, "Preset Exists",
                    f"A preset named '{metadata.name}' already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
                    
            # Create and save preset
            preset = Preset(config=self.current_config, metadata=metadata)
            self.presets[metadata.name] = preset
            self._save_presets()
            self._refresh_preset_list()
            
            self.preset_saved.emit(preset)
            self.logger.info(f"Saved preset: {metadata.name}")
            
    @Slot()
    def _on_edit_clicked(self) -> None:
        """Handle edit button click."""
        preset = self.preset_list.get_selected_preset()
        if preset:
            self._edit_preset(preset)
            
    @Slot()
    def _on_delete_clicked(self) -> None:
        """Handle delete button click."""
        preset = self.preset_list.get_selected_preset()
        if preset:
            self._delete_preset(preset)
            
    @Slot()
    def _on_import_clicked(self) -> None:
        """Handle import button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Presets", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self._import_presets(file_path)
            
    @Slot()
    def _on_export_clicked(self) -> None:
        """Handle export button click."""
        preset = self.preset_list.get_selected_preset()
        if preset:
            self._export_preset(preset)
            
    @Slot()
    def _on_export_all_clicked(self) -> None:
        """Handle export all button click."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export All Presets", "sdr_presets.json", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self._export_all_presets(file_path)
            
    def _update_details_display(self, preset: Preset) -> None:
        """Update preset details display."""
        config = preset.config
        metadata = preset.metadata
        
        details = f"""<h3>{metadata.name}</h3>
<p><b>Category:</b> {metadata.category}</p>
<p><b>Description:</b> {metadata.description or 'No description'}</p>

<h4>Configuration:</h4>
<p><b>Frequency:</b> {config.center_freq_hz/1e6:.3f} MHz</p>
<p><b>Sample Rate:</b> {config.sample_rate_hz/1e6:.3f} MSps</p>
<p><b>Bandwidth:</b> {config.bandwidth_hz/1e6:.3f} MHz</p>
<p><b>Gain:</b> {config.gain_db:.1f} dB {'(AGC Enabled)' if config.agc_enabled else ''}</p>
<p><b>Mode:</b> {config.mode.value.title()}</p>

<h4>Metadata:</h4>
<p><b>Tags:</b> {', '.join(metadata.tags) if metadata.tags else 'None'}</p>
<p><b>Author:</b> {metadata.author or 'Unknown'}</p>
<p><b>Version:</b> {metadata.version}</p>
<p><b>Compatibility:</b> {', '.join(metadata.compatibility)}</p>
<p><b>Usage Count:</b> {metadata.usage_count}</p>
<p><b>Created:</b> {datetime.fromisoformat(metadata.created_date).strftime('%Y-%m-%d %H:%M')}</p>
<p><b>Modified:</b> {datetime.fromisoformat(metadata.modified_date).strftime('%Y-%m-%d %H:%M')}</p>
"""
        
        self.details_text.setHtml(details)
        
    def _load_preset(self, preset: Preset) -> None:
        """Load a preset."""
        # Increment usage count
        preset.metadata.usage_count += 1
        self._save_presets()
        self._update_details_display(preset)
        
        self.preset_loaded.emit(preset)
        self.logger.info(f"Loaded preset: {preset.metadata.name}")
        
    def _edit_preset(self, preset: Preset) -> None:
        """Edit a preset."""
        dialog = PresetEditDialog(preset, parent=self)
        if dialog.exec() == QDialog.Accepted:
            new_metadata = dialog.get_metadata()
            
            if not new_metadata.name:
                QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty.")
                return
                
            # Handle name change
            old_name = preset.metadata.name
            if new_metadata.name != old_name:
                if new_metadata.name in self.presets:
                    QMessageBox.warning(self, "Name Exists", 
                                      f"A preset named '{new_metadata.name}' already exists.")
                    return
                    
                # Remove old entry
                del self.presets[old_name]
                
            # Update preset
            preset.metadata = new_metadata
            self.presets[new_metadata.name] = preset
            
            self._save_presets()
            self._refresh_preset_list()
            self._update_details_display(preset)
            
            self.logger.info(f"Edited preset: {new_metadata.name}")
            
    def _duplicate_preset(self, preset: Preset) -> None:
        """Duplicate a preset."""
        # Create copy with new name
        new_metadata = PresetMetadata(
            name=f"{preset.metadata.name} (Copy)",
            description=preset.metadata.description,
            category=preset.metadata.category,
            tags=preset.metadata.tags.copy(),
            author=preset.metadata.author,
            compatibility=preset.metadata.compatibility.copy()
        )
        
        new_preset = Preset(config=preset.config, metadata=new_metadata)
        
        # Edit the duplicate
        dialog = PresetEditDialog(new_preset, parent=self)
        if dialog.exec() == QDialog.Accepted:
            final_metadata = dialog.get_metadata()
            
            if not final_metadata.name:
                QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty.")
                return
                
            if final_metadata.name in self.presets:
                QMessageBox.warning(self, "Name Exists", 
                                  f"A preset named '{final_metadata.name}' already exists.")
                return
                
            new_preset.metadata = final_metadata
            self.presets[final_metadata.name] = new_preset
            
            self._save_presets()
            self._refresh_preset_list()
            
            self.logger.info(f"Duplicated preset: {final_metadata.name}")
            
    def _delete_preset(self, preset: Preset) -> None:
        """Delete a preset."""
        reply = QMessageBox.question(
            self, "Delete Preset",
            f"Are you sure you want to delete the preset '{preset.metadata.name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            del self.presets[preset.metadata.name]
            self._save_presets()
            self._refresh_preset_list()
            
            # Clear selection
            self.load_btn.setEnabled(False)
            self.edit_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.details_text.clear()
            
            self.preset_deleted.emit(preset.metadata.name)
            self.logger.info(f"Deleted preset: {preset.metadata.name}")
            
    def _export_preset(self, preset: Preset) -> None:
        """Export a single preset."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Preset", f"{preset.metadata.name}.json", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                data = {
                    "version": "1.0",
                    "exported": datetime.now().isoformat(),
                    "presets": [preset.to_dict()]
                }
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                QMessageBox.information(self, "Export Successful", 
                                      f"Preset exported to {file_path}")
                self.logger.info(f"Exported preset: {preset.metadata.name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to export preset: {e}")
                self.logger.error(f"Failed to export preset: {e}")
                
    def _export_all_presets(self, file_path: str) -> None:
        """Export all presets."""
        try:
            data = {
                "version": "1.0",
                "exported": datetime.now().isoformat(),
                "presets": [preset.to_dict() for preset in self.presets.values()]
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            QMessageBox.information(self, "Export Successful", 
                                  f"All presets exported to {file_path}")
            self.logger.info(f"Exported all presets to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export presets: {e}")
            self.logger.error(f"Failed to export presets: {e}")
            
    def _import_presets(self, file_path: str) -> None:
        """Import presets from file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            imported_count = 0
            skipped_count = 0
            
            for preset_data in data.get("presets", []):
                try:
                    preset = Preset.from_dict(preset_data)
                    
                    # Check for duplicates
                    if preset.metadata.name in self.presets:
                        reply = QMessageBox.question(
                            self, "Preset Exists",
                            f"Preset '{preset.metadata.name}' already exists. Overwrite?",
                            QMessageBox.Yes | QMessageBox.No | QMessageBox.YesToAll | QMessageBox.NoToAll
                        )
                        
                        if reply == QMessageBox.No:
                            skipped_count += 1
                            continue
                        elif reply == QMessageBox.NoToAll:
                            break
                            
                    self.presets[preset.metadata.name] = preset
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to import preset: {e}")
                    skipped_count += 1
                    
            self._save_presets()
            self._refresh_preset_list()
            
            QMessageBox.information(
                self, "Import Complete",
                f"Imported {imported_count} presets. Skipped {skipped_count} presets."
            )
            self.logger.info(f"Imported {imported_count} presets from {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Failed to import presets: {e}")
            self.logger.error(f"Failed to import presets: {e}")
            
    def set_current_config(self, config: SDRConfig) -> None:
        """Set current SDR configuration for saving."""
        self.current_config = config
        
    def get_preset_count(self) -> int:
        """Get number of stored presets."""
        return len(self.presets)
        
    def get_preset_by_name(self, name: str) -> Optional[Preset]:
        """Get preset by name."""
        return self.presets.get(name)
        
    def get_presets_by_category(self, category: str) -> List[Preset]:
        """Get presets by category."""
        return [preset for preset in self.presets.values() 
                if preset.metadata.category == category]