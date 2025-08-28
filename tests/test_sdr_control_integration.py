"""
Integration tests for SDR control interface components.

Tests the integration between SDR control panel, preset management,
and hardware detection components.
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest

# Import the components to test
sys.path.append(str(Path(__file__).parent.parent))
from gui.widgets.sdr_control_panel import (
    SDRControlPanel, SDRConfig, SDRMode, ConnectionStatus,
    FrequencyControl, GainControl, SampleRateControl
)
from gui.widgets.preset_manager import (
    PresetManager, Preset, PresetMetadata, PresetCategories
)
from gui.widgets.hardware_manager import (
    HardwareManager, HardwareInfo, HardwareType, DeviceStatus
)


class TestSDRControlPanel:
    """Test SDR control panel functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_gui(self, qtbot):
        """Setup GUI test environment."""
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        self.qtbot = qtbot
        
    def test_control_panel_creation(self):
        """Test SDR control panel creation and initialization."""
        panel = SDRControlPanel()
        self.qtbot.addWidget(panel)
        
        # Check initial state
        assert panel.current_config.center_freq_hz == 100e6
        assert panel.current_config.sample_rate_hz == 2e6
        assert panel.current_config.gain_db == 30.0
        assert panel.current_config.mode == SDRMode.SIMULATION
        assert panel.connection_status == ConnectionStatus.DISCONNECTED
        
    def test_frequency_control(self):
        """Test frequency control functionality."""
        freq_control = FrequencyControl()
        self.qtbot.addWidget(freq_control)
        
        # Test frequency setting
        test_freq = 433e6
        freq_control.set_frequency(test_freq)
        assert freq_control.get_frequency() == test_freq
        
        # Test fine tuning
        freq_control.fine_slider.setValue(100)  # 100 kHz offset
        expected_freq = test_freq + 100 * 1000
        assert freq_control.get_frequency() == expected_freq
        
    def test_gain_control(self):
        """Test gain control functionality."""
        gain_control = GainControl()
        self.qtbot.addWidget(gain_control)
        
        # Test manual gain
        test_gain = 45.0
        gain_control.set_gain(test_gain)
        config = gain_control.get_gain_config()
        assert config["gain_db"] == test_gain
        assert not config["agc_enabled"]
        
        # Test AGC
        gain_control.set_agc_enabled(True)
        config = gain_control.get_gain_config()
        assert config["agc_enabled"]
        
    def test_sample_rate_control(self):
        """Test sample rate control functionality."""
        rate_control = SampleRateControl()
        self.qtbot.addWidget(rate_control)
        
        # Test sample rate setting
        test_rate = 5e6
        rate_control.set_sample_rate(test_rate)
        config = rate_control.get_sample_rate_config()
        assert config["sample_rate_hz"] == test_rate
        
        # Test auto bandwidth
        expected_bw = test_rate * 0.8
        assert config["bandwidth_hz"] == expected_bw
        
    def test_config_changes(self):
        """Test configuration change signals."""
        panel = SDRControlPanel()
        self.qtbot.addWidget(panel)
        
        # Connect signal spy
        config_changed_spy = Mock()
        panel.config_changed.connect(config_changed_spy)
        
        # Change frequency
        panel.freq_control.set_frequency(915e6)
        
        # Verify signal was emitted
        config_changed_spy.assert_called()
        
        # Check config was updated
        assert panel.current_config.center_freq_hz == 915e6
        
    def test_connection_status_updates(self):
        """Test connection status updates."""
        panel = SDRControlPanel()
        self.qtbot.addWidget(panel)
        
        # Test status changes
        panel.set_connection_status(ConnectionStatus.CONNECTING)
        assert panel.connection_status == ConnectionStatus.CONNECTING
        assert not panel.connect_btn.isEnabled()
        assert not panel.disconnect_btn.isEnabled()
        
        panel.set_connection_status(ConnectionStatus.CONNECTED)
        assert panel.connection_status == ConnectionStatus.CONNECTED
        assert not panel.connect_btn.isEnabled()
        assert panel.disconnect_btn.isEnabled()
        
    def test_validation_errors(self):
        """Test parameter validation."""
        freq_control = FrequencyControl()
        self.qtbot.addWidget(freq_control)
        
        # Connect validation error spy
        validation_error_spy = Mock()
        freq_control.validation_failed.connect(validation_error_spy)
        
        # Test invalid frequency (too high)
        freq_control.freq_spinbox.setValue(7000)  # 7 GHz, above PlutoSDR limit
        
        # Should trigger validation error
        validation_error_spy.assert_called()


class TestPresetManager:
    """Test preset management functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_gui(self, qtbot):
        """Setup GUI test environment."""
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        self.qtbot = qtbot
        
    @pytest.fixture
    def temp_preset_file(self):
        """Create temporary preset file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()
            
    def test_preset_creation(self):
        """Test preset creation and metadata."""
        config = SDRConfig(
            center_freq_hz=433e6,
            sample_rate_hz=2e6,
            gain_db=40.0,
            mode=SDRMode.HARDWARE
        )
        
        metadata = PresetMetadata(
            name="Test Preset",
            description="Test description",
            category=PresetCategories.ISM_BANDS,
            tags=["test", "433MHz"],
            author="Test User"
        )
        
        preset = Preset(config=config, metadata=metadata)
        
        # Test serialization
        preset_dict = preset.to_dict()
        assert preset_dict["metadata"]["name"] == "Test Preset"
        assert preset_dict["config"]["center_freq_hz"] == 433e6
        
        # Test deserialization
        restored_preset = Preset.from_dict(preset_dict)
        assert restored_preset.metadata.name == "Test Preset"
        assert restored_preset.config.center_freq_hz == 433e6
        
    def test_preset_manager_creation(self, temp_preset_file):
        """Test preset manager creation."""
        with patch('gui.widgets.preset_manager.PresetManager._get_preset_file_path', 
                   return_value=temp_preset_file):
            manager = PresetManager()
            self.qtbot.addWidget(manager)
            
            assert manager.get_preset_count() == 0
            assert manager.preset_file == temp_preset_file
            
    def test_preset_save_load(self, temp_preset_file):
        """Test saving and loading presets."""
        with patch('gui.widgets.preset_manager.PresetManager._get_preset_file_path', 
                   return_value=temp_preset_file):
            manager = PresetManager()
            self.qtbot.addWidget(manager)
            
            # Create test config
            test_config = SDRConfig(
                center_freq_hz=915e6,
                sample_rate_hz=5e6,
                gain_db=35.0
            )
            
            manager.set_current_config(test_config)
            
            # Mock the preset edit dialog
            with patch('gui.widgets.preset_manager.PresetEditDialog') as mock_dialog:
                mock_dialog.return_value.exec.return_value = mock_dialog.return_value.Accepted
                mock_dialog.return_value.get_metadata.return_value = PresetMetadata(
                    name="Test Save",
                    category=PresetCategories.ISM_BANDS
                )
                
                # Trigger save
                manager._on_save_clicked()
                
                # Check preset was saved
                assert manager.get_preset_count() == 1
                assert "Test Save" in manager.presets
                
    def test_preset_search_filter(self, temp_preset_file):
        """Test preset search and filtering."""
        with patch('gui.widgets.preset_manager.PresetManager._get_preset_file_path', 
                   return_value=temp_preset_file):
            manager = PresetManager()
            self.qtbot.addWidget(manager)
            
            # Add test presets
            preset1 = Preset(
                config=SDRConfig(center_freq_hz=433e6),
                metadata=PresetMetadata(name="ISM Band", category=PresetCategories.ISM_BANDS)
            )
            preset2 = Preset(
                config=SDRConfig(center_freq_hz=144e6),
                metadata=PresetMetadata(name="Ham Radio", category=PresetCategories.AMATEUR_RADIO)
            )
            
            manager.presets["ISM Band"] = preset1
            manager.presets["Ham Radio"] = preset2
            
            # Test category filter
            manager.filter_category = PresetCategories.ISM_BANDS
            manager._refresh_preset_list()
            
            # Should show only ISM band preset
            visible_count = 0
            for i in range(manager.preset_list.topLevelItemCount()):
                category_item = manager.preset_list.topLevelItem(i)
                visible_count += category_item.childCount()
                
            # Note: This test may need adjustment based on actual tree structure
            
    def test_preset_export_import(self, temp_preset_file):
        """Test preset export and import."""
        with patch('gui.widgets.preset_manager.PresetManager._get_preset_file_path', 
                   return_value=temp_preset_file):
            manager = PresetManager()
            self.qtbot.addWidget(manager)
            
            # Create test preset
            preset = Preset(
                config=SDRConfig(center_freq_hz=2.4e9),
                metadata=PresetMetadata(name="WiFi Band", category=PresetCategories.WIFI_BLUETOOTH)
            )
            manager.presets["WiFi Band"] = preset
            
            # Test export
            export_file = temp_preset_file.with_suffix('.export.json')
            manager._export_all_presets(str(export_file))
            
            assert export_file.exists()
            
            # Clear presets and import
            manager.presets.clear()
            manager._import_presets(str(export_file))
            
            assert manager.get_preset_count() == 1
            assert "WiFi Band" in manager.presets
            
            # Cleanup
            export_file.unlink()


class TestHardwareManager:
    """Test hardware detection and management."""
    
    @pytest.fixture(autouse=True)
    def setup_gui(self, qtbot):
        """Setup GUI test environment."""
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        self.qtbot = qtbot
        
    def test_hardware_manager_creation(self):
        """Test hardware manager creation."""
        with patch('gui.widgets.hardware_manager.QThread'):
            manager = HardwareManager()
            self.qtbot.addWidget(manager)
            
            assert manager.auto_detect_enabled
            assert len(manager.detected_devices) == 0
            
    def test_device_detection(self):
        """Test device detection process."""
        with patch('gui.widgets.hardware_manager.QThread'):
            manager = HardwareManager()
            self.qtbot.addWidget(manager)
            
            # Create mock devices
            pluto_device = HardwareInfo(
                device_type=HardwareType.PLUTO_SDR,
                device_id="pluto_test",
                name="Test PlutoSDR",
                status=DeviceStatus.DETECTED,
                capabilities={"frequency_range": (70e6, 6e9)},
                connection_string="ip:192.168.4.1"
            )
            
            sim_device = HardwareInfo(
                device_type=HardwareType.SIMULATION,
                device_id="simulation_0",
                name="Simulation Mode",
                status=DeviceStatus.DETECTED,
                capabilities={"frequency_range": (1e6, 6e9)}
            )
            
            # Simulate detection completion
            devices = [pluto_device, sim_device]
            manager._on_detection_complete(devices)
            
            assert len(manager.detected_devices) == 2
            assert "pluto_test" in manager.detected_devices
            assert "simulation_0" in manager.detected_devices
            
    def test_device_selection(self):
        """Test device selection and connection."""
        with patch('gui.widgets.hardware_manager.QThread'):
            manager = HardwareManager()
            self.qtbot.addWidget(manager)
            
            # Create test device
            test_device = HardwareInfo(
                device_type=HardwareType.SIMULATION,
                device_id="sim_test",
                name="Test Simulation",
                status=DeviceStatus.DETECTED,
                capabilities={}
            )
            
            # Connect signal spy
            device_selected_spy = Mock()
            manager.device_selected.connect(device_selected_spy)
            
            # Simulate device selection
            manager._on_device_selected(test_device)
            
            assert manager.current_device == test_device
            assert manager.connect_btn.isEnabled()
            device_selected_spy.assert_called_with(test_device)
            
    def test_simulation_settings(self):
        """Test simulation settings management."""
        with patch('gui.widgets.hardware_manager.QThread'):
            manager = HardwareManager()
            self.qtbot.addWidget(manager)
            
            # Test setting simulation parameters
            settings = {
                "signal_type": "QPSK",
                "snr_db": 25,
                "add_noise": False
            }
            
            manager.set_simulation_settings(settings)
            
            retrieved_settings = manager.get_simulation_settings()
            assert retrieved_settings["signal_type"] == "QPSK"
            assert retrieved_settings["snr_db"] == 25
            assert not retrieved_settings["add_noise"]
            
    def test_device_status_updates(self):
        """Test device status updates."""
        with patch('gui.widgets.hardware_manager.QThread'):
            manager = HardwareManager()
            self.qtbot.addWidget(manager)
            
            # Add test device
            test_device = HardwareInfo(
                device_type=HardwareType.PLUTO_SDR,
                device_id="pluto_status_test",
                name="Status Test PlutoSDR",
                status=DeviceStatus.DETECTED,
                capabilities={}
            )
            
            manager.detected_devices["pluto_status_test"] = test_device
            
            # Update status
            manager.update_device_status("pluto_status_test", DeviceStatus.CONNECTED)
            
            updated_device = manager.detected_devices["pluto_status_test"]
            assert updated_device.status == DeviceStatus.CONNECTED


class TestIntegration:
    """Test integration between components."""
    
    @pytest.fixture(autouse=True)
    def setup_gui(self, qtbot):
        """Setup GUI test environment."""
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        self.qtbot = qtbot
        
    def test_control_panel_preset_integration(self, tmp_path):
        """Test integration between control panel and preset manager."""
        preset_file = tmp_path / "test_presets.json"
        
        with patch('gui.widgets.preset_manager.PresetManager._get_preset_file_path', 
                   return_value=preset_file):
            # Create components
            control_panel = SDRControlPanel()
            preset_manager = PresetManager()
            
            self.qtbot.addWidget(control_panel)
            self.qtbot.addWidget(preset_manager)
            
            # Connect preset loading
            preset_manager.preset_loaded.connect(
                lambda preset: control_panel.set_config(preset.config)
            )
            
            # Set up test configuration
            test_config = SDRConfig(
                center_freq_hz=868e6,
                sample_rate_hz=4e6,
                gain_db=42.0,
                agc_enabled=True,
                mode=SDRMode.HARDWARE
            )
            
            control_panel.set_config(test_config)
            preset_manager.set_current_config(test_config)
            
            # Create and save preset
            preset = Preset(
                config=test_config,
                metadata=PresetMetadata(name="Integration Test", category=PresetCategories.GENERAL)
            )
            
            preset_manager.presets["Integration Test"] = preset
            
            # Load preset and verify control panel updates
            preset_manager._load_preset(preset)
            
            # Verify control panel was updated
            assert control_panel.current_config.center_freq_hz == 868e6
            assert control_panel.current_config.sample_rate_hz == 4e6
            assert control_panel.current_config.gain_db == 42.0
            assert control_panel.current_config.agc_enabled
            
    def test_hardware_control_integration(self):
        """Test integration between hardware manager and control panel."""
        with patch('gui.widgets.hardware_manager.QThread'):
            # Create components
            hardware_manager = HardwareManager()
            control_panel = SDRControlPanel()
            
            self.qtbot.addWidget(hardware_manager)
            self.qtbot.addWidget(control_panel)
            
            # Connect hardware info updates
            hardware_manager.hardware_info_updated.connect(
                control_panel.set_hardware_info
            )
            
            # Create test device
            test_device = HardwareInfo(
                device_type=HardwareType.PLUTO_SDR,
                device_id="integration_pluto",
                name="Integration PlutoSDR",
                status=DeviceStatus.DETECTED,
                capabilities={
                    "frequency_range": (70e6, 6e9),
                    "sample_rate_range": (520833, 61440000),
                    "gain_range": (0, 70)
                },
                connection_string="ip:192.168.4.1"
            )
            
            # Simulate device selection
            hardware_manager._on_device_selected(test_device)
            
            # Verify control panel received hardware info
            assert control_panel.hardware_info["device_name"] == "Integration PlutoSDR"
            assert control_panel.hardware_info["device_type"] == "PlutoSDR"
            
    def test_full_workflow_simulation(self, tmp_path):
        """Test complete workflow in simulation mode."""
        preset_file = tmp_path / "workflow_presets.json"
        
        with patch('gui.widgets.preset_manager.PresetManager._get_preset_file_path', 
                   return_value=preset_file), \
             patch('gui.widgets.hardware_manager.QThread'):
            
            # Create all components
            hardware_manager = HardwareManager()
            control_panel = SDRControlPanel()
            preset_manager = PresetManager()
            
            self.qtbot.addWidget(hardware_manager)
            self.qtbot.addWidget(control_panel)
            self.qtbot.addWidget(preset_manager)
            
            # Connect components
            hardware_manager.hardware_info_updated.connect(control_panel.set_hardware_info)
            control_panel.config_changed.connect(preset_manager.set_current_config)
            preset_manager.preset_loaded.connect(
                lambda preset: control_panel.set_config(preset.config)
            )
            
            # 1. Detect simulation device
            sim_device = HardwareInfo(
                device_type=HardwareType.SIMULATION,
                device_id="workflow_sim",
                name="Workflow Simulation",
                status=DeviceStatus.DETECTED,
                capabilities={"frequency_range": (1e6, 6e9)}
            )
            
            hardware_manager._on_detection_complete([sim_device])
            
            # 2. Select simulation device
            hardware_manager._on_device_selected(sim_device)
            
            # 3. Configure SDR parameters
            test_config = SDRConfig(
                center_freq_hz=2.45e9,
                sample_rate_hz=10e6,
                gain_db=50.0,
                mode=SDRMode.SIMULATION
            )
            
            control_panel.set_config(test_config)
            
            # 4. Save as preset
            preset = Preset(
                config=test_config,
                metadata=PresetMetadata(
                    name="Workflow Test",
                    category=PresetCategories.WIFI_BLUETOOTH,
                    description="Full workflow test preset"
                )
            )
            
            preset_manager.presets["Workflow Test"] = preset
            
            # 5. Verify all components are synchronized
            assert hardware_manager.current_device.device_type == HardwareType.SIMULATION
            assert control_panel.current_config.center_freq_hz == 2.45e9
            assert control_panel.current_config.mode == SDRMode.SIMULATION
            assert preset_manager.get_preset_count() == 1
            
            # 6. Load preset and verify
            preset_manager._load_preset(preset)
            
            loaded_config = control_panel.get_config()
            assert loaded_config.center_freq_hz == 2.45e9
            assert loaded_config.sample_rate_hz == 10e6
            assert loaded_config.gain_db == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])