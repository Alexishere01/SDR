"""
Tests for Waterfall Widget

This module contains tests for the waterfall visualization widget,
including OpenGL rendering, color mapping, and performance tests.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import time

# Import the waterfall widget components
from gui.widgets.waterfall_widget import (
    WaterfallWidget, WaterfallRenderer, WaterfallConfig, 
    ColorMap, ColorMapGenerator
)


class TestWaterfallConfig:
    """Test waterfall configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WaterfallConfig()
        
        assert config.history_size == 1000
        assert config.color_map == ColorMap.VIRIDIS
        assert config.intensity_min == -80.0
        assert config.intensity_max == -20.0
        assert config.time_span_seconds == 10.0
        assert config.auto_scale is True
        assert config.show_colorbar is True
        assert config.interpolation is True
        assert config.update_rate_fps == 30
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WaterfallConfig(
            history_size=2000,
            color_map=ColorMap.JET,
            intensity_min=-100.0,
            intensity_max=0.0,
            auto_scale=False
        )
        
        assert config.history_size == 2000
        assert config.color_map == ColorMap.JET
        assert config.intensity_min == -100.0
        assert config.intensity_max == 0.0
        assert config.auto_scale is False


class TestColorMapGenerator:
    """Test color map generation functionality."""
    
    def test_colormap_generation(self):
        """Test color map generation for all types."""
        for colormap in ColorMap:
            colors = ColorMapGenerator.generate_colormap(colormap, 256)
            
            # Check array properties
            assert colors.shape == (256, 3)
            assert colors.dtype == np.float32
            
            # Check value ranges
            assert np.all(colors >= 0.0)
            assert np.all(colors <= 1.0)
            
            # Check that colors vary across the map (not all the same)
            if colormap != ColorMap.GRAY:
                # For non-gray colormaps, there should be variation
                assert np.std(colors) > 0.01
                
    def test_colormap_sizes(self):
        """Test color map generation with different sizes."""
        sizes = [64, 128, 256, 512]
        
        for size in sizes:
            colors = ColorMapGenerator.generate_colormap(ColorMap.VIRIDIS, size)
            assert colors.shape == (size, 3)
            assert colors.dtype == np.float32
            
    def test_specific_colormaps(self):
        """Test specific color map properties."""
        # Test grayscale
        gray_colors = ColorMapGenerator.generate_colormap(ColorMap.GRAY, 256)
        # All channels should be equal for grayscale
        for i in range(256):
            assert abs(gray_colors[i, 0] - gray_colors[i, 1]) < 1e-6
            assert abs(gray_colors[i, 1] - gray_colors[i, 2]) < 1e-6
            
        # Test jet colormap has expected color progression
        jet_colors = ColorMapGenerator.generate_colormap(ColorMap.JET, 256)
        # First color should be blue-ish (low red, high blue)
        assert jet_colors[0, 0] < 0.5  # Low red
        assert jet_colors[0, 2] > 0.5  # High blue
        
        # Last color should be red-ish (high red, low blue)
        assert jet_colors[-1, 0] > 0.5  # High red
        assert jet_colors[-1, 2] < 0.5  # Low blue


class TestWaterfallRenderer:
    """Test waterfall renderer functionality (without OpenGL context)."""
    
    def test_renderer_initialization(self):
        """Test renderer initialization."""
        # Mock OpenGL context to avoid requiring actual OpenGL
        with patch('gui.widgets.waterfall_widget.QOpenGLWidget.__init__'):
            renderer = WaterfallRenderer()
            
            # Check initial state
            assert renderer.config.history_size == 1000
            assert len(renderer.spectrum_history) == 0
            assert len(renderer.time_stamps) == 0
            assert len(renderer.frequencies) == 0
            assert renderer.zoom_factor == 1.0
            assert renderer.pan_offset_x == 0.0
            assert renderer.pan_offset_y == 0.0
            
    def test_spectrum_data_addition(self):
        """Test adding spectrum data to renderer."""
        with patch('gui.widgets.waterfall_widget.QOpenGLWidget.__init__'):
            renderer = WaterfallRenderer()
            
            # Add spectrum data
            spectrum1 = np.random.randn(1024) * 10 - 50
            spectrum2 = np.random.randn(1024) * 10 - 45
            
            renderer.add_spectrum_data(spectrum1, 1.0)
            renderer.add_spectrum_data(spectrum2, 2.0)
            
            # Check data was added
            assert len(renderer.spectrum_history) == 2
            assert len(renderer.time_stamps) == 2
            assert renderer.time_stamps[0] == 1.0
            assert renderer.time_stamps[1] == 2.0
            assert np.array_equal(renderer.spectrum_history[0], spectrum1)
            assert np.array_equal(renderer.spectrum_history[1], spectrum2)
            
    def test_history_size_limit(self):
        """Test history size limiting."""
        with patch('gui.widgets.waterfall_widget.QOpenGLWidget.__init__'):
            renderer = WaterfallRenderer()
            config = WaterfallConfig(history_size=5)
            renderer.set_config(config)
            
            # Add more spectra than history size
            for i in range(10):
                spectrum = np.random.randn(100) - 50
                renderer.add_spectrum_data(spectrum, float(i))
                
            # Should only keep last 5
            assert len(renderer.spectrum_history) == 5
            assert len(renderer.time_stamps) == 5
            assert renderer.time_stamps == [5.0, 6.0, 7.0, 8.0, 9.0]
            
    def test_frequency_setting(self):
        """Test setting frequency array."""
        with patch('gui.widgets.waterfall_widget.QOpenGLWidget.__init__'):
            renderer = WaterfallRenderer()
            
            frequencies = np.linspace(99e6, 101e6, 1024)
            renderer.set_frequencies(frequencies)
            
            assert np.array_equal(renderer.frequencies, frequencies)
            
    def test_sdr_parameters(self):
        """Test SDR parameter setting."""
        with patch('gui.widgets.waterfall_widget.QOpenGLWidget.__init__'):
            renderer = WaterfallRenderer()
            
            sample_rate = 10e6
            center_freq = 2.4e9
            
            renderer.set_sdr_parameters(sample_rate, center_freq)
            
            assert renderer.sample_rate == sample_rate
            assert renderer.center_freq == center_freq
            
    def test_zoom_operations(self):
        """Test zoom and pan operations."""
        with patch('gui.widgets.waterfall_widget.QOpenGLWidget.__init__'):
            renderer = WaterfallRenderer()
            
            # Test zoom in
            initial_zoom = renderer.zoom_factor
            renderer.zoom_in(2.0)
            assert renderer.zoom_factor == initial_zoom * 2.0
            
            # Test zoom out
            renderer.zoom_out(2.0)
            assert abs(renderer.zoom_factor - initial_zoom) < 1e-6
            
            # Test reset
            renderer.pan_offset_x = 0.5
            renderer.pan_offset_y = 0.3
            renderer.zoom_factor = 3.0
            
            renderer.reset_zoom()
            assert renderer.zoom_factor == 1.0
            assert renderer.pan_offset_x == 0.0
            assert renderer.pan_offset_y == 0.0
            
    def test_clear_history(self):
        """Test clearing waterfall history."""
        with patch('gui.widgets.waterfall_widget.QOpenGLWidget.__init__'):
            renderer = WaterfallRenderer()
            
            # Add some data
            for i in range(5):
                spectrum = np.random.randn(100)
                renderer.add_spectrum_data(spectrum, float(i))
                
            assert len(renderer.spectrum_history) == 5
            
            # Clear history
            renderer.clear_history()
            
            assert len(renderer.spectrum_history) == 0
            assert len(renderer.time_stamps) == 0


@pytest.fixture
def qapp():
    """Create QApplication for GUI tests."""
    from PySide6.QtWidgets import QApplication
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    yield app


class TestWaterfallWidget:
    """Test waterfall widget GUI functionality."""
    
    def test_widget_creation(self, qapp, qtbot):
        """Test waterfall widget creation and initialization."""
        # Mock OpenGL to avoid requiring actual OpenGL context
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer'):
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Check initial state
            assert widget.config.history_size == 1000
            assert widget.config.color_map == ColorMap.VIRIDIS
            
            # Check UI elements exist
            assert widget.waterfall_display is not None
            assert widget.colormap_combo is not None
            assert widget.history_spinbox is not None
            assert widget.intensity_min_spinbox is not None
            assert widget.intensity_max_spinbox is not None
            
    def test_colormap_update(self, qapp, qtbot):
        """Test color map update functionality."""
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer'):
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Change color map
            widget.colormap_combo.setCurrentText("Jet")
            
            # Verify configuration updated
            assert widget.config.color_map == ColorMap.JET
            
    def test_history_size_update(self, qapp, qtbot):
        """Test history size update."""
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer'):
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Change history size
            widget.history_spinbox.setValue(2000)
            
            # Verify configuration updated
            assert widget.config.history_size == 2000
            
    def test_intensity_range_update(self, qapp, qtbot):
        """Test intensity range updates."""
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer'):
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Change intensity range
            widget.intensity_min_spinbox.setValue(-100.0)
            widget.intensity_max_spinbox.setValue(0.0)
            
            # Verify configuration updated
            assert widget.config.intensity_min == -100.0
            assert widget.config.intensity_max == 0.0
            
    def test_auto_scale_toggle(self, qapp, qtbot):
        """Test auto scale toggle."""
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer'):
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Toggle auto scale
            widget.auto_scale_cb.setChecked(False)
            assert widget.config.auto_scale is False
            
            widget.auto_scale_cb.setChecked(True)
            assert widget.config.auto_scale is True
            
    def test_interpolation_toggle(self, qapp, qtbot):
        """Test interpolation toggle."""
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer'):
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Toggle interpolation
            widget.interpolation_cb.setChecked(False)
            assert widget.config.interpolation is False
            
            widget.interpolation_cb.setChecked(True)
            assert widget.config.interpolation is True
            
    def test_spectrum_data_addition(self, qapp, qtbot):
        """Test adding spectrum data to widget."""
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer') as mock_renderer:
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Create mock renderer instance
            mock_instance = Mock()
            mock_renderer.return_value = mock_instance
            widget.waterfall_display = mock_instance
            
            # Add spectrum data
            spectrum = np.random.randn(1024) * 10 - 50
            widget.add_spectrum_data(spectrum, 1.0)
            
            # Verify renderer was called
            mock_instance.add_spectrum_data.assert_called_once_with(spectrum, 1.0)
            
    def test_frequency_setting(self, qapp, qtbot):
        """Test setting frequencies."""
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer') as mock_renderer:
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Create mock renderer instance
            mock_instance = Mock()
            mock_renderer.return_value = mock_instance
            widget.waterfall_display = mock_instance
            
            # Set frequencies
            frequencies = np.linspace(99e6, 101e6, 1024)
            widget.set_frequencies(frequencies)
            
            # Verify renderer was called
            mock_instance.set_frequencies.assert_called_once()
            
    def test_sdr_parameters_setting(self, qapp, qtbot):
        """Test setting SDR parameters."""
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer') as mock_renderer:
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Create mock renderer instance
            mock_instance = Mock()
            mock_renderer.return_value = mock_instance
            widget.waterfall_display = mock_instance
            
            # Set SDR parameters
            sample_rate = 10e6
            center_freq = 2.4e9
            widget.set_sdr_parameters(sample_rate, center_freq)
            
            # Verify renderer was called
            mock_instance.set_sdr_parameters.assert_called_once_with(sample_rate, center_freq)
            
    def test_config_get_set(self, qapp, qtbot):
        """Test configuration get/set functionality."""
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer'):
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Create custom config
            custom_config = WaterfallConfig(
                history_size=2000,
                color_map=ColorMap.PLASMA,
                intensity_min=-100.0,
                intensity_max=0.0,
                auto_scale=False,
                interpolation=False
            )
            
            # Set config
            widget.set_config(custom_config)
            
            # Verify config applied
            retrieved_config = widget.get_config()
            assert retrieved_config.history_size == 2000
            assert retrieved_config.color_map == ColorMap.PLASMA
            assert retrieved_config.intensity_min == -100.0
            assert retrieved_config.intensity_max == 0.0
            assert retrieved_config.auto_scale is False
            assert retrieved_config.interpolation is False
            
            # Verify UI updated
            assert widget.colormap_combo.currentText() == "Plasma"
            assert widget.history_spinbox.value() == 2000
            assert widget.intensity_min_spinbox.value() == -100.0
            assert widget.intensity_max_spinbox.value() == 0.0
            assert widget.auto_scale_cb.isChecked() is False
            assert widget.interpolation_cb.isChecked() is False


class TestWaterfallWidgetIntegration:
    """Integration tests for waterfall widget."""
    
    def test_signal_emission(self, qapp, qtbot):
        """Test signal emission from widget."""
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer'):
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Connect to signals
            frequency_signals = []
            config_signals = []
            
            widget.frequency_selected.connect(lambda f: frequency_signals.append(f))
            widget.config_changed.connect(lambda c: config_signals.append(c))
            
            # Trigger config change
            widget.colormap_combo.setCurrentText("Jet")
            
            # Verify signal emitted
            assert len(config_signals) > 0
            assert config_signals[-1].color_map == ColorMap.JET
            
    def test_performance_with_large_data(self, qapp, qtbot):
        """Test performance with large datasets."""
        with patch('gui.widgets.waterfall_widget.WaterfallRenderer') as mock_renderer:
            widget = WaterfallWidget()
            qtbot.addWidget(widget)
            
            # Create mock renderer instance
            mock_instance = Mock()
            mock_renderer.return_value = mock_instance
            widget.waterfall_display = mock_instance
            
            # Add large amount of data
            start_time = time.time()
            
            for i in range(100):
                spectrum = np.random.randn(4096) * 10 - 50
                widget.add_spectrum_data(spectrum, float(i))
                
            end_time = time.time()
            
            # Should complete quickly
            processing_time = end_time - start_time
            assert processing_time < 1.0  # Should complete within 1 second
            
            # Verify all data was processed
            assert mock_instance.add_spectrum_data.call_count == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])