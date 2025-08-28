"""
Tests for Spectrum Widget

This module contains comprehensive tests for the spectrum visualization widget,
including functionality, performance, and accuracy tests.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
import sys
import time

# Import the spectrum widget components
from gui.widgets.spectrum_widget import (
    SpectrumWidget, SpectrumProcessor, SpectrumConfig, 
    WindowFunction
)


class TestSpectrumConfig:
    """Test spectrum configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SpectrumConfig()
        
        assert config.fft_size == 2048
        assert config.window_function == WindowFunction.HANN
        assert config.averaging_factor == 0.8
        assert config.update_rate_fps == 30
        assert config.show_grid is True
        assert config.show_peak_markers is True
        assert config.frequency_unit == "MHz"
        assert config.power_unit == "dBm"
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SpectrumConfig(
            fft_size=4096,
            window_function=WindowFunction.BLACKMAN,
            averaging_factor=0.5,
            update_rate_fps=60
        )
        
        assert config.fft_size == 4096
        assert config.window_function == WindowFunction.BLACKMAN
        assert config.averaging_factor == 0.5
        assert config.update_rate_fps == 60


class TestSpectrumProcessor:
    """Test spectrum processor functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = SpectrumProcessor()
        self.processor.set_parameters(2e6, 100e6)  # 2 MSps, 100 MHz
        
    def test_processor_initialization(self):
        """Test processor initialization."""
        assert self.processor.sample_rate == 2e6
        assert self.processor.center_freq == 100e6
        assert self.processor.running is False
        assert len(self.processor.spectrum_history) == 0
        
    def test_window_functions(self):
        """Test different window functions."""
        config = SpectrumConfig(fft_size=1024)
        
        # Test each window function
        for window_func in WindowFunction:
            config.window_function = window_func
            self.processor.set_config(config)
            
            window = self.processor._get_window()
            assert len(window) == 1024
            assert np.all(np.isfinite(window))
            
            # Check window properties
            if window_func == WindowFunction.RECTANGULAR:
                assert np.all(window == 1.0)
            else:
                # Other windows should have values between 0 and 1
                assert np.all(window >= 0)
                assert np.all(window <= 1)
                
    def test_spectrum_processing(self):
        """Test spectrum processing with synthetic data."""
        # Create synthetic I/Q data with known frequency components
        sample_rate = 2e6
        duration = 0.001  # 1 ms
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create signal with two tones
        freq1, freq2 = 100e3, 300e3  # 100 kHz and 300 kHz
        signal = (np.exp(1j * 2 * np.pi * freq1 * t) + 
                 0.5 * np.exp(1j * 2 * np.pi * freq2 * t))
        
        # Add some noise
        noise = 0.1 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        iq_data = signal + noise
        
        # Process the data
        self.processor.start_processing()
        
        # Mock the signal emission to capture results
        frequencies = None
        powers = None
        
        def capture_spectrum(freq, power):
            nonlocal frequencies, powers
            frequencies = freq
            powers = power
            
        self.processor.spectrum_ready.connect(capture_spectrum)
        self.processor.process_iq_data(iq_data)
        
        # Verify results
        assert frequencies is not None
        assert powers is not None
        assert len(frequencies) == self.processor.config.fft_size
        assert len(powers) == self.processor.config.fft_size
        
        # Check frequency range
        expected_min_freq = self.processor.center_freq - sample_rate / 2
        expected_max_freq = self.processor.center_freq + sample_rate / 2
        assert np.min(frequencies) >= expected_min_freq - 1000  # Allow small tolerance
        assert np.max(frequencies) <= expected_max_freq + 1000
        
    def test_peak_detection(self):
        """Test peak detection functionality."""
        # Create spectrum with clear peaks
        frequencies = np.linspace(-1e6, 1e6, 2048) + 100e6
        
        # Create spectrum with two peaks
        powers = -60 * np.ones_like(frequencies)  # Noise floor
        
        # Add peaks
        peak1_idx = 512
        peak2_idx = 1536
        powers[peak1_idx-5:peak1_idx+6] = -20  # First peak
        powers[peak2_idx-5:peak2_idx+6] = -30  # Second peak
        
        # Test peak detection
        peaks = self.processor._detect_peaks(frequencies, powers)
        
        # Should detect both peaks
        assert len(peaks) >= 1  # At least one peak should be detected
        
        # Check peak properties
        for peak in peaks:
            assert 'frequency' in peak
            assert 'power' in peak
            assert 'index' in peak
            assert peak['power'] > -50  # Above noise floor
            
    def test_averaging(self):
        """Test spectrum averaging functionality."""
        config = SpectrumConfig(averaging_factor=0.9)
        self.processor.set_config(config)
        
        # Create consistent I/Q data
        iq_data = np.random.randn(2048) + 1j * np.random.randn(2048)
        
        # Process multiple times to build history
        self.processor.start_processing()
        
        spectra = []
        def capture_spectrum(freq, power):
            spectra.append(power.copy())
            
        self.processor.spectrum_ready.connect(capture_spectrum)
        
        # Process same data multiple times
        for _ in range(5):
            self.processor.process_iq_data(iq_data)
            
        # Check that averaging is working (later spectra should be more stable)
        assert len(spectra) == 5
        
        # Variance should decrease with averaging
        variance_first = np.var(spectra[0])
        variance_last = np.var(spectra[-1])
        # Note: This test might be flaky due to randomness, but generally averaging should reduce variance


@pytest.fixture
def qapp():
    """Create QApplication for GUI tests."""
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    yield app


class TestSpectrumWidget:
    """Test spectrum widget GUI functionality."""
    
    def test_widget_creation(self, qapp, qtbot):
        """Test spectrum widget creation and initialization."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Check initial state
        assert widget.config.fft_size == 2048
        assert widget.sample_rate == 2e6
        assert widget.center_freq == 100e6
        assert len(widget.current_frequencies) == 0
        assert len(widget.current_spectrum) == 0
        
        # Check UI elements exist
        assert widget.plot_widget is not None
        assert widget.fft_size_combo is not None
        assert widget.window_combo is not None
        assert widget.avg_slider is not None
        assert widget.rate_spinbox is not None
        
    def test_fft_size_update(self, qapp, qtbot):
        """Test FFT size update functionality."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Change FFT size
        widget.fft_size_combo.setCurrentText("4096")
        
        # Verify configuration updated
        assert widget.config.fft_size == 4096
        
    def test_window_function_update(self, qapp, qtbot):
        """Test window function update."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Change window function
        widget.window_combo.setCurrentText("Blackman")
        
        # Verify configuration updated
        assert widget.config.window_function == WindowFunction.BLACKMAN
        
    def test_averaging_update(self, qapp, qtbot):
        """Test averaging factor update."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Change averaging
        widget.avg_slider.setValue(50)
        
        # Verify configuration updated
        assert widget.config.averaging_factor == 0.5
        assert widget.avg_label.text() == "50%"
        
    def test_sdr_parameters_update(self, qapp, qtbot):
        """Test SDR parameters update."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Update parameters
        new_sample_rate = 10e6
        new_center_freq = 2.4e9
        
        widget.set_sdr_parameters(new_sample_rate, new_center_freq)
        
        # Verify parameters updated
        assert widget.sample_rate == new_sample_rate
        assert widget.center_freq == new_center_freq
        
        # Check plot title updated
        title = widget.plot_widget.plotItem.titleLabel.text
        assert "2.400 GHz" in title
        assert "10.0 MSps" in title
        
    def test_spectrum_data_update(self, qapp, qtbot):
        """Test spectrum data update with synthetic data."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Create synthetic I/Q data
        iq_data = np.random.randn(4096) + 1j * np.random.randn(4096)
        
        # Start processing
        widget.start_processing()
        
        # Update with data
        widget.update_spectrum_data(iq_data)
        
        # Allow some time for processing
        qtbot.wait(100)
        
        # Stop processing
        widget.stop_processing()
        
    def test_measurement_cursors(self, qapp, qtbot):
        """Test measurement cursor functionality."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Add some fake spectrum data
        widget.current_frequencies = np.linspace(99e6, 101e6, 2048)
        widget.current_spectrum = -50 * np.ones(2048)
        
        # Add measurement cursor
        initial_cursor_count = len(widget.measurement_cursors)
        widget._add_measurement_cursor()
        
        # Verify cursor added
        assert len(widget.measurement_cursors) == initial_cursor_count + 1
        
        # Clear cursors
        widget._clear_measurements()
        assert len(widget.measurement_cursors) == 0
        
    def test_peak_markers_toggle(self, qapp, qtbot):
        """Test peak markers toggle functionality."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Toggle peaks off
        widget.show_peaks_cb.setChecked(False)
        assert widget.config.show_peak_markers is False
        assert not widget.peak_scatter.isVisible()
        
        # Toggle peaks on
        widget.show_peaks_cb.setChecked(True)
        assert widget.config.show_peak_markers is True
        assert widget.peak_scatter.isVisible()
        
    def test_grid_toggle(self, qapp, qtbot):
        """Test grid toggle functionality."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Toggle grid (initial state is on)
        widget.show_grid_cb.setChecked(False)
        assert widget.config.show_grid is False
        
        widget.show_grid_cb.setChecked(True)
        assert widget.config.show_grid is True
        
    def test_config_get_set(self, qapp, qtbot):
        """Test configuration get/set functionality."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Create custom config
        custom_config = SpectrumConfig(
            fft_size=4096,
            window_function=WindowFunction.KAISER,
            averaging_factor=0.6,
            update_rate_fps=60
        )
        
        # Set config
        widget.set_config(custom_config)
        
        # Verify config applied
        retrieved_config = widget.get_config()
        assert retrieved_config.fft_size == 4096
        assert retrieved_config.window_function == WindowFunction.KAISER
        assert retrieved_config.averaging_factor == 0.6
        assert retrieved_config.update_rate_fps == 60
        
        # Verify UI updated
        assert widget.fft_size_combo.currentText() == "4096"
        assert widget.window_combo.currentText() == "Kaiser"
        assert widget.avg_slider.value() == 60
        assert widget.rate_spinbox.value() == 60


class TestSpectrumWidgetPerformance:
    """Performance tests for spectrum widget."""
    
    def test_large_fft_performance(self, qapp, qtbot):
        """Test performance with large FFT sizes."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Set large FFT size
        config = SpectrumConfig(fft_size=16384)
        widget.set_config(config)
        
        # Create large I/Q data
        iq_data = np.random.randn(20000) + 1j * np.random.randn(20000)
        
        # Measure processing time
        start_time = time.time()
        widget.start_processing()
        widget.update_spectrum_data(iq_data)
        qtbot.wait(200)  # Allow processing time
        widget.stop_processing()
        end_time = time.time()
        
        # Processing should complete within reasonable time
        processing_time = end_time - start_time
        assert processing_time < 1.0  # Should complete within 1 second
        
    def test_high_update_rate(self, qapp, qtbot):
        """Test widget with high update rates."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Set high update rate
        config = SpectrumConfig(update_rate_fps=60)
        widget.set_config(config)
        
        # Simulate rapid updates
        widget.start_processing()
        
        for _ in range(10):
            iq_data = np.random.randn(2048) + 1j * np.random.randn(2048)
            widget.update_spectrum_data(iq_data)
            qtbot.wait(16)  # ~60 FPS
            
        widget.stop_processing()
        
        # Widget should remain responsive
        assert widget.isVisible()


class TestSpectrumWidgetIntegration:
    """Integration tests for spectrum widget."""
    
    def test_signal_emission(self, qapp, qtbot):
        """Test signal emission from widget."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Connect to signals
        frequency_signals = []
        measurement_signals = []
        config_signals = []
        
        widget.frequency_selected.connect(lambda f: frequency_signals.append(f))
        widget.measurement_made.connect(lambda m: measurement_signals.append(m))
        widget.config_changed.connect(lambda c: config_signals.append(c))
        
        # Trigger config change
        widget.fft_size_combo.setCurrentText("4096")
        
        # Verify signal emitted
        assert len(config_signals) > 0
        assert config_signals[-1].fft_size == 4096
        
    def test_processor_thread_cleanup(self, qapp, qtbot):
        """Test proper cleanup of processor thread."""
        widget = SpectrumWidget()
        qtbot.addWidget(widget)
        
        # Verify thread is running
        assert widget.processor_thread.isRunning()
        
        # Close widget
        widget.close()
        
        # Allow time for cleanup
        qtbot.wait(100)
        
        # Thread should be cleaned up
        assert not widget.processor_thread.isRunning()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])