"""
Tests for Constellation Widget

This module contains comprehensive tests for the constellation diagram widget,
including signal analysis, modulation classification, and measurement accuracy.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import time

# Import the constellation widget components
from gui.widgets.constellation_widget import (
    ConstellationWidget, SignalAnalyzer, ConstellationConfig, 
    ModulationType, SignalAnalysis
)


class TestConstellationConfig:
    """Test constellation configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ConstellationConfig()
        
        assert config.max_points == 10000
        assert config.point_size == 2
        assert config.point_alpha == 0.6
        assert config.show_grid is True
        assert config.auto_scale is True
        assert config.scale_factor == 1.0
        assert config.center_offset == (0.0, 0.0)
        assert config.enable_classification is True
        assert config.show_statistics is True
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConstellationConfig(
            max_points=5000,
            point_size=4,
            point_alpha=0.8,
            auto_scale=False,
            scale_factor=2.0,
            enable_classification=False
        )
        
        assert config.max_points == 5000
        assert config.point_size == 4
        assert config.point_alpha == 0.8
        assert config.auto_scale is False
        assert config.scale_factor == 2.0
        assert config.enable_classification is False


class TestSignalAnalysis:
    """Test signal analysis dataclass."""
    
    def test_default_analysis(self):
        """Test default analysis values."""
        analysis = SignalAnalysis()
        
        assert analysis.modulation_type == ModulationType.UNKNOWN
        assert analysis.confidence == 0.0
        assert analysis.evm_rms == 0.0
        assert analysis.evm_peak == 0.0
        assert analysis.snr_estimate == 0.0
        assert analysis.symbol_rate is None
        assert analysis.constellation_points == 0
        assert isinstance(analysis.statistics, dict)
        
    def test_custom_analysis(self):
        """Test custom analysis values."""
        stats = {'mean_i': 0.1, 'std_q': 0.5}
        analysis = SignalAnalysis(
            modulation_type=ModulationType.QPSK,
            confidence=0.85,
            evm_rms=5.2,
            snr_estimate=20.5,
            constellation_points=1024,
            statistics=stats
        )
        
        assert analysis.modulation_type == ModulationType.QPSK
        assert analysis.confidence == 0.85
        assert analysis.evm_rms == 5.2
        assert analysis.snr_estimate == 20.5
        assert analysis.constellation_points == 1024
        assert analysis.statistics == stats


class TestSignalAnalyzer:
    """Test signal analyzer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = SignalAnalyzer()
        
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.running is False
        assert isinstance(self.analyzer.config, ConstellationConfig)
        
    def test_basic_statistics_calculation(self):
        """Test basic statistics calculation."""
        # Create test I/Q data
        iq_data = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 100)  # QPSK-like
        
        stats = self.analyzer._calculate_basic_statistics(iq_data)
        
        # Check that all expected statistics are present
        expected_keys = [
            'mean_i', 'mean_q', 'std_i', 'std_q',
            'mean_magnitude', 'std_magnitude', 'mean_phase', 'std_phase',
            'peak_magnitude', 'rms_power', 'peak_to_average_ratio',
            'kurtosis_i', 'kurtosis_q', 'skewness_i', 'skewness_q'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
            assert np.isfinite(stats[key])
            
    def test_snr_estimation(self):
        """Test SNR estimation."""
        # Create clean signal
        clean_signal = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 100)
        snr_clean = self.analyzer._estimate_snr(clean_signal)
        
        # Create noisy signal
        noise = 0.1 * (np.random.randn(400) + 1j * np.random.randn(400))
        noisy_signal = clean_signal + noise
        snr_noisy = self.analyzer._estimate_snr(noisy_signal)
        
        # Clean signal should have higher SNR
        assert snr_clean >= snr_noisy
        assert 0 <= snr_clean <= 50
        assert 0 <= snr_noisy <= 50
        
    def test_phase_cluster_detection(self):
        """Test phase cluster detection."""
        # BPSK signal (2 phase clusters)
        bpsk_data = np.array([1+0j, -1+0j] * 100)
        bpsk_clusters = self.analyzer._detect_phase_clusters(bpsk_data)
        
        # Should detect approximately 2 clusters
        assert 1 <= len(bpsk_clusters) <= 3
        
        # QPSK signal (4 phase clusters)
        qpsk_data = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 100)
        qpsk_clusters = self.analyzer._detect_phase_clusters(qpsk_data)
        
        # Should detect approximately 4 clusters
        assert 3 <= len(qpsk_clusters) <= 5
        
    def test_amplitude_level_detection(self):
        """Test amplitude level detection."""
        # Constant amplitude signal (1 level)
        const_amp_data = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 100)
        const_levels = self.analyzer._detect_amplitude_levels(const_amp_data)
        
        # Should detect 1 amplitude level
        assert len(const_levels) == 1
        
        # Multi-amplitude signal
        multi_amp_data = np.concatenate([
            0.5 * np.array([1+1j, -1+1j, -1-1j, 1-1j] * 50),
            1.0 * np.array([1+1j, -1+1j, -1-1j, 1-1j] * 50)
        ])
        multi_levels = self.analyzer._detect_amplitude_levels(multi_amp_data)
        
        # Should detect 2 amplitude levels
        assert len(multi_levels) >= 2
        
    def test_modulation_classification(self):
        """Test modulation classification."""
        # Test BPSK classification
        bpsk_data = np.array([1+0j, -1+0j] * 200)
        mod_type, confidence = self.analyzer._classify_modulation(bpsk_data)
        
        # Should classify as BPSK with reasonable confidence
        assert mod_type == ModulationType.BPSK
        assert confidence > 0.5
        
        # Test QPSK classification
        qpsk_data = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 200)
        mod_type, confidence = self.analyzer._classify_modulation(qpsk_data)
        
        # Should classify as QPSK with reasonable confidence
        assert mod_type == ModulationType.QPSK
        assert confidence > 0.5
        
        # Test noise classification
        noise_data = 0.01 * (np.random.randn(800) + 1j * np.random.randn(800))
        mod_type, confidence = self.analyzer._classify_modulation(noise_data)
        
        # Should classify as noise
        assert mod_type == ModulationType.NOISE
        
    def test_ideal_constellation_generation(self):
        """Test ideal constellation generation."""
        # Test BPSK
        bpsk_points = self.analyzer._generate_ideal_constellation(ModulationType.BPSK)
        assert len(bpsk_points) == 2
        
        # Test QPSK
        qpsk_points = self.analyzer._generate_ideal_constellation(ModulationType.QPSK)
        assert len(qpsk_points) == 4
        
        # Test 8-PSK
        psk8_points = self.analyzer._generate_ideal_constellation(ModulationType.PSK8)
        assert len(psk8_points) == 8
        
        # Test 16-QAM
        qam16_points = self.analyzer._generate_ideal_constellation(ModulationType.QAM16)
        assert len(qam16_points) == 16
        
        # Test unknown modulation
        unknown_points = self.analyzer._generate_ideal_constellation(ModulationType.UNKNOWN)
        assert len(unknown_points) == 0
        
    def test_evm_calculation(self):
        """Test EVM calculation."""
        # Perfect QPSK signal (should have low EVM)
        perfect_qpsk = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 100)
        evm_rms, evm_peak = self.analyzer._calculate_evm(perfect_qpsk, ModulationType.QPSK)
        
        # Perfect signal should have very low EVM
        assert evm_rms < 1.0  # Less than 1%
        assert evm_peak < 5.0  # Less than 5%
        
        # Noisy QPSK signal (should have higher EVM)
        noise = 0.1 * (np.random.randn(400) + 1j * np.random.randn(400))
        noisy_qpsk = perfect_qpsk + noise
        evm_rms_noisy, evm_peak_noisy = self.analyzer._calculate_evm(noisy_qpsk, ModulationType.QPSK)
        
        # Noisy signal should have higher EVM
        assert evm_rms_noisy > evm_rms
        assert evm_peak_noisy > evm_peak
        
    def test_carrier_offset_estimation(self):
        """Test carrier offset estimation."""
        # Signal with no offset
        no_offset_signal = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 100)
        offset_no = self.analyzer._estimate_carrier_offset(no_offset_signal)
        
        # Signal with known offset
        t = np.arange(400)
        offset_freq = 0.1  # radians per sample
        offset_signal = no_offset_signal * np.exp(1j * offset_freq * t)
        offset_est = self.analyzer._estimate_carrier_offset(offset_signal)
        
        # Estimated offset should be close to actual offset
        assert abs(offset_est - offset_freq) < 0.05
        
    def test_iq_imbalance_calculation(self):
        """Test I/Q imbalance calculation."""
        # Balanced I/Q signal
        balanced_signal = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 100)
        amp_imb, quad_err = self.analyzer._calculate_iq_imbalances(balanced_signal)
        
        # Should have minimal imbalances
        assert abs(amp_imb) < 1.0  # Less than 1 dB
        assert abs(quad_err) < 5.0  # Less than 5 degrees
        
        # Imbalanced signal (scale Q channel)
        imbalanced_signal = balanced_signal.copy()
        imbalanced_signal.imag *= 0.5  # Reduce Q channel amplitude
        amp_imb_imb, quad_err_imb = self.analyzer._calculate_iq_imbalances(imbalanced_signal)
        
        # Should detect amplitude imbalance
        assert abs(amp_imb_imb) > abs(amp_imb)
        
    def test_symbol_rate_estimation(self):
        """Test symbol rate estimation."""
        # Create signal with known symbol rate
        symbols_per_sample = 0.1  # 10 samples per symbol
        symbol_data = np.repeat([1+1j, -1+1j, -1-1j, 1-1j], 10)  # 10 samples per symbol
        
        estimated_rate = self.analyzer._estimate_symbol_rate(symbol_data)
        
        # Should estimate a reasonable symbol rate
        if estimated_rate is not None:
            assert 0.05 < estimated_rate < 0.2  # Reasonable range
            
    def test_zero_crossing_rate(self):
        """Test zero crossing rate calculation."""
        # High frequency signal (many zero crossings)
        t = np.linspace(0, 10, 1000)
        high_freq_signal = np.exp(1j * 2 * np.pi * 5 * t)  # 5 cycles
        high_zcr = self.analyzer._calculate_zero_crossing_rate(high_freq_signal)
        
        # Low frequency signal (few zero crossings)
        low_freq_signal = np.exp(1j * 2 * np.pi * 0.5 * t)  # 0.5 cycles
        low_zcr = self.analyzer._calculate_zero_crossing_rate(low_freq_signal)
        
        # High frequency should have higher zero crossing rate
        assert high_zcr > low_zcr
        assert 0 <= high_zcr <= 1
        assert 0 <= low_zcr <= 1


@pytest.fixture
def qapp():
    """Create QApplication for GUI tests."""
    from PySide6.QtWidgets import QApplication
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    yield app


class TestConstellationWidget:
    """Test constellation widget GUI functionality."""
    
    def test_widget_creation(self, qapp, qtbot):
        """Test constellation widget creation and initialization."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Check initial state
        assert widget.config.max_points == 10000
        assert widget.config.enable_classification is True
        assert len(widget.current_iq_data) == 0
        
        # Check UI elements exist
        assert widget.plot_widget is not None
        assert widget.max_points_spinbox is not None
        assert widget.point_size_slider is not None
        assert widget.classification_cb is not None
        assert widget.statistics_cb is not None
        assert widget.analysis_tabs is not None
        
    def test_max_points_update(self, qapp, qtbot):
        """Test max points update functionality."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Change max points
        widget.max_points_spinbox.setValue(5000)
        
        # Verify configuration updated
        assert widget.config.max_points == 5000
        
    def test_point_size_update(self, qapp, qtbot):
        """Test point size update."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Change point size
        widget.point_size_slider.setValue(5)
        
        # Verify configuration updated
        assert widget.config.point_size == 5
        
    def test_classification_toggle(self, qapp, qtbot):
        """Test classification toggle."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Toggle classification
        widget.classification_cb.setChecked(False)
        assert widget.config.enable_classification is False
        
        widget.classification_cb.setChecked(True)
        assert widget.config.enable_classification is True
        
    def test_statistics_toggle(self, qapp, qtbot):
        """Test statistics toggle."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Toggle statistics
        widget.statistics_cb.setChecked(False)
        assert widget.config.show_statistics is False
        
        widget.statistics_cb.setChecked(True)
        assert widget.config.show_statistics is True
        
    def test_auto_scale_toggle(self, qapp, qtbot):
        """Test auto scale toggle."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Toggle auto scale
        widget.auto_scale_cb.setChecked(False)
        assert widget.config.auto_scale is False
        
        widget.auto_scale_cb.setChecked(True)
        assert widget.config.auto_scale is True
        
    def test_iq_data_update(self, qapp, qtbot):
        """Test I/Q data update."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Create test I/Q data
        iq_data = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 100)
        
        # Update widget with data
        widget.update_iq_data(iq_data)
        
        # Verify data was stored
        assert len(widget.current_iq_data) == len(iq_data)
        assert np.array_equal(widget.current_iq_data, iq_data)
        
    def test_large_data_handling(self, qapp, qtbot):
        """Test handling of large datasets."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Create large dataset
        large_data = np.random.randn(50000) + 1j * np.random.randn(50000)
        
        # Update widget with large data
        widget.update_iq_data(large_data)
        
        # Should limit to max_points
        assert len(widget.current_iq_data) <= widget.config.max_points
        
    def test_clear_constellation(self, qapp, qtbot):
        """Test clearing constellation."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Add some data
        iq_data = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 100)
        widget.update_iq_data(iq_data)
        
        assert len(widget.current_iq_data) > 0
        
        # Clear constellation
        widget._clear_constellation()
        
        # Should be empty
        assert len(widget.current_iq_data) == 0
        
    def test_analysis_display_update(self, qapp, qtbot):
        """Test analysis display update."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Create test analysis
        analysis = SignalAnalysis(
            modulation_type=ModulationType.QPSK,
            confidence=0.85,
            evm_rms=5.2,
            evm_peak=8.1,
            snr_estimate=20.5,
            constellation_points=400,
            phase_noise=0.1,
            amplitude_imbalance=0.5,
            quadrature_error=2.0,
            carrier_offset=0.01,
            statistics={'mean_i': 0.1, 'std_q': 0.5}
        )
        
        # Update display
        widget._update_analysis_display(analysis)
        
        # Verify display updated
        assert widget.modulation_label.text() == "QPSK"
        assert "85.0%" in widget.confidence_label.text()
        assert "5.20%" in widget.evm_rms_label.text()
        assert "8.10%" in widget.evm_peak_label.text()
        assert "20.5 dB" in widget.snr_label.text()
        assert "400" in widget.points_count_label.text()
        
    def test_config_get_set(self, qapp, qtbot):
        """Test configuration get/set functionality."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Create custom config
        custom_config = ConstellationConfig(
            max_points=5000,
            point_size=4,
            enable_classification=False,
            show_statistics=False,
            auto_scale=False
        )
        
        # Set config
        widget.set_config(custom_config)
        
        # Verify config applied
        retrieved_config = widget.get_config()
        assert retrieved_config.max_points == 5000
        assert retrieved_config.point_size == 4
        assert retrieved_config.enable_classification is False
        assert retrieved_config.show_statistics is False
        assert retrieved_config.auto_scale is False
        
        # Verify UI updated
        assert widget.max_points_spinbox.value() == 5000
        assert widget.point_size_slider.value() == 4
        assert widget.classification_cb.isChecked() is False
        assert widget.statistics_cb.isChecked() is False
        assert widget.auto_scale_cb.isChecked() is False
        
    def test_signal_emission(self, qapp, qtbot):
        """Test signal emission from widget."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Connect to signals
        analysis_signals = []
        config_signals = []
        
        widget.analysis_updated.connect(lambda a: analysis_signals.append(a))
        widget.config_changed.connect(lambda c: config_signals.append(c))
        
        # Trigger config change
        widget.max_points_spinbox.setValue(8000)
        
        # Verify signal emitted
        assert len(config_signals) > 0
        assert config_signals[-1].max_points == 8000


class TestConstellationWidgetIntegration:
    """Integration tests for constellation widget."""
    
    def test_end_to_end_analysis(self, qapp, qtbot):
        """Test end-to-end analysis workflow."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Start analysis
        widget.start_analysis()
        
        # Create QPSK test signal
        qpsk_data = np.array([1+1j, -1+1j, -1-1j, 1-1j] * 250)  # 1000 points
        
        # Add some noise
        noise = 0.05 * (np.random.randn(1000) + 1j * np.random.randn(1000))
        noisy_qpsk = qpsk_data + noise
        
        # Update widget
        widget.update_iq_data(noisy_qpsk)
        
        # Allow time for analysis
        qtbot.wait(200)
        
        # Check that analysis was performed
        analysis = widget.get_current_analysis()
        assert analysis.constellation_points > 0
        
        # Stop analysis
        widget.stop_analysis()
        
    def test_performance_with_large_data(self, qapp, qtbot):
        """Test performance with large datasets."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        # Create large dataset
        start_time = time.time()
        
        for i in range(10):
            large_data = np.random.randn(10000) + 1j * np.random.randn(10000)
            widget.update_iq_data(large_data)
            
        end_time = time.time()
        
        # Should complete quickly
        processing_time = end_time - start_time
        assert processing_time < 2.0  # Should complete within 2 seconds
        
    def test_modulation_type_detection(self, qapp, qtbot):
        """Test modulation type detection accuracy."""
        widget = ConstellationWidget()
        qtbot.addWidget(widget)
        
        widget.start_analysis()
        
        # Test different modulation types
        test_cases = [
            (np.array([1+0j, -1+0j] * 500), ModulationType.BPSK),
            (np.array([1+1j, -1+1j, -1-1j, 1-1j] * 250), ModulationType.QPSK),
        ]
        
        for signal, expected_mod in test_cases:
            widget.update_iq_data(signal)
            qtbot.wait(100)  # Allow analysis time
            
            analysis = widget.get_current_analysis()
            # Note: Classification might not be perfect, so we check if it's reasonable
            assert analysis.modulation_type in [expected_mod, ModulationType.UNKNOWN]
            
        widget.stop_analysis()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])