"""
Constellation Diagram Widget

This module provides I/Q constellation visualization with automatic signal detection,
classification overlays, and comprehensive signal analysis tools.
"""

import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSlider, QComboBox, QCheckBox, QSpinBox,
                               QDoubleSpinBox, QPushButton, QGroupBox,
                               QFrame, QTableWidget, QTableWidgetItem,
                               QSplitter, QTextEdit, QTabWidget)
from PySide6.QtCore import QTimer, Signal, Slot, Qt, QThread, QObject
from PySide6.QtGui import QFont, QPen, QColor, QBrush
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import scipy.signal
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import time


class ModulationType(Enum):
    """Detected modulation types."""
    UNKNOWN = "Unknown"
    BPSK = "BPSK"
    QPSK = "QPSK"
    PSK8 = "8-PSK"
    QAM16 = "16-QAM"
    QAM64 = "64-QAM"
    QAM256 = "256-QAM"
    FSK = "FSK"
    ASK = "ASK"
    OFDM = "OFDM"
    NOISE = "Noise"


@dataclass
class ConstellationConfig:
    """Configuration for constellation display."""
    max_points: int = 10000
    point_size: int = 2
    point_alpha: float = 0.6
    show_grid: bool = True
    show_axes: bool = True
    show_reference_circles: bool = False
    auto_scale: bool = True
    scale_factor: float = 1.0
    center_offset: Tuple[float, float] = (0.0, 0.0)
    update_rate_fps: int = 30
    enable_classification: bool = True
    classification_threshold: float = 0.8
    show_statistics: bool = True


@dataclass
class SignalAnalysis:
    """Signal analysis results."""
    modulation_type: ModulationType = ModulationType.UNKNOWN
    confidence: float = 0.0
    evm_rms: float = 0.0  # Error Vector Magnitude
    evm_peak: float = 0.0
    snr_estimate: float = 0.0
    symbol_rate: Optional[float] = None
    carrier_offset: float = 0.0
    phase_noise: float = 0.0
    amplitude_imbalance: float = 0.0
    quadrature_error: float = 0.0
    constellation_points: int = 0
    statistics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.statistics is None:
            self.statistics = {}


class SignalAnalyzer(QObject):
    """Background signal analysis processor."""
    
    analysis_ready = Signal(SignalAnalysis)
    
    def __init__(self):
        super().__init__()
        self.config = ConstellationConfig()
        self.running = False
        
    def set_config(self, config: ConstellationConfig) -> None:
        """Update analyzer configuration."""
        self.config = config
        
    def analyze_iq_data(self, iq_data: np.ndarray) -> None:
        """Analyze I/Q data and detect modulation characteristics."""
        if not self.running or len(iq_data) < 100:
            return
            
        try:
            analysis = SignalAnalysis()
            
            # Basic statistics
            analysis.constellation_points = len(iq_data)
            
            # Calculate basic signal properties
            analysis.statistics = self._calculate_basic_statistics(iq_data)
            
            # Estimate SNR
            analysis.snr_estimate = self._estimate_snr(iq_data)
            
            # Detect modulation type
            if self.config.enable_classification:
                analysis.modulation_type, analysis.confidence = self._classify_modulation(iq_data)
                
            # Calculate EVM if modulation is detected
            if analysis.modulation_type != ModulationType.UNKNOWN:
                analysis.evm_rms, analysis.evm_peak = self._calculate_evm(iq_data, analysis.modulation_type)
                
            # Estimate carrier offset
            analysis.carrier_offset = self._estimate_carrier_offset(iq_data)
            
            # Calculate phase noise
            analysis.phase_noise = self._estimate_phase_noise(iq_data)
            
            # Calculate amplitude and quadrature imbalances
            analysis.amplitude_imbalance, analysis.quadrature_error = self._calculate_iq_imbalances(iq_data)
            
            # Estimate symbol rate
            analysis.symbol_rate = self._estimate_symbol_rate(iq_data)
            
            self.analysis_ready.emit(analysis)
            
        except Exception as e:
            print(f"Signal analysis error: {e}")
            
    def _calculate_basic_statistics(self, iq_data: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistical properties."""
        i_data = np.real(iq_data)
        q_data = np.imag(iq_data)
        magnitude = np.abs(iq_data)
        phase = np.angle(iq_data)
        
        stats = {
            'mean_i': np.mean(i_data),
            'mean_q': np.mean(q_data),
            'std_i': np.std(i_data),
            'std_q': np.std(q_data),
            'mean_magnitude': np.mean(magnitude),
            'std_magnitude': np.std(magnitude),
            'mean_phase': np.mean(phase),
            'std_phase': np.std(phase),
            'peak_magnitude': np.max(magnitude),
            'rms_power': np.sqrt(np.mean(magnitude**2)),
            'peak_to_average_ratio': np.max(magnitude) / np.mean(magnitude),
            'kurtosis_i': scipy.stats.kurtosis(i_data),
            'kurtosis_q': scipy.stats.kurtosis(q_data),
            'skewness_i': scipy.stats.skew(i_data),
            'skewness_q': scipy.stats.skew(q_data)
        }
        
        return stats
        
    def _estimate_snr(self, iq_data: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        try:
            # Simple SNR estimation based on signal variance
            magnitude = np.abs(iq_data)
            
            # Assume signal power is the mean of the top 10% of samples
            sorted_mag = np.sort(magnitude)
            signal_power = np.mean(sorted_mag[-len(sorted_mag)//10:])
            
            # Noise power estimated from variance of magnitude
            noise_power = np.var(magnitude)
            
            if noise_power > 0:
                snr_linear = signal_power / noise_power
                snr_db = 10 * np.log10(snr_linear)
                return max(0, min(50, snr_db))  # Clamp to reasonable range
            else:
                return 50.0  # Very high SNR
                
        except Exception:
            return 0.0
            
    def _classify_modulation(self, iq_data: np.ndarray) -> Tuple[ModulationType, float]:
        """Classify modulation type using statistical features."""
        try:
            # Extract features for classification
            features = self._extract_modulation_features(iq_data)
            
            # Simple rule-based classification
            confidence = 0.0
            modulation = ModulationType.UNKNOWN
            
            # Check for noise (very low signal power, high randomness)
            if features['signal_power'] < 0.01:
                return ModulationType.NOISE, 0.9
                
            # Check for PSK modulations based on phase clustering
            phase_clusters = self._detect_phase_clusters(iq_data)
            
            if len(phase_clusters) == 2:
                modulation = ModulationType.BPSK
                confidence = 0.8
            elif len(phase_clusters) == 4:
                modulation = ModulationType.QPSK
                confidence = 0.8
            elif len(phase_clusters) == 8:
                modulation = ModulationType.PSK8
                confidence = 0.7
                
            # Check for QAM based on amplitude and phase clustering
            if modulation == ModulationType.UNKNOWN:
                amplitude_levels = self._detect_amplitude_levels(iq_data)
                
                if len(amplitude_levels) >= 2 and len(phase_clusters) >= 4:
                    total_points = len(amplitude_levels) * len(phase_clusters)
                    
                    if 14 <= total_points <= 18:
                        modulation = ModulationType.QAM16
                        confidence = 0.7
                    elif 60 <= total_points <= 68:
                        modulation = ModulationType.QAM64
                        confidence = 0.6
                    elif 250 <= total_points <= 260:
                        modulation = ModulationType.QAM256
                        confidence = 0.5
                        
            # Check for FSK based on frequency content
            if modulation == ModulationType.UNKNOWN:
                if self._detect_fsk_characteristics(iq_data):
                    modulation = ModulationType.FSK
                    confidence = 0.6
                    
            # Check for OFDM based on spectral characteristics
            if modulation == ModulationType.UNKNOWN:
                if self._detect_ofdm_characteristics(iq_data):
                    modulation = ModulationType.OFDM
                    confidence = 0.5
                    
            return modulation, confidence
            
        except Exception:
            return ModulationType.UNKNOWN, 0.0
            
    def _extract_modulation_features(self, iq_data: np.ndarray) -> Dict[str, float]:
        """Extract features for modulation classification."""
        magnitude = np.abs(iq_data)
        phase = np.angle(iq_data)
        
        features = {
            'signal_power': np.mean(magnitude**2),
            'magnitude_variance': np.var(magnitude),
            'phase_variance': np.var(phase),
            'magnitude_kurtosis': scipy.stats.kurtosis(magnitude),
            'phase_kurtosis': scipy.stats.kurtosis(phase),
            'peak_to_average_ratio': np.max(magnitude) / np.mean(magnitude),
            'zero_crossing_rate': self._calculate_zero_crossing_rate(iq_data)
        }
        
        return features
        
    def _detect_phase_clusters(self, iq_data: np.ndarray, max_clusters: int = 16) -> List[float]:
        """Detect phase clusters in the constellation."""
        try:
            phase = np.angle(iq_data)
            
            # Use clustering to find phase levels
            best_clusters = []
            best_score = -np.inf
            
            for n_clusters in range(2, min(max_clusters + 1, len(iq_data) // 10)):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(phase.reshape(-1, 1))
                    score = kmeans.score(phase.reshape(-1, 1))
                    
                    if score > best_score:
                        best_score = score
                        best_clusters = sorted(kmeans.cluster_centers_.flatten())
                        
                except Exception:
                    continue
                    
            return best_clusters
            
        except Exception:
            return []
            
    def _detect_amplitude_levels(self, iq_data: np.ndarray, max_levels: int = 8) -> List[float]:
        """Detect amplitude levels in the constellation."""
        try:
            magnitude = np.abs(iq_data)
            
            # Use clustering to find amplitude levels
            best_levels = []
            best_score = -np.inf
            
            for n_levels in range(1, min(max_levels + 1, len(iq_data) // 20)):
                try:
                    kmeans = KMeans(n_clusters=n_levels, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(magnitude.reshape(-1, 1))
                    score = kmeans.score(magnitude.reshape(-1, 1))
                    
                    if score > best_score:
                        best_score = score
                        best_levels = sorted(kmeans.cluster_centers_.flatten())
                        
                except Exception:
                    continue
                    
            return best_levels
            
        except Exception:
            return []
            
    def _detect_fsk_characteristics(self, iq_data: np.ndarray) -> bool:
        """Detect FSK modulation characteristics."""
        try:
            # FSK typically has constant amplitude and frequency shifts
            magnitude = np.abs(iq_data)
            magnitude_variation = np.std(magnitude) / np.mean(magnitude)
            
            # Low amplitude variation suggests FSK
            return magnitude_variation < 0.1
            
        except Exception:
            return False
            
    def _detect_ofdm_characteristics(self, iq_data: np.ndarray) -> bool:
        """Detect OFDM modulation characteristics."""
        try:
            # OFDM typically has high peak-to-average ratio
            magnitude = np.abs(iq_data)
            papr = np.max(magnitude) / np.mean(magnitude)
            
            # High PAPR suggests OFDM
            return papr > 3.0
            
        except Exception:
            return False
            
    def _calculate_evm(self, iq_data: np.ndarray, modulation: ModulationType) -> Tuple[float, float]:
        """Calculate Error Vector Magnitude."""
        try:
            # Generate ideal constellation points based on modulation type
            ideal_points = self._generate_ideal_constellation(modulation)
            
            if len(ideal_points) == 0:
                return 0.0, 0.0
                
            # Find nearest ideal point for each received point
            errors = []
            
            for point in iq_data:
                distances = [abs(point - ideal) for ideal in ideal_points]
                min_distance = min(distances)
                errors.append(min_distance)
                
            errors = np.array(errors)
            
            # Calculate RMS and peak EVM
            signal_power = np.mean(np.abs(iq_data)**2)
            
            if signal_power > 0:
                evm_rms = np.sqrt(np.mean(errors**2)) / np.sqrt(signal_power) * 100
                evm_peak = np.max(errors) / np.sqrt(signal_power) * 100
            else:
                evm_rms = 0.0
                evm_peak = 0.0
                
            return evm_rms, evm_peak
            
        except Exception:
            return 0.0, 0.0
            
    def _generate_ideal_constellation(self, modulation: ModulationType) -> List[complex]:
        """Generate ideal constellation points for a modulation type."""
        if modulation == ModulationType.BPSK:
            return [1+0j, -1+0j]
        elif modulation == ModulationType.QPSK:
            return [1+1j, 1-1j, -1+1j, -1-1j]
        elif modulation == ModulationType.PSK8:
            points = []
            for k in range(8):
                angle = 2 * np.pi * k / 8
                points.append(np.exp(1j * angle))
            return points
        elif modulation == ModulationType.QAM16:
            points = []
            for i in [-3, -1, 1, 3]:
                for q in [-3, -1, 1, 3]:
                    points.append(i + 1j*q)
            return points
        else:
            return []
            
    def _estimate_carrier_offset(self, iq_data: np.ndarray) -> float:
        """Estimate carrier frequency offset."""
        try:
            # Simple carrier offset estimation using phase derivative
            phase = np.angle(iq_data)
            phase_diff = np.diff(np.unwrap(phase))
            carrier_offset = np.mean(phase_diff)
            return carrier_offset
            
        except Exception:
            return 0.0
            
    def _estimate_phase_noise(self, iq_data: np.ndarray) -> float:
        """Estimate phase noise."""
        try:
            phase = np.angle(iq_data)
            phase_unwrapped = np.unwrap(phase)
            
            # Remove linear trend (carrier offset)
            t = np.arange(len(phase_unwrapped))
            coeffs = np.polyfit(t, phase_unwrapped, 1)
            phase_detrended = phase_unwrapped - np.polyval(coeffs, t)
            
            # Phase noise is the standard deviation of detrended phase
            phase_noise = np.std(phase_detrended)
            return phase_noise
            
        except Exception:
            return 0.0
            
    def _calculate_iq_imbalances(self, iq_data: np.ndarray) -> Tuple[float, float]:
        """Calculate I/Q amplitude and quadrature imbalances."""
        try:
            i_data = np.real(iq_data)
            q_data = np.imag(iq_data)
            
            # Amplitude imbalance
            i_power = np.mean(i_data**2)
            q_power = np.mean(q_data**2)
            
            if q_power > 0:
                amplitude_imbalance = 10 * np.log10(i_power / q_power)
            else:
                amplitude_imbalance = 0.0
                
            # Quadrature error (deviation from 90 degrees)
            correlation = np.corrcoef(i_data, q_data)[0, 1]
            quadrature_error = np.arccos(abs(correlation)) * 180 / np.pi - 90
            
            return amplitude_imbalance, quadrature_error
            
        except Exception:
            return 0.0, 0.0
            
    def _estimate_symbol_rate(self, iq_data: np.ndarray) -> Optional[float]:
        """Estimate symbol rate from signal characteristics."""
        try:
            # Simple symbol rate estimation using autocorrelation
            magnitude = np.abs(iq_data)
            
            # Calculate autocorrelation
            autocorr = np.correlate(magnitude, magnitude, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            peaks, _ = scipy.signal.find_peaks(autocorr[1:], height=0.1*np.max(autocorr))
            
            if len(peaks) > 0:
                # Symbol rate is inverse of the first significant peak
                symbol_period = peaks[0] + 1  # +1 because we started from index 1
                symbol_rate = 1.0 / symbol_period  # Normalized rate
                return symbol_rate
            else:
                return None
                
        except Exception:
            return None
            
    def _calculate_zero_crossing_rate(self, iq_data: np.ndarray) -> float:
        """Calculate zero crossing rate."""
        try:
            i_data = np.real(iq_data)
            q_data = np.imag(iq_data)
            
            # Count zero crossings in I and Q channels
            i_crossings = np.sum(np.diff(np.sign(i_data)) != 0)
            q_crossings = np.sum(np.diff(np.sign(q_data)) != 0)
            
            total_crossings = i_crossings + q_crossings
            crossing_rate = total_crossings / (2 * len(iq_data))
            
            return crossing_rate
            
        except Exception:
            return 0.0
            
    def start_analysis(self) -> None:
        """Start signal analysis."""
        self.running = True
        
    def stop_analysis(self) -> None:
        """Stop signal analysis."""
        self.running = False


class ConstellationWidget(QWidget):
    """I/Q constellation diagram display with signal analysis."""
    
    # Signals
    analysis_updated = Signal(SignalAnalysis)
    config_changed = Signal(ConstellationConfig)
    measurement_made = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configuration
        self.config = ConstellationConfig()
        
        # Data storage
        self.current_iq_data = np.array([])
        self.current_analysis = SignalAnalysis()
        
        # Setup analyzer thread
        self.analyzer_thread = QThread()
        self.analyzer = SignalAnalyzer()
        self.analyzer.moveToThread(self.analyzer_thread)
        self.analyzer_thread.start()
        
        self._setup_ui()
        self._setup_plot()
        self._setup_connections()
        
    def _setup_ui(self) -> None:
        """Setup the constellation widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for plot and analysis
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left side: constellation plot and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Control panel
        controls = self._create_controls()
        left_layout.addWidget(controls)
        
        # Constellation plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Quadrature (Q)')
        self.plot_widget.setLabel('bottom', 'In-phase (I)')
        self.plot_widget.showGrid(True, True, alpha=0.3)
        self.plot_widget.setBackground('k')
        self.plot_widget.setAspectLocked(True)
        
        # Set equal axis scaling
        self.plot_widget.setXRange(-2, 2)
        self.plot_widget.setYRange(-2, 2)
        
        left_layout.addWidget(self.plot_widget)
        splitter.addWidget(left_widget)
        
        # Right side: analysis results
        analysis_widget = self._create_analysis_widget()
        splitter.addWidget(analysis_widget)
        
        # Set splitter proportions
        splitter.setSizes([600, 400])
        
        # Status bar
        self.status_label = QLabel("Constellation Display - Ready")
        self.status_label.setStyleSheet("QLabel { color: #888888; font-size: 10px; }")
        layout.addWidget(self.status_label)
        
    def _create_controls(self) -> QWidget:
        """Create constellation control panel."""
        controls = QFrame()
        controls.setFrameStyle(QFrame.StyledPanel)
        controls.setMaximumHeight(100)
        
        layout = QHBoxLayout(controls)
        
        # Display Settings Group
        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout(display_group)
        
        # Max points
        points_layout = QHBoxLayout()
        points_layout.addWidget(QLabel("Max Points:"))
        self.max_points_spinbox = QSpinBox()
        self.max_points_spinbox.setRange(1000, 100000)
        self.max_points_spinbox.setValue(self.config.max_points)
        self.max_points_spinbox.valueChanged.connect(self._update_max_points)
        points_layout.addWidget(self.max_points_spinbox)
        display_layout.addLayout(points_layout)
        
        # Point size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Point Size:"))
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(1, 10)
        self.point_size_slider.setValue(self.config.point_size)
        self.point_size_slider.valueChanged.connect(self._update_point_size)
        size_layout.addWidget(self.point_size_slider)
        display_layout.addLayout(size_layout)
        
        layout.addWidget(display_group)
        
        # Analysis Settings Group
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Enable classification
        self.classification_cb = QCheckBox("Auto Classification")
        self.classification_cb.setChecked(self.config.enable_classification)
        self.classification_cb.toggled.connect(self._toggle_classification)
        analysis_layout.addWidget(self.classification_cb)
        
        # Show statistics
        self.statistics_cb = QCheckBox("Show Statistics")
        self.statistics_cb.setChecked(self.config.show_statistics)
        self.statistics_cb.toggled.connect(self._toggle_statistics)
        analysis_layout.addWidget(self.statistics_cb)
        
        # Auto scale
        self.auto_scale_cb = QCheckBox("Auto Scale")
        self.auto_scale_cb.setChecked(self.config.auto_scale)
        self.auto_scale_cb.toggled.connect(self._toggle_auto_scale)
        analysis_layout.addWidget(self.auto_scale_cb)
        
        layout.addWidget(analysis_group)
        
        # Control Buttons
        button_group = QGroupBox("Controls")
        button_layout = QVBoxLayout(button_group)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_constellation)
        button_layout.addWidget(self.clear_btn)
        
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self._trigger_analysis)
        button_layout.addWidget(self.analyze_btn)
        
        layout.addWidget(button_group)
        
        return controls
        
    def _create_analysis_widget(self) -> QWidget:
        """Create analysis results widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Tab widget for different analysis views
        self.analysis_tabs = QTabWidget()
        
        # Classification tab
        classification_tab = self._create_classification_tab()
        self.analysis_tabs.addTab(classification_tab, "Classification")
        
        # Statistics tab
        statistics_tab = self._create_statistics_tab()
        self.analysis_tabs.addTab(statistics_tab, "Statistics")
        
        # Quality metrics tab
        quality_tab = self._create_quality_tab()
        self.analysis_tabs.addTab(quality_tab, "Quality")
        
        layout.addWidget(self.analysis_tabs)
        
        return widget
        
    def _create_classification_tab(self) -> QWidget:
        """Create modulation classification tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Modulation type display
        mod_layout = QHBoxLayout()
        mod_layout.addWidget(QLabel("Modulation:"))
        self.modulation_label = QLabel("Unknown")
        self.modulation_label.setStyleSheet("QLabel { font-weight: bold; color: #00ff00; }")
        mod_layout.addWidget(self.modulation_label)
        layout.addLayout(mod_layout)
        
        # Confidence display
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.confidence_label = QLabel("0.0%")
        conf_layout.addWidget(self.confidence_label)
        layout.addLayout(conf_layout)
        
        # Symbol rate
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Symbol Rate:"))
        self.symbol_rate_label = QLabel("Unknown")
        rate_layout.addWidget(self.symbol_rate_label)
        layout.addLayout(rate_layout)
        
        # Constellation points
        points_layout = QHBoxLayout()
        points_layout.addWidget(QLabel("Points:"))
        self.points_count_label = QLabel("0")
        points_layout.addWidget(self.points_count_label)
        layout.addLayout(points_layout)
        
        layout.addStretch()
        
        return widget
        
    def _create_statistics_tab(self) -> QWidget:
        """Create statistics display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Statistics table
        self.statistics_table = QTableWidget()
        self.statistics_table.setColumnCount(2)
        self.statistics_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.statistics_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.statistics_table)
        
        return widget
        
    def _create_quality_tab(self) -> QWidget:
        """Create signal quality metrics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # EVM display
        evm_layout = QHBoxLayout()
        evm_layout.addWidget(QLabel("EVM RMS:"))
        self.evm_rms_label = QLabel("0.0%")
        evm_layout.addWidget(self.evm_rms_label)
        layout.addLayout(evm_layout)
        
        evm_peak_layout = QHBoxLayout()
        evm_peak_layout.addWidget(QLabel("EVM Peak:"))
        self.evm_peak_label = QLabel("0.0%")
        evm_peak_layout.addWidget(self.evm_peak_label)
        layout.addLayout(evm_peak_layout)
        
        # SNR display
        snr_layout = QHBoxLayout()
        snr_layout.addWidget(QLabel("SNR Estimate:"))
        self.snr_label = QLabel("0.0 dB")
        snr_layout.addWidget(self.snr_label)
        layout.addLayout(snr_layout)
        
        # Phase noise
        phase_layout = QHBoxLayout()
        phase_layout.addWidget(QLabel("Phase Noise:"))
        self.phase_noise_label = QLabel("0.0 rad")
        phase_layout.addWidget(self.phase_noise_label)
        layout.addLayout(phase_layout)
        
        # I/Q imbalances
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(QLabel("Amplitude Imbalance:"))
        self.amp_imbalance_label = QLabel("0.0 dB")
        amp_layout.addWidget(self.amp_imbalance_label)
        layout.addLayout(amp_layout)
        
        quad_layout = QHBoxLayout()
        quad_layout.addWidget(QLabel("Quadrature Error:"))
        self.quad_error_label = QLabel("0.0°")
        quad_layout.addWidget(self.quad_error_label)
        layout.addLayout(quad_layout)
        
        # Carrier offset
        carrier_layout = QHBoxLayout()
        carrier_layout.addWidget(QLabel("Carrier Offset:"))
        self.carrier_offset_label = QLabel("0.0 rad/sample")
        carrier_layout.addWidget(self.carrier_offset_label)
        layout.addLayout(carrier_layout)
        
        layout.addStretch()
        
        return widget
        
    def _setup_plot(self) -> None:
        """Setup the constellation plot."""
        # Main constellation scatter plot
        self.constellation_scatter = pg.ScatterPlotItem(
            size=self.config.point_size,
            brush=pg.mkBrush(255, 255, 0, int(255 * self.config.point_alpha)),
            pen=None
        )
        self.plot_widget.addItem(self.constellation_scatter)
        
        # Reference circles (optional)
        self.reference_circles = []
        
        # Ideal constellation overlay (for EVM calculation)
        self.ideal_scatter = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkBrush(255, 0, 0, 150),
            pen=pg.mkPen('r', width=2),
            symbol='o'
        )
        self.plot_widget.addItem(self.ideal_scatter)
        self.ideal_scatter.setVisible(False)
        
    def _setup_connections(self) -> None:
        """Setup signal-slot connections."""
        # Connect analyzer signals
        self.analyzer.analysis_ready.connect(self._update_analysis_display)
        
        # Update analyzer configuration
        self.analyzer.set_config(self.config)
        
    @Slot(int)
    def _update_max_points(self, value: int) -> None:
        """Update maximum points to display."""
        self.config.max_points = value
        self.analyzer.set_config(self.config)
        self.config_changed.emit(self.config)
        
    @Slot(int)
    def _update_point_size(self, value: int) -> None:
        """Update point size."""
        self.config.point_size = value
        self.constellation_scatter.setSize(value)
        self.config_changed.emit(self.config)
        
    @Slot(bool)
    def _toggle_classification(self, enabled: bool) -> None:
        """Toggle automatic classification."""
        self.config.enable_classification = enabled
        self.analyzer.set_config(self.config)
        
    @Slot(bool)
    def _toggle_statistics(self, enabled: bool) -> None:
        """Toggle statistics display."""
        self.config.show_statistics = enabled
        
    @Slot(bool)
    def _toggle_auto_scale(self, enabled: bool) -> None:
        """Toggle auto scaling."""
        self.config.auto_scale = enabled
        
    def _clear_constellation(self) -> None:
        """Clear constellation display."""
        self.constellation_scatter.clear()
        self.current_iq_data = np.array([])
        self.status_label.setText("Constellation Display - Cleared")
        
    def _trigger_analysis(self) -> None:
        """Trigger manual analysis of current data."""
        if len(self.current_iq_data) > 0:
            self.analyzer.analyze_iq_data(self.current_iq_data)
            
    @Slot(SignalAnalysis)
    def _update_analysis_display(self, analysis: SignalAnalysis) -> None:
        """Update analysis display with new results."""
        self.current_analysis = analysis
        
        # Update classification tab
        self.modulation_label.setText(analysis.modulation_type.value)
        self.confidence_label.setText(f"{analysis.confidence * 100:.1f}%")
        
        if analysis.symbol_rate is not None:
            self.symbol_rate_label.setText(f"{analysis.symbol_rate:.3f}")
        else:
            self.symbol_rate_label.setText("Unknown")
            
        self.points_count_label.setText(str(analysis.constellation_points))
        
        # Update quality tab
        self.evm_rms_label.setText(f"{analysis.evm_rms:.2f}%")
        self.evm_peak_label.setText(f"{analysis.evm_peak:.2f}%")
        self.snr_label.setText(f"{analysis.snr_estimate:.1f} dB")
        self.phase_noise_label.setText(f"{analysis.phase_noise:.4f} rad")
        self.amp_imbalance_label.setText(f"{analysis.amplitude_imbalance:.2f} dB")
        self.quad_error_label.setText(f"{analysis.quadrature_error:.2f}°")
        self.carrier_offset_label.setText(f"{analysis.carrier_offset:.6f} rad/sample")
        
        # Update statistics table
        if self.config.show_statistics and analysis.statistics:
            self._update_statistics_table(analysis.statistics)
            
        # Show ideal constellation if modulation is detected
        if analysis.modulation_type != ModulationType.UNKNOWN:
            self._show_ideal_constellation(analysis.modulation_type)
        else:
            self.ideal_scatter.setVisible(False)
            
        # Emit analysis signal
        self.analysis_updated.emit(analysis)
        
    def _update_statistics_table(self, statistics: Dict[str, float]) -> None:
        """Update statistics table with new data."""
        self.statistics_table.setRowCount(len(statistics))
        
        for row, (param, value) in enumerate(statistics.items()):
            param_item = QTableWidgetItem(param.replace('_', ' ').title())
            
            # Format value based on parameter type
            if 'phase' in param.lower():
                value_item = QTableWidgetItem(f"{value:.4f}")
            elif 'power' in param.lower() or 'magnitude' in param.lower():
                value_item = QTableWidgetItem(f"{value:.6f}")
            else:
                value_item = QTableWidgetItem(f"{value:.3f}")
                
            self.statistics_table.setItem(row, 0, param_item)
            self.statistics_table.setItem(row, 1, value_item)
            
    def _show_ideal_constellation(self, modulation: ModulationType) -> None:
        """Show ideal constellation points for comparison."""
        ideal_points = self.analyzer._generate_ideal_constellation(modulation)
        
        if ideal_points:
            x_coords = [np.real(point) for point in ideal_points]
            y_coords = [np.imag(point) for point in ideal_points]
            
            self.ideal_scatter.setData(x_coords, y_coords)
            self.ideal_scatter.setVisible(True)
        else:
            self.ideal_scatter.setVisible(False)
            
    @Slot(np.ndarray)
    def update_iq_data(self, iq_data: np.ndarray) -> None:
        """Update constellation with new I/Q data."""
        if len(iq_data) == 0:
            return
            
        # Limit number of points for performance
        if len(iq_data) > self.config.max_points:
            # Take evenly spaced samples
            indices = np.linspace(0, len(iq_data) - 1, self.config.max_points, dtype=int)
            iq_data = iq_data[indices]
            
        self.current_iq_data = iq_data.copy()
        
        # Extract I and Q components
        i_data = np.real(iq_data)
        q_data = np.imag(iq_data)
        
        # Auto scale if enabled
        if self.config.auto_scale:
            max_val = max(np.max(np.abs(i_data)), np.max(np.abs(q_data)))
            if max_val > 0:
                scale = 1.5 / max_val  # Leave some margin
                self.plot_widget.setXRange(-1.5, 1.5)
                self.plot_widget.setYRange(-1.5, 1.5)
            else:
                scale = 1.0
        else:
            scale = self.config.scale_factor
            
        # Apply scaling and offset
        i_scaled = i_data * scale + self.config.center_offset[0]
        q_scaled = q_data * scale + self.config.center_offset[1]
        
        # Update scatter plot
        self.constellation_scatter.setData(i_scaled, q_scaled)
        
        # Trigger analysis if enabled
        if self.config.enable_classification:
            self.analyzer.analyze_iq_data(iq_data)
            
        # Update status
        rms_power = np.sqrt(np.mean(np.abs(iq_data)**2))
        peak_power = np.max(np.abs(iq_data))
        self.status_label.setText(
            f"Points: {len(iq_data)} | RMS: {rms_power:.3f} | Peak: {peak_power:.3f} | "
            f"Modulation: {self.current_analysis.modulation_type.value}"
        )
        
    def start_analysis(self) -> None:
        """Start signal analysis."""
        self.analyzer.start_analysis()
        
    def stop_analysis(self) -> None:
        """Stop signal analysis."""
        self.analyzer.stop_analysis()
        
    def get_config(self) -> ConstellationConfig:
        """Get current constellation configuration."""
        return self.config
        
    def set_config(self, config: ConstellationConfig) -> None:
        """Set constellation configuration."""
        self.config = config
        self.analyzer.set_config(config)
        
        # Update UI controls
        self.max_points_spinbox.setValue(config.max_points)
        self.point_size_slider.setValue(config.point_size)
        self.classification_cb.setChecked(config.enable_classification)
        self.statistics_cb.setChecked(config.show_statistics)
        self.auto_scale_cb.setChecked(config.auto_scale)
        
        # Update plot settings
        self.constellation_scatter.setSize(config.point_size)
        
    def get_current_analysis(self) -> SignalAnalysis:
        """Get current signal analysis results."""
        return self.current_analysis
        
    def closeEvent(self, event) -> None:
        """Handle widget close event."""
        self.stop_analysis()
        if hasattr(self, 'analyzer_thread'):
            self.analyzer_thread.quit()
            self.analyzer_thread.wait()
        event.accept()