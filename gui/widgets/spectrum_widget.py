"""
Spectrum Visualization Widget

This module provides high-performance spectrum visualization with PyQtGraph integration,
configurable FFT processing, and interactive measurement tools.
"""

import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSlider, QComboBox, QCheckBox, QSpinBox,
                               QDoubleSpinBox, QPushButton, QGroupBox,
                               QSplitter, QFrame)
from PySide6.QtCore import QTimer, Signal, Slot, Qt, QThread, QObject
from PySide6.QtGui import QFont, QPen, QColor
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import scipy.signal
from dataclasses import dataclass
from enum import Enum


class WindowFunction(Enum):
    """Available FFT window functions."""
    RECTANGULAR = "rectangular"
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    KAISER = "kaiser"
    FLATTOP = "flattop"


@dataclass
class SpectrumConfig:
    """Configuration for spectrum display."""
    fft_size: int = 2048
    window_function: WindowFunction = WindowFunction.HANN
    averaging_factor: float = 0.8
    update_rate_fps: int = 30
    show_grid: bool = True
    show_peak_markers: bool = True
    show_measurements: bool = True
    frequency_unit: str = "MHz"  # Hz, kHz, MHz, GHz
    power_unit: str = "dBm"  # dBm, dBFS, linear


class SpectrumProcessor(QObject):
    """Background spectrum processing thread."""
    
    spectrum_ready = Signal(np.ndarray, np.ndarray)  # frequencies, powers
    peaks_detected = Signal(list)  # peak frequencies and powers
    
    def __init__(self):
        super().__init__()
        self.config = SpectrumConfig()
        self.sample_rate = 2e6
        self.center_freq = 100e6
        self.running = False
        self.spectrum_history = []
        
    def set_config(self, config: SpectrumConfig) -> None:
        """Update spectrum processing configuration."""
        self.config = config
        
    def set_parameters(self, sample_rate: float, center_freq: float) -> None:
        """Set SDR parameters."""
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        
    def process_iq_data(self, iq_data: np.ndarray) -> None:
        """Process I/Q data and generate spectrum."""
        if not self.running or len(iq_data) < self.config.fft_size:
            return
            
        # Apply window function
        window = self._get_window()
        
        # Compute FFT
        windowed_data = iq_data[:self.config.fft_size] * window
        fft_result = np.fft.fftshift(np.fft.fft(windowed_data))
        
        # Convert to power spectrum (dBm)
        power_spectrum = 20 * np.log10(np.abs(fft_result) + 1e-12)
        
        # Apply averaging
        if len(self.spectrum_history) > 0:
            alpha = 1.0 - self.config.averaging_factor
            power_spectrum = (alpha * power_spectrum + 
                            self.config.averaging_factor * self.spectrum_history[-1])
        
        # Store in history
        self.spectrum_history.append(power_spectrum.copy())
        if len(self.spectrum_history) > 100:  # Keep last 100 spectra
            self.spectrum_history.pop(0)
            
        # Generate frequency array
        frequencies = np.fft.fftshift(np.fft.fftfreq(self.config.fft_size, 1/self.sample_rate))
        frequencies += self.center_freq
        
        # Detect peaks if enabled
        peaks = []
        if self.config.show_peak_markers:
            peaks = self._detect_peaks(frequencies, power_spectrum)
            
        # Emit results
        self.spectrum_ready.emit(frequencies, power_spectrum)
        if peaks:
            self.peaks_detected.emit(peaks)
            
    def _get_window(self) -> np.ndarray:
        """Get window function array."""
        if self.config.window_function == WindowFunction.RECTANGULAR:
            return np.ones(self.config.fft_size)
        elif self.config.window_function == WindowFunction.HANN:
            return scipy.signal.windows.hann(self.config.fft_size)
        elif self.config.window_function == WindowFunction.HAMMING:
            return scipy.signal.windows.hamming(self.config.fft_size)
        elif self.config.window_function == WindowFunction.BLACKMAN:
            return scipy.signal.windows.blackman(self.config.fft_size)
        elif self.config.window_function == WindowFunction.KAISER:
            return scipy.signal.windows.kaiser(self.config.fft_size, beta=8.6)
        elif self.config.window_function == WindowFunction.FLATTOP:
            return scipy.signal.windows.flattop(self.config.fft_size)
        else:
            return scipy.signal.windows.hann(self.config.fft_size)
            
    def _detect_peaks(self, frequencies: np.ndarray, 
                     power_spectrum: np.ndarray) -> List[Dict[str, float]]:
        """Detect peaks in spectrum."""
        # Find peaks with minimum height and distance
        peak_indices, properties = scipy.signal.find_peaks(
            power_spectrum, 
            height=np.max(power_spectrum) - 20,  # 20 dB below max
            distance=self.config.fft_size // 20  # Minimum separation
        )
        
        peaks = []
        for idx in peak_indices:
            peaks.append({
                'frequency': frequencies[idx],
                'power': power_spectrum[idx],
                'index': idx
            })
            
        return peaks
        
    def start_processing(self) -> None:
        """Start spectrum processing."""
        self.running = True
        
    def stop_processing(self) -> None:
        """Stop spectrum processing."""
        self.running = False


class SpectrumWidget(QWidget):
    """High-performance spectrum visualization widget."""
    
    # Signals
    frequency_selected = Signal(float)
    measurement_made = Signal(dict)
    config_changed = Signal(SpectrumConfig)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configuration
        self.config = SpectrumConfig()
        self.sample_rate = 2e6
        self.center_freq = 100e6
        
        # Data storage
        self.current_frequencies = np.array([])
        self.current_spectrum = np.array([])
        self.peak_markers = []
        self.measurement_cursors = []
        
        # Setup processor thread
        self.processor_thread = QThread()
        self.processor = SpectrumProcessor()
        self.processor.moveToThread(self.processor_thread)
        self.processor_thread.start()
        
        self._setup_ui()
        self._setup_plot()
        self._setup_connections()
        
    def _setup_ui(self) -> None:
        """Setup the spectrum widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for controls and plot
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)
        
        # Control panel
        controls = self._create_controls()
        splitter.addWidget(controls)
        
        # Spectrum plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Power', 'dBm')
        self.plot_widget.setLabel('bottom', 'Frequency', 'Hz')
        self.plot_widget.showGrid(True, True, alpha=0.3)
        self.plot_widget.setBackground('k')
        
        # Enable mouse interaction
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.enableAutoRange(axis='y')
        
        splitter.addWidget(self.plot_widget)
        
        # Set splitter proportions (controls smaller, plot larger)
        splitter.setSizes([150, 600])
        
        # Status bar
        self.status_label = QLabel("Ready - No data")
        self.status_label.setStyleSheet("QLabel { color: #888888; font-size: 10px; }")
        layout.addWidget(self.status_label)
        
    def _create_controls(self) -> QWidget:
        """Create spectrum control panel."""
        controls = QFrame()
        controls.setFrameStyle(QFrame.StyledPanel)
        controls.setMaximumHeight(140)
        
        layout = QHBoxLayout(controls)
        
        # FFT Settings Group
        fft_group = QGroupBox("FFT Settings")
        fft_layout = QVBoxLayout(fft_group)
        
        # FFT Size
        fft_size_layout = QHBoxLayout()
        fft_size_layout.addWidget(QLabel("FFT Size:"))
        self.fft_size_combo = QComboBox()
        fft_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
        for size in fft_sizes:
            self.fft_size_combo.addItem(str(size), size)
        self.fft_size_combo.setCurrentText(str(self.config.fft_size))
        self.fft_size_combo.currentTextChanged.connect(self._update_fft_size)
        fft_size_layout.addWidget(self.fft_size_combo)
        fft_layout.addLayout(fft_size_layout)
        
        # Window Function
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        for window in WindowFunction:
            self.window_combo.addItem(window.value.title(), window)
        self.window_combo.setCurrentText(self.config.window_function.value.title())
        self.window_combo.currentTextChanged.connect(self._update_window_function)
        window_layout.addWidget(self.window_combo)
        fft_layout.addLayout(window_layout)
        
        layout.addWidget(fft_group)
        
        # Display Settings Group
        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout(display_group)
        
        # Averaging
        avg_layout = QHBoxLayout()
        avg_layout.addWidget(QLabel("Averaging:"))
        self.avg_slider = QSlider(Qt.Horizontal)
        self.avg_slider.setRange(0, 99)
        self.avg_slider.setValue(int(self.config.averaging_factor * 100))
        self.avg_slider.valueChanged.connect(self._update_averaging)
        avg_layout.addWidget(self.avg_slider)
        self.avg_label = QLabel(f"{int(self.config.averaging_factor * 100)}%")
        self.avg_label.setMinimumWidth(30)
        avg_layout.addWidget(self.avg_label)
        display_layout.addLayout(avg_layout)
        
        # Update Rate
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Update Rate:"))
        self.rate_spinbox = QSpinBox()
        self.rate_spinbox.setRange(1, 60)
        self.rate_spinbox.setValue(self.config.update_rate_fps)
        self.rate_spinbox.setSuffix(" FPS")
        self.rate_spinbox.valueChanged.connect(self._update_rate)
        rate_layout.addWidget(self.rate_spinbox)
        display_layout.addLayout(rate_layout)
        
        layout.addWidget(display_group)
        
        # Measurement Tools Group
        measurement_group = QGroupBox("Measurements")
        measurement_layout = QVBoxLayout(measurement_group)
        
        # Checkboxes for features
        self.show_peaks_cb = QCheckBox("Show Peaks")
        self.show_peaks_cb.setChecked(self.config.show_peak_markers)
        self.show_peaks_cb.toggled.connect(self._toggle_peaks)
        measurement_layout.addWidget(self.show_peaks_cb)
        
        self.show_grid_cb = QCheckBox("Show Grid")
        self.show_grid_cb.setChecked(self.config.show_grid)
        self.show_grid_cb.toggled.connect(self._toggle_grid)
        measurement_layout.addWidget(self.show_grid_cb)
        
        # Measurement buttons
        button_layout = QHBoxLayout()
        self.cursor_btn = QPushButton("Add Cursor")
        self.cursor_btn.clicked.connect(self._add_measurement_cursor)
        button_layout.addWidget(self.cursor_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_measurements)
        button_layout.addWidget(self.clear_btn)
        
        measurement_layout.addLayout(button_layout)
        layout.addWidget(measurement_group)
        
        return controls
        
    def _setup_plot(self) -> None:
        """Setup the spectrum plot."""
        # Main spectrum curve
        self.spectrum_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='y', width=1.5),
            name='Spectrum'
        )
        
        # Peak markers
        self.peak_scatter = pg.ScatterPlotItem(
            size=8, 
            brush=pg.mkBrush(255, 0, 0, 120),
            pen=pg.mkPen('r', width=1)
        )
        self.plot_widget.addItem(self.peak_scatter)
        
        # Crosshair cursor
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, 
                                          pen=pg.mkPen('w', width=1, style=Qt.DashLine))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False,
                                          pen=pg.mkPen('w', width=1, style=Qt.DashLine))
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Connect mouse events
        self.plot_widget.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._mouse_clicked)
        
    def _setup_connections(self) -> None:
        """Setup signal-slot connections."""
        # Connect processor signals
        self.processor.spectrum_ready.connect(self._update_spectrum_display)
        self.processor.peaks_detected.connect(self._update_peak_markers)
        
        # Update processor configuration
        self.processor.set_config(self.config)
        self.processor.set_parameters(self.sample_rate, self.center_freq)
        
    @Slot(str)
    def _update_fft_size(self, size_text: str) -> None:
        """Update FFT size based on combo box selection."""
        try:
            new_size = int(size_text)
            self.config.fft_size = new_size
            self.processor.set_config(self.config)
            self.config_changed.emit(self.config)
        except ValueError:
            pass
            
    @Slot(str)
    def _update_window_function(self, window_text: str) -> None:
        """Update window function."""
        for window in WindowFunction:
            if window.value.title() == window_text:
                self.config.window_function = window
                self.processor.set_config(self.config)
                self.config_changed.emit(self.config)
                break
                
    @Slot(int)
    def _update_averaging(self, value: int) -> None:
        """Update averaging factor."""
        self.config.averaging_factor = value / 100.0
        self.avg_label.setText(f"{value}%")
        self.processor.set_config(self.config)
        self.config_changed.emit(self.config)
        
    @Slot(int)
    def _update_rate(self, fps: int) -> None:
        """Update display update rate."""
        self.config.update_rate_fps = fps
        self.config_changed.emit(self.config)
        
    @Slot(bool)
    def _toggle_peaks(self, enabled: bool) -> None:
        """Toggle peak detection display."""
        self.config.show_peak_markers = enabled
        self.peak_scatter.setVisible(enabled)
        self.processor.set_config(self.config)
        
    @Slot(bool)
    def _toggle_grid(self, enabled: bool) -> None:
        """Toggle grid display."""
        self.config.show_grid = enabled
        self.plot_widget.showGrid(enabled, enabled, alpha=0.3)
        
    @Slot(np.ndarray, np.ndarray)
    def _update_spectrum_display(self, frequencies: np.ndarray, powers: np.ndarray) -> None:
        """Update spectrum display with new data."""
        self.current_frequencies = frequencies
        self.current_spectrum = powers
        
        # Update main spectrum curve
        self.spectrum_curve.setData(frequencies, powers)
        
        # Update status
        if len(frequencies) > 0:
            freq_span = (frequencies[-1] - frequencies[0]) / 1e6  # MHz
            max_power = np.max(powers)
            self.status_label.setText(
                f"Span: {freq_span:.1f} MHz | Max: {max_power:.1f} dBm | "
                f"FFT: {self.config.fft_size} | Avg: {int(self.config.averaging_factor * 100)}%"
            )
        
    @Slot(list)
    def _update_peak_markers(self, peaks: List[Dict[str, float]]) -> None:
        """Update peak markers on the display."""
        if not self.config.show_peak_markers or not peaks:
            self.peak_scatter.clear()
            return
            
        # Extract peak positions
        peak_freqs = [peak['frequency'] for peak in peaks]
        peak_powers = [peak['power'] for peak in peaks]
        
        # Update scatter plot
        self.peak_scatter.setData(peak_freqs, peak_powers)
        
    def _mouse_moved(self, pos) -> None:
        """Handle mouse movement for crosshair cursor."""
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.crosshair_v.setPos(mouse_point.x())
            self.crosshair_h.setPos(mouse_point.y())
            
    def _mouse_clicked(self, event) -> None:
        """Handle mouse clicks for frequency selection."""
        if event.double():
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
            frequency = mouse_point.x()
            self.frequency_selected.emit(frequency)
            
    def _add_measurement_cursor(self) -> None:
        """Add a measurement cursor to the plot."""
        if len(self.current_frequencies) == 0:
            return
            
        # Add vertical line at center frequency
        center_freq = (self.current_frequencies[0] + self.current_frequencies[-1]) / 2
        cursor = pg.InfiniteLine(
            pos=center_freq,
            angle=90,
            movable=True,
            pen=pg.mkPen('g', width=2)
        )
        
        # Connect cursor movement to measurement update
        cursor.sigPositionChanged.connect(lambda: self._update_cursor_measurement(cursor))
        
        self.plot_widget.addItem(cursor)
        self.measurement_cursors.append(cursor)
        
    def _update_cursor_measurement(self, cursor) -> None:
        """Update measurement for a cursor position."""
        freq = cursor.value()
        
        # Find nearest spectrum point
        if len(self.current_frequencies) > 0:
            idx = np.argmin(np.abs(self.current_frequencies - freq))
            power = self.current_spectrum[idx]
            
            measurement = {
                'frequency': freq,
                'power': power,
                'cursor_id': id(cursor)
            }
            self.measurement_made.emit(measurement)
            
    def _clear_measurements(self) -> None:
        """Clear all measurement cursors."""
        for cursor in self.measurement_cursors:
            self.plot_widget.removeItem(cursor)
        self.measurement_cursors.clear()
        
    def set_sdr_parameters(self, sample_rate: float, center_freq: float) -> None:
        """Update SDR parameters."""
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.processor.set_parameters(sample_rate, center_freq)
        
        # Update plot labels
        if sample_rate >= 1e6:
            rate_str = f"{sample_rate/1e6:.1f} MSps"
        elif sample_rate >= 1e3:
            rate_str = f"{sample_rate/1e3:.1f} kSps"
        else:
            rate_str = f"{sample_rate:.0f} Sps"
            
        if center_freq >= 1e9:
            freq_str = f"{center_freq/1e9:.3f} GHz"
        elif center_freq >= 1e6:
            freq_str = f"{center_freq/1e6:.3f} MHz"
        elif center_freq >= 1e3:
            freq_str = f"{center_freq/1e3:.3f} kHz"
        else:
            freq_str = f"{center_freq:.0f} Hz"
            
        self.plot_widget.setTitle(f"Spectrum - {freq_str} @ {rate_str}")
        
    @Slot(np.ndarray)
    def update_spectrum_data(self, iq_data: np.ndarray) -> None:
        """Update spectrum with new I/Q data."""
        self.processor.process_iq_data(iq_data)
        
    def start_processing(self) -> None:
        """Start spectrum processing."""
        self.processor.start_processing()
        
    def stop_processing(self) -> None:
        """Stop spectrum processing."""
        self.processor.stop_processing()
        
    def get_config(self) -> SpectrumConfig:
        """Get current spectrum configuration."""
        return self.config
        
    def set_config(self, config: SpectrumConfig) -> None:
        """Set spectrum configuration."""
        self.config = config
        self.processor.set_config(config)
        
        # Update UI controls
        self.fft_size_combo.setCurrentText(str(config.fft_size))
        self.window_combo.setCurrentText(config.window_function.value.title())
        self.avg_slider.setValue(int(config.averaging_factor * 100))
        self.rate_spinbox.setValue(config.update_rate_fps)
        self.show_peaks_cb.setChecked(config.show_peak_markers)
        self.show_grid_cb.setChecked(config.show_grid)
        
    def closeEvent(self, event) -> None:
        """Handle widget close event."""
        self.stop_processing()
        if hasattr(self, 'processor_thread'):
            self.processor_thread.quit()
            self.processor_thread.wait()
        event.accept()