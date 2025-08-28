#!/usr/bin/env python3
"""
Spectrum Visualization Demo

This demo showcases the complete spectrum visualization system including:
- High-performance spectrum display with PyQtGraph
- Waterfall display with OpenGL acceleration  
- Constellation diagram with signal analysis
- Integration between all three visualization widgets

Usage:
    python examples/spectrum_visualization_demo.py
"""

import sys
import numpy as np
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                               QHBoxLayout, QWidget, QTabWidget, QLabel,
                               QPushButton, QGroupBox, QSlider, QComboBox,
                               QCheckBox, QSpinBox, QSplitter)
from PySide6.QtCore import QTimer, Signal, Slot, Qt
from PySide6.QtGui import QFont

# Import our visualization widgets
from gui.widgets.spectrum_widget import SpectrumWidget, SpectrumConfig, WindowFunction
from gui.widgets.waterfall_widget import WaterfallWidget, WaterfallConfig, ColorMap
from gui.widgets.constellation_widget import ConstellationWidget, ConstellationConfig, ModulationType


class SignalGenerator:
    """Generate various test signals for demonstration."""
    
    def __init__(self, sample_rate: float = 2e6, center_freq: float = 100e6):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.time_offset = 0.0
        
    def generate_qpsk_signal(self, duration: float = 0.001, 
                           symbol_rate: float = 100e3,
                           snr_db: float = 20.0) -> np.ndarray:
        """Generate QPSK modulated signal."""
        num_samples = int(duration * self.sample_rate)
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        
        # Generate random symbols
        num_symbols = num_samples // samples_per_symbol
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_symbols)
        
        # Upsample symbols
        upsampled = np.repeat(symbols, samples_per_symbol)
        
        # Add carrier frequency
        t = np.arange(len(upsampled)) / self.sample_rate + self.time_offset
        carrier = np.exp(1j * 2 * np.pi * 50e3 * t)  # 50 kHz offset from center
        signal = upsampled * carrier
        
        # Add noise
        noise_power = 10**(-snr_db/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                         1j * np.random.randn(len(signal)))
        
        self.time_offset += duration
        return signal + noise
        
    def generate_bpsk_signal(self, duration: float = 0.001,
                           symbol_rate: float = 50e3,
                           snr_db: float = 15.0) -> np.ndarray:
        """Generate BPSK modulated signal."""
        num_samples = int(duration * self.sample_rate)
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        
        # Generate random symbols
        num_symbols = num_samples // samples_per_symbol
        symbols = np.random.choice([1+0j, -1+0j], num_symbols)
        
        # Upsample symbols
        upsampled = np.repeat(symbols, samples_per_symbol)
        
        # Add carrier frequency
        t = np.arange(len(upsampled)) / self.sample_rate + self.time_offset
        carrier = np.exp(1j * 2 * np.pi * (-30e3) * t)  # -30 kHz offset
        signal = upsampled * carrier
        
        # Add noise
        noise_power = 10**(-snr_db/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                         1j * np.random.randn(len(signal)))
        
        self.time_offset += duration
        return signal + noise
        
    def generate_fsk_signal(self, duration: float = 0.001,
                          symbol_rate: float = 25e3,
                          freq_deviation: float = 10e3,
                          snr_db: float = 18.0) -> np.ndarray:
        """Generate FSK modulated signal."""
        num_samples = int(duration * self.sample_rate)
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        
        # Generate random symbols
        num_symbols = num_samples // samples_per_symbol
        symbols = np.random.choice([0, 1], num_symbols)
        
        # Generate FSK signal
        signal = np.zeros(num_samples, dtype=complex)
        t = np.arange(num_samples) / self.sample_rate + self.time_offset
        
        for i, symbol in enumerate(symbols):
            start_idx = i * samples_per_symbol
            end_idx = min(start_idx + samples_per_symbol, num_samples)
            
            freq_offset = freq_deviation if symbol else -freq_deviation
            phase = 2 * np.pi * (100e3 + freq_offset) * t[start_idx:end_idx]
            signal[start_idx:end_idx] = np.exp(1j * phase)
            
        # Add noise
        noise_power = 10**(-snr_db/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                         1j * np.random.randn(len(signal)))
        
        self.time_offset += duration
        return signal + noise
        
    def generate_noise(self, duration: float = 0.001,
                      power_db: float = -40.0) -> np.ndarray:
        """Generate white noise."""
        num_samples = int(duration * self.sample_rate)
        power_linear = 10**(power_db/10)
        
        noise = np.sqrt(power_linear/2) * (np.random.randn(num_samples) + 
                                          1j * np.random.randn(num_samples))
        
        self.time_offset += duration
        return noise
        
    def generate_multi_tone(self, duration: float = 0.001,
                          frequencies: list = None,
                          amplitudes: list = None,
                          snr_db: float = 25.0) -> np.ndarray:
        """Generate multi-tone signal."""
        if frequencies is None:
            frequencies = [-75e3, -25e3, 25e3, 75e3]
        if amplitudes is None:
            amplitudes = [1.0, 0.8, 0.6, 0.4]
            
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples) / self.sample_rate + self.time_offset
        
        signal = np.zeros(num_samples, dtype=complex)
        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.exp(1j * 2 * np.pi * freq * t)
            
        # Add noise
        noise_power = 10**(-snr_db/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                         1j * np.random.randn(len(signal)))
        
        self.time_offset += duration
        return signal + noise


class VisualizationDemo(QMainWindow):
    """Main demo application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("GeminiSDR Spectrum Visualization Demo")
        self.setGeometry(100, 100, 1400, 900)
        
        # Signal generator
        self.signal_generator = SignalGenerator()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_displays)
        
        # Current signal type
        self.current_signal_type = "QPSK"
        
        self._setup_ui()
        self._setup_connections()
        
        # Start demo
        self.start_demo()
        
    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Control panel
        controls = self._create_control_panel()
        layout.addWidget(controls)
        
        # Main visualization area
        viz_area = self._create_visualization_area()
        layout.addWidget(viz_area)
        
        # Status bar
        self.statusBar().showMessage("Demo Ready - Click Start to begin")
        
    def _create_control_panel(self) -> QWidget:
        """Create the control panel."""
        panel = QGroupBox("Demo Controls")
        layout = QHBoxLayout(panel)
        
        # Signal type selection
        layout.addWidget(QLabel("Signal Type:"))
        self.signal_combo = QComboBox()
        self.signal_combo.addItems(["QPSK", "BPSK", "FSK", "Multi-tone", "Noise"])
        self.signal_combo.currentTextChanged.connect(self._change_signal_type)
        layout.addWidget(self.signal_combo)
        
        # Update rate control
        layout.addWidget(QLabel("Update Rate:"))
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(1, 60)
        self.rate_slider.setValue(30)
        self.rate_slider.valueChanged.connect(self._change_update_rate)
        layout.addWidget(self.rate_slider)
        
        self.rate_label = QLabel("30 FPS")
        layout.addWidget(self.rate_label)
        
        # Control buttons
        self.start_btn = QPushButton("Start Demo")
        self.start_btn.clicked.connect(self.start_demo)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Demo")
        self.stop_btn.clicked.connect(self.stop_demo)
        layout.addWidget(self.stop_btn)
        
        self.reset_btn = QPushButton("Reset Views")
        self.reset_btn.clicked.connect(self.reset_views)
        layout.addWidget(self.reset_btn)
        
        layout.addStretch()
        
        return panel
        
    def _create_visualization_area(self) -> QWidget:
        """Create the main visualization area."""
        # Create tab widget for different views
        self.viz_tabs = QTabWidget()
        
        # Combined view tab
        combined_tab = self._create_combined_view()
        self.viz_tabs.addTab(combined_tab, "Combined View")
        
        # Individual tabs for each widget
        spectrum_tab = self._create_spectrum_tab()
        self.viz_tabs.addTab(spectrum_tab, "Spectrum Analyzer")
        
        waterfall_tab = self._create_waterfall_tab()
        self.viz_tabs.addTab(waterfall_tab, "Waterfall Display")
        
        constellation_tab = self._create_constellation_tab()
        self.viz_tabs.addTab(constellation_tab, "Constellation Diagram")
        
        return self.viz_tabs
        
    def _create_combined_view(self) -> QWidget:
        """Create combined view with all three widgets."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Top row: Spectrum and Waterfall
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Spectrum widget
        self.spectrum_widget = SpectrumWidget()
        top_splitter.addWidget(self.spectrum_widget)
        
        # Waterfall widget
        self.waterfall_widget = WaterfallWidget()
        top_splitter.addWidget(self.waterfall_widget)
        
        # Set equal sizes
        top_splitter.setSizes([500, 500])
        
        # Bottom row: Constellation
        self.constellation_widget = ConstellationWidget()
        
        # Main splitter
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.constellation_widget)
        main_splitter.setSizes([400, 300])
        
        layout.addWidget(main_splitter)
        
        return widget
        
    def _create_spectrum_tab(self) -> QWidget:
        """Create spectrum analyzer tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create separate spectrum widget for this tab
        self.spectrum_widget_tab = SpectrumWidget()
        layout.addWidget(self.spectrum_widget_tab)
        
        return widget
        
    def _create_waterfall_tab(self) -> QWidget:
        """Create waterfall display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create separate waterfall widget for this tab
        self.waterfall_widget_tab = WaterfallWidget()
        layout.addWidget(self.waterfall_widget_tab)
        
        return widget
        
    def _create_constellation_tab(self) -> QWidget:
        """Create constellation diagram tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create separate constellation widget for this tab
        self.constellation_widget_tab = ConstellationWidget()
        layout.addWidget(self.constellation_widget_tab)
        
        return widget
        
    def _setup_connections(self):
        """Setup signal connections between widgets."""
        # Connect spectrum widget frequency selection to other widgets
        self.spectrum_widget.frequency_selected.connect(self._on_frequency_selected)
        
        # Connect constellation analysis updates
        self.constellation_widget.analysis_updated.connect(self._on_analysis_updated)
        
        # Synchronize configurations
        self.spectrum_widget.config_changed.connect(self._sync_spectrum_config)
        self.waterfall_widget.config_changed.connect(self._sync_waterfall_config)
        self.constellation_widget.config_changed.connect(self._sync_constellation_config)
        
    @Slot(str)
    def _change_signal_type(self, signal_type: str):
        """Change the signal type being generated."""
        self.current_signal_type = signal_type
        self.statusBar().showMessage(f"Signal type changed to: {signal_type}")
        
    @Slot(int)
    def _change_update_rate(self, rate: int):
        """Change the update rate."""
        self.rate_label.setText(f"{rate} FPS")
        if self.update_timer.isActive():
            self.update_timer.start(1000 // rate)
            
    @Slot(float)
    def _on_frequency_selected(self, frequency: float):
        """Handle frequency selection from spectrum widget."""
        freq_mhz = frequency / 1e6
        self.statusBar().showMessage(f"Selected frequency: {freq_mhz:.3f} MHz")
        
    @Slot(object)
    def _on_analysis_updated(self, analysis):
        """Handle constellation analysis updates."""
        mod_type = analysis.modulation_type.value
        confidence = analysis.confidence * 100
        snr = analysis.snr_estimate
        
        message = f"Detected: {mod_type} (confidence: {confidence:.1f}%, SNR: {snr:.1f} dB)"
        self.statusBar().showMessage(message)
        
    def _sync_spectrum_config(self, config):
        """Synchronize spectrum configuration across widgets."""
        if hasattr(self, 'spectrum_widget_tab'):
            self.spectrum_widget_tab.set_config(config)
            
    def _sync_waterfall_config(self, config):
        """Synchronize waterfall configuration across widgets."""
        if hasattr(self, 'waterfall_widget_tab'):
            self.waterfall_widget_tab.set_config(config)
            
    def _sync_constellation_config(self, config):
        """Synchronize constellation configuration across widgets."""
        if hasattr(self, 'constellation_widget_tab'):
            self.constellation_widget_tab.set_config(config)
            
    def start_demo(self):
        """Start the demonstration."""
        # Configure widgets
        sample_rate = 2e6
        center_freq = 100e6
        
        # Set SDR parameters for all widgets
        widgets = [
            self.spectrum_widget, self.waterfall_widget,
            getattr(self, 'spectrum_widget_tab', None),
            getattr(self, 'waterfall_widget_tab', None)
        ]
        
        for widget in widgets:
            if widget is not None:
                widget.set_sdr_parameters(sample_rate, center_freq)
                
        # Start processing
        self.spectrum_widget.start_processing()
        self.constellation_widget.start_analysis()
        
        if hasattr(self, 'spectrum_widget_tab'):
            self.spectrum_widget_tab.start_processing()
        if hasattr(self, 'constellation_widget_tab'):
            self.constellation_widget_tab.start_analysis()
            
        # Start update timer
        rate = self.rate_slider.value()
        self.update_timer.start(1000 // rate)
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Demo started - Generating signals...")
        
    def stop_demo(self):
        """Stop the demonstration."""
        # Stop timer
        self.update_timer.stop()
        
        # Stop processing
        self.spectrum_widget.stop_processing()
        self.constellation_widget.stop_analysis()
        
        if hasattr(self, 'spectrum_widget_tab'):
            self.spectrum_widget_tab.stop_processing()
        if hasattr(self, 'constellation_widget_tab'):
            self.constellation_widget_tab.stop_analysis()
            
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Demo stopped")
        
    def reset_views(self):
        """Reset all visualization views."""
        # Clear displays
        widgets = [
            self.spectrum_widget, self.waterfall_widget, self.constellation_widget,
            getattr(self, 'spectrum_widget_tab', None),
            getattr(self, 'waterfall_widget_tab', None),
            getattr(self, 'constellation_widget_tab', None)
        ]
        
        for widget in widgets:
            if widget is not None:
                if hasattr(widget, 'clear_history'):
                    widget.clear_history()
                if hasattr(widget, '_clear_constellation'):
                    widget._clear_constellation()
                    
        self.statusBar().showMessage("Views reset")
        
    def _update_displays(self):
        """Update all displays with new signal data."""
        try:
            # Generate signal based on current type
            if self.current_signal_type == "QPSK":
                signal = self.signal_generator.generate_qpsk_signal()
            elif self.current_signal_type == "BPSK":
                signal = self.signal_generator.generate_bpsk_signal()
            elif self.current_signal_type == "FSK":
                signal = self.signal_generator.generate_fsk_signal()
            elif self.current_signal_type == "Multi-tone":
                signal = self.signal_generator.generate_multi_tone()
            elif self.current_signal_type == "Noise":
                signal = self.signal_generator.generate_noise()
            else:
                signal = self.signal_generator.generate_qpsk_signal()
                
            # Update all widgets
            widgets = [
                (self.spectrum_widget, 'update_spectrum_data'),
                (self.waterfall_widget, 'add_spectrum_data'),
                (self.constellation_widget, 'update_iq_data'),
                (getattr(self, 'spectrum_widget_tab', None), 'update_spectrum_data'),
                (getattr(self, 'waterfall_widget_tab', None), 'add_spectrum_data'),
                (getattr(self, 'constellation_widget_tab', None), 'update_iq_data')
            ]
            
            for widget, method_name in widgets:
                if widget is not None and hasattr(widget, method_name):
                    method = getattr(widget, method_name)
                    
                    if method_name == 'add_spectrum_data':
                        # For waterfall, we need spectrum data, not I/Q
                        # Simple FFT for demonstration
                        fft_data = np.fft.fftshift(np.fft.fft(signal))
                        spectrum = 20 * np.log10(np.abs(fft_data) + 1e-12)
                        method(spectrum)
                    else:
                        method(signal)
                        
        except Exception as e:
            print(f"Error updating displays: {e}")
            
    def closeEvent(self, event):
        """Handle application close."""
        self.stop_demo()
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("GeminiSDR Visualization Demo")
    app.setApplicationVersion("1.0.0")
    
    # Apply dark theme
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #404040;
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 5px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #505050;
        }
        QPushButton:pressed {
            background-color: #353535;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            color: #666666;
        }
        QTabWidget::pane {
            border: 1px solid #555555;
        }
        QTabBar::tab {
            background-color: #404040;
            border: 1px solid #555555;
            padding: 5px 10px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #505050;
        }
    """)
    
    # Create and show demo window
    demo = VisualizationDemo()
    demo.show()
    
    # Run application
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())