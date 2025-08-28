# core/visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading

class RealTimeVisualizer:
    """Real-time visualization for SDR signals."""
    
    def __init__(self, sample_rate=1e6, window_size=1024):
        self.sample_rate = sample_rate
        self.window_size = window_size
        
        # Data buffers
        self.time_buffer = deque(maxlen=window_size)
        self.freq_buffer = deque(maxlen=window_size//2)
        self.constellation_buffer = deque(maxlen=1000)
        
        self.is_running = False
        self.fig = None
        self.axes = None
    
    def start_realtime_plot(self):
        """Start real-time plotting."""
        self.is_running = True
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('Real-Time SDR Signal Analysis')
        
        # Time domain setup
        self.ax_time_i = self.axes[0, 0]
        self.ax_time_q = self.axes[0, 1]
        self.ax_time_i.set_title('I Component (Time)')
        self.ax_time_q.set_title('Q Component (Time)')
        
        # Frequency domain setup
        self.ax_freq = self.axes[1, 0]
        self.ax_freq.set_title('Frequency Spectrum')
        self.ax_freq.set_xlabel('Frequency (kHz)')
        self.ax_freq.set_ylabel('Magnitude (dB)')
        
        # Constellation setup
        self.ax_const = self.axes[1, 1]
        self.ax_const.set_title('Constellation Diagram')
        self.ax_const.set_xlabel('I')
        self.ax_const.set_ylabel('Q')
        self.ax_const.set_aspect('equal')
        
        # Initialize plot lines
        self.line_i, = self.ax_time_i.plot([], [], 'b-')
        self.line_q, = self.ax_time_q.plot([], [], 'r-')
        self.line_freq, = self.ax_freq.plot([], [], 'g-')
        self.scatter_const = self.ax_const.scatter([], [], alpha=0.5, s=1)
        
        # Set up animation
        self.anim = FuncAnimation(
            self.fig, 
            self._update_plot, 
            interval=50,  # 20 FPS
            blit=False  # Changed to False for better compatibility
        )
        
        plt.tight_layout()
        plt.show()
    
    def update_data(self, samples):
        """Update data buffers with new samples."""
        if not self.is_running:
            return
        
        # Update time domain buffers
        self.time_buffer.extend(samples[:self.window_size])
        
        # Compute and update frequency spectrum
        if len(samples) >= self.window_size:
            fft = np.fft.fft(samples[:self.window_size])
            fft_mag = 20 * np.log10(np.abs(fft[:self.window_size//2]) + 1e-12)
            self.freq_buffer = deque(fft_mag, maxlen=self.window_size//2)
        
        # Update constellation
        decimation = max(1, len(samples) // 100)  # Limit points for performance
        self.constellation_buffer.extend(samples[::decimation])
    
    def _update_plot(self, frame):
        """Animation update function."""
        # Update time domain plots
        if len(self.time_buffer) > 0:
            time_data = np.array(self.time_buffer)
            time_axis = np.arange(len(time_data)) / self.sample_rate * 1000  # ms
            
            self.line_i.set_data(time_axis, np.real(time_data))
            self.line_q.set_data(time_axis, np.imag(time_data))
            
            # Auto-scale
            self.ax_time_i.relim()
            self.ax_time_i.autoscale_view()
            self.ax_time_q.relim()
            self.ax_time_q.autoscale_view()
        
        # Update frequency plot
        if len(self.freq_buffer) > 0:
            freq_data = np.array(self.freq_buffer)
            freq_axis = np.linspace(0, self.sample_rate/2/1000, len(freq_data))  # kHz
            
            self.line_freq.set_data(freq_axis, freq_data)
            self.ax_freq.relim()
            self.ax_freq.autoscale_view()
        
        # Update constellation
        if len(self.constellation_buffer) > 0:
            const_data = np.array(list(self.constellation_buffer))
            self.ax_const.clear()
            self.ax_const.scatter(np.real(const_data), np.imag(const_data), alpha=0.5, s=1)
            self.ax_const.set_title('Constellation Diagram')
            self.ax_const.set_xlabel('I')
            self.ax_const.set_ylabel('Q')
            self.ax_const.set_aspect('equal')
            self.ax_const.grid(True, alpha=0.3)
            
            # Auto-scale constellation
            max_val = np.max(np.abs(const_data)) * 1.1
            self.ax_const.set_xlim(-max_val, max_val)
            self.ax_const.set_ylim(-max_val, max_val)
        
        return []
    
    def stop(self):
        """Stop real-time plotting."""
        self.is_running = False
        if hasattr(self, 'anim') and self.anim:
            self.anim.event_source.stop()
        if self.fig:
            plt.close(self.fig)

class StaticVisualizer:
    """Static visualization tools for analysis."""
    
    @staticmethod
    def plot_modulation_comparison(signals_dict, sample_rate=1e6):
        """Compare different modulation schemes."""
        num_mods = len(signals_dict)
        fig, axes = plt.subplots(num_mods, 3, figsize=(15, 4*num_mods))
        
        if num_mods == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (mod_type, signal) in enumerate(signals_dict.items()):
            # Time domain
            ax = axes[idx, 0]
            time_ms = np.arange(len(signal)) / sample_rate * 1000
            ax.plot(time_ms[:500], np.real(signal[:500]), 'b', alpha=0.7, label='I')
            ax.plot(time_ms[:500], np.imag(signal[:500]), 'r', alpha=0.7, label='Q')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{mod_type} - Time Domain')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Frequency domain
            ax = axes[idx, 1]
            fft = np.fft.fftshift(np.fft.fft(signal))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/sample_rate))
            ax.plot(freqs/1e3, 20*np.log10(np.abs(fft) + 1e-12))
            ax.set_xlabel('Frequency (kHz)')
            ax.set_ylabel('Magnitude (dB)')
            ax.set_title(f'{mod_type} - Frequency Domain')
            ax.grid(True, alpha=0.3)
            
            # Constellation
            ax = axes[idx, 2]
            # Downsample for clarity
            decimated = signal[::8][:1000]
            ax.scatter(np.real(decimated), np.imag(decimated), alpha=0.5, s=1)
            ax.set_xlabel('I')
            ax.set_ylabel('Q')
            ax.set_title(f'{mod_type} - Constellation')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_snr_performance(snr_values, accuracies, title="SNR vs Accuracy"):
        """Plot performance vs SNR."""
        plt.figure(figsize=(10, 6))
        plt.plot(snr_values, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy (%)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()