# real_signal_capture.py
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from core.sdr_interface import PlutoSDRInterface
from core.visualizer import StaticVisualizer
from ml.traditional_amr import TraditionalAMR
from ml.neural_amr import NeuralAMR
import h5py
import os

class RealSignalCapture:
    """Capture and analyze real signals from PlutoSDR."""
    
    def __init__(self):
        self.pluto = PlutoSDRInterface()
        self.trad_amr = TraditionalAMR()
        self.neural_amr = NeuralAMR()
        
        # Load pre-trained models
        try:
            self.trad_amr.load_model('models/traditional_amr.pkl')
            print("✓ Loaded traditional AMR model")
        except:
            print("✗ Traditional model not found - need to train first")
            
        try:
            self.neural_amr.load_model('models/neural_amr.pth')
            print("✓ Loaded neural AMR model")
        except:
            print("✗ Neural model not found - need to train first")
    
    def find_active_signals(self, center_freq=100e6, span=2e6, threshold_db=-50):
        """Scan spectrum to find active signals."""
        print(f"\nScanning spectrum around {center_freq/1e6:.1f} MHz...")
        
        if not self.pluto.connect():
            return None
        
        # Configure for spectrum scanning
        self.pluto.configure(
            center_freq=center_freq,
            sample_rate=span,
            bandwidth=span * 0.8,
            gain=30
        )
        
        # Capture samples
        samples = self.pluto.capture_batch(0.1)  # 100ms capture
        
        if samples is None:
            return None
        
        # Compute spectrum
        fft = np.fft.fftshift(np.fft.fft(samples))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/span))
        power_db = 20 * np.log10(np.abs(fft) + 1e-12)
        
        # Find peaks above threshold
        peaks = []
        peak_indices = np.where(power_db > threshold_db)[0]
        
        if len(peak_indices) > 0:
            # Group nearby peaks
            groups = []
            current_group = [peak_indices[0]]
            
            for i in range(1, len(peak_indices)):
                if peak_indices[i] - peak_indices[i-1] < 100:  # Within 100 bins
                    current_group.append(peak_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [peak_indices[i]]
            groups.append(current_group)
            
            # Find center of each group
            for group in groups:
                center_idx = group[len(group)//2]
                freq = center_freq + freqs[center_idx]
                power = power_db[center_idx]
                peaks.append({'freq': freq, 'power': power})
        
        # Plot spectrum
        plt.figure(figsize=(12, 6))
        plt.plot(center_freq + freqs, power_db)
        plt.axhline(y=threshold_db, color='r', linestyle='--', label=f'Threshold ({threshold_db} dB)')
        
        for peak in peaks:
            plt.axvline(x=peak['freq'], color='g', alpha=0.5)
            plt.text(peak['freq'], peak['power'] + 5, 
                    f"{peak['freq']/1e6:.3f} MHz", 
                    rotation=90, ha='center')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.title('Spectrum Scan - Active Signals')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
        print(f"Found {len(peaks)} active signals")
        return peaks
    
    def capture_and_classify(self, frequency, sample_rate=1e6, capture_time=0.1):
        """Capture signal at specific frequency and classify it."""
        print(f"\nCapturing signal at {frequency/1e6:.3f} MHz...")
        
        # Configure SDR
        self.pluto.configure(
            center_freq=frequency,
            sample_rate=sample_rate,
            bandwidth=sample_rate * 0.8,
            gain=40  # Higher gain for weak signals
        )
        
        # Capture signal
        raw_samples = self.pluto.capture_batch(capture_time)
        
        if raw_samples is None:
            print("Failed to capture samples")
            return None
        
        print(f"Captured {len(raw_samples)} samples")
        
        # Analyze signal quality
        signal_power = np.mean(np.abs(raw_samples)**2)
        noise_floor = np.percentile(np.abs(raw_samples)**2, 10)
        snr_estimate = 10 * np.log10(signal_power / noise_floor)
        
        print(f"Estimated SNR: {snr_estimate:.1f} dB")
        
        # Extract signal segments for classification
        segment_length = 1024
        hop_length = 512
        classifications = []
        
        for i in range(0, len(raw_samples) - segment_length, hop_length):
            segment = raw_samples[i:i + segment_length]
            
            # Traditional classification
            trad_pred, trad_probs = self.trad_amr.predict(segment)
            
            # Neural classification
            neural_pred, neural_probs = self.neural_amr.predict(segment)
            
            classifications.append({
                'traditional': trad_pred,
                'neural': neural_pred,
                'trad_confidence': np.max(trad_probs),
                'neural_confidence': np.max(neural_probs)
            })
        
        # Aggregate results (majority voting)
        trad_votes = {}
        neural_votes = {}
        
        for c in classifications:
            trad_votes[c['traditional']] = trad_votes.get(c['traditional'], 0) + 1
            neural_votes[c['neural']] = neural_votes.get(c['neural'], 0) + 1
        
        trad_result = max(trad_votes, key=trad_votes.get)
        neural_result = max(neural_votes, key=neural_votes.get)
        
        # Display results
        print("\n=== Classification Results ===")
        print(f"Traditional AMR: {trad_result} (confidence: {trad_votes[trad_result]/len(classifications)*100:.1f}%)")
        print(f"Neural AMR: {neural_result} (confidence: {neural_votes[neural_result]/len(classifications)*100:.1f}%)")
        
        # Visualize captured signal
        self._visualize_captured_signal(raw_samples, frequency, sample_rate)
        
        return {
            'frequency': frequency,
            'samples': raw_samples,
            'traditional_prediction': trad_result,
            'neural_prediction': neural_result,
            'snr_estimate': snr_estimate,
            'classifications': classifications
        }
    
    def _visualize_captured_signal(self, samples, center_freq, sample_rate):
        """Visualize the captured signal."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time domain
        ax = axes[0, 0]
        time_ms = np.arange(len(samples)) / sample_rate * 1000
        ax.plot(time_ms[:1000], np.real(samples[:1000]), 'b', alpha=0.7, label='I')
        ax.plot(time_ms[:1000], np.imag(samples[:1000]), 'r', alpha=0.7, label='Q')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Time Domain')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Frequency domain
        ax = axes[0, 1]
        fft = np.fft.fftshift(np.fft.fft(samples))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sample_rate))
        ax.plot(freqs/1e3, 20*np.log10(np.abs(fft) + 1e-12))
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(f'Spectrum centered at {center_freq/1e6:.1f} MHz')
        ax.grid(True, alpha=0.3)
        
        # Constellation
        ax = axes[1, 0]
        # Downsample for visibility
        decimated = samples[::10][:1000]
        ax.scatter(np.real(decimated), np.imag(decimated), alpha=0.5, s=1)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title('Constellation Diagram')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Spectrogram
        ax = axes[1, 1]
        from scipy import signal
        f, t, Sxx = signal.spectrogram(samples, sample_rate, nperseg=256)
        ax.pcolormesh(t*1000, f/1e3, 10*np.log10(Sxx + 1e-12))
        ax.set_ylabel('Frequency (kHz)')
        ax.set_xlabel('Time (ms)')
        ax.set_title('Spectrogram')
        
        plt.tight_layout()
        plt.show()
    
    def capture_known_signals(self):
        """Capture known test signals for validation."""
        print("\n=== Capturing Known Signals ===")
        
        # Common signal sources to try
        test_frequencies = [
            {'freq': 100.1e6, 'name': 'FM Radio', 'expected': 'FM'},
            {'freq': 162.4e6, 'name': 'NOAA Weather', 'expected': 'FM'},
            {'freq': 462.5625e6, 'name': 'FRS/GMRS', 'expected': 'FM'},
            {'freq': 915e6, 'name': 'ISM Band', 'expected': 'Various'},
        ]
        
        results = []
        
        for test in test_frequencies:
            print(f"\nTrying {test['name']} at {test['freq']/1e6:.1f} MHz...")
            result = self.capture_and_classify(test['freq'])
            
            if result:
                result['expected'] = test['expected']
                result['name'] = test['name']
                results.append(result)
        
        return results
    
    def save_captured_signals(self, results, filename=None):
        """Save captured signals for later analysis."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_signals_{timestamp}.h5"
        
        with h5py.File(filename, 'w') as f:
            for i, result in enumerate(results):
                grp = f.create_group(f'signal_{i}')
                grp.create_dataset('samples', data=result['samples'])
                grp.attrs['frequency'] = result['frequency']
                grp.attrs['traditional_prediction'] = result['traditional_prediction']
                grp.attrs['neural_prediction'] = result['neural_prediction']
                grp.attrs['snr_estimate'] = result['snr_estimate']
                
                if 'name' in result:
                    grp.attrs['name'] = result['name']
                if 'expected' in result:
                    grp.attrs['expected'] = result['expected']
        
        print(f"\nSaved {len(results)} captured signals to {filename}")
    
    def continuous_monitoring(self, frequency, duration=60):
        """Continuously monitor and classify a frequency."""
        print(f"\nMonitoring {frequency/1e6:.1f} MHz for {duration} seconds...")
        
        if not self.pluto.connect():
            return
        
        # Configure for monitoring
        self.pluto.configure(
            center_freq=frequency,
            sample_rate=1e6,
            gain=40
        )
        
        # Real-time classification
        start_time = time.time()
        classifications = []
        
        def process_samples(samples):
            """Callback for real-time processing."""
            if len(samples) >= 1024:
                # Classify the middle segment
                segment = samples[len(samples)//2 - 512:len(samples)//2 + 512]
                
                trad_pred, _ = self.trad_amr.predict(segment)
                neural_pred, _ = self.neural_amr.predict(segment)
                
                timestamp = time.time() - start_time
                classifications.append({
                    'time': timestamp,
                    'traditional': trad_pred,
                    'neural': neural_pred
                })
                
                print(f"\r[{timestamp:.1f}s] Traditional: {trad_pred}, Neural: {neural_pred}", end='')
        
        # Start streaming
        self.pluto.start_streaming(callback_func=process_samples)
        
        # Monitor for specified duration
        time.sleep(duration)
        
        # Stop streaming
        self.pluto.stop_streaming()
        
        print("\n\nMonitoring complete!")
        
        # Analyze results
        if classifications:
            trad_counts = {}
            neural_counts = {}
            
            for c in classifications:
                trad_counts[c['traditional']] = trad_counts.get(c['traditional'], 0) + 1
                neural_counts[c['neural']] = neural_counts.get(c['neural'], 0) + 1
            
            print("\nClassification Summary:")
            print("Traditional AMR:")
            for mod, count in sorted(trad_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {mod}: {count} ({count/len(classifications)*100:.1f}%)")
            
            print("\nNeural AMR:")
            for mod, count in sorted(neural_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {mod}: {count} ({count/len(classifications)*100:.1f}%)")

def main():
    """Main function for real signal capture and classification."""
    capture = RealSignalCapture()
    
    while True:
        print("\n=== Real Signal Capture Menu ===")
        print("1. Scan for active signals")
        print("2. Capture and classify specific frequency")
        print("3. Capture known test signals")
        print("4. Continuous monitoring")
        print("5. Exit")
        
        choice = input("\nSelect option: ")
        
        if choice == '1':
            freq = float(input("Center frequency (MHz) [100]: ") or "100") * 1e6
            span = float(input("Span (MHz) [2]: ") or "2") * 1e6
            peaks = capture.find_active_signals(freq, span)
            
            if peaks:
                print("\nFound signals at:")
                for i, peak in enumerate(peaks):
                    print(f"  {i+1}. {peak['freq']/1e6:.3f} MHz ({peak['power']:.1f} dB)")
        
        elif choice == '2':
            freq = float(input("Frequency (MHz): ")) * 1e6
            result = capture.capture_and_classify(freq)
            
            if result:
                save = input("\nSave this capture? (y/n): ")
                if save.lower() == 'y':
                    capture.save_captured_signals([result])
        
        elif choice == '3':
            results = capture.capture_known_signals()
            if results:
                capture.save_captured_signals(results)
        
        elif choice == '4':
            freq = float(input("Frequency to monitor (MHz): ")) * 1e6
            duration = int(input("Duration (seconds) [60]: ") or "60")
            capture.continuous_monitoring(freq, duration)
        
        elif choice == '5':
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()