# core/signal_generator.py
import numpy as np
import h5py
from datetime import datetime
import os

class SDRSignalGenerator:
    """Advanced signal generator for AMR and other SDR applications."""
    
    def __init__(self, sample_rate=1e6):
        self.sample_rate = sample_rate
        self.modulation_types = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']
    
    def generate_complex_noise(self, num_samples, noise_power=0.1):
        """Generate complex Gaussian noise."""
        noise = np.sqrt(noise_power/2) * (np.random.randn(num_samples) + 
                                           1j * np.random.randn(num_samples))
        return noise
    
    def add_channel_effects(self, signal, snr_db=20, freq_offset=0, phase_offset=0):
        """Add realistic channel impairments."""
        # Add frequency offset
        time = np.arange(len(signal)) / self.sample_rate
        signal = signal * np.exp(1j * 2 * np.pi * freq_offset * time)
        
        # Add phase offset
        signal = signal * np.exp(1j * phase_offset)
        
        # Add noise based on SNR
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = self.generate_complex_noise(len(signal), noise_power)
        
        return signal + noise
    
    def generate_bpsk(self, num_symbols, samples_per_symbol=8):
        """Generate BPSK modulated signal."""
        # Random binary data
        data = np.random.randint(0, 2, num_symbols)
        
        # Map to phase: 0 -> 0, 1 -> π
        symbols = 2 * data - 1  # Maps to -1, +1
        
        # Upsample and apply pulse shaping
        upsampled = np.repeat(symbols, samples_per_symbol)
        
        return self._apply_pulse_shaping(upsampled, samples_per_symbol)
    
    def generate_qpsk(self, num_symbols, samples_per_symbol=8):
        """Generate QPSK modulated signal."""
        # Random data (2 bits per symbol)
        data = np.random.randint(0, 4, num_symbols)
        
        # Map to constellation points
        constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        symbols = constellation[data]
        
        # Upsample
        upsampled = np.repeat(symbols, samples_per_symbol)
        
        return self._apply_pulse_shaping(upsampled, samples_per_symbol)
    
    def generate_qam(self, num_symbols, order=16, samples_per_symbol=8):
        """Generate QAM modulated signal (16-QAM or 64-QAM)."""
        # Generate random data
        data = np.random.randint(0, order, num_symbols)
        
        # Create QAM constellation
        if order == 16:
            # 16-QAM constellation
            levels = np.array([-3, -1, 1, 3])
            constellation = []
            for i in levels:
                for q in levels:
                    constellation.append(i + 1j*q)
            constellation = np.array(constellation) / np.sqrt(10)
        elif order == 64:
            # 64-QAM constellation
            levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            constellation = []
            for i in levels:
                for q in levels:
                    constellation.append(i + 1j*q)
            constellation = np.array(constellation) / np.sqrt(42)
        else:
            raise ValueError(f"Unsupported QAM order: {order}")
        
        # Map data to constellation
        symbols = constellation[data]
        
        # Upsample
        upsampled = np.repeat(symbols, samples_per_symbol)
        
        return self._apply_pulse_shaping(upsampled, samples_per_symbol)
    
    def generate_8psk(self, num_symbols, samples_per_symbol=8):
        """Generate 8-PSK modulated signal."""
        # Random data (3 bits per symbol)
        data = np.random.randint(0, 8, num_symbols)
        
        # 8-PSK constellation points
        angles = np.pi * np.arange(8) / 4
        constellation = np.exp(1j * angles)
        symbols = constellation[data]
        
        # Upsample
        upsampled = np.repeat(symbols, samples_per_symbol)
        
        return self._apply_pulse_shaping(upsampled, samples_per_symbol)
    
    def _apply_pulse_shaping(self, signal, samples_per_symbol, alpha=0.35):
        """Apply root raised cosine pulse shaping."""
        # Simple approximation of RRC filter
        filter_length = 4 * samples_per_symbol + 1
        t = np.arange(filter_length) - filter_length // 2
        t = t / samples_per_symbol
        
        # Root raised cosine impulse response
        h = np.zeros_like(t, dtype=float)
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 1 - alpha + 4*alpha/np.pi
            elif abs(ti) == 1/(4*alpha):
                h[i] = (alpha/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*alpha)) +
                                               (1-2/np.pi)*np.cos(np.pi/(4*alpha)))
            else:
                h[i] = (np.sin(np.pi*ti*(1-alpha)) +
                        4*alpha*ti*np.cos(np.pi*ti*(1+alpha))) / \
                       (np.pi*ti*(1-(4*alpha*ti)**2))
        
        # Normalize filter
        h = h / np.sqrt(np.sum(h**2))
        
        # Apply filter
        filtered = np.convolve(signal, h, mode='same')
        return filtered
    
    def generate_modulated_signal(self, modulation_type, num_symbols=1000,
                                  samples_per_symbol=8, snr_db=20):
        """Generate modulated signal with specified parameters."""
        if modulation_type == 'BPSK':
            signal = self.generate_bpsk(num_symbols, samples_per_symbol)
        elif modulation_type == 'QPSK':
            signal = self.generate_qpsk(num_symbols, samples_per_symbol)
        elif modulation_type == '8PSK':
            signal = self.generate_8psk(num_symbols, samples_per_symbol)
        elif modulation_type == '16QAM':
            signal = self.generate_qam(num_symbols, 16, samples_per_symbol)
        elif modulation_type == '64QAM':
            signal = self.generate_qam(num_symbols, 64, samples_per_symbol)
        else:
            raise ValueError(f"Unsupported modulation type: {modulation_type}")
        
        # Add channel effects
        signal = self.add_channel_effects(signal, snr_db)
        
        return signal

class SDRDatasetManager:
    """Manage datasets for training and testing."""
    
    def __init__(self, base_path='./datasets'):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def generate_dataset(self, num_samples_per_class=100, snr_range=(0, 30),
                         signal_length=1024, modulations=None):
        """Generate comprehensive dataset for AMR."""
        if modulations is None:
            modulations = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']
        
        generator = SDRSignalGenerator()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.base_path, f'amr_dataset_{timestamp}.h5')
        
        # Calculate total samples
        snr_steps = list(range(snr_range[0], snr_range[1], 5))
        num_total = num_samples_per_class * len(modulations) * len(snr_steps)
        
        with h5py.File(filename, 'w') as f:
            # Pre-allocate arrays
            signals = f.create_dataset('signals', (num_total, signal_length), dtype=np.complex64)
            labels = f.create_dataset('labels', (num_total,), dtype='S10')
            snrs = f.create_dataset('snrs', (num_total,), dtype=np.float32)
            
            idx = 0
            for mod in modulations:
                print(f"Generating {mod} signals...")
                for snr in snr_steps:
                    for _ in range(num_samples_per_class):
                        # Generate signal
                        sig = generator.generate_modulated_signal(
                            mod, 
                            num_symbols=signal_length//8,
                            snr_db=snr
                        )
                        
                        # Store in dataset
                        signals[idx] = sig[:signal_length]
                        labels[idx] = mod.encode()
                        snrs[idx] = snr
                        idx += 1
            
            # Store metadata
            f.attrs['modulations'] = [m.encode() for m in modulations]
            f.attrs['sample_rate'] = generator.sample_rate
            f.attrs['creation_time'] = timestamp
        
        print(f"Dataset saved to {filename}")
        return filename
    
    def load_dataset(self, filename):
        """Load dataset from HDF5 file."""
        with h5py.File(filename, 'r') as f:
            signals = f['signals'][:]
            labels = f['labels'][:]
            snrs = f['snrs'][:]
            
            # Decode labels
            labels = np.array([l.decode() for l in labels])
            
            return signals, labels, snrs