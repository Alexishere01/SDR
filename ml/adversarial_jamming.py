# ml/adversarial_jamming.py
import numpy as np
import torch
import torch.nn as nn
import random
import time
from collections import deque

class AdversarialJammer:
    """Intelligent jammer that adapts to counter the AI receiver."""
    
    def __init__(self, power_budget=100, learning_rate=0.01):
        self.power_budget = power_budget
        self.learning_rate = learning_rate
        
        # Jamming strategies
        self.strategies = {
            'narrowband': self._narrowband_jamming,
            'wideband': self._wideband_jamming,
            'sweep': self._sweep_jamming,
            'pulse': self._pulse_jamming,
            'adaptive': self._adaptive_jamming,
            'deceptive': self._deceptive_jamming,
            'smart_follow': self._smart_follow_jamming
        }
        
        # Learning components
        self.receiver_history = deque(maxlen=100)
        self.success_rates = {strategy: 0.5 for strategy in self.strategies}
        self.strategy_usage = {strategy: 0 for strategy in self.strategies}
        
        # Current state
        self.current_strategy = 'narrowband'
        self.target_freq = 100e6
        self.jamming_active = False
        
        # Advanced parameters
        self.frequency_memory = deque(maxlen=50)  # Remember where receiver goes
        self.power_allocation = np.ones(len(self.strategies)) / len(self.strategies)
        
    def observe_receiver(self, freq, gain, snr_achieved):
        """Learn from receiver behavior."""
        observation = {
            'freq': freq,
            'gain': gain,
            'snr': snr_achieved,
            'timestamp': time.time(),
            'strategy_used': self.current_strategy
        }
        
        self.receiver_history.append(observation)
        self.frequency_memory.append(freq)
        
        # Update strategy success rates
        jamming_success = snr_achieved < 10  # Consider jamming successful if SNR < 10dB
        
        # Update success rate with exponential moving average
        old_rate = self.success_rates[self.current_strategy]
        self.success_rates[self.current_strategy] = (
            old_rate * 0.9 + jamming_success * 0.1
        )
        
        # Update power allocation (multi-armed bandit approach)
        self._update_power_allocation()
    
    def _update_power_allocation(self):
        """Update power allocation across strategies using UCB algorithm."""
        total_usage = sum(self.strategy_usage.values()) + 1e-6
        
        for i, strategy in enumerate(self.strategies.keys()):
            # Upper Confidence Bound calculation
            success_rate = self.success_rates[strategy]
            usage_count = self.strategy_usage[strategy] + 1
            confidence = np.sqrt(2 * np.log(total_usage) / usage_count)
            
            self.power_allocation[i] = success_rate + confidence
        
        # Normalize
        self.power_allocation = self.power_allocation / np.sum(self.power_allocation)
    
    def select_strategy(self):
        """Select jamming strategy based on learned effectiveness."""
        # Use epsilon-greedy with UCB
        if random.random() < 0.1:  # 10% exploration
            strategy = random.choice(list(self.strategies.keys()))
        else:
            # Select based on power allocation (exploitation)
            strategy_idx = np.random.choice(
                len(self.strategies), 
                p=self.power_allocation
            )
            strategy = list(self.strategies.keys())[strategy_idx]
        
        self.current_strategy = strategy
        self.strategy_usage[strategy] += 1
        return strategy
    
    def generate_jamming(self, signal, receiver_freq, receiver_gain, sample_rate):
        """Generate adaptive jamming signal."""
        if not self.jamming_active:
            return signal
        
        # Select and apply jamming strategy
        strategy = self.select_strategy()
        jammed_signal = self.strategies[strategy](
            signal, receiver_freq, receiver_gain, sample_rate
        )
        
        return jammed_signal
    
    def _narrowband_jamming(self, signal, rx_freq, rx_gain, sample_rate):
        """Narrowband jamming at receiver frequency."""
        # Target the exact receiver frequency
        jam_freq = rx_freq
        jam_power = min(self.power_budget * 0.8, 50)  # Concentrate power
        
        t = np.arange(len(signal)) / sample_rate
        jamming = jam_power * np.exp(1j * 2 * np.pi * jam_freq * t)
        
        return signal + jamming
    
    def _wideband_jamming(self, signal, rx_freq, rx_gain, sample_rate):
        """Wideband noise jamming."""
        jam_power = self.power_budget * 0.3  # Spread power across bandwidth
        
        # Generate wideband noise
        noise = jam_power * (np.random.randn(len(signal)) + 
                           1j * np.random.randn(len(signal)))
        
        return signal + noise
    
    def _sweep_jamming(self, signal, rx_freq, rx_gain, sample_rate):
        """Frequency sweeping jamming."""
        jam_power = self.power_budget * 0.6
        
        # Sweep around receiver frequency
        sweep_range = 2e6  # 2 MHz sweep
        sweep_rate = 1e6   # 1 MHz/second
        
        t = np.arange(len(signal)) / sample_rate
        sweep_freq = rx_freq + sweep_range * np.sin(2 * np.pi * sweep_rate * t)
        
        jamming = jam_power * np.exp(1j * 2 * np.pi * sweep_freq * t)
        
        return signal + jamming
    
    def _pulse_jamming(self, signal, rx_freq, rx_gain, sample_rate):
        """Pulse jamming with adaptive timing."""
        jam_power = self.power_budget * 1.2  # High power pulses
        
        # Adaptive pulse parameters based on receiver behavior
        if len(self.receiver_history) > 5:
            # Analyze receiver adaptation time
            recent_changes = [abs(h['freq'] - self.receiver_history[i-1]['freq']) 
                            for i, h in enumerate(self.receiver_history[-5:]) if i > 0]
            avg_change = np.mean(recent_changes) if recent_changes else 1e6
            
            # Adjust pulse rate based on receiver agility
            pulse_rate = max(10, min(100, 50 / (avg_change / 1e6 + 1)))
        else:
            pulse_rate = 50
        
        # Generate pulse pattern
        pulse_width = int(sample_rate / pulse_rate / 10)  # 10% duty cycle
        pulse_period = int(sample_rate / pulse_rate)
        
        jamming = np.zeros_like(signal)
        for i in range(0, len(signal), pulse_period):
            end_idx = min(i + pulse_width, len(signal))
            jamming[i:end_idx] = jam_power * (np.random.randn(end_idx - i) + 
                                            1j * np.random.randn(end_idx - i))
        
        return signal + jamming
    
    def _adaptive_jamming(self, signal, rx_freq, rx_gain, sample_rate):
        """Adaptive jamming that learns receiver patterns."""
        if len(self.frequency_memory) < 10:
            # Not enough data, use narrowband
            return self._narrowband_jamming(signal, rx_freq, rx_gain, sample_rate)
        
        # Predict where receiver will go next
        freq_changes = np.diff(list(self.frequency_memory)[-10:])
        predicted_change = np.mean(freq_changes) if len(freq_changes) > 0 else 0
        predicted_freq = rx_freq + predicted_change
        
        # Jam both current and predicted frequencies
        jam_power = self.power_budget * 0.4
        
        t = np.arange(len(signal)) / sample_rate
        
        # Current frequency jamming
        jamming1 = jam_power * np.exp(1j * 2 * np.pi * rx_freq * t)
        
        # Predicted frequency jamming
        jamming2 = jam_power * np.exp(1j * 2 * np.pi * predicted_freq * t)
        
        return signal + jamming1 + jamming2
    
    def _deceptive_jamming(self, signal, rx_freq, rx_gain, sample_rate):
        """Deceptive jamming that mimics real signals."""
        from core.signal_generator import SDRSignalGenerator
        
        # Generate fake signals that look real
        generator = SDRSignalGenerator(sample_rate)
        
        # Create multiple fake signals at different frequencies
        fake_signals = []
        jam_power = self.power_budget * 0.3
        
        for offset in [-1e6, -0.5e6, 0.5e6, 1e6]:  # Spread fake signals
            fake_freq = rx_freq + offset
            
            # Generate realistic modulated signal
            fake_mod = random.choice(['BPSK', 'QPSK', '8PSK'])
            fake_signal = generator.generate_modulated_signal(
                fake_mod, num_symbols=len(signal)//8, snr_db=15
            )
            
            # Frequency shift to target location
            t = np.arange(len(signal)) / sample_rate
            fake_signal = fake_signal[:len(signal)]  # Ensure same length
            fake_signal *= np.exp(1j * 2 * np.pi * offset * t)
            fake_signal *= jam_power
            
            fake_signals.append(fake_signal)
        
        # Combine all fake signals
        total_jamming = sum(fake_signals)
        
        return signal + total_jamming
    
    def _smart_follow_jamming(self, signal, rx_freq, rx_gain, sample_rate):
        """Smart jamming that follows receiver with delay."""
        if len(self.receiver_history) < 3:
            return self._narrowband_jamming(signal, rx_freq, rx_gain, sample_rate)
        
        # Analyze receiver movement pattern
        recent_freqs = [h['freq'] for h in self.receiver_history[-5:]]
        
        # Calculate receiver velocity (frequency change rate)
        if len(recent_freqs) >= 2:
            freq_velocity = (recent_freqs[-1] - recent_freqs[-2])
            
            # Lead the target based on velocity
            lead_time = 0.1  # 100ms lead time
            predicted_freq = rx_freq + freq_velocity * lead_time
        else:
            predicted_freq = rx_freq
        
        # Multi-tone jamming around predicted location
        jam_power = self.power_budget * 0.25
        t = np.arange(len(signal)) / sample_rate
        
        jamming = np.zeros_like(signal, dtype=complex)
        
        # Create jamming tones around predicted frequency
        for offset in [-200e3, -100e3, 0, 100e3, 200e3]:
            jam_freq = predicted_freq + offset
            tone = jam_power * np.exp(1j * 2 * np.pi * jam_freq * t)
            jamming += tone
        
        return signal + jamming
    
    def get_strategy_stats(self):
        """Get statistics about jamming strategy effectiveness."""
        stats = {
            'current_strategy': self.current_strategy,
            'success_rates': self.success_rates.copy(),
            'strategy_usage': self.strategy_usage.copy(),
            'power_allocation': dict(zip(self.strategies.keys(), self.power_allocation)),
            'total_observations': len(self.receiver_history)
        }
        
        return stats
    
    def set_jamming_active(self, active):
        """Enable/disable jamming."""
        self.jamming_active = active
        if active:
            print("🔴 Adversarial jamming ACTIVATED")
        else:
            print("🟢 Adversarial jamming DEACTIVATED")

class AdversarialSDREnvironment:
    """SDR environment with intelligent adversarial jamming."""
    
    def __init__(self, target_freq=100e6, target_snr=20, jammer_power=50):
        self.target_freq = target_freq
        self.target_snr = target_snr
        
        # Initialize jammer
        self.jammer = AdversarialJammer(power_budget=jammer_power)
        self.jammer.set_jamming_active(True)
        
        # Environment state
        self.current_freq = 100e6
        self.current_gain = 30
        self.sample_rate = 2e6
        self.steps = 0
        self.max_steps = 100  # Longer episodes for adversarial scenarios
        
        # Performance tracking
        self.receiver_wins = 0
        self.jammer_wins = 0
        
    def reset(self):
        """Reset environment with jammer active."""
        # Randomize starting position
        self.current_freq = self.target_freq + np.random.uniform(-5e6, 5e6)
        self.current_gain = np.random.uniform(10, 50)
        self.steps = 0
        
        # Reset jammer learning
        self.jammer.frequency_memory.clear()
        
        return self._get_observation()
    
    def step(self, action):
        """Step with adversarial jamming."""
        self.steps += 1
        
        # Apply receiver action
        freq_adjust, gain_adjust, bw_factor = action
        
        self.current_freq = np.clip(
            self.current_freq + freq_adjust,
            70e6, 200e6
        )
        self.current_gain = np.clip(
            self.current_gain + gain_adjust,
            0, 70
        )
        
        # Generate clean signal
        clean_snr = self._get_clean_snr()
        
        # Generate signal
        from core.signal_generator import SDRSignalGenerator
        generator = SDRSignalGenerator(self.sample_rate)
        signal = generator.generate_modulated_signal(
            'QPSK', num_symbols=256, snr_db=clean_snr
        )
        
        # Apply adversarial jamming
        jammed_signal = self.jammer.generate_jamming(
            signal, self.current_freq, self.current_gain, self.sample_rate
        )
        
        # Calculate final SNR after jamming
        final_snr = self._calculate_snr_with_jamming(signal, jammed_signal)
        
        # Let jammer learn from this interaction
        self.jammer.observe_receiver(self.current_freq, self.current_gain, final_snr)
        
        # Calculate reward (receiver perspective)
        reward = self._calculate_adversarial_reward(final_snr, clean_snr)
        
        # Episode termination
        receiver_success = final_snr > 15  # Receiver wins if SNR > 15dB
        jammer_success = final_snr < 5    # Jammer wins if SNR < 5dB
        
        if receiver_success:
            self.receiver_wins += 1
        elif jammer_success:
            self.jammer_wins += 1
        
        done = receiver_success or jammer_success or (self.steps >= self.max_steps)
        
        info = {
            'snr': final_snr,
            'clean_snr': clean_snr,
            'freq': self.current_freq,
            'gain': self.current_gain,
            'jammer_strategy': self.jammer.current_strategy,
            'receiver_wins': self.receiver_wins,
            'jammer_wins': self.jammer_wins,
            'jamming_effectiveness': clean_snr - final_snr
        }
        
        observation = self._get_observation_with_jamming(jammed_signal)
        
        return observation, reward, done, info
    
    def _get_clean_snr(self):
        """Calculate SNR without jamming."""
        freq_error_mhz = abs(self.current_freq - self.target_freq) / 1e6
        
        if freq_error_mhz < 0.1:
            snr = self.target_snr
        elif freq_error_mhz < 1:
            snr = self.target_snr - 10 * freq_error_mhz
        else:
            snr = self.target_snr - 10 - 20 * (freq_error_mhz - 1)
        
        # Gain affects SNR
        optimal_gain = 40
        gain_error = abs(self.current_gain - optimal_gain)
        snr -= gain_error / 10
        
        return np.clip(snr, -20, 40)
    
    def _calculate_snr_with_jamming(self, clean_signal, jammed_signal):
        """Calculate SNR after jamming is applied."""
        signal_power = np.mean(np.abs(clean_signal)**2)
        
        # Jamming power is the difference
        jamming_power = np.mean(np.abs(jammed_signal - clean_signal)**2)
        
        # Total noise includes original noise plus jamming
        total_noise_power = signal_power / (10**(self._get_clean_snr()/10)) + jamming_power
        
        if total_noise_power > 0:
            snr_with_jamming = 10 * np.log10(signal_power / total_noise_power)
        else:
            snr_with_jamming = self._get_clean_snr()
        
        return snr_with_jamming
    
    def _calculate_adversarial_reward(self, final_snr, clean_snr):
        """Calculate reward in adversarial environment."""
        # Base reward on final SNR
        reward = final_snr
        
        # Bonus for overcoming jamming
        jamming_overcome = max(0, final_snr - 10)  # Bonus if SNR > 10 despite jamming
        reward += jamming_overcome * 2
        
        # Penalty for being jammed effectively
        jamming_damage = max(0, clean_snr - final_snr)
        reward -= jamming_damage * 0.5
        
        # Big bonus for achieving good SNR under jamming
        if final_snr > 15:
            reward += 30
        
        # Penalty for very poor performance
        if final_snr < 0:
            reward -= 20
        
        return reward
    
    def _get_observation(self):
        """Get observation without jamming (for comparison)."""
        from core.signal_generator import SDRSignalGenerator
        
        generator = SDRSignalGenerator(self.sample_rate)
        signal = generator.generate_modulated_signal(
            'QPSK', num_symbols=256, snr_db=self._get_clean_snr()
        )
        
        return self._create_feature_vector(signal)
    
    def _get_observation_with_jamming(self, jammed_signal):
        """Get observation with jamming effects."""
        return self._create_feature_vector(jammed_signal)
    
    def _create_feature_vector(self, signal):
        """Create feature vector from signal."""
        # Compute spectrum
        if len(signal) >= 2048:
            fft = np.fft.fft(signal[:2048])
        else:
            padded_signal = np.pad(signal, (0, 2048-len(signal)))
            fft = np.fft.fft(padded_signal)
            
        spectrum = np.abs(fft[:1024])
        
        # Downsample to 250 bins
        spectrum_bins = []
        for i in range(0, min(1000, len(spectrum)), 4):
            spectrum_bins.append(np.mean(spectrum[i:i+4]))
        
        spectrum_bins = np.array(spectrum_bins)
        if len(spectrum_bins) < 250:
            spectrum_bins = np.pad(spectrum_bins, (0, 250 - len(spectrum_bins)))
        else:
            spectrum_bins = spectrum_bins[:250]
        
        # Normalize
        if np.max(spectrum_bins) > 0:
            spectrum_bins = spectrum_bins / np.max(spectrum_bins)
        
        # Create feature vector
        features = np.concatenate([
            spectrum_bins,  # 250 values
            [self.current_freq / 1e9],
            [self.current_gain / 70],
            [0.8],  # bandwidth factor
            [0],    # placeholder SNR
            [0.1],  # signal power
            [0.01]  # noise floor
        ]).astype(np.float32)
        
        return features
    
    def get_battle_stats(self):
        """Get adversarial battle statistics."""
        total_battles = self.receiver_wins + self.jammer_wins
        
        stats = {
            'receiver_wins': self.receiver_wins,
            'jammer_wins': self.jammer_wins,
            'total_battles': total_battles,
            'receiver_win_rate': self.receiver_wins / max(1, total_battles),
            'jammer_stats': self.jammer.get_strategy_stats()
        }
        
        return stats