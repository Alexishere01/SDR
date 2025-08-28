# ml/intelligent_receiver.py
from core.signal_generator import SDRSignalGenerator
from geminisdr.core.error_handling import (
    ErrorHandler, GeminiSDRError, HardwareError, ModelError, MemoryError,
    ErrorSeverity, ErrorContext, retry_with_backoff, fallback_to_simulation
)
from geminisdr.core.logging_manager import StructuredLogger
from geminisdr.config.config_manager import get_config_manager, SystemConfig
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import logging

# Use gymnasium instead of deprecated gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
        print("Warning: Using deprecated gym. Consider upgrading to gymnasium.")
    except ImportError:
        print("Warning: Neither gymnasium nor gym is available. Creating minimal fallback.")
        GYM_AVAILABLE = False
        
        # Create minimal fallback classes
        class spaces:
            class Box:
                def __init__(self, low, high, shape=None, dtype=None):
                    self.low = low
                    self.high = high
                    self.shape = shape
                    self.dtype = dtype
        
        class gym:
            class Env:
                def __init__(self):
                    self.action_space = None
                    self.observation_space = None

class SimulatedSDREnvironment(gym.Env):
    """Simulated environment for fast training without real SDR."""
    
    def __init__(self, target_freq=100e6, target_snr=20):
        super(SimulatedSDREnvironment, self).__init__()
        
        self.target_freq = target_freq
        self.target_snr = target_snr
        
        # Action space: [freq_adjust, gain_adjust, bandwidth_adjust]
        self.action_space = spaces.Box(
            low=np.array([-1e6, -10, -0.5]),
            high=np.array([1e6, 10, 0.5]),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(256,),
            dtype=np.float32
        )
        
        # Current state
        self.current_freq = 100e6
        self.current_gain = 30
        self.current_bandwidth = 1e6
        self.sample_rate = 2e6
        self.steps = 0
        self.max_steps = 50
    
    def reset(self, seed=None, options=None):
        """Reset to random initial state."""
        if seed is not None:
            np.random.seed(seed)
            
        # Randomize starting position away from target
        self.current_freq = self.target_freq + np.random.uniform(-5e6, 5e6)
        self.current_gain = np.random.uniform(10, 50)
        self.current_bandwidth = self.sample_rate * 0.8
        self.steps = 0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Apply action and return new state."""
        self.steps += 1
        
        # Apply adjustments
        freq_adjust, gain_adjust, bw_factor = action
        
        # Update parameters
        self.current_freq = np.clip(
            self.current_freq + freq_adjust,
            70e6, 200e6
        )
        self.current_gain = np.clip(
            self.current_gain + gain_adjust,
            0, 70
        )
        self.current_bandwidth = np.clip(
            self.sample_rate * (0.8 + bw_factor),
            self.sample_rate * 0.1,
            self.sample_rate * 0.9
        )
        
        # Get observation and reward
        observation = self._get_observation()
        reward = self._calculate_reward()
        
        # Episode ends if we find good signal or exceed max steps
        terminated = (reward > 50) or (self.steps >= self.max_steps)
        truncated = False
        
        info = {
            'snr': self._get_current_snr(),
            'freq': self.current_freq,
            'gain': self.current_gain,
            'freq_error': abs(self.current_freq - self.target_freq)
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Generate observation based on current tuning."""
        try:
            from geminisdr.ml.neural_amr import NeuralAMR
        except ImportError:
            from ml.neural_amr import NeuralAMR
        
        # Calculate how well-tuned we are
        current_snr = self._get_current_snr()
        
        # Generate signal with appropriate SNR
        generator = SDRSignalGenerator(self.sample_rate)
        signal = generator.generate_modulated_signal(
            'QPSK',
            num_symbols=256,
            snr_db=current_snr
        )
        
        # Compute spectrum
        if len(signal) >= 2048:
            fft = np.fft.fft(signal[:2048])
        else:
            padded_signal = np.pad(signal, (0, 2048-len(signal)))
            fft = np.fft.fft(padded_signal)
            
        spectrum = np.abs(fft[:1024])
        
        # Create feature vector similar to real SDR
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
            [self.current_bandwidth / self.sample_rate],
            [current_snr / 50],
            [0.1],  # Dummy signal power
            [0.01]  # Dummy noise floor
        ]).astype(np.float32)
        
        return features
    
    def _get_current_snr(self):
        """Calculate SNR based on tuning accuracy."""
        freq_error_mhz = abs(self.current_freq - self.target_freq) / 1e6
        
        # SNR drops off with frequency error
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
        
        # Add some noise
        snr += np.random.normal(0, 2)
        
        return np.clip(snr, -20, 40)
    
    def _calculate_reward(self):
        """Calculate reward based on current state."""
        snr = self._get_current_snr()
        freq_error_mhz = abs(self.current_freq - self.target_freq) / 1e6
        
        # Base reward on SNR (more aggressive)
        reward = snr * 2  # Double SNR importance
        
        # Progressive frequency accuracy rewards
        if freq_error_mhz < 0.05:  # Very close (50 kHz)
            reward += 50
        elif freq_error_mhz < 0.1:  # Close (100 kHz)
            reward += 30
        elif freq_error_mhz < 0.5:  # Reasonable (500 kHz)
            reward += 15
        elif freq_error_mhz < 1.0:  # Acceptable (1 MHz)
            reward += 5
        else:
            # Penalty for being far off
            reward -= min(50, freq_error_mhz * 10)  # Cap penalty
        
        # Strong bonus for good SNR
        if snr > 20:
            reward += 25
        elif snr > 15:
            reward += 15
        elif snr > 10:
            reward += 5
        
        # Penalty for very poor SNR
        if snr < -10:
            reward -= 20
        
        return reward

class DeepQLearningReceiver(nn.Module):
    """Deep Q-Network for receiver control."""
    
    def __init__(self, state_size=256, action_size=3, hidden_size=512):
        super(DeepQLearningReceiver, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 256)
        
        # Dueling DQN architecture
        self.value_head = nn.Linear(256, 1)
        self.advantage_head = nn.Linear(256, action_size * 11)  # 11 discrete levels per action
        
        self.action_size = action_size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        
        # Dueling network
        value = self.value_head(x)
        advantages = self.advantage_head(x).view(-1, self.action_size, 11)
        
        # Combine value and advantages
        q_values = value.unsqueeze(2) + (advantages - advantages.mean(2, keepdim=True))
        
        return q_values

class IntelligentReceiverML:
    """ML-driven intelligent receiver with cross-platform compatibility and error handling."""
    
    def __init__(self, sdr_interface, hardware_abstraction=None, config: SystemConfig = None):
        self.sdr = sdr_interface
        
        # Initialize error handling and logging
        self.logger = StructuredLogger(__name__, config.logging if config else None)
        self.error_handler = ErrorHandler(self.logger.logger)
        self._register_recovery_strategies()
        
        # Load configuration
        if config is None:
            try:
                config_manager = get_config_manager()
                self.config = config_manager.get_config()
                if self.config is None:
                    self.config = config_manager.load_config()
            except Exception as e:
                self.logger.log_error_with_context(e, component="IntelligentReceiverML", operation="config_load")
                # Use fallback configuration
                from geminisdr.config.config_models import SystemConfig, HardwareConfig, MLConfig, LoggingConfig, PerformanceConfig
                self.config = SystemConfig(
                    hardware=HardwareConfig(),
                    ml=MLConfig(),
                    logging=LoggingConfig(),
                    performance=PerformanceConfig()
                )
        else:
            self.config = config
        
        # Import hardware abstraction if available
        if hardware_abstraction is None:
            try:
                from environments.hardware_abstraction import HardwareAbstraction
                self.hw_abstraction = HardwareAbstraction()
            except (ImportError, Exception) as e:
                self.logger.logger.warning(f"Hardware abstraction not available, using fallback: {e}")
                self.hw_abstraction = None
        else:
            self.hw_abstraction = hardware_abstraction
        
        # Set device using hardware abstraction or fallback with error handling
        try:
            with self.error_handler.error_context("device_initialization", component="IntelligentReceiverML"):
                # Always use fallback for now to avoid hardware abstraction issues
                self.device = self._fallback_device_selection()
                self.optimization_config = self._get_fallback_optimizations()
                
                self.logger.logger.info(f"Using device: {self.device}")
                if self.optimization_config:
                    self.logger.logger.info(f"Platform optimizations: {self.optimization_config}")
                
                # Apply platform-specific configurations
                self._configure_platform_settings()
        except Exception as e:
            # Final fallback to CPU
            self.device = torch.device('cpu')
            self.optimization_config = {'device': 'cpu', 'recommended_batch_size': 16}
            self.logger.logger.warning(f"Device initialization failed, using CPU fallback: {e}")
        
        # Initialize DQN with platform-optimized settings and error handling
        try:
            with self.error_handler.error_context("model_initialization", component="IntelligentReceiverML"):
                self.q_network = DeepQLearningReceiver().to(self.device)
                self.target_network = DeepQLearningReceiver().to(self.device)
                
                # Use configuration-driven parameters
                self.batch_size = self.config.ml.batch_size or self.optimization_config.get('recommended_batch_size', 32)
                base_lr = self.config.ml.learning_rate
                
                # Platform-specific learning rate adjustments
                if self.device.type == 'mps':
                    # MPS may benefit from slightly higher learning rate
                    lr = base_lr * 1.2
                elif self.device.type == 'cpu':
                    # CPU training may benefit from lower learning rate
                    lr = base_lr * 0.8
                else:
                    lr = base_lr
                
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
                
                # Experience replay with platform-optimized memory size
                memory_size = self._get_optimal_memory_size()
                self.memory = deque(maxlen=memory_size)
                
                # Exploration parameters
                self.epsilon = 1.0
                self.epsilon_decay = 0.99  # Faster decay for better learning
                self.epsilon_min = 0.01
                
                # Action discretization
                self.freq_actions = np.linspace(-500e3, 500e3, 11)
                self.gain_actions = np.linspace(-5, 5, 11)
                self.bw_actions = np.linspace(-0.2, 0.2, 11)
                
                self.logger.logger.info(f"Initialized with batch_size={self.batch_size}, memory_size={memory_size}, lr={lr}")
                
                # Validate device compatibility
                self._validate_device_compatibility()
        except Exception as e:
            raise ModelError(
                f"Failed to initialize ML model: {str(e)}",
                model_name="DeepQLearningReceiver",
                severity=ErrorSeverity.HIGH,
                cause=e
            )
    
    def _register_recovery_strategies(self):
        """Register error recovery strategies."""
        # Hardware error recovery - fallback to simulation
        def hardware_fallback_strategy(error: Exception, context: ErrorContext) -> bool:
            try:
                self.logger.logger.warning(f"Hardware error detected, attempting fallback to simulation: {error}")
                # Switch to simulation mode if possible
                if hasattr(self, 'sdr') and hasattr(self.sdr, 'simulation_mode'):
                    self.sdr.simulation_mode = True
                    self.logger.logger.info("Successfully switched to simulation mode")
                    return True
                return False
            except Exception as e:
                self.logger.logger.error(f"Hardware fallback strategy failed: {e}")
                return False
        
        # Memory error recovery - reduce batch size
        def memory_recovery_strategy(error: Exception, context: ErrorContext) -> bool:
            try:
                if hasattr(self, 'batch_size') and self.batch_size > 8:
                    old_batch_size = self.batch_size
                    self.batch_size = max(8, self.batch_size // 2)
                    self.logger.logger.warning(f"Reduced batch size from {old_batch_size} to {self.batch_size} due to memory error")
                    return True
                return False
            except Exception as e:
                self.logger.logger.error(f"Memory recovery strategy failed: {e}")
                return False
        
        # Model error recovery - reinitialize model
        def model_recovery_strategy(error: Exception, context: ErrorContext) -> bool:
            try:
                self.logger.logger.warning("Attempting to reinitialize model due to model error")
                # Try to reinitialize the model with CPU fallback
                if self.device.type != 'cpu':
                    self.device = torch.device('cpu')
                    self.q_network = DeepQLearningReceiver().to(self.device)
                    self.target_network = DeepQLearningReceiver().to(self.device)
                    self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
                    self.logger.logger.info("Successfully reinitialized model on CPU")
                    return True
                return False
            except Exception as e:
                self.logger.logger.error(f"Model recovery strategy failed: {e}")
                return False
        
        # Register strategies
        self.error_handler.register_recovery_strategy(HardwareError, hardware_fallback_strategy)
        self.error_handler.register_recovery_strategy(MemoryError, memory_recovery_strategy)
        self.error_handler.register_recovery_strategy(ModelError, model_recovery_strategy)
    
    def _validate_device_compatibility(self):
        """Validate that the selected device works correctly."""
        try:
            with self.error_handler.error_context("device_validation", component="IntelligentReceiverML"):
                # Test basic tensor operations on the device
                test_tensor = torch.randn(2, 2).to(self.device)
                test_result = test_tensor @ test_tensor.T
                
                if self.device.type == 'mps':
                    # Test MPS-specific operations
                    test_conv = torch.nn.Conv1d(2, 4, 3).to(self.device)
                    test_input = torch.randn(1, 2, 10).to(self.device)
                    _ = test_conv(test_input)
                    self.logger.logger.info("✓ MPS device validation passed")
                elif self.device.type == 'cuda':
                    self.logger.logger.info(f"✓ CUDA device validation passed (GPU: {torch.cuda.get_device_name()})")
                else:
                    self.logger.logger.info("✓ CPU device validation passed")
                    
        except Exception as e:
            self.logger.log_error_with_context(e, component="IntelligentReceiverML", operation="device_validation")
            self.logger.logger.warning("Device validation warning, continuing with current configuration")
    
    def _fallback_device_selection(self):
        """Fallback device selection when hardware abstraction is unavailable."""
        try:
            # Check MPS availability (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device('mps')
            # Check CUDA availability
            elif torch.cuda.is_available():
                return torch.device('cuda')
            # Fallback to CPU
            else:
                return torch.device('cpu')
        except Exception as e:
            print(f"Warning: Error in device selection, falling back to CPU: {e}")
            return torch.device('cpu')
    
    def _get_fallback_optimizations(self):
        """Get basic optimizations when hardware abstraction is unavailable."""
        config = {
            'device': self.device.type,
            'recommended_batch_size': 32,
            'pin_memory': self.device.type in ['cuda', 'mps'],
            'thread_count': 4
        }
        
        if self.device.type == 'mps':
            config.update({
                'mixed_precision': False,
                'mps_fallback': True
            })
        elif self.device.type == 'cuda':
            config.update({
                'mixed_precision': True,
                'benchmark_mode': True
            })
        
        return config
    
    def _configure_platform_settings(self):
        """Configure platform-specific PyTorch settings."""
        # Configure MPS fallback if needed
        if self.device.type == 'mps':
            import os
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
            print("Enabled MPS fallback for compatibility")
        
        # Configure CPU threading
        if self.device.type == 'cpu':
            thread_count = self.optimization_config.get('thread_count', 4)
            torch.set_num_threads(thread_count)
            print(f"Set CPU threads to {thread_count}")
        
        # Configure CUDA optimizations
        elif self.device.type == 'cuda':
            if self.optimization_config.get('benchmark_mode', False):
                torch.backends.cudnn.benchmark = True
                print("Enabled cuDNN benchmark mode")
    
    def _get_optimal_memory_size(self):
        """Get optimal replay memory size based on platform capabilities."""
        if self.device.type == 'cpu':
            return 5000  # Conservative for CPU
        elif self.device.type == 'mps':
            return 8000  # Moderate for M1
        else:  # CUDA
            return 10000  # Full size for GPU
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, exceptions=(ModelError, MemoryError))
    def train_intelligent_search(self, num_episodes=500):
        """Train the receiver to find signals intelligently with platform optimizations and error handling."""
        try:
            with self.error_handler.error_context("training", component="IntelligentReceiverML", num_episodes=num_episodes):
                env = SimulatedSDREnvironment()
                
                self.logger.logger.info(f"Training intelligent receiver on {self.device} for {num_episodes} episodes...")
                rewards_history = []
                
                # Platform-specific training parameters
                target_update_freq = self._get_target_update_frequency()
                progress_report_freq = self._get_progress_report_frequency()
        except Exception as e:
            raise ModelError(
                f"Failed to initialize training: {str(e)}",
                model_name="DeepQLearningReceiver",
                severity=ErrorSeverity.HIGH,
                cause=e
            )
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 50:
                # Choose action
                action = self._choose_action(state)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.memory.append((state, action, reward, next_state, done))
                
                # Train with platform-optimized frequency
                if len(self.memory) > self.batch_size:
                    self._train_step()
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    if info['snr'] > 15:
                        print(f"Episode {episode}: Found signal! SNR: {info['snr']:.1f} dB, "
                              f"Freq error: {info['freq_error']/1e3:.1f} kHz")
                    break
            
            rewards_history.append(total_reward)
            
            # Update target network with platform-optimized frequency
            if episode % target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Progress reporting with platform-optimized frequency
            if episode % progress_report_freq == 0:
                avg_reward = np.mean(rewards_history[-progress_report_freq:]) if len(rewards_history) >= progress_report_freq else np.mean(rewards_history)
                device_info = f"({self.device})" if self.device.type != 'cpu' else ""
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Best: {max(rewards_history):.2f}, Epsilon: {self.epsilon:.3f} {device_info}")
        
        print(f"Training completed on {self.device}")
        return rewards_history
    
    def _get_target_update_frequency(self):
        """Get platform-optimized target network update frequency."""
        if self.device.type == 'cpu':
            return 20  # Less frequent updates for CPU to reduce overhead
        else:
            return 10  # More frequent updates for GPU/MPS
    
    def _get_progress_report_frequency(self):
        """Get platform-optimized progress reporting frequency."""
        if self.device.type == 'cpu':
            return 100  # Less frequent reporting for CPU
        else:
            return 50  # More frequent reporting for GPU/MPS
    
    def _choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            # Random exploration
            action_indices = np.random.randint(0, 11, 3)
        else:
            # Exploit learned policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_indices = q_values.argmax(dim=2).squeeze().cpu().numpy()
        
        # Convert indices to actual actions
        action = np.array([
            self.freq_actions[action_indices[0]],
            self.gain_actions[action_indices[1]],
            self.bw_actions[action_indices[2]]
        ])
        
        return action
    
    def _train_step(self):
        """Train the Q-network on a batch of experiences with platform optimizations."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Use platform-optimized tensor creation
        pin_memory = self.optimization_config.get('pin_memory', False)
        
        # Create tensors with platform-specific optimizations
        # Convert to numpy arrays first for better performance
        states_np = np.array(states, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)
        
        try:
            states = torch.from_numpy(states_np).to(self.device, non_blocking=pin_memory)
            next_states = torch.from_numpy(next_states_np).to(self.device, non_blocking=pin_memory)
            rewards = torch.from_numpy(rewards_np).to(self.device, non_blocking=pin_memory)
            dones = torch.from_numpy(dones_np).to(self.device, non_blocking=pin_memory)
        except Exception as e:
            # Fallback without non_blocking if it causes issues
            print(f"Warning: Non-blocking tensor transfer failed, using blocking: {e}")
            states = torch.from_numpy(states_np).to(self.device)
            next_states = torch.from_numpy(next_states_np).to(self.device)
            rewards = torch.from_numpy(rewards_np).to(self.device)
            dones = torch.from_numpy(dones_np).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states)
        
        # Next Q values from target network with platform-specific optimizations
        with torch.no_grad():  # More efficient for target network
            next_q = self.target_network(next_states).max(2)[0].max(1)[0]
            target_q = rewards + 0.99 * next_q * (1 - dones)
        
        # Loss computation
        current_q_values = current_q.max(2)[0].max(1)[0]
        loss = nn.MSELoss()(current_q_values, target_q)
        
        # Optimize with platform-specific settings
        self.optimizer.zero_grad()
        loss.backward()
        
        # Platform-specific gradient clipping
        if self.device.type == 'mps':
            # MPS benefits from gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        elif self.device.type == 'cuda':
            # CUDA can handle larger gradients but still clip for safety
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=2.0)
        # CPU typically doesn't need gradient clipping for this model
        
        self.optimizer.step()
    
    def find_signal_intelligently(self, search_time=30):
        """Use trained model to find signal in real-time."""
        print(f"\nSearching for signal intelligently for {search_time} seconds...")
        
        # Create environment with real SDR
        env = RealSDREnvironment(self.sdr)
        state = env.reset()
        
        start_time = time.time()
        best_result = {'snr': -np.inf, 'freq': None, 'gain': None}
        
        # Set to exploitation mode
        old_epsilon = self.epsilon
        self.epsilon = 0  # No exploration during deployment
        
        try:
            while time.time() - start_time < search_time:
                # Use learned policy
                action = self._choose_action(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Track best result
                if info['snr'] > best_result['snr']:
                    best_result = info
                    print(f"\rBest SNR: {info['snr']:.1f} dB at "
                          f"{info['freq']/1e6:.1f} MHz, Gain: {info['gain']:.0f} dB",
                           end='')
                
                state = next_state
                
                if done:
                    print(f"\n✓ Strong signal found!")
                    break
        finally:
            # Restore epsilon
            self.epsilon = old_epsilon
        
        return best_result
    
    def save_model(self, filepath):
        """Save trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model."""
        try:
            # Try with weights_only=False for compatibility
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            checkpoint = torch.load(filepath, map_location=self.device)
        except Exception as e:
            print(f"Warning: Error loading model with weights_only=False: {e}")
            # Final fallback
            checkpoint = torch.load(filepath, map_location=self.device)
            
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        print(f"Model loaded from {filepath} on device: {self.device}")

class RealSDREnvironment:
    """Environment wrapper for real SDR hardware."""
    
    def __init__(self, sdr_interface):
        self.sdr = sdr_interface
        self.current_freq = 100e6
        self.current_gain = 30
        self.sample_rate = 2e6
    
    def reset(self):
        """Reset to initial state."""
        self.current_freq = 100e6 + np.random.uniform(-2e6, 2e6)
        self.current_gain = 30
        
        self.sdr.configure(
            center_freq=self.current_freq,
            sample_rate=self.sample_rate,
            gain=self.current_gain
        )
        
        return self._get_observation()
    
    def step(self, action):
        """Apply action and return new state."""
        freq_adjust, gain_adjust, _ = action
        
        # Update parameters
        self.current_freq = np.clip(
            self.current_freq + freq_adjust,
            70e6, 6e9
        )
        self.current_gain = np.clip(
            self.current_gain + gain_adjust,
            0, 70
        )
        
        # Reconfigure SDR
        self.sdr.configure(
            center_freq=self.current_freq,
            sample_rate=self.sample_rate,
            gain=self.current_gain
        )
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward based on signal quality
        snr = self._estimate_snr()
        reward = snr
        
        # Check if we found good signal
        done = reward > 50
        
        info = {
            'snr': snr,
            'freq': self.current_freq,
            'gain': self.current_gain
        }
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """Get observation from real SDR."""
        # Capture samples
        samples = self.sdr.capture_batch(0.01)  # 10ms
        
        if samples is None or len(samples) < 2048:
            return np.zeros(256, dtype=np.float32)
        
        # Compute spectrum
        fft = np.fft.fft(samples[:2048])
        spectrum = np.abs(fft[:1024])
        
        # Downsample to 250 bins
        spectrum_bins = []
        for i in range(0, min(1000, len(spectrum)-4), 4):
            spectrum_bins.append(np.mean(spectrum[i:i+4]))
        spectrum_bins = np.array(spectrum_bins)
        
        # Pad or trim to exactly 250
        if len(spectrum_bins) < 250:
            spectrum_bins = np.pad(spectrum_bins, (0, 250 - len(spectrum_bins)))
        else:
            spectrum_bins = spectrum_bins[:250]
        
        # Normalize
        max_val = np.max(spectrum_bins)
        if max_val > 0:
            spectrum_norm = spectrum_bins / max_val
        else:
            spectrum_norm = spectrum_bins
        
        # Create feature vector
        features = np.concatenate([
            spectrum_norm,  # 250 values
            [self.current_freq / 1e9],
            [self.current_gain / 70],
            [0.8],  # bandwidth factor
            [0],    # placeholder SNR
            [0.1],  # signal power
            [0.01]  # noise floor
        ]).astype(np.float32)
        
        return features
    
    def _estimate_snr(self):
        """Estimate SNR from captured samples."""
        samples = self.sdr.capture_batch(0.01)
        
        if samples is None or len(samples) == 0:
            return -30.0
        
        # Simple SNR estimation
        signal_power = np.mean(np.abs(samples)**2)
        noise_floor = np.percentile(np.abs(samples)**2, 10)
        snr = 10 * np.log10(signal_power / (noise_floor + 1e-12))
        
        return float(snr)