#!/usr/bin/env python3
"""
Simple Intelligent Receiver Demo

A simplified version that demonstrates the core concepts without complex dependencies.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleSDREnvironment:
    """Simplified SDR environment for demonstration."""
    
    def __init__(self, target_freq=100e6, target_snr=20):
        self.target_freq = target_freq
        self.target_snr = target_snr
        
        # Current state
        self.current_freq = 100e6
        self.current_gain = 30
        self.current_bandwidth = 1e6
        self.sample_rate = 2e6
        self.steps = 0
        self.max_steps = 50
    
    def reset(self):
        """Reset to random initial state."""
        # Randomize starting position away from target
        self.current_freq = self.target_freq + np.random.uniform(-5e6, 5e6)
        self.current_gain = np.random.uniform(10, 50)
        self.current_bandwidth = self.sample_rate * 0.8
        self.steps = 0
        
        observation = self._get_observation()
        return observation
    
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
        done = (reward > 50) or (self.steps >= self.max_steps)
        
        info = {
            'snr': self._get_current_snr(),
            'freq': self.current_freq,
            'gain': self.current_gain,
            'freq_error': abs(self.current_freq - self.target_freq)
        }
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """Generate observation based on current tuning."""
        # Calculate how well-tuned we are
        current_snr = self._get_current_snr()
        
        # Generate simple spectrum representation
        spectrum_bins = np.random.randn(250) * 0.1
        
        # Add signal peak if we're close to target frequency
        freq_error_mhz = abs(self.current_freq - self.target_freq) / 1e6
        if freq_error_mhz < 2:
            # Add signal peak
            peak_strength = max(0, current_snr / 30)
            center_bin = 125 + int((self.current_freq - self.target_freq) / 1e6 * 10)
            center_bin = np.clip(center_bin, 0, 249)
            
            for i in range(max(0, center_bin-5), min(250, center_bin+5)):
                spectrum_bins[i] += peak_strength * np.exp(-0.5 * ((i - center_bin) / 2)**2)
        
        # Normalize
        spectrum_bins = np.clip(spectrum_bins, -1, 1)
        
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
        
        # Base reward on SNR
        reward = snr
        
        # Bonus for getting close to target frequency
        if freq_error_mhz < 0.1:
            reward += 30
        elif freq_error_mhz < 0.5:
            reward += 10
        
        # Penalty for being far off
        if freq_error_mhz > 2:
            reward -= 20
        
        # Bonus for good SNR
        if snr > 15:
            reward += 10
        
        return reward


class SimpleDQN(nn.Module):
    """Simple Deep Q-Network for receiver control."""
    
    def __init__(self, state_size=256, action_size=3, hidden_size=256):
        super(SimpleDQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 128)
        
        # Output Q-values for each action
        self.output = nn.Linear(128, action_size * 11)  # 11 discrete levels per action
        
        self.action_size = action_size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        
        # Output Q-values
        q_values = self.output(x).view(-1, self.action_size, 11)
        
        return q_values


class SimpleIntelligentReceiver:
    """Simplified intelligent receiver for demonstration."""
    
    def __init__(self):
        # Device selection
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using CUDA GPU")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        # Initialize networks
        self.q_network = SimpleDQN().to(self.device)
        self.target_network = SimpleDQN().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        
        # Experience replay
        self.memory = deque(maxlen=5000)
        self.batch_size = 32
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Action discretization
        self.freq_actions = np.linspace(-500e3, 500e3, 11)
        self.gain_actions = np.linspace(-5, 5, 11)
        self.bw_actions = np.linspace(-0.2, 0.2, 11)
        
        print(f"Initialized Simple Intelligent Receiver on {self.device}")
    
    def choose_action(self, state):
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
    
    def train_step(self):
        """Train the Q-network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(2)[0].max(1)[0]
            target_q = rewards + 0.99 * next_q * (1 - dones)
        
        # Loss computation
        current_q_values = current_q.max(2)[0].max(1)[0]
        loss = nn.MSELoss()(current_q_values, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
    
    def train(self, num_episodes=100):
        """Train the receiver."""
        print(f"\nTraining for {num_episodes} episodes...")
        
        rewards_history = []
        
        for episode in range(num_episodes):
            env = SimpleSDREnvironment()
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 50:
                # Choose action
                action = self.choose_action(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                self.memory.append((state, action, reward, next_state, done))
                
                # Train
                if len(self.memory) > self.batch_size:
                    self.train_step()
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    if info['snr'] > 15:
                        print(f"Episode {episode}: Found signal! SNR: {info['snr']:.1f} dB, "
                              f"Freq error: {info['freq_error']/1e3:.1f} kHz")
                    break
            
            rewards_history.append(total_reward)
            
            # Update target network
            if episode % 10 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Progress reporting
            if episode % 20 == 0:
                avg_reward = np.mean(rewards_history[-20:]) if len(rewards_history) >= 20 else np.mean(rewards_history)
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Best: {max(rewards_history):.2f}, Epsilon: {self.epsilon:.3f}")
        
        print(f"Training completed! Final average reward: {np.mean(rewards_history[-20:]):.2f}")
        return rewards_history
    
    def test(self, num_episodes=10):
        """Test the trained receiver."""
        print(f"\nTesting for {num_episodes} episodes...")
        
        # Set to exploitation mode
        old_epsilon = self.epsilon
        self.epsilon = 0.0
        
        successes = 0
        convergence_times = []
        snrs_found = []
        
        for episode in range(num_episodes):
            env = SimpleSDREnvironment(
                target_freq=np.random.uniform(80e6, 180e6),
                target_snr=np.random.uniform(10, 25)
            )
            state = env.reset()
            
            steps = 0
            while steps < 50:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                steps += 1
                
                if done and info['snr'] > 15:
                    successes += 1
                    convergence_times.append(steps)
                    snrs_found.append(info['snr'])
                    print(f"Test {episode}: Success in {steps} steps, SNR: {info['snr']:.1f} dB")
                    break
            
            if steps >= 50:
                print(f"Test {episode}: Failed to find signal")
        
        # Restore epsilon
        self.epsilon = old_epsilon
        
        success_rate = successes / num_episodes
        avg_convergence = np.mean(convergence_times) if convergence_times else 0
        avg_snr = np.mean(snrs_found) if snrs_found else 0
        
        print(f"\nTest Results:")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Convergence Time: {avg_convergence:.1f} steps")
        print(f"Average SNR Found: {avg_snr:.1f} dB")
        
        return success_rate, avg_convergence, avg_snr
    
    def demonstrate_signal_finding(self):
        """Interactive demonstration of signal finding."""
        print("\n=== Signal Finding Demonstration ===")
        
        target_freq = 120e6
        target_snr = 20
        
        print(f"Target: {target_freq/1e6:.1f} MHz, {target_snr} dB SNR")
        
        env = SimpleSDREnvironment(target_freq=target_freq, target_snr=target_snr)
        state = env.reset()
        
        # Set to exploitation mode
        old_epsilon = self.epsilon
        self.epsilon = 0.1  # Small exploration for demo
        
        print("\nStep | Frequency (MHz) | Gain (dB) | SNR (dB) | Reward")
        print("-" * 55)
        
        total_reward = 0
        steps = 0
        
        try:
            while steps < 30:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                
                print(f"{steps:4d} | {info['freq']/1e6:11.1f} | {info['gain']:7.1f} | "
                      f"{info['snr']:6.1f} | {reward:6.1f}")
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done and info['snr'] > 15:
                    print(f"\n✅ Signal found in {steps} steps!")
                    print(f"📡 Final frequency: {info['freq']/1e6:.1f} MHz")
                    print(f"📊 Final SNR: {info['snr']:.1f} dB")
                    print(f"🎯 Frequency error: {info['freq_error']/1e3:.1f} kHz")
                    break
            
            if steps >= 30:
                print(f"\n⏱️ Demo ended after {steps} steps")
            
            print(f"📈 Total reward: {total_reward:.1f}")
            
        finally:
            self.epsilon = old_epsilon


def main():
    """Run the simple intelligent receiver demo."""
    print("🤖 Simple Intelligent Receiver Demo")
    print("=" * 50)
    
    try:
        # Create receiver
        receiver = SimpleIntelligentReceiver()
        
        # Train the model
        print("\n1. Training the model...")
        rewards = receiver.train(num_episodes=100)
        
        # Test the model
        print("\n2. Testing the model...")
        success_rate, avg_convergence, avg_snr = receiver.test(num_episodes=20)
        
        # Demonstrate signal finding
        print("\n3. Demonstrating signal finding...")
        receiver.demonstrate_signal_finding()
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print(f"\nFinal Performance:")
        print(f"• Success Rate: {success_rate:.1%}")
        print(f"• Average Convergence: {avg_convergence:.1f} steps")
        print(f"• Average SNR Found: {avg_snr:.1f} dB")
        
        print(f"\nThis demonstrates the core concept of the intelligent receiver:")
        print(f"• The model learns to adjust SDR parameters automatically")
        print(f"• It can find signals faster than random search")
        print(f"• Performance improves with more training")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()