#!/usr/bin/env python3
"""
Intelligent Receiver Demo

This example demonstrates how to use the intelligent receiver for automatic
signal finding and optimization.
"""

import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geminisdr.config.config_manager import ConfigManager
from ml.intelligent_receiver import IntelligentReceiverML, SimulatedSDREnvironment


def demo_basic_usage():
    """Demonstrate basic intelligent receiver usage."""
    print("=== Basic Intelligent Receiver Demo ===\n")
    
    # Create mock SDR interface for demo
    class MockSDRInterface:
        def __init__(self):
            self.sample_rate = 2e6
            self.center_frequency = 100e6
            self.gain = 30
            self.simulation_mode = True
        
        def set_sample_rate(self, rate):
            self.sample_rate = rate
        
        def set_center_frequency(self, freq):
            self.center_frequency = freq
        
        def set_gain(self, gain):
            self.gain = gain
    
    # Initialize components
    config = ConfigManager().load_config()
    mock_sdr = MockSDRInterface()
    
    print("1. Initializing Intelligent Receiver...")
    receiver = IntelligentReceiverML(sdr_interface=mock_sdr, config=config)
    print(f"   ✓ Using device: {receiver.device}")
    print(f"   ✓ Batch size: {receiver.batch_size}")
    print(f"   ✓ Memory size: {len(receiver.memory) if hasattr(receiver, 'memory') else 'N/A'}")
    
    return receiver


def demo_training():
    """Demonstrate model training."""
    print("\n=== Training Demo ===\n")
    
    receiver = demo_basic_usage()
    
    print("2. Training intelligent receiver (short demo)...")
    print("   Note: This is a short demo. For full training, use the training scripts.")
    
    # Short training demo (just a few episodes)
    try:
        rewards = receiver.train_intelligent_search(num_episodes=20)
        
        print(f"   ✓ Training completed!")
        print(f"   ✓ Episodes: {len(rewards)}")
        print(f"   ✓ Average reward: {np.mean(rewards):.2f}")
        print(f"   ✓ Best reward: {max(rewards):.2f}")
        print(f"   ✓ Final epsilon: {receiver.epsilon:.3f}")
        
        return receiver
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return None


def demo_signal_finding():
    """Demonstrate intelligent signal finding."""
    print("\n=== Signal Finding Demo ===\n")
    
    # Create a trained receiver (or use untrained for demo)
    receiver = demo_basic_usage()
    
    print("3. Demonstrating signal finding...")
    
    # Create test environment
    target_freq = 120e6  # 120 MHz
    target_snr = 20      # 20 dB
    
    print(f"   Target: {target_freq/1e6:.1f} MHz, {target_snr} dB SNR")
    
    env = SimulatedSDREnvironment(target_freq=target_freq, target_snr=target_snr)
    state, _ = env.reset()
    
    # Set to exploitation mode for demo
    original_epsilon = receiver.epsilon
    receiver.epsilon = 0.1  # Small exploration for demo
    
    print("\n   Step | Frequency (MHz) | Gain (dB) | SNR (dB) | Reward")
    print("   " + "-" * 55)
    
    total_reward = 0
    steps = 0
    found_signal = False
    
    try:
        while steps < 20:  # Limit steps for demo
            # Choose action using the model
            action = receiver._choose_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Display progress
            print(f"   {steps:4d} | {info['freq']/1e6:11.1f} | {info['gain']:7.1f} | "
                  f"{info['snr']:6.1f} | {reward:6.1f}")
            
            total_reward += reward
            state = next_state
            steps += 1
            
            # Check if signal found
            if done and info['snr'] > 15:
                found_signal = True
                print(f"\n   ✅ Signal found in {steps} steps!")
                print(f"   📡 Final frequency: {info['freq']/1e6:.1f} MHz")
                print(f"   📊 Final SNR: {info['snr']:.1f} dB")
                print(f"   🎯 Frequency error: {info['freq_error']/1e3:.1f} kHz")
                break
        
        if not found_signal:
            print(f"\n   ⏱️ Demo ended after {steps} steps (signal not found)")
            print("   Note: Untrained model may not find signals efficiently")
        
        print(f"   📈 Total reward: {total_reward:.1f}")
        
    finally:
        # Restore original epsilon
        receiver.epsilon = original_epsilon


def demo_performance_comparison():
    """Compare intelligent vs random search."""
    print("\n=== Performance Comparison Demo ===\n")
    
    print("4. Comparing intelligent vs random search...")
    
    # Test scenarios
    scenarios = [
        (100e6, 25),  # Easy: 100 MHz, 25 dB
        (150e6, 15),  # Medium: 150 MHz, 15 dB
        (80e6, 10),   # Hard: 80 MHz, 10 dB
    ]
    
    receiver = demo_basic_usage()
    receiver.epsilon = 0.0  # No exploration for fair comparison
    
    print("\n   Scenario | Method     | Success | Steps | SNR Found")
    print("   " + "-" * 50)
    
    for i, (freq, snr) in enumerate(scenarios):
        scenario_name = ["Easy", "Medium", "Hard"][i]
        
        # Test intelligent search
        env = SimulatedSDREnvironment(target_freq=freq, target_snr=snr)
        state, _ = env.reset()
        
        intelligent_success = False
        intelligent_steps = 0
        intelligent_snr = 0
        
        for step in range(30):
            action = receiver._choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            intelligent_steps = step + 1
            
            if done and info['snr'] > max(5, snr - 10):
                intelligent_success = True
                intelligent_snr = info['snr']
                break
        
        # Test random search
        env = SimulatedSDREnvironment(target_freq=freq, target_snr=snr)
        state, _ = env.reset()
        
        random_success = False
        random_steps = 0
        random_snr = 0
        
        for step in range(30):
            # Random action
            action = np.array([
                np.random.uniform(-500e3, 500e3),  # freq
                np.random.uniform(-5, 5),          # gain
                np.random.uniform(-0.2, 0.2)       # bandwidth
            ])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            random_steps = step + 1
            
            if done and info['snr'] > max(5, snr - 10):
                random_success = True
                random_snr = info['snr']
                break
        
        # Display results
        print(f"   {scenario_name:8s} | Intelligent | {'✓' if intelligent_success else '✗':7s} | "
              f"{intelligent_steps:5d} | {intelligent_snr:7.1f}")
        print(f"   {' ':8s} | Random      | {'✓' if random_success else '✗':7s} | "
              f"{random_steps:5d} | {random_snr:7.1f}")
        print("   " + "-" * 50)


def demo_model_save_load():
    """Demonstrate saving and loading models."""
    print("\n=== Model Save/Load Demo ===\n")
    
    print("5. Demonstrating model save and load...")
    
    # Create and train a small model
    receiver = demo_basic_usage()
    
    # Quick training
    print("   Training small model...")
    try:
        rewards = receiver.train_intelligent_search(num_episodes=10)
        print(f"   ✓ Training completed (avg reward: {np.mean(rewards):.2f})")
    except Exception as e:
        print(f"   ⚠️ Training failed, continuing with untrained model: {e}")
    
    # Save model
    model_path = "demo_intelligent_receiver.pth"
    print(f"   Saving model to {model_path}...")
    
    try:
        receiver.save_model(model_path)
        print("   ✓ Model saved successfully")
        
        # Create new receiver and load model
        print("   Loading model into new receiver...")
        new_receiver = demo_basic_usage()
        new_receiver.load_model(model_path)
        print("   ✓ Model loaded successfully")
        
        # Compare parameters
        print(f"   Original epsilon: {receiver.epsilon:.3f}")
        print(f"   Loaded epsilon: {new_receiver.epsilon:.3f}")
        
        # Clean up
        os.remove(model_path)
        print("   ✓ Demo model file cleaned up")
        
    except Exception as e:
        print(f"   ❌ Save/load failed: {e}")


def main():
    """Run all demos."""
    print("🤖 Intelligent Receiver Demo")
    print("=" * 60)
    
    try:
        # Run demos
        demo_basic_usage()
        demo_signal_finding()
        demo_performance_comparison()
        demo_model_save_load()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("\nNext steps:")
        print("• Run full training: python scripts/run_intelligent_receiver_pipeline.py")
        print("• Try interactive mode: python scripts/run_intelligent_receiver_pipeline.py --interactive")
        print("• Read the guide: docs/intelligent_receiver_training.md")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()