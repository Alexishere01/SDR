#!/usr/bin/env python3
"""
Intelligent Receiver Model Testing Script

This script provides comprehensive testing capabilities for trained intelligent receiver models.
It can test models on various scenarios and generate detailed performance reports.

Usage:
    python scripts/test_intelligent_receiver.py --model models/intelligent_receiver_best.pth
    python scripts/test_intelligent_receiver.py --model models/intelligent_receiver_best.pth --scenarios interference
    python scripts/test_intelligent_receiver.py --model models/intelligent_receiver_best.pth --interactive
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geminisdr.config.config_manager import ConfigManager
from geminisdr.core.logging_manager import StructuredLogger
from ml.intelligent_receiver import IntelligentReceiverML, SimulatedSDREnvironment


class IntelligentReceiverTester:
    """Comprehensive testing suite for intelligent receiver models."""
    
    def __init__(self, model_path: str, config_path: str = None):
        """Initialize tester with model and configuration."""
        self.model_path = model_path
        
        # Load configuration
        self.config_manager = ConfigManager()
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self.config_manager.load_config()
        
        # Initialize logging
        self.logger = StructuredLogger("IntelligentReceiverTester", self.config.logging)
        
        # Create output directory
        self.output_dir = Path("outputs/intelligent_receiver_testing")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.receiver = self._load_model()
        
        self.logger.info(f"Intelligent Receiver Tester initialized with model: {model_path}")
    
    def _load_model(self) -> IntelligentReceiverML:
        """Load the trained intelligent receiver model."""
        # Create mock SDR interface
        class MockSDRInterface:
            def __init__(self):
                self.sample_rate = 2e6
                self.center_frequency = 100e6
                self.gain = 30
                self.simulation_mode = True
        
        mock_sdr = MockSDRInterface()
        
        try:
            receiver = IntelligentReceiverML(sdr_interface=mock_sdr, config=self.config)
            receiver.load_model(self.model_path)
            
            # Set to evaluation mode (no exploration)
            receiver.epsilon = 0.0
            
            self.logger.info("Model loaded successfully")
            return receiver
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def test_basic_scenarios(self, num_episodes: int = 50) -> Dict:
        """Test model on basic signal finding scenarios."""
        self.logger.info(f"Testing basic scenarios ({num_episodes} episodes)")
        
        results = {
            'episodes': [],
            'success_count': 0,
            'total_episodes': num_episodes,
            'avg_convergence_time': 0,
            'avg_snr_found': 0,
            'avg_frequency_error': 0
        }
        
        convergence_times = []
        snrs_found = []
        frequency_errors = []
        
        for episode in range(num_episodes):
            # Create random scenario
            target_freq = np.random.uniform(80e6, 180e6)
            target_snr = np.random.uniform(10, 25)
            
            env = SimulatedSDREnvironment(target_freq=target_freq, target_snr=target_snr)
            state, _ = env.reset()
            
            episode_result = {
                'episode': episode,
                'target_freq': target_freq,
                'target_snr': target_snr,
                'found_signal': False,
                'convergence_time': 0,
                'final_snr': 0,
                'frequency_error': 0,
                'total_reward': 0
            }
            
            total_reward = 0
            steps = 0
            
            while steps < 50:
                action = self.receiver._choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done and info['snr'] > 15:
                    episode_result['found_signal'] = True
                    episode_result['convergence_time'] = steps
                    episode_result['final_snr'] = info['snr']
                    episode_result['frequency_error'] = info['freq_error']
                    
                    results['success_count'] += 1
                    convergence_times.append(steps)
                    snrs_found.append(info['snr'])
                    frequency_errors.append(info['freq_error'])
                    break
            
            episode_result['total_reward'] = total_reward
            results['episodes'].append(episode_result)
            
            if (episode + 1) % 10 == 0:
                current_success_rate = results['success_count'] / (episode + 1)
                print(f"Progress: {episode + 1}/{num_episodes}, Success rate: {current_success_rate:.2%}")
        
        # Calculate averages
        results['success_rate'] = results['success_count'] / num_episodes
        results['avg_convergence_time'] = np.mean(convergence_times) if convergence_times else 0
        results['avg_snr_found'] = np.mean(snrs_found) if snrs_found else 0
        results['avg_frequency_error'] = np.mean(frequency_errors) if frequency_errors else 0
        
        self.logger.info("Basic scenarios test completed", extra={
            'success_rate': results['success_rate'],
            'avg_convergence_time': results['avg_convergence_time'],
            'avg_snr_found': results['avg_snr_found']
        })
        
        return results
    
    def test_interference_scenarios(self, num_episodes: int = 30) -> Dict:
        """Test model performance with interference present."""
        self.logger.info(f"Testing interference scenarios ({num_episodes} episodes)")
        
        results = {
            'episodes': [],
            'success_count': 0,
            'total_episodes': num_episodes,
            'interference_types': ['narrowband', 'wideband', 'pulsed']
        }
        
        for episode in range(num_episodes):
            # Create interference scenario
            target_freq = np.random.uniform(90e6, 170e6)
            target_snr = np.random.uniform(5, 20)  # Lower SNR due to interference
            
            env = SimulatedSDREnvironment(target_freq=target_freq, target_snr=target_snr)
            state, _ = env.reset()
            
            episode_result = {
                'episode': episode,
                'target_freq': target_freq,
                'target_snr': target_snr,
                'interference_type': np.random.choice(results['interference_types']),
                'found_signal': False,
                'convergence_time': 0,
                'final_snr': 0
            }
            
            steps = 0
            while steps < 50:
                action = self.receiver._choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                state = next_state
                steps += 1
                
                if done and info['snr'] > 10:  # Lower threshold for interference scenarios
                    episode_result['found_signal'] = True
                    episode_result['convergence_time'] = steps
                    episode_result['final_snr'] = info['snr']
                    results['success_count'] += 1
                    break
            
            results['episodes'].append(episode_result)
        
        results['success_rate'] = results['success_count'] / num_episodes
        
        self.logger.info("Interference scenarios test completed", extra={
            'success_rate': results['success_rate']
        })
        
        return results
    
    def test_frequency_sweep(self, freq_start: float = 70e6, freq_end: float = 200e6, 
                           num_points: int = 20) -> Dict:
        """Test model performance across different frequency ranges."""
        self.logger.info(f"Testing frequency sweep from {freq_start/1e6:.1f} to {freq_end/1e6:.1f} MHz")
        
        frequencies = np.linspace(freq_start, freq_end, num_points)
        results = {
            'frequencies': frequencies.tolist(),
            'success_rates': [],
            'avg_convergence_times': [],
            'avg_snrs_found': []
        }
        
        for freq in frequencies:
            # Test multiple episodes at this frequency
            episodes_per_freq = 10
            successes = 0
            convergence_times = []
            snrs_found = []
            
            for _ in range(episodes_per_freq):
                env = SimulatedSDREnvironment(target_freq=freq, target_snr=20)
                state, _ = env.reset()
                
                steps = 0
                while steps < 50:
                    action = self.receiver._choose_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    state = next_state
                    steps += 1
                    
                    if done and info['snr'] > 15:
                        successes += 1
                        convergence_times.append(steps)
                        snrs_found.append(info['snr'])
                        break
            
            success_rate = successes / episodes_per_freq
            avg_convergence = np.mean(convergence_times) if convergence_times else 0
            avg_snr = np.mean(snrs_found) if snrs_found else 0
            
            results['success_rates'].append(success_rate)
            results['avg_convergence_times'].append(avg_convergence)
            results['avg_snrs_found'].append(avg_snr)
            
            print(f"Freq: {freq/1e6:.1f} MHz, Success: {success_rate:.2%}, "
                  f"Avg convergence: {avg_convergence:.1f} steps")
        
        self.logger.info("Frequency sweep test completed")
        return results
    
    def test_snr_performance(self, snr_range: Tuple[float, float] = (-10, 30), 
                           num_points: int = 15) -> Dict:
        """Test model performance across different SNR levels."""
        self.logger.info(f"Testing SNR performance from {snr_range[0]} to {snr_range[1]} dB")
        
        snr_levels = np.linspace(snr_range[0], snr_range[1], num_points)
        results = {
            'snr_levels': snr_levels.tolist(),
            'success_rates': [],
            'avg_convergence_times': []
        }
        
        for snr in snr_levels:
            # Test multiple episodes at this SNR
            episodes_per_snr = 10
            successes = 0
            convergence_times = []
            
            for _ in range(episodes_per_snr):
                env = SimulatedSDREnvironment(target_freq=100e6, target_snr=snr)
                state, _ = env.reset()
                
                steps = 0
                while steps < 50:
                    action = self.receiver._choose_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    state = next_state
                    steps += 1
                    
                    # Adjust success threshold based on target SNR
                    success_threshold = max(5, min(15, snr - 5))
                    if done and info['snr'] > success_threshold:
                        successes += 1
                        convergence_times.append(steps)
                        break
            
            success_rate = successes / episodes_per_snr
            avg_convergence = np.mean(convergence_times) if convergence_times else 0
            
            results['success_rates'].append(success_rate)
            results['avg_convergence_times'].append(avg_convergence)
            
            print(f"SNR: {snr:+.1f} dB, Success: {success_rate:.2%}, "
                  f"Avg convergence: {avg_convergence:.1f} steps")
        
        self.logger.info("SNR performance test completed")
        return results
    
    def interactive_test(self):
        """Interactive testing mode for manual evaluation."""
        print("\n" + "="*60)
        print("INTERACTIVE INTELLIGENT RECEIVER TESTING")
        print("="*60)
        print("Commands:")
        print("  'test' - Run a single test episode")
        print("  'scenario <freq_mhz> <snr_db>' - Test specific scenario")
        print("  'sweep' - Quick frequency sweep")
        print("  'quit' - Exit interactive mode")
        print("="*60)
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'quit':
                    break
                
                elif command == 'test':
                    self._run_single_test()
                
                elif command.startswith('scenario'):
                    parts = command.split()
                    if len(parts) == 3:
                        freq_mhz = float(parts[1])
                        snr_db = float(parts[2])
                        self._run_scenario_test(freq_mhz * 1e6, snr_db)
                    else:
                        print("Usage: scenario <freq_mhz> <snr_db>")
                
                elif command == 'sweep':
                    self._run_quick_sweep()
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
            
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _run_single_test(self):
        """Run a single test episode with random parameters."""
        target_freq = np.random.uniform(80e6, 180e6)
        target_snr = np.random.uniform(5, 25)
        
        print(f"\nTesting: {target_freq/1e6:.1f} MHz, {target_snr:.1f} dB SNR")
        
        env = SimulatedSDREnvironment(target_freq=target_freq, target_snr=target_snr)
        state, _ = env.reset()
        
        steps = 0
        found_signal = False
        
        print("Step | Freq (MHz) | Gain (dB) | SNR (dB) | Reward")
        print("-" * 50)
        
        while steps < 50:
            action = self.receiver._choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"{steps:4d} | {info['freq']/1e6:8.1f} | {info['gain']:7.1f} | "
                  f"{info['snr']:6.1f} | {reward:6.1f}")
            
            state = next_state
            steps += 1
            
            if done and info['snr'] > 15:
                found_signal = True
                print(f"\n✅ Signal found in {steps} steps!")
                print(f"Final SNR: {info['snr']:.1f} dB")
                print(f"Frequency error: {info['freq_error']/1e3:.1f} kHz")
                break
        
        if not found_signal:
            print(f"\n❌ Signal not found after {steps} steps")
    
    def _run_scenario_test(self, freq: float, snr: float):
        """Run test with specific frequency and SNR."""
        print(f"\nTesting scenario: {freq/1e6:.1f} MHz, {snr:.1f} dB SNR")
        
        env = SimulatedSDREnvironment(target_freq=freq, target_snr=snr)
        state, _ = env.reset()
        
        steps = 0
        while steps < 50:
            action = self.receiver._choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            steps += 1
            
            if done and info['snr'] > max(5, snr - 10):
                print(f"✅ Found signal in {steps} steps")
                print(f"Final SNR: {info['snr']:.1f} dB")
                return
        
        print(f"❌ Signal not found after {steps} steps")
    
    def _run_quick_sweep(self):
        """Run a quick frequency sweep."""
        print("\nRunning quick frequency sweep...")
        
        frequencies = np.linspace(80e6, 180e6, 10)
        for freq in frequencies:
            env = SimulatedSDREnvironment(target_freq=freq, target_snr=20)
            state, _ = env.reset()
            
            steps = 0
            found = False
            
            while steps < 30:
                action = self.receiver._choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                state = next_state
                steps += 1
                
                if done and info['snr'] > 15:
                    found = True
                    break
            
            status = "✅" if found else "❌"
            print(f"{freq/1e6:6.1f} MHz: {status} ({steps:2d} steps)")
    
    def create_performance_plots(self, test_results: Dict):
        """Create visualization plots from test results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Intelligent Receiver Test Results', fontsize=16)
        
        # Basic scenarios success rate
        if 'basic_scenarios' in test_results:
            basic = test_results['basic_scenarios']
            axes[0, 0].bar(['Success', 'Failure'], 
                          [basic['success_count'], basic['total_episodes'] - basic['success_count']])
            axes[0, 0].set_title('Basic Scenarios Success Rate')
            axes[0, 0].set_ylabel('Episodes')
        
        # Frequency sweep results
        if 'frequency_sweep' in test_results:
            freq_sweep = test_results['frequency_sweep']
            axes[0, 1].plot(np.array(freq_sweep['frequencies'])/1e6, freq_sweep['success_rates'], 'o-')
            axes[0, 1].set_title('Success Rate vs Frequency')
            axes[0, 1].set_xlabel('Frequency (MHz)')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].grid(True)
        
        # SNR performance
        if 'snr_performance' in test_results:
            snr_perf = test_results['snr_performance']
            axes[1, 0].plot(snr_perf['snr_levels'], snr_perf['success_rates'], 'o-')
            axes[1, 0].set_title('Success Rate vs SNR')
            axes[1, 0].set_xlabel('SNR (dB)')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].grid(True)
        
        # Convergence time distribution
        if 'basic_scenarios' in test_results:
            basic = test_results['basic_scenarios']
            convergence_times = [ep['convergence_time'] for ep in basic['episodes'] 
                               if ep['found_signal']]
            if convergence_times:
                axes[1, 1].hist(convergence_times, bins=10, alpha=0.7)
                axes[1, 1].set_title('Convergence Time Distribution')
                axes[1, 1].set_xlabel('Steps to Convergence')
                axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to {plot_file}")
    
    def run_comprehensive_test(self) -> Dict:
        """Run all test scenarios and generate comprehensive report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE INTELLIGENT RECEIVER TESTING")
        print("="*60)
        
        all_results = {}
        
        # Test 1: Basic scenarios
        print("\n1. Testing basic signal finding scenarios...")
        all_results['basic_scenarios'] = self.test_basic_scenarios(50)
        
        # Test 2: Interference scenarios
        print("\n2. Testing interference scenarios...")
        all_results['interference_scenarios'] = self.test_interference_scenarios(30)
        
        # Test 3: Frequency sweep
        print("\n3. Testing frequency sweep...")
        all_results['frequency_sweep'] = self.test_frequency_sweep()
        
        # Test 4: SNR performance
        print("\n4. Testing SNR performance...")
        all_results['snr_performance'] = self.test_snr_performance()
        
        # Generate plots
        self.create_performance_plots(all_results)
        
        # Save results
        results_file = self.output_dir / f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        self._print_test_summary(all_results)
        
        return all_results
    
    def _print_test_summary(self, results: Dict):
        """Print a summary of all test results."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        if 'basic_scenarios' in results:
            basic = results['basic_scenarios']
            print(f"Basic Scenarios Success Rate: {basic['success_rate']:.1%}")
            print(f"Average Convergence Time: {basic['avg_convergence_time']:.1f} steps")
            print(f"Average SNR Found: {basic['avg_snr_found']:.1f} dB")
        
        if 'interference_scenarios' in results:
            interference = results['interference_scenarios']
            print(f"Interference Scenarios Success Rate: {interference['success_rate']:.1%}")
        
        if 'frequency_sweep' in results:
            freq_sweep = results['frequency_sweep']
            avg_success = np.mean(freq_sweep['success_rates'])
            print(f"Average Success Rate Across Frequencies: {avg_success:.1%}")
        
        if 'snr_performance' in results:
            snr_perf = results['snr_performance']
            avg_success = np.mean(snr_perf['success_rates'])
            print(f"Average Success Rate Across SNR Levels: {avg_success:.1%}")
        
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Intelligent Receiver Model Testing")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--mode", choices=['comprehensive', 'basic', 'interference', 'frequency', 'snr', 'interactive'],
                       default='comprehensive', help="Test mode")
    parser.add_argument("--episodes", type=int, default=50, help="Number of test episodes")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Initialize tester
    tester = IntelligentReceiverTester(args.model, args.config)
    
    try:
        if args.mode == 'comprehensive':
            tester.run_comprehensive_test()
        
        elif args.mode == 'basic':
            results = tester.test_basic_scenarios(args.episodes)
            print(f"Success rate: {results['success_rate']:.1%}")
        
        elif args.mode == 'interference':
            results = tester.test_interference_scenarios(args.episodes)
            print(f"Success rate with interference: {results['success_rate']:.1%}")
        
        elif args.mode == 'frequency':
            tester.test_frequency_sweep()
        
        elif args.mode == 'snr':
            tester.test_snr_performance()
        
        elif args.mode == 'interactive':
            tester.interactive_test()
        
        print("\n✅ Testing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()