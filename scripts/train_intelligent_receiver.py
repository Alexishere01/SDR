#!/usr/bin/env python3
"""
Intelligent Receiver Training Pipeline

This script provides a complete pipeline for:
1. Generating synthetic training data for intelligent receiver
2. Training the Deep Q-Learning model
3. Testing and evaluating the trained model
4. Saving model artifacts and performance metrics

Usage:
    python scripts/train_intelligent_receiver.py --mode full
    python scripts/train_intelligent_receiver.py --mode train_only
    python scripts/train_intelligent_receiver.py --mode test_only --model_path models/intelligent_receiver.pth
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
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geminisdr.config.config_manager import ConfigManager
from geminisdr.core.logging_manager import StructuredLogger
from geminisdr.core.memory_manager import MemoryManager
from geminisdr.core.model_manager import ModelManager
from geminisdr.core.metrics_collector import MetricsCollector
from ml.intelligent_receiver import IntelligentReceiverML, SimulatedSDREnvironment
from core.signal_generator import SDRSignalGenerator


class IntelligentReceiverTrainer:
    """Complete training pipeline for intelligent receiver."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize trainer with configuration."""
        # Load configuration
        self.config_manager = ConfigManager()
        try:
            if config_path:
                # Try to load specific config, fallback to default if not found
                try:
                    self.config = self.config_manager.load_config(config_path)
                except Exception as e:
                    print(f"Warning: Could not load {config_path}, using default config: {e}")
                    self.config = self.config_manager.load_config()
            else:
                self.config = self.config_manager.load_config()
        except Exception as e:
            print(f"Warning: Could not load config, using fallback: {e}")
            # Create fallback configuration
            from geminisdr.config.config_models import SystemConfig, HardwareConfig, MLConfig, LoggingConfig, PerformanceConfig
            self.config = SystemConfig(
                hardware=HardwareConfig(),
                ml=MLConfig(),
                logging=LoggingConfig(),
                performance=PerformanceConfig()
            )
        
        # Initialize components
        self.logger = StructuredLogger("IntelligentReceiverTrainer", self.config.logging)
        self.memory_manager = MemoryManager(self.config)
        self.model_manager = ModelManager()
        self.metrics_collector = MetricsCollector(self.config)
        
        # Create output directories
        self.output_dir = Path("outputs/intelligent_receiver")
        self.models_dir = Path("models")
        self.data_dir = Path("data/intelligent_receiver")
        
        for directory in [self.output_dir, self.models_dir, self.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.training_params = {
            'num_episodes': 1000,
            'test_episodes': 100,
            'validation_episodes': 50,
            'save_interval': 100,
            'eval_interval': 50
        }
        
        self.logger.info("Intelligent Receiver Trainer initialized", extra={
            'output_dir': str(self.output_dir),
            'models_dir': str(self.models_dir),
            'data_dir': str(self.data_dir)
        })
    
    def generate_training_scenarios(self, num_scenarios: int = 1000) -> List[Dict]:
        """Generate diverse training scenarios for the intelligent receiver."""
        self.logger.info(f"Generating {num_scenarios} training scenarios...")
        
        scenarios = []
        signal_generator = SDRSignalGenerator(sample_rate=2e6)
        
        # Define scenario parameters
        modulation_types = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
        frequency_ranges = [
            (70e6, 100e6),    # VHF Low
            (100e6, 200e6),   # VHF High
            (200e6, 400e6),   # UHF Low
            (400e6, 800e6),   # UHF High
        ]
        snr_ranges = [
            (-10, 0),   # Very poor
            (0, 10),    # Poor
            (10, 20),   # Good
            (20, 30),   # Excellent
        ]
        
        for i in range(num_scenarios):
            # Random scenario parameters
            modulation = np.random.choice(modulation_types)
            freq_range = frequency_ranges[np.random.randint(len(frequency_ranges))]
            snr_range = snr_ranges[np.random.randint(len(snr_ranges))]
            
            target_freq = np.random.uniform(freq_range[0], freq_range[1])
            target_snr = np.random.uniform(snr_range[0], snr_range[1])
            
            # Add interference scenarios
            has_interference = np.random.random() < 0.3
            interference_type = None
            if has_interference:
                interference_types = ['narrowband', 'wideband', 'pulsed']
                interference_type = interference_types[np.random.randint(len(interference_types))]
            
            scenario = {
                'id': i,
                'modulation': modulation,
                'target_frequency': target_freq,
                'target_snr': target_snr,
                'has_interference': has_interference,
                'interference_type': interference_type,
                'bandwidth': np.random.uniform(100e3, 2e6),
                'symbol_rate': np.random.uniform(10e3, 1e6),
                'difficulty': self._calculate_scenario_difficulty(target_snr, has_interference)
            }
            
            scenarios.append(scenario)
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"Generated {i + 1}/{num_scenarios} scenarios")
        
        # Save scenarios
        scenarios_file = self.data_dir / "training_scenarios.json"
        with open(scenarios_file, 'w') as f:
            json.dump(scenarios, f, indent=2)
        
        self.logger.info(f"Training scenarios saved to {scenarios_file}")
        return scenarios
    
    def _calculate_scenario_difficulty(self, snr: float, has_interference: bool) -> str:
        """Calculate difficulty level of a scenario."""
        if snr < 0:
            base_difficulty = 'hard'
        elif snr < 10:
            base_difficulty = 'medium'
        else:
            base_difficulty = 'easy'
        
        if has_interference:
            if base_difficulty == 'easy':
                return 'medium'
            elif base_difficulty == 'medium':
                return 'hard'
            else:
                return 'very_hard'
        
        return base_difficulty
    
    def create_mock_sdr_interface(self):
        """Create a mock SDR interface for training."""
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
            
            def capture_samples(self, num_samples):
                # Return mock I/Q samples
                return np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        
        return MockSDRInterface()
    
    def train_model(self, num_episodes: int = None, resume_from: str = None) -> Dict:
        """Train the intelligent receiver model."""
        if num_episodes is None:
            num_episodes = self.training_params['num_episodes']
        
        self.logger.info(f"Starting training for {num_episodes} episodes")
        
        # Create mock SDR interface
        mock_sdr = self.create_mock_sdr_interface()
        
        # Initialize intelligent receiver
        try:
            receiver = IntelligentReceiverML(
                sdr_interface=mock_sdr,
                config=self.config
            )
            
            # Resume from checkpoint if specified
            if resume_from and os.path.exists(resume_from):
                receiver.load_model(resume_from)
                self.logger.info(f"Resumed training from {resume_from}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize intelligent receiver: {e}")
            raise
        
        # Training metrics
        training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'average_snr_found': [],
            'training_loss': [],
            'epsilon_values': [],
            'memory_usage': []
        }
        
        # Training loop with enhanced monitoring
        start_time = time.time()
        best_avg_reward = -np.inf
        episodes_since_improvement = 0
        
        try:
            for episode in range(num_episodes):
                episode_start = time.time()
                
                # Monitor memory usage
                memory_stats = self.memory_manager.get_memory_stats()
                training_metrics['memory_usage'].append(memory_stats.used_ram_mb)
                
                # Train one episode
                episode_reward, episode_length, episode_info = self._train_episode(receiver, episode)
                
                # Record metrics
                training_metrics['episode_rewards'].append(episode_reward)
                training_metrics['episode_lengths'].append(episode_length)
                training_metrics['epsilon_values'].append(receiver.epsilon)
                
                # Calculate success rate (last 100 episodes)
                recent_rewards = training_metrics['episode_rewards'][-100:]
                success_rate = sum(1 for r in recent_rewards if r > 50) / len(recent_rewards)
                training_metrics['success_rate'].append(success_rate)
                
                # Record average SNR found
                if 'snr' in episode_info:
                    training_metrics['average_snr_found'].append(episode_info['snr'])
                
                # Log metrics to collector
                self.metrics_collector.record_ml_metrics(
                    model_name="intelligent_receiver",
                    operation="training",
                    epoch=episode,
                    batch_size=receiver.batch_size,
                    loss=episode_info.get('loss', 0),
                    accuracy=success_rate,
                    duration=time.time() - episode_start,
                    memory_usage_mb=memory_stats.used_ram_mb
                )
                
                # Evaluation and checkpointing
                if (episode + 1) % self.training_params['eval_interval'] == 0:
                    avg_reward = np.mean(training_metrics['episode_rewards'][-50:])
                    
                    self.logger.info(f"Episode {episode + 1}/{num_episodes}", extra={
                        'avg_reward': avg_reward,
                        'success_rate': success_rate,
                        'epsilon': receiver.epsilon,
                        'memory_mb': memory_stats.used_ram_mb
                    })
                    
                    # Check for improvement
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        episodes_since_improvement = 0
                        
                        # Save best model
                        best_model_path = self.models_dir / "intelligent_receiver_best.pth"
                        receiver.save_model(str(best_model_path))
                        self.logger.info(f"New best model saved: {avg_reward:.2f}")
                    else:
                        episodes_since_improvement += self.training_params['eval_interval']
                
                # Save checkpoint
                if (episode + 1) % self.training_params['save_interval'] == 0:
                    checkpoint_path = self.models_dir / f"intelligent_receiver_ep_{episode + 1}.pth"
                    receiver.save_model(str(checkpoint_path))
                
                # Early stopping check
                if episodes_since_improvement > 200:
                    self.logger.info(f"Early stopping at episode {episode + 1} (no improvement for 200 episodes)")
                    break
                
                # Memory cleanup
                if episode % 50 == 0:
                    self.memory_manager.cleanup_memory()
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        # Final model save
        final_model_path = self.models_dir / "intelligent_receiver_final.pth"
        receiver.save_model(str(final_model_path))
        
        # Training summary
        training_time = time.time() - start_time
        final_metrics = {
            'training_time_hours': training_time / 3600,
            'total_episodes': len(training_metrics['episode_rewards']),
            'final_avg_reward': np.mean(training_metrics['episode_rewards'][-100:]),
            'best_avg_reward': best_avg_reward,
            'final_success_rate': training_metrics['success_rate'][-1] if training_metrics['success_rate'] else 0,
            'final_epsilon': receiver.epsilon
        }
        
        self.logger.info("Training completed", extra=final_metrics)
        
        # Save training metrics
        metrics_file = self.output_dir / f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'training_metrics': training_metrics,
                'final_metrics': final_metrics,
                'training_params': self.training_params
            }, f, indent=2)
        
        return final_metrics
    
    def _train_episode(self, receiver: IntelligentReceiverML, episode: int) -> Tuple[float, int, Dict]:
        """Train a single episode."""
        env = SimulatedSDREnvironment()
        state, _ = env.reset()
        
        total_reward = 0
        steps = 0
        episode_info = {}
        
        while steps < 50:
            # Choose and take action
            action = receiver._choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            receiver.memory.append((state, action, reward, next_state, done))
            
            # Train if enough experience
            if len(receiver.memory) > receiver.batch_size:
                receiver._train_step()
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                episode_info = info
                break
        
        return total_reward, steps, episode_info
    
    def test_model(self, model_path: str, num_test_episodes: int = None) -> Dict:
        """Test the trained model on various scenarios."""
        if num_test_episodes is None:
            num_test_episodes = self.training_params['test_episodes']
        
        self.logger.info(f"Testing model from {model_path} for {num_test_episodes} episodes")
        
        # Load model
        mock_sdr = self.create_mock_sdr_interface()
        receiver = IntelligentReceiverML(sdr_interface=mock_sdr, config=self.config)
        
        try:
            receiver.load_model(model_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
        # Test scenarios
        test_results = {
            'episode_rewards': [],
            'success_episodes': 0,
            'average_snr_found': [],
            'frequency_accuracy': [],
            'convergence_times': [],
            'scenario_difficulties': []
        }
        
        # Set to evaluation mode (no exploration)
        original_epsilon = receiver.epsilon
        receiver.epsilon = 0.0
        
        try:
            for episode in range(num_test_episodes):
                # Create test environment with random scenario
                env = SimulatedSDREnvironment(
                    target_freq=np.random.uniform(80e6, 180e6),
                    target_snr=np.random.uniform(5, 25)
                )
                
                state, _ = env.reset()
                total_reward = 0
                steps = 0
                found_signal = False
                
                while steps < 50:
                    action = receiver._choose_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    total_reward += reward
                    state = next_state
                    steps += 1
                    
                    if done and info['snr'] > 15:
                        found_signal = True
                        test_results['convergence_times'].append(steps)
                        test_results['average_snr_found'].append(info['snr'])
                        test_results['frequency_accuracy'].append(info['freq_error'])
                        break
                
                test_results['episode_rewards'].append(total_reward)
                if found_signal:
                    test_results['success_episodes'] += 1
                
                # Calculate scenario difficulty
                difficulty = 'easy' if env.target_snr > 15 else 'medium' if env.target_snr > 5 else 'hard'
                test_results['scenario_difficulties'].append(difficulty)
                
                if (episode + 1) % 20 == 0:
                    success_rate = test_results['success_episodes'] / (episode + 1)
                    avg_reward = np.mean(test_results['episode_rewards'])
                    self.logger.info(f"Test progress: {episode + 1}/{num_test_episodes}, "
                                   f"Success rate: {success_rate:.2f}, Avg reward: {avg_reward:.2f}")
        
        finally:
            # Restore original epsilon
            receiver.epsilon = original_epsilon
        
        # Calculate final test metrics
        final_test_metrics = {
            'total_episodes': num_test_episodes,
            'success_rate': test_results['success_episodes'] / num_test_episodes,
            'average_reward': np.mean(test_results['episode_rewards']),
            'average_snr_when_found': np.mean(test_results['average_snr_found']) if test_results['average_snr_found'] else 0,
            'average_convergence_time': np.mean(test_results['convergence_times']) if test_results['convergence_times'] else 0,
            'frequency_accuracy_khz': np.mean(test_results['frequency_accuracy']) / 1e3 if test_results['frequency_accuracy'] else 0
        }
        
        self.logger.info("Testing completed", extra=final_test_metrics)
        
        # Save test results
        test_file = self.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(test_file, 'w') as f:
            json.dump({
                'test_results': test_results,
                'final_metrics': final_test_metrics,
                'model_path': model_path
            }, f, indent=2)
        
        return final_test_metrics
    
    def create_performance_plots(self, metrics_file: str):
        """Create performance visualization plots."""
        self.logger.info(f"Creating performance plots from {metrics_file}")
        
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        training_metrics = data['training_metrics']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Intelligent Receiver Training Performance', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(training_metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Success rate
        if training_metrics['success_rate']:
            axes[0, 1].plot(training_metrics['success_rate'])
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].grid(True)
        
        # Epsilon decay
        axes[0, 2].plot(training_metrics['epsilon_values'])
        axes[0, 2].set_title('Epsilon Decay')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Epsilon')
        axes[0, 2].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(training_metrics['episode_lengths'])
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Memory usage
        if training_metrics['memory_usage']:
            axes[1, 1].plot(training_metrics['memory_usage'])
            axes[1, 1].set_title('Memory Usage')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Memory (MB)')
            axes[1, 1].grid(True)
        
        # Average SNR found
        if training_metrics['average_snr_found']:
            axes[1, 2].plot(training_metrics['average_snr_found'])
            axes[1, 2].set_title('Average SNR Found')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('SNR (dB)')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"training_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance plots saved to {plot_file}")
    
    def run_full_pipeline(self, num_episodes: int = None, num_test_episodes: int = None):
        """Run the complete training and testing pipeline."""
        self.logger.info("Starting full intelligent receiver training pipeline")
        
        try:
            # Step 1: Generate training scenarios
            scenarios = self.generate_training_scenarios()
            
            # Step 2: Train model
            training_metrics = self.train_model(num_episodes)
            
            # Step 3: Test best model
            best_model_path = self.models_dir / "intelligent_receiver_best.pth"
            if best_model_path.exists():
                test_metrics = self.test_model(str(best_model_path), num_test_episodes)
            else:
                self.logger.warning("Best model not found, testing final model")
                final_model_path = self.models_dir / "intelligent_receiver_final.pth"
                test_metrics = self.test_model(str(final_model_path), num_test_episodes)
            
            # Step 4: Create performance plots
            metrics_files = list(self.output_dir.glob("training_metrics_*.json"))
            if metrics_files:
                latest_metrics = max(metrics_files, key=os.path.getctime)
                self.create_performance_plots(str(latest_metrics))
            
            # Step 5: Generate final report
            self._generate_final_report(training_metrics, test_metrics)
            
            self.logger.info("Full pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _generate_final_report(self, training_metrics: Dict, test_metrics: Dict):
        """Generate a final training report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_summary': training_metrics,
            'test_summary': test_metrics,
            'model_files': {
                'best_model': str(self.models_dir / "intelligent_receiver_best.pth"),
                'final_model': str(self.models_dir / "intelligent_receiver_final.pth")
            },
            'recommendations': self._generate_recommendations(training_metrics, test_metrics)
        }
        
        report_file = self.output_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Final report saved to {report_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("INTELLIGENT RECEIVER TRAINING SUMMARY")
        print("="*60)
        print(f"Training Episodes: {training_metrics['total_episodes']}")
        print(f"Training Time: {training_metrics['training_time_hours']:.2f} hours")
        print(f"Final Success Rate: {training_metrics['final_success_rate']:.2%}")
        print(f"Best Average Reward: {training_metrics['best_avg_reward']:.2f}")
        print(f"\nTest Success Rate: {test_metrics['success_rate']:.2%}")
        print(f"Average SNR Found: {test_metrics['average_snr_when_found']:.1f} dB")
        print(f"Average Convergence Time: {test_metrics['average_convergence_time']:.1f} steps")
        print("="*60)
    
    def _generate_recommendations(self, training_metrics: Dict, test_metrics: Dict) -> List[str]:
        """Generate recommendations based on training results."""
        recommendations = []
        
        if test_metrics['success_rate'] < 0.7:
            recommendations.append("Consider increasing training episodes or adjusting reward function")
        
        if test_metrics['average_convergence_time'] > 30:
            recommendations.append("Model may benefit from curriculum learning or better exploration strategy")
        
        if training_metrics['final_success_rate'] < 0.8:
            recommendations.append("Training may have stopped too early - consider longer training")
        
        if test_metrics['frequency_accuracy_khz'] > 100:
            recommendations.append("Frequency accuracy could be improved with finer action discretization")
        
        if not recommendations:
            recommendations.append("Model performance looks good! Consider testing on real hardware")
        
        return recommendations


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Intelligent Receiver Training Pipeline")
    parser.add_argument("--mode", choices=['full', 'train_only', 'test_only', 'generate_data'], 
                       default='full', help="Training mode")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--test_episodes", type=int, default=100, help="Number of test episodes")
    parser.add_argument("--model_path", type=str, help="Path to model for testing")
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = IntelligentReceiverTrainer(args.config)
    
    try:
        if args.mode == 'full':
            trainer.run_full_pipeline(args.episodes, args.test_episodes)
        
        elif args.mode == 'train_only':
            trainer.train_model(args.episodes, args.resume_from)
        
        elif args.mode == 'test_only':
            if not args.model_path:
                print("Error: --model_path required for test_only mode")
                sys.exit(1)
            trainer.test_model(args.model_path, args.test_episodes)
        
        elif args.mode == 'generate_data':
            trainer.generate_training_scenarios()
        
        print("\n✅ Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()