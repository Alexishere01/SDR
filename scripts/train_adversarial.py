#!/usr/bin/env python3
# train_adversarial.py - Train AI receiver against intelligent jamming
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from ml.intelligent_receiver import IntelligentReceiverML, DeepQLearningReceiver
from geminisdr.ml.adversarial_jamming import AdversarialSDREnvironment, AdversarialJammer
from geminisdr.config.config_manager import get_config_manager, load_config
from geminisdr.core.error_handling import ErrorHandler, ModelError, ErrorSeverity
from geminisdr.core.logging_manager import StructuredLogger
import time
import logging

class AdversarialIntelligentReceiver(IntelligentReceiverML):
    """Enhanced intelligent receiver for adversarial environments with configuration management."""
    
    def __init__(self, sdr_interface, config=None):
        # Load configuration if not provided
        if config is None:
            try:
                config = load_config("config")
            except Exception as e:
                logging.warning(f"Failed to load configuration, using defaults: {e}")
                config = None
        
        super().__init__(sdr_interface, config=config)
        
        # Enhanced for adversarial scenarios - use configuration where possible
        self.epsilon_min = 0.02  # Higher minimum exploration
        self.epsilon_decay = 0.9998  # Slower decay for more exploration
        
        # Use configuration for memory and batch size
        if config and config.ml.batch_size:
            self.batch_size = min(config.ml.batch_size * 2, 128)  # Double for adversarial training
        else:
            self.batch_size = 64  # Larger batch size
            
        # Larger memory for adversarial scenarios
        self.memory = torch.collections.deque(maxlen=20000)
        
        # Anti-jamming specific parameters
        self.jamming_memory = torch.collections.deque(maxlen=1000)
        self.anti_jam_bonus = 0.0
    
    def train_against_jammer(self, num_episodes=1500, jammer_power=50):
        """Train receiver to fight against intelligent jammer."""
        env = AdversarialSDREnvironment(jammer_power=jammer_power)
        
        print(f"Training receiver against intelligent jammer...")
        print(f"Episodes: {num_episodes}, Jammer Power: {jammer_power}")
        
        rewards_history = []
        snr_history = []
        battle_stats = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            episode_snrs = []
            steps = 0
            
            while steps < 100:  # Max steps per episode
                # Choose action with anti-jamming awareness
                action = self._choose_anti_jam_action(state, env)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Enhanced reward for anti-jamming
                enhanced_reward = self._calculate_anti_jam_reward(reward, info)
                
                # Store experience
                self.memory.append((state, action, enhanced_reward, next_state, done))
                
                # Train more frequently in adversarial environment
                if len(self.memory) > self.batch_size and steps % 2 == 0:
                    self._train_step()
                
                total_reward += enhanced_reward
                episode_snrs.append(info['snr'])
                state = next_state
                steps += 1
                
                if done:
                    result = "🏆 RECEIVER WIN" if info['snr'] > 15 else "💥 JAMMED"
                    if episode % 50 == 0:
                        print(f"Episode {episode}: {result} - "
                              f"SNR: {info['snr']:.1f}dB, "
                              f"Strategy: {info['jammer_strategy']}")
                    break
            
            rewards_history.append(total_reward)
            snr_history.append(np.mean(episode_snrs))
            
            # Track battle statistics
            if episode % 100 == 0:
                stats = env.get_battle_stats()
                battle_stats.append(stats)
                
                print(f"\nEpisode {episode} Battle Stats:")
                print(f"  Receiver Wins: {stats['receiver_wins']}")
                print(f"  Jammer Wins: {stats['jammer_wins']}")
                print(f"  Win Rate: {stats['receiver_win_rate']:.1%}")
            
            # Update target network
            if episode % 20 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return rewards_history, snr_history, battle_stats
    
    def _choose_anti_jam_action(self, state, env):
        """Choose action with anti-jamming intelligence."""
        if np.random.random() < self.epsilon:
            # Enhanced exploration for anti-jamming
            action_indices = np.random.randint(0, 11, 3)
            
            # Bias towards more aggressive frequency changes when jammed
            if hasattr(env, 'jammer') and len(env.jammer.receiver_history) > 0:
                last_snr = env.jammer.receiver_history[-1]['snr']
                if last_snr < 10:  # If being jammed, be more aggressive
                    action_indices[0] = np.random.choice([0, 1, 9, 10])  # Extreme freq changes
        else:
            # Use learned policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_indices = q_values.argmax(dim=2).squeeze().cpu().numpy()
        
        # Convert to actions
        action = np.array([
            self.freq_actions[action_indices[0]],
            self.gain_actions[action_indices[1]],
            self.bw_actions[action_indices[2]]
        ])
        
        return action
    
    def _calculate_anti_jam_reward(self, base_reward, info):
        """Calculate enhanced reward for anti-jamming scenarios."""
        enhanced_reward = base_reward
        
        # Bonus for maintaining good SNR under jamming
        if 'jamming_effectiveness' in info:
            jamming_damage = info['jamming_effectiveness']
            if jamming_damage > 5:  # Significant jamming present
                if info['snr'] > 10:  # But still maintaining decent SNR
                    enhanced_reward += 20  # Big bonus for anti-jam success
        
        # Bonus for frequency agility (avoiding predictable patterns)
        if len(self.jamming_memory) > 5:
            recent_freqs = [m['freq'] for m in list(self.jamming_memory)[-5:]]
            freq_variance = np.var(recent_freqs)
            if freq_variance > 1e12:  # High frequency variance = good agility
                enhanced_reward += 5
        
        # Store jamming encounter for learning
        self.jamming_memory.append({
            'freq': info['freq'],
            'snr': info['snr'],
            'strategy': info.get('jammer_strategy', 'unknown')
        })
        
        return enhanced_reward

def train_escalating_adversarial(episodes_per_level=300):
    """Train with escalating jammer difficulty using configuration management."""
    # Load configuration
    try:
        config = load_config("config")
        logger = StructuredLogger(__name__, config.logging)
        error_handler = ErrorHandler(logger.logger)
    except Exception as e:
        logging.warning(f"Failed to load configuration: {e}")
        config = None
        logger = None
        error_handler = None
    
    if logger:
        logger.logger.info("🎯 ESCALATING ADVERSARIAL TRAINING")
    else:
        print("🎯 ESCALATING ADVERSARIAL TRAINING")
    print("="*60)
    
    # Create dummy SDR
    class DummySDR:
        def configure(self, **kwargs):
            return True
    
    dummy_sdr = DummySDR()
    
    try:
        receiver = AdversarialIntelligentReceiver(dummy_sdr, config=config)
    except Exception as e:
        if error_handler:
            error_handler.handle_error(ModelError(
                f"Failed to initialize adversarial receiver: {str(e)}",
                model_name="AdversarialIntelligentReceiver",
                severity=ErrorSeverity.HIGH,
                cause=e
            ))
        raise
    
    # Escalating difficulty levels
    difficulty_levels = [
        {'power': 20, 'name': 'Weak Jammer'},
        {'power': 40, 'name': 'Moderate Jammer'},
        {'power': 60, 'name': 'Strong Jammer'},
        {'power': 80, 'name': 'Powerful Jammer'},
        {'power': 100, 'name': 'Elite Jammer'}
    ]
    
    all_results = {}
    
    for level, config in enumerate(difficulty_levels):
        print(f"\n🔥 Level {level + 1}: {config['name']} (Power: {config['power']})")
        print("-" * 50)
        
        # Train at this difficulty level
        rewards, snrs, battles = receiver.train_against_jammer(
            num_episodes=episodes_per_level,
            jammer_power=config['power']
        )
        
        all_results[config['name']] = {
            'rewards': rewards,
            'snrs': snrs,
            'battles': battles,
            'final_win_rate': battles[-1]['receiver_win_rate'] if battles else 0
        }
        
        print(f"✅ Level {level + 1} Complete!")
        print(f"   Final Win Rate: {all_results[config['name']]['final_win_rate']:.1%}")
        print(f"   Average SNR: {np.mean(snrs[-50:]):.1f} dB")
    
    # Save the battle-hardened model
    receiver.save_model('models/adversarial_receiver.pth')
    
    return receiver, all_results

def plot_adversarial_results(results):
    """Plot comprehensive adversarial training results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Adversarial Training Results: AI Receiver vs Intelligent Jammer', fontsize=16)
    
    # Plot 1: Win rates across difficulty levels
    ax = axes[0, 0]
    levels = list(results.keys())
    win_rates = [results[level]['final_win_rate'] * 100 for level in levels]
    
    bars = ax.bar(levels, win_rates, color=['green', 'yellow', 'orange', 'red', 'darkred'])
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Final Win Rate vs Jammer Difficulty')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # Plot 2: SNR performance over time
    ax = axes[0, 1]
    for level, data in results.items():
        snrs = data['snrs']
        # Moving average for smoother plot
        window = 50
        if len(snrs) >= window:
            moving_avg = np.convolve(snrs, np.ones(window)/window, mode='valid')
            ax.plot(moving_avg, label=level, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average SNR (dB)')
    ax.set_title('SNR Performance Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Reward progression
    ax = axes[0, 2]
    for level, data in results.items():
        rewards = data['rewards']
        # Moving average
        window = 50
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(moving_avg, label=level, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Reward Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Battle statistics heatmap
    ax = axes[1, 0]
    battle_data = []
    for level in levels:
        if results[level]['battles']:
            final_battle = results[level]['battles'][-1]
            jammer_stats = final_battle['jammer_stats']
            success_rates = list(jammer_stats['success_rates'].values())
            battle_data.append(success_rates)
    
    if battle_data:
        im = ax.imshow(battle_data, cmap='RdYlBu_r', aspect='auto')
        ax.set_xlabel('Jamming Strategy')
        ax.set_ylabel('Difficulty Level')
        ax.set_title('Jammer Strategy Effectiveness')
        
        # Add strategy labels
        if results[levels[0]]['battles']:
            strategies = list(results[levels[0]]['battles'][-1]['jammer_stats']['success_rates'].keys())
            ax.set_xticks(range(len(strategies)))
            ax.set_xticklabels(strategies, rotation=45)
        
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels(levels)
        plt.colorbar(im, ax=ax, label='Success Rate')
    
    # Plot 5: Learning curves comparison
    ax = axes[1, 1]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (level, data) in enumerate(results.items()):
        rewards = data['rewards']
        # Calculate learning curve (cumulative average)
        cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        ax.plot(cumulative_avg, color=colors[i], label=level, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Average Reward')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Final performance summary
    ax = axes[1, 2]
    metrics = ['Win Rate', 'Avg SNR', 'Avg Reward']
    
    # Normalize metrics for comparison
    win_rates_norm = [r / 100 for r in win_rates]  # Already in %
    avg_snrs = [np.mean(results[level]['snrs'][-50:]) for level in levels]
    avg_snrs_norm = [(s + 10) / 50 for s in avg_snrs]  # Normalize -10 to 40 dB range
    avg_rewards = [np.mean(results[level]['rewards'][-50:]) for level in levels]
    avg_rewards_norm = [(r + 50) / 150 for r in avg_rewards]  # Rough normalization
    
    x = np.arange(len(levels))
    width = 0.25
    
    ax.bar(x - width, win_rates_norm, width, label='Win Rate', alpha=0.8)
    ax.bar(x, avg_snrs_norm, width, label='Avg SNR (norm)', alpha=0.8)
    ax.bar(x + width, avg_rewards_norm, width, label='Avg Reward (norm)', alpha=0.8)
    
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Normalized Performance')
    ax.set_title('Final Performance Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/adversarial_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_adversarial_receiver():
    """Test the trained adversarial receiver."""
    print("\n🧪 TESTING ADVERSARIAL RECEIVER")
    print("="*50)
    
    # Create dummy SDR
    class DummySDR:
        def configure(self, **kwargs):
            return True
    
    dummy_sdr = DummySDR()
    
    # Load adversarial receiver
    if os.path.exists('models/adversarial_receiver.pth'):
        receiver = AdversarialIntelligentReceiver(dummy_sdr)
        receiver.load_model('models/adversarial_receiver.pth')
        print("✅ Loaded adversarial receiver model")
    else:
        print("❌ No adversarial receiver model found")
        return
    
    # Test against different jammer powers
    test_powers = [30, 60, 90]
    
    for power in test_powers:
        print(f"\n🎯 Testing against {power}W jammer...")
        
        env = AdversarialSDREnvironment(jammer_power=power)
        
        # Run test episodes
        wins = 0
        total_episodes = 20
        snrs = []
        
        for episode in range(total_episodes):
            state = env.reset()
            steps = 0
            
            while steps < 50:
                action = receiver._choose_anti_jam_action(state, env)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                steps += 1
                
                if done:
                    if info['snr'] > 15:
                        wins += 1
                    snrs.append(info['snr'])
                    break
        
        win_rate = wins / total_episodes * 100
        avg_snr = np.mean(snrs)
        
        print(f"   Win Rate: {win_rate:.0f}%")
        print(f"   Average SNR: {avg_snr:.1f} dB")
        
        # Performance assessment
        if win_rate >= 70:
            print("   🏆 EXCELLENT - Receiver dominates!")
        elif win_rate >= 50:
            print("   ✅ GOOD - Receiver holds its own")
        elif win_rate >= 30:
            print("   ⚠️  FAIR - Struggling but fighting")
        else:
            print("   💥 POOR - Getting jammed hard")

def main():
    """Main adversarial training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adversarial SDR Training')
    parser.add_argument('--episodes', type=int, default=300,
                       help='Episodes per difficulty level')
    parser.add_argument('--test', action='store_true',
                       help='Test existing adversarial model')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with fewer episodes')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    if args.test:
        test_adversarial_receiver()
        return
    
    episodes = 100 if args.quick else args.episodes
    
    print("🔥 ADVERSARIAL SDR TRAINING")
    print("="*60)
    print("Training AI receiver to fight intelligent jamming!")
    print(f"Episodes per level: {episodes}")
    print()
    
    # Train with escalating difficulty
    receiver, results = train_escalating_adversarial(episodes_per_level=episodes)
    
    # Plot results
    plot_adversarial_results(results)
    
    # Test the trained model
    test_adversarial_receiver()
    
    print("\n🎉 ADVERSARIAL TRAINING COMPLETE!")
    print("="*60)
    print("Your AI receiver is now battle-hardened against intelligent jamming!")
    print("\nModel saved: models/adversarial_receiver.pth")
    print("Results plot: plots/adversarial_training_results.png")
    print("\nTry: python demo.py --demo 4  (to test with real SDR)")

if __name__ == "__main__":
    main()