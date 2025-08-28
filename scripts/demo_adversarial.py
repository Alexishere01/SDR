#!/usr/bin/env python3
# demo_adversarial.py - Demonstrate AI receiver vs intelligent jammer
import numpy as np
import matplotlib.pyplot as plt
import time
from ml.adversarial_jamming import AdversarialSDREnvironment, AdversarialJammer
from ml.intelligent_receiver import IntelligentReceiverML
from geminisdr.core.sdr_interface import PlutoSDRInterface
import os

class AdversarialDemo:
    """Demonstrate adversarial AI vs AI scenarios."""
    
    def __init__(self):
        # Create dummy SDR for demo
        class DummySDR:
            def configure(self, **kwargs):
                return True
        
        self.dummy_sdr = DummySDR()
        
        # Load models
        self.regular_receiver = None
        self.adversarial_receiver = None
        self._load_models()
    
    def _load_models(self):
        """Load receiver models."""
        try:
            if os.path.exists('models/intelligent_receiver.pth'):
                self.regular_receiver = IntelligentReceiverML(self.dummy_sdr)
                self.regular_receiver.load_model('models/intelligent_receiver.pth')
                print("✅ Loaded regular receiver model")
        except Exception as e:
            print(f"⚠️  Failed to load regular receiver: {e}")
        
        try:
            if os.path.exists('models/adversarial_receiver.pth'):
                from geminisdr.scripts.train_adversarial import AdversarialIntelligentReceiver
                self.adversarial_receiver = AdversarialIntelligentReceiver(self.dummy_sdr)
                self.adversarial_receiver.load_model('models/adversarial_receiver.pth')
                print("✅ Loaded adversarial receiver model")
        except Exception as e:
            print(f"⚠️  Failed to load adversarial receiver: {e}")
    
    def demo_jammer_strategies(self):
        """Demonstrate different jamming strategies."""
        print("\n" + "="*60)
        print("🔴 JAMMING STRATEGIES DEMONSTRATION")
        print("="*60)
        
        # Create jammer
        jammer = AdversarialJammer(power_budget=60)
        jammer.set_jamming_active(True)
        
        # Test each strategy
        strategies = ['narrowband', 'wideband', 'sweep', 'pulse', 'adaptive', 'deceptive', 'smart_follow']
        
        from geminisdr.core.signal_generator import generate_signal
        generator = SDRSignalGenerator()
        
        # Generate clean signal
        clean_signal = generator.generate_modulated_signal('QPSK', num_symbols=256, snr_db=20)
        
        print("Testing jamming strategies on clean QPSK signal (20 dB SNR):")
        print()
        
        results = {}
        
        for strategy in strategies:
            # Force jammer to use specific strategy
            jammer.current_strategy = strategy
            
            # Apply jamming
            jammed_signal = jammer.strategies[strategy](
                clean_signal, 100e6, 30, 2e6
            )
            
            # Calculate SNR degradation
            clean_power = np.mean(np.abs(clean_signal)**2)
            jam_power = np.mean(np.abs(jammed_signal - clean_signal)**2)
            
            if jam_power > 0:
                snr_degradation = 10 * np.log10(clean_power / jam_power)
            else:
                snr_degradation = 0
            
            results[strategy] = {
                'snr_degradation': snr_degradation,
                'jamming_power': jam_power
            }
            
            print(f"  {strategy:15s}: SNR degradation = {snr_degradation:6.1f} dB")
        
        # Plot jamming effectiveness
        self._plot_jamming_strategies(results)
        
        return results
    
    def demo_ai_vs_ai_battle(self, num_rounds=10, jammer_power=60):
        """Demonstrate AI receiver vs AI jammer battle."""
        print("\n" + "="*60)
        print("⚔️  AI vs AI BATTLE DEMONSTRATION")
        print("="*60)
        print(f"Rounds: {num_rounds}, Jammer Power: {jammer_power}W")
        
        if not self.regular_receiver:
            print("❌ No receiver model loaded")
            return
        
        # Create adversarial environment
        env = AdversarialSDREnvironment(jammer_power=jammer_power)
        
        battle_log = []
        
        print("\n🥊 BATTLE BEGINS!")
        print("-" * 40)
        
        for round_num in range(num_rounds):
            print(f"\n🔥 Round {round_num + 1}")
            
            # Reset environment
            state = env.reset()
            
            # Battle variables
            round_steps = 0
            max_steps = 50
            round_snrs = []
            
            while round_steps < max_steps:
                # Receiver chooses action
                action = self.regular_receiver._choose_action(state)
                
                # Environment step (includes jammer response)
                next_state, reward, done, info = env.step(action)
                
                round_snrs.append(info['snr'])
                state = next_state
                round_steps += 1
                
                # Show real-time battle
                if round_steps % 10 == 0:
                    print(f"   Step {round_steps:2d}: SNR = {info['snr']:5.1f} dB, "
                          f"Jammer using {info['jammer_strategy']}")
                
                if done:
                    break
            
            # Determine round winner
            final_snr = info['snr']
            if final_snr > 15:
                winner = "🤖 RECEIVER"
                result = "WIN"
            elif final_snr < 5:
                winner = "🔴 JAMMER"
                result = "WIN"
            else:
                winner = "🤝 DRAW"
                result = "DRAW"
            
            print(f"   Result: {winner} - Final SNR: {final_snr:.1f} dB")
            
            battle_log.append({
                'round': round_num + 1,
                'winner': result,
                'final_snr': final_snr,
                'avg_snr': np.mean(round_snrs),
                'steps': round_steps,
                'jammer_strategy': info['jammer_strategy']
            })
        
        # Battle summary
        receiver_wins = sum(1 for b in battle_log if b['winner'] == 'WIN' and b['final_snr'] > 15)
        jammer_wins = sum(1 for b in battle_log if b['winner'] == 'WIN' and b['final_snr'] < 5)
        draws = sum(1 for b in battle_log if b['winner'] == 'DRAW')
        
        print(f"\n🏆 BATTLE RESULTS:")
        print(f"   Receiver Wins: {receiver_wins}")
        print(f"   Jammer Wins: {jammer_wins}")
        print(f"   Draws: {draws}")
        print(f"   Receiver Win Rate: {receiver_wins/num_rounds*100:.0f}%")
        
        # Get jammer learning stats
        jammer_stats = env.get_battle_stats()
        print(f"\n🧠 JAMMER LEARNING:")
        for strategy, success_rate in jammer_stats['jammer_stats']['success_rates'].items():
            print(f"   {strategy:15s}: {success_rate:.1%} success rate")
        
        # Plot battle results
        self._plot_battle_results(battle_log)
        
        return battle_log
    
    def demo_receiver_comparison(self, jammer_power=60):
        """Compare regular vs adversarial-trained receiver."""
        print("\n" + "="*60)
        print("🆚 RECEIVER COMPARISON")
        print("="*60)
        
        if not self.regular_receiver:
            print("❌ No regular receiver model loaded")
            return
        
        if not self.adversarial_receiver:
            print("❌ No adversarial receiver model loaded")
            return
        
        receivers = {
            'Regular': self.regular_receiver,
            'Adversarial': self.adversarial_receiver
        }
        
        results = {}
        
        for name, receiver in receivers.items():
            print(f"\n🧪 Testing {name} Receiver...")
            
            # Create fresh environment
            env = AdversarialSDREnvironment(jammer_power=jammer_power)
            
            wins = 0
            total_tests = 20
            snrs = []
            
            for test in range(total_tests):
                state = env.reset()
                steps = 0
                
                while steps < 30:
                    if hasattr(receiver, '_choose_anti_jam_action'):
                        action = receiver._choose_anti_jam_action(state, env)
                    else:
                        action = receiver._choose_action(state)
                    
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    steps += 1
                    
                    if done:
                        if info['snr'] > 15:
                            wins += 1
                        snrs.append(info['snr'])
                        break
            
            win_rate = wins / total_tests * 100
            avg_snr = np.mean(snrs)
            
            results[name] = {
                'win_rate': win_rate,
                'avg_snr': avg_snr,
                'snrs': snrs
            }
            
            print(f"   Win Rate: {win_rate:.0f}%")
            print(f"   Average SNR: {avg_snr:.1f} dB")
        
        # Plot comparison
        self._plot_receiver_comparison(results)
        
        return results
    
    def demo_escalating_jamming(self):
        """Demonstrate performance against escalating jamming power."""
        print("\n" + "="*60)
        print("📈 ESCALATING JAMMING DEMONSTRATION")
        print("="*60)
        
        if not self.regular_receiver:
            print("❌ No receiver model loaded")
            return
        
        # Test different jammer power levels
        power_levels = [20, 40, 60, 80, 100]
        results = {}
        
        for power in power_levels:
            print(f"\n⚡ Testing against {power}W jammer...")
            
            env = AdversarialSDREnvironment(jammer_power=power)
            
            # Run multiple tests
            wins = 0
            snrs = []
            strategies_used = []
            
            for test in range(15):
                state = env.reset()
                steps = 0
                
                while steps < 25:
                    action = self.regular_receiver._choose_action(state)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    steps += 1
                    
                    if done:
                        if info['snr'] > 15:
                            wins += 1
                        snrs.append(info['snr'])
                        strategies_used.append(info['jammer_strategy'])
                        break
            
            win_rate = wins / 15 * 100
            avg_snr = np.mean(snrs)
            
            results[power] = {
                'win_rate': win_rate,
                'avg_snr': avg_snr,
                'strategies': strategies_used
            }
            
            print(f"   Win Rate: {win_rate:.0f}%")
            print(f"   Average SNR: {avg_snr:.1f} dB")
            
            # Performance assessment
            if win_rate >= 70:
                print("   🏆 Receiver dominates!")
            elif win_rate >= 50:
                print("   ✅ Receiver holds ground")
            elif win_rate >= 30:
                print("   ⚠️  Receiver struggling")
            else:
                print("   💥 Receiver overwhelmed")
        
        # Plot escalation results
        self._plot_escalation_results(results)
        
        return results
    
    def _plot_jamming_strategies(self, results):
        """Plot jamming strategy effectiveness."""
        strategies = list(results.keys())
        degradations = [results[s]['snr_degradation'] for s in strategies]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(strategies, degradations, color='red', alpha=0.7)
        plt.ylabel('SNR Degradation (dB)')
        plt.title('Jamming Strategy Effectiveness')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, deg in zip(bars, degradations):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{deg:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/jamming_strategies.png', dpi=300)
        plt.show()
    
    def _plot_battle_results(self, battle_log):
        """Plot AI vs AI battle results."""
        rounds = [b['round'] for b in battle_log]
        snrs = [b['final_snr'] for b in battle_log]
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: SNR over rounds
        plt.subplot(2, 2, 1)
        colors = ['green' if snr > 15 else 'red' if snr < 5 else 'yellow' for snr in snrs]
        plt.scatter(rounds, snrs, c=colors, alpha=0.7, s=100)
        plt.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Receiver Win')
        plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Jammer Win')
        plt.xlabel('Round')
        plt.ylabel('Final SNR (dB)')
        plt.title('Battle Results: Final SNR per Round')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Strategy usage
        plt.subplot(2, 2, 2)
        strategies = [b['jammer_strategy'] for b in battle_log]
        strategy_counts = {}
        for s in strategies:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        
        plt.pie(strategy_counts.values(), labels=strategy_counts.keys(), autopct='%1.0f%%')
        plt.title('Jammer Strategy Usage')
        
        # Plot 3: Win/Loss distribution
        plt.subplot(2, 2, 3)
        outcomes = []
        for b in battle_log:
            if b['final_snr'] > 15:
                outcomes.append('Receiver Win')
            elif b['final_snr'] < 5:
                outcomes.append('Jammer Win')
            else:
                outcomes.append('Draw')
        
        outcome_counts = {}
        for o in outcomes:
            outcome_counts[o] = outcome_counts.get(o, 0) + 1
        
        colors = ['green', 'red', 'yellow']
        plt.bar(outcome_counts.keys(), outcome_counts.values(), color=colors, alpha=0.7)
        plt.ylabel('Count')
        plt.title('Battle Outcomes')
        
        # Plot 4: SNR progression
        plt.subplot(2, 2, 4)
        avg_snrs = [b['avg_snr'] for b in battle_log]
        plt.plot(rounds, avg_snrs, 'b-o', alpha=0.7)
        plt.xlabel('Round')
        plt.ylabel('Average SNR (dB)')
        plt.title('Average SNR Progression')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/ai_battle_results.png', dpi=300)
        plt.show()
    
    def _plot_receiver_comparison(self, results):
        """Plot receiver comparison results."""
        receivers = list(results.keys())
        win_rates = [results[r]['win_rate'] for r in receivers]
        avg_snrs = [results[r]['avg_snr'] for r in receivers]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Win rates
        bars1 = ax1.bar(receivers, win_rates, color=['blue', 'orange'], alpha=0.7)
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_title('Receiver Win Rate Comparison')
        
        for bar, rate in zip(bars1, win_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.0f}%', ha='center', va='bottom')
        
        # Average SNRs
        bars2 = ax2.bar(receivers, avg_snrs, color=['blue', 'orange'], alpha=0.7)
        ax2.set_ylabel('Average SNR (dB)')
        ax2.set_title('Average SNR Comparison')
        
        for bar, snr in zip(bars2, avg_snrs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{snr:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/receiver_comparison.png', dpi=300)
        plt.show()
    
    def _plot_escalation_results(self, results):
        """Plot escalating jamming results."""
        powers = list(results.keys())
        win_rates = [results[p]['win_rate'] for p in powers]
        avg_snrs = [results[p]['avg_snr'] for p in powers]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Win rate vs power
        ax1.plot(powers, win_rates, 'ro-', linewidth=2, markersize=8)
        ax1.set_xlabel('Jammer Power (W)')
        ax1.set_ylabel('Receiver Win Rate (%)')
        ax1.set_title('Performance vs Jamming Power')
        ax1.grid(True, alpha=0.3)
        
        # SNR vs power
        ax2.plot(powers, avg_snrs, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Jammer Power (W)')
        ax2.set_ylabel('Average SNR (dB)')
        ax2.set_title('SNR vs Jamming Power')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/escalation_results.png', dpi=300)
        plt.show()

def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adversarial SDR Demo')
    parser.add_argument('--demo', type=int, choices=range(1, 5),
                       help='Run specific demo (1-4)')
    parser.add_argument('--all', action='store_true',
                       help='Run all demonstrations')
    
    args = parser.parse_args()
    
    demo = AdversarialDemo()
    
    print("🔥 ADVERSARIAL SDR DEMONSTRATION")
    print("="*60)
    print("AI Receiver vs Intelligent Jammer")
    print()
    
    if args.all or not args.demo:
        # Run all demos
        demo.demo_jammer_strategies()
        demo.demo_ai_vs_ai_battle()
        demo.demo_receiver_comparison()
        demo.demo_escalating_jamming()
    else:
        # Run specific demo
        if args.demo == 1:
            demo.demo_jammer_strategies()
        elif args.demo == 2:
            demo.demo_ai_vs_ai_battle()
        elif args.demo == 3:
            demo.demo_receiver_comparison()
        elif args.demo == 4:
            demo.demo_escalating_jamming()
    
    print("\n🎉 ADVERSARIAL DEMONSTRATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()