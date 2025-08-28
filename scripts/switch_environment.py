#!/usr/bin/env python3
"""
Environment Switching Utilities

This script provides easy environment switching, status checking,
environment health validation, and comparison/benchmarking tools.

Requirements addressed: 7.1, 7.2, 5.3
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import argparse



try:
    from platform_detector import PlatformDetector, PlatformInfo
    from environment_manager import EnvironmentManager, EnvironmentStatus
except ImportError as e:
    print(f"Error importing environment modules: {e}")
    print("Make sure the environments/ directory exists with required modules")
    print(f"Looking for modules in: {environments_dir}")
    sys.exit(1)


class EnvironmentSwitcher:
    """
    Environment switching and management utilities.
    
    Provides easy switching between M1 and VM environments,
    status checking, health validation, and benchmarking tools.
    """
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.platform_detector = PlatformDetector()
        self.environment_manager = EnvironmentManager(str(self.project_root))
        self.platform_info = self.platform_detector.get_platform_info()
        
        # Environment configurations
        self.environments = self.environment_manager.environments
        
    def list_environments(self) -> Dict[str, EnvironmentStatus]:
        """
        List all available environments and their status.
        
        Returns:
            Dict[str, EnvironmentStatus]: Environment statuses
        """
        return self.environment_manager.list_environments()
    
    def switch_to_environment(self, env_name: str, verbose: bool = True) -> bool:
        """
        Switch to the specified environment.
        
        Args:
            env_name: Environment name to switch to
            verbose: Print status messages
            
        Returns:
            bool: True if switch successful
        """
        if env_name not in self.environments:
            if verbose:
                print(f"❌ Unknown environment: {env_name}")
                print(f"Available environments: {', '.join(self.environments.keys())}")
            return False
        
        # Check if environment exists
        status = self.environment_manager.validate_environment(env_name)
        
        if not status.exists:
            if verbose:
                print(f"❌ Environment '{env_name}' does not exist.")
                print(f"Run 'python setup_dual_env.py' to create it.")
            return False
        
        if not status.healthy:
            if verbose:
                print(f"⚠️  Environment '{env_name}' has issues:")
                for issue in status.issues[:3]:
                    print(f"   - {issue}")
                print("Consider running health check and repairs.")
        
        # Activate the environment
        success = self.environment_manager.activate_environment(env_name)
        
        if success and verbose:
            config = self.environments[env_name]
            print(f"✅ Switched to {env_name} environment")
            print(f"   Python: {config.python_path}")
            print(f"   Device: {config.device_type}")
            print(f"   Threads: {config.max_threads}")
            print(f"   Activation script: {config.activation_script}")
            print()
            print("To activate in your shell, run:")
            print(f"   source {config.activation_script}")
        
        return success
    
    def get_current_environment(self) -> Optional[str]:
        """
        Detect the currently active environment.
        
        Returns:
            Optional[str]: Current environment name or None
        """
        # Check environment variables to detect active environment
        virtual_env = os.environ.get('VIRTUAL_ENV', '')
        
        if '.venv_m1' in virtual_env:
            return 'm1_native'
        elif '.venv_vm' in virtual_env:
            return 'vm_ubuntu'
        
        # Check if any environment is marked as active in manager
        return self.environment_manager.get_active_environment()
    
    def display_status(self, detailed: bool = False):
        """
        Display status of all environments.
        
        Args:
            detailed: Show detailed status information
        """
        print("🔍 Environment Status")
        print("=" * 40)
        
        current_env = self.get_current_environment()
        statuses = self.list_environments()
        
        for env_name, status in statuses.items():
            # Status indicators
            exists_icon = "✅" if status.exists else "❌"
            health_icon = "🟢" if status.healthy else ("🟡" if status.exists else "🔴")
            active_icon = "👉" if env_name == current_env else "  "
            
            print(f"{active_icon} {exists_icon} {health_icon} {env_name}")
            
            if status.exists:
                print(f"     Packages: {status.package_count}")
                print(f"     Python: {status.python_version}")
                
                if detailed and status.issues:
                    print("     Issues:")
                    for issue in status.issues[:5]:
                        print(f"       - {issue}")
                    if len(status.issues) > 5:
                        print(f"       - ... and {len(status.issues) - 5} more")
            else:
                print("     Not installed")
            
            print()
        
        # Legend
        print("Legend:")
        print("  👉 Currently active")
        print("  ✅ Installed  ❌ Not installed")
        print("  🟢 Healthy   🟡 Issues   🔴 Not available")
        
        if current_env:
            print(f"\nCurrent environment: {current_env}")
        else:
            print("\nNo environment currently active")
    
    def health_check(self, env_name: Optional[str] = None, fix_issues: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive health check on environment(s).
        
        Args:
            env_name: Environment to check (all if None)
            fix_issues: Attempt to fix detected issues
            
        Returns:
            Dict[str, Any]: Health check results
        """
        if env_name:
            environments_to_check = [env_name]
        else:
            environments_to_check = list(self.environments.keys())
        
        print("🏥 Environment Health Check")
        print("=" * 40)
        
        all_results = {}
        
        for env in environments_to_check:
            print(f"\n🔍 Checking {env}...")
            
            health_report = self.environment_manager.health_check(env)
            all_results[env] = health_report
            
            # Display results
            overall_health = health_report['overall_health']
            health_icon = "✅" if overall_health == 'healthy' else "⚠️"
            
            print(f"  {health_icon} Overall Health: {overall_health}")
            
            status = health_report['status']
            print(f"     Exists: {status['exists']}")
            print(f"     Healthy: {status['healthy']}")
            print(f"     Packages: {status['package_count']}")
            print(f"     Python: {status['python_version']}")
            
            # Show issues
            if health_report['issues']:
                print("     Issues:")
                for issue in health_report['issues']:
                    print(f"       - {issue}")
            
            # Show recommendations
            if health_report['recommendations']:
                print("     Recommendations:")
                for rec in health_report['recommendations']:
                    print(f"       → {rec}")
            
            # Platform-specific info
            if 'mps_available' in health_report:
                mps_icon = "✅" if health_report['mps_available'] else "❌"
                print(f"     MPS Available: {mps_icon}")
            
            if 'simulation_mode' in health_report:
                sim_icon = "🎮" if health_report['simulation_mode'] else "🔧"
                mode = "Simulation" if health_report['simulation_mode'] else "Hardware"
                print(f"     Mode: {sim_icon} {mode}")
            
            if 'dependency_conflicts' in health_report:
                conflicts = health_report['dependency_conflicts']
                if conflicts > 0:
                    print(f"     Dependency Conflicts: ⚠️ {conflicts}")
                else:
                    print(f"     Dependency Conflicts: ✅ None")
            
            # Attempt fixes if requested
            if fix_issues and health_report['issues']:
                print(f"\n🔧 Attempting to fix issues in {env}...")
                self._attempt_fixes(env, health_report)
        
        return all_results
    
    def _attempt_fixes(self, env_name: str, health_report: Dict[str, Any]):
        """
        Attempt to fix common issues in an environment.
        
        Args:
            env_name: Environment name
            health_report: Health check results
        """
        issues = health_report['issues']
        fixed_count = 0
        
        for issue in issues:
            if 'not importable' in issue.lower():
                # Try to reinstall requirements
                print(f"  🔧 Reinstalling requirements for {env_name}...")
                try:
                    success = self.environment_manager.setup_environment(env_name)
                    if success:
                        print(f"  ✅ Requirements reinstalled")
                        fixed_count += 1
                    else:
                        print(f"  ❌ Failed to reinstall requirements")
                except Exception as e:
                    print(f"  ❌ Error reinstalling: {e}")
                break  # Only try once
            
            elif 'activation script not found' in issue.lower():
                # Try to recreate activation script
                print(f"  🔧 Recreating activation script for {env_name}...")
                try:
                    config = self.environments[env_name]
                    # The environment manager should recreate the script
                    success = self.environment_manager._create_activation_script(config)
                    if success:
                        print(f"  ✅ Activation script recreated")
                        fixed_count += 1
                    else:
                        print(f"  ❌ Failed to recreate activation script")
                except Exception as e:
                    print(f"  ❌ Error recreating script: {e}")
        
        if fixed_count > 0:
            print(f"  ✅ Fixed {fixed_count} issue(s)")
        else:
            print(f"  ⚠️  No automatic fixes available")
    
    def compare_environments(self) -> Dict[str, Any]:
        """
        Compare available environments and their capabilities.
        
        Returns:
            Dict[str, Any]: Comparison results
        """
        print("⚖️  Environment Comparison")
        print("=" * 40)
        
        statuses = self.list_environments()
        comparison = {
            'environments': {},
            'summary': {},
            'recommendations': []
        }
        
        # Collect environment info
        for env_name, status in statuses.items():
            if not status.exists:
                comparison['environments'][env_name] = {
                    'status': 'not_installed',
                    'packages': 0,
                    'python_version': 'unknown',
                    'device_type': self.environments[env_name].device_type,
                    'max_threads': self.environments[env_name].max_threads
                }
                continue
            
            config = self.environments[env_name]
            health_report = self.environment_manager.health_check(env_name)
            
            comparison['environments'][env_name] = {
                'status': 'healthy' if status.healthy else 'issues',
                'packages': status.package_count,
                'python_version': status.python_version,
                'device_type': config.device_type,
                'max_threads': config.max_threads,
                'issues_count': len(status.issues),
                'mps_available': health_report.get('mps_available', False),
                'simulation_mode': health_report.get('simulation_mode', False)
            }
        
        # Display comparison table
        print(f"{'Environment':<15} {'Status':<10} {'Packages':<10} {'Device':<8} {'Threads':<8}")
        print("-" * 60)
        
        for env_name, info in comparison['environments'].items():
            status_icon = {
                'healthy': '✅',
                'issues': '⚠️',
                'not_installed': '❌'
            }.get(info['status'], '❓')
            
            print(f"{env_name:<15} {status_icon} {info['status']:<8} {info['packages']:<10} "
                  f"{info['device_type']:<8} {info['max_threads']:<8}")
        
        # Generate recommendations
        healthy_envs = [env for env, info in comparison['environments'].items() 
                       if info['status'] == 'healthy']
        
        if not healthy_envs:
            comparison['recommendations'].append("No healthy environments found. Run setup first.")
        elif len(healthy_envs) == 1:
            comparison['recommendations'].append(f"Only {healthy_envs[0]} is available. Consider setting up both environments.")
        else:
            # Recommend based on platform
            recommended = self.platform_info.platform
            if recommended in healthy_envs:
                comparison['recommendations'].append(f"Use {recommended} for optimal performance on this platform.")
            
            # Performance recommendations
            m1_info = comparison['environments'].get('m1_native', {})
            vm_info = comparison['environments'].get('vm_ubuntu', {})
            
            if m1_info.get('mps_available'):
                comparison['recommendations'].append("M1 environment has MPS acceleration available.")
            
            if vm_info.get('simulation_mode'):
                comparison['recommendations'].append("VM environment is in simulation mode (no SDR hardware).")
        
        # Display recommendations
        if comparison['recommendations']:
            print("\n💡 Recommendations:")
            for rec in comparison['recommendations']:
                print(f"   • {rec}")
        
        return comparison
    
    def benchmark_environments(self, quick: bool = False) -> Dict[str, Any]:
        """
        Benchmark performance of available environments.
        
        Args:
            quick: Run quick benchmark (less comprehensive)
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        print("🏃 Environment Performance Benchmark")
        print("=" * 40)
        
        if quick:
            print("Running quick benchmark...")
        else:
            print("Running comprehensive benchmark...")
        
        statuses = self.list_environments()
        healthy_envs = [env for env, status in statuses.items() if status.healthy]
        
        if not healthy_envs:
            print("❌ No healthy environments found for benchmarking.")
            return {'error': 'No healthy environments'}
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'platform': self.platform_info.platform,
            'environments': {},
            'comparison': {}
        }
        
        for env_name in healthy_envs:
            print(f"\n🔍 Benchmarking {env_name}...")
            
            config = self.environments[env_name]
            python_path = config.python_path
            
            env_results = {
                'environment': env_name,
                'device_type': config.device_type,
                'max_threads': config.max_threads,
                'tests': {}
            }
            
            # Test 1: Python startup time
            print("  📊 Testing Python startup time...")
            startup_times = []
            for _ in range(3):
                start_time = time.time()
                try:
                    subprocess.run([python_path, '-c', 'pass'], 
                                 capture_output=True, check=True, timeout=10)
                    startup_times.append(time.time() - start_time)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    startup_times.append(float('inf'))
            
            env_results['tests']['python_startup'] = {
                'avg_time': sum(startup_times) / len(startup_times),
                'min_time': min(startup_times),
                'unit': 'seconds'
            }
            
            # Test 2: NumPy performance
            print("  📊 Testing NumPy performance...")
            numpy_test = """
import time
import numpy as np

# Matrix multiplication test
start = time.time()
a = np.random.randn(1000, 1000)
b = np.random.randn(1000, 1000)
c = np.dot(a, b)
numpy_time = time.time() - start

print(f"numpy_time:{numpy_time}")
"""
            
            try:
                result = subprocess.run([python_path, '-c', numpy_test],
                                      capture_output=True, text=True, 
                                      check=True, timeout=30)
                
                # Parse result
                for line in result.stdout.split('\n'):
                    if 'numpy_time:' in line:
                        numpy_time = float(line.split(':')[1])
                        env_results['tests']['numpy_matmul'] = {
                            'time': numpy_time,
                            'unit': 'seconds'
                        }
                        break
                else:
                    env_results['tests']['numpy_matmul'] = {'error': 'Could not parse result'}
                    
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                env_results['tests']['numpy_matmul'] = {'error': str(e)}
            
            # Test 3: PyTorch performance (if not quick)
            if not quick:
                print("  📊 Testing PyTorch performance...")
                pytorch_test = f"""
import time
import torch

# Set device
device = torch.device('{config.device_type}' if torch.backends.mps.is_available() and '{config.device_type}' == 'mps' else 'cpu')

# Tensor operations test
start = time.time()
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
c = torch.matmul(a, b)
if device.type == 'mps':
    torch.mps.synchronize()
pytorch_time = time.time() - start

print(f"pytorch_time:{pytorch_time}")
print(f"device:{device}")
"""
                
                try:
                    result = subprocess.run([python_path, '-c', pytorch_test],
                                          capture_output=True, text=True,
                                          check=True, timeout=60)
                    
                    # Parse result
                    pytorch_time = None
                    device_used = None
                    
                    for line in result.stdout.split('\n'):
                        if 'pytorch_time:' in line:
                            pytorch_time = float(line.split(':')[1])
                        elif 'device:' in line:
                            device_used = line.split(':', 1)[1].strip()
                    
                    if pytorch_time is not None:
                        env_results['tests']['pytorch_matmul'] = {
                            'time': pytorch_time,
                            'device': device_used,
                            'unit': 'seconds'
                        }
                    else:
                        env_results['tests']['pytorch_matmul'] = {'error': 'Could not parse result'}
                        
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    env_results['tests']['pytorch_matmul'] = {'error': str(e)}
            
            benchmark_results['environments'][env_name] = env_results
            
            # Display results for this environment
            print(f"    Python startup: {env_results['tests']['python_startup']['avg_time']:.3f}s")
            
            if 'numpy_matmul' in env_results['tests'] and 'time' in env_results['tests']['numpy_matmul']:
                print(f"    NumPy matmul: {env_results['tests']['numpy_matmul']['time']:.3f}s")
            
            if 'pytorch_matmul' in env_results['tests'] and 'time' in env_results['tests']['pytorch_matmul']:
                pytorch_result = env_results['tests']['pytorch_matmul']
                print(f"    PyTorch matmul: {pytorch_result['time']:.3f}s ({pytorch_result.get('device', 'unknown')})")
        
        # Generate comparison
        if len(benchmark_results['environments']) > 1:
            print(f"\n📊 Performance Comparison:")
            
            # Compare startup times
            startup_times = {env: results['tests']['python_startup']['avg_time'] 
                           for env, results in benchmark_results['environments'].items()}
            fastest_startup = min(startup_times, key=startup_times.get)
            print(f"   Fastest startup: {fastest_startup} ({startup_times[fastest_startup]:.3f}s)")
            
            # Compare NumPy performance
            numpy_times = {}
            for env, results in benchmark_results['environments'].items():
                if 'numpy_matmul' in results['tests'] and 'time' in results['tests']['numpy_matmul']:
                    numpy_times[env] = results['tests']['numpy_matmul']['time']
            
            if numpy_times:
                fastest_numpy = min(numpy_times, key=numpy_times.get)
                print(f"   Fastest NumPy: {fastest_numpy} ({numpy_times[fastest_numpy]:.3f}s)")
                
                benchmark_results['comparison']['numpy_winner'] = fastest_numpy
            
            # Compare PyTorch performance
            pytorch_times = {}
            for env, results in benchmark_results['environments'].items():
                if 'pytorch_matmul' in results['tests'] and 'time' in results['tests']['pytorch_matmul']:
                    pytorch_times[env] = results['tests']['pytorch_matmul']['time']
            
            if pytorch_times:
                fastest_pytorch = min(pytorch_times, key=pytorch_times.get)
                print(f"   Fastest PyTorch: {fastest_pytorch} ({pytorch_times[fastest_pytorch]:.3f}s)")
                
                benchmark_results['comparison']['pytorch_winner'] = fastest_pytorch
        
        # Save benchmark results
        results_file = self.project_root / 'benchmark_results.json'
        try:
            with open(results_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            print(f"\n💾 Results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")
        
        return benchmark_results
    
    def interactive_menu(self):
        """Run interactive environment switching menu."""
        while True:
            print("\n" + "=" * 50)
            print("🔄 Environment Switcher")
            print("=" * 50)
            
            current_env = self.get_current_environment()
            if current_env:
                print(f"Current environment: {current_env}")
            else:
                print("No environment currently active")
            
            print("\nOptions:")
            print("1. Switch environment")
            print("2. Show status")
            print("3. Health check")
            print("4. Compare environments")
            print("5. Benchmark performance")
            print("6. Exit")
            
            try:
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == "1":
                    self._interactive_switch()
                elif choice == "2":
                    detailed = input("Show detailed status? (y/N): ").strip().lower() == 'y'
                    self.display_status(detailed=detailed)
                elif choice == "3":
                    env = input("Environment to check (or Enter for all): ").strip()
                    fix = input("Attempt to fix issues? (y/N): ").strip().lower() == 'y'
                    self.health_check(env if env else None, fix_issues=fix)
                elif choice == "4":
                    self.compare_environments()
                elif choice == "5":
                    quick = input("Quick benchmark? (Y/n): ").strip().lower() != 'n'
                    self.benchmark_environments(quick=quick)
                elif choice == "6":
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please enter 1-6.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _interactive_switch(self):
        """Interactive environment switching."""
        statuses = self.list_environments()
        available_envs = [env for env, status in statuses.items() if status.exists]
        
        if not available_envs:
            print("❌ No environments are installed.")
            print("Run 'python setup_dual_env.py' to create environments.")
            return
        
        print("\nAvailable environments:")
        for i, env_name in enumerate(available_envs, 1):
            status = statuses[env_name]
            health_icon = "✅" if status.healthy else "⚠️"
            print(f"{i}. {health_icon} {env_name} ({status.package_count} packages)")
        
        try:
            choice = input(f"\nSelect environment (1-{len(available_envs)}): ").strip()
            index = int(choice) - 1
            
            if 0 <= index < len(available_envs):
                env_name = available_envs[index]
                self.switch_to_environment(env_name)
            else:
                print("Invalid selection.")
                
        except (ValueError, IndexError):
            print("Invalid input. Please enter a number.")


def main():
    """Main entry point for environment switching utilities."""
    
    parser = argparse.ArgumentParser(
        description="Environment switching and management utilities"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Switch command
    switch_parser = subparsers.add_parser('switch', help='Switch to an environment')
    switch_parser.add_argument('environment', choices=['m1_native', 'vm_ubuntu'],
                              help='Environment to switch to')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show environment status')
    status_parser.add_argument('--detailed', action='store_true',
                              help='Show detailed status information')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Run health check')
    health_parser.add_argument('environment', nargs='?',
                              help='Environment to check (all if not specified)')
    health_parser.add_argument('--fix', action='store_true',
                              help='Attempt to fix detected issues')
    
    # Compare command
    subparsers.add_parser('compare', help='Compare environments')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark performance')
    benchmark_parser.add_argument('--quick', action='store_true',
                                 help='Run quick benchmark')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Run interactive menu')
    
    args = parser.parse_args()
    
    # Initialize switcher
    switcher = EnvironmentSwitcher()
    
    try:
        if args.command == 'switch':
            success = switcher.switch_to_environment(args.environment)
            return 0 if success else 1
            
        elif args.command == 'status':
            switcher.display_status(detailed=args.detailed)
            return 0
            
        elif args.command == 'health':
            switcher.health_check(args.environment, fix_issues=args.fix)
            return 0
            
        elif args.command == 'compare':
            switcher.compare_environments()
            return 0
            
        elif args.command == 'benchmark':
            switcher.benchmark_environments(quick=args.quick)
            return 0
            
        elif args.command == 'interactive':
            switcher.interactive_menu()
            return 0
            
        else:
            # No command specified, show status by default
            switcher.display_status()
            return 0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())