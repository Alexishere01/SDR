#!/usr/bin/env python3
"""
Master Dual Environment Setup Script

This script detects the platform and runs the appropriate environment setup
with an interactive setup wizard and validation of created environments.

Requirements addressed: 3.1, 3.4, 7.1
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add environments directory to path for imports
environments_dir = Path(__file__).parent / "environments"
if environments_dir.exists():
    sys.path.insert(0, str(environments_dir))

try:
    from geminisdr.environments.platform_detector import PlatformDetector, PlatformInfo
    from geminisdr.environments.environment_manager import EnvironmentManager, EnvironmentStatus
except ImportError as e:
    print(f"Error importing environment modules: {e}")
    print("Make sure the environments/ directory exists with required modules")
    sys.exit(1)


class DualEnvironmentSetup:
    """
    Master setup system for dual M1/VM environments.
    
    Provides interactive setup wizard with platform detection,
    automatic environment selection, and comprehensive validation.
    """
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.platform_detector = PlatformDetector()
        self.environment_manager = EnvironmentManager(str(self.project_root))
        self.platform_info = self.platform_detector.get_platform_info()
        self.capabilities = self.platform_detector.get_hardware_capabilities()
        
    def display_welcome(self):
        """Display welcome message and system information."""
        print("🚀 SDR AI Dual Environment Setup")
        print("=" * 50)
        print("This script will set up optimized development environments")
        print("for both M1 Mac native and Ubuntu VM development.")
        print()
        
        # Display detected platform info
        print("📋 System Information:")
        print(f"  Platform: {self.platform_info.platform}")
        print(f"  Architecture: {self.platform_info.architecture}")
        print(f"  Python Version: {self.platform_info.python_version}")
        print(f"  MPS Available: {'Yes' if self.platform_info.has_mps else 'No'}")
        print(f"  SDR Hardware: {'Yes' if self.platform_info.has_sdr else 'No (Simulation Mode)'}")
        print(f"  CPU Cores: {self.capabilities.cpu_cores}")
        print(f"  Memory: {self.capabilities.max_memory} MB")
        print()
    
    def validate_system_requirements(self) -> bool:
        """
        Validate system requirements before setup.
        
        Returns:
            bool: True if system meets requirements
        """
        print("🔍 Validating System Requirements...")
        
        validation_results = self.platform_detector.validate_environment()
        all_passed = True
        
        for check, result in validation_results.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check.replace('_', ' ').title()}: {result}")
            if not result:
                all_passed = False
        
        if not all_passed:
            print("\n⚠️  Some system requirements are not met.")
            print("Please address the issues above before continuing.")
            return False
        
        print("✅ All system requirements met!")
        return True
    
    def get_user_preferences(self) -> Dict[str, any]:
        """
        Get user preferences through interactive prompts.
        
        Returns:
            Dict[str, any]: User preferences
        """
        print("\n🎯 Setup Preferences")
        print("-" * 20)
        
        preferences = {}
        
        # Determine recommended environment
        recommended_env = self.platform_info.platform
        if recommended_env == 'unknown':
            recommended_env = 'vm_ubuntu'
        
        print(f"Recommended environment for your system: {recommended_env}")
        
        # Ask which environments to set up
        print("\nWhich environments would you like to set up?")
        print("1. Recommended environment only")
        print("2. Both environments (M1 + VM)")
        print("3. M1 environment only")
        print("4. VM environment only")
        print("5. Custom selection")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == "1":
                    preferences['environments'] = [recommended_env]
                    break
                elif choice == "2":
                    preferences['environments'] = ['m1_native', 'vm_ubuntu']
                    break
                elif choice == "3":
                    preferences['environments'] = ['m1_native']
                    break
                elif choice == "4":
                    preferences['environments'] = ['vm_ubuntu']
                    break
                elif choice == "5":
                    # Custom selection
                    envs = []
                    if self._ask_yes_no("Set up M1 native environment?"):
                        envs.append('m1_native')
                    if self._ask_yes_no("Set up VM Ubuntu environment?"):
                        envs.append('vm_ubuntu')
                    
                    if not envs:
                        print("No environments selected. Please choose at least one.")
                        continue
                    
                    preferences['environments'] = envs
                    break
                else:
                    print("Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\n\nSetup cancelled by user.")
                sys.exit(0)
        
        # Ask about additional options
        preferences['skip_existing'] = self._ask_yes_no(
            "\nSkip setup for environments that already exist?", default=True
        )
        
        preferences['run_tests'] = self._ask_yes_no(
            "Run validation tests after setup?", default=True
        )
        
        preferences['create_activation_scripts'] = self._ask_yes_no(
            "Create activation scripts?", default=True
        )
        
        return preferences
    
    def _ask_yes_no(self, question: str, default: bool = False) -> bool:
        """
        Ask a yes/no question with default.
        
        Args:
            question: Question to ask
            default: Default answer
            
        Returns:
            bool: User's answer
        """
        default_str = "Y/n" if default else "y/N"
        
        while True:
            try:
                answer = input(f"{question} ({default_str}): ").strip().lower()
                
                if not answer:
                    return default
                elif answer in ['y', 'yes']:
                    return True
                elif answer in ['n', 'no']:
                    return False
                else:
                    print("Please answer 'y' or 'n'")
                    
            except KeyboardInterrupt:
                print("\n\nSetup cancelled by user.")
                sys.exit(0)
    
    def check_existing_environments(self) -> Dict[str, EnvironmentStatus]:
        """
        Check status of existing environments.
        
        Returns:
            Dict[str, EnvironmentStatus]: Environment statuses
        """
        print("\n🔍 Checking Existing Environments...")
        
        statuses = self.environment_manager.list_environments()
        
        for env_name, status in statuses.items():
            if status.exists:
                health_icon = "✅" if status.healthy else "⚠️"
                print(f"  {health_icon} {env_name}: Exists ({status.package_count} packages)")
                if status.issues:
                    for issue in status.issues[:3]:  # Show first 3 issues
                        print(f"    - {issue}")
                    if len(status.issues) > 3:
                        print(f"    - ... and {len(status.issues) - 3} more issues")
            else:
                print(f"  ❌ {env_name}: Not found")
        
        return statuses
    
    def setup_environment(self, env_name: str, skip_existing: bool = False) -> bool:
        """
        Set up a specific environment.
        
        Args:
            env_name: Environment name to set up
            skip_existing: Skip if environment already exists
            
        Returns:
            bool: True if setup successful
        """
        print(f"\n🔧 Setting up {env_name} environment...")
        
        # Check if environment already exists
        status = self.environment_manager.validate_environment(env_name)
        
        if status.exists and skip_existing:
            if status.healthy:
                print(f"✅ {env_name} environment already exists and is healthy. Skipping.")
                return True
            else:
                print(f"⚠️  {env_name} environment exists but has issues. Recreating...")
        
        # Run platform-specific setup
        success = self.environment_manager.setup_environment(env_name)
        
        if success:
            print(f"✅ {env_name} environment setup completed!")
        else:
            print(f"❌ {env_name} environment setup failed!")
        
        return success
    
    def run_validation_tests(self, environments: List[str]) -> Dict[str, bool]:
        """
        Run comprehensive validation tests on environments.
        
        Args:
            environments: List of environment names to test
            
        Returns:
            Dict[str, bool]: Test results for each environment
        """
        print("\n🧪 Running Validation Tests...")
        
        results = {}
        
        for env_name in environments:
            print(f"\n  Testing {env_name}...")
            
            # Get health check results
            health_report = self.environment_manager.health_check(env_name)
            
            is_healthy = health_report['overall_health'] == 'healthy'
            results[env_name] = is_healthy
            
            if is_healthy:
                print(f"    ✅ {env_name}: All tests passed")
                print(f"       Packages: {health_report['status']['package_count']}")
                print(f"       Python: {health_report['status']['python_version']}")
            else:
                print(f"    ❌ {env_name}: Issues detected")
                for issue in health_report['issues'][:3]:
                    print(f"       - {issue}")
                
                if health_report['recommendations']:
                    print("       Recommendations:")
                    for rec in health_report['recommendations'][:2]:
                        print(f"       → {rec}")
        
        return results
    
    def create_activation_scripts(self, environments: List[str]) -> bool:
        """
        Create activation scripts for environments.
        
        Args:
            environments: List of environment names
            
        Returns:
            bool: True if all scripts created successfully
        """
        print("\n📝 Creating Activation Scripts...")
        
        success = True
        
        for env_name in environments:
            try:
                config = self.environment_manager.environments[env_name]
                script_path = Path(config.activation_script)
                
                if script_path.exists():
                    print(f"  ✅ {script_path.name}: Already exists")
                else:
                    # The environment manager should have created this during setup
                    if script_path.exists():
                        print(f"  ✅ {script_path.name}: Created")
                    else:
                        print(f"  ❌ {script_path.name}: Failed to create")
                        success = False
                        
            except Exception as e:
                print(f"  ❌ Error creating script for {env_name}: {e}")
                success = False
        
        return success
    
    def display_setup_summary(self, results: Dict[str, bool], preferences: Dict[str, any]):
        """
        Display setup summary and next steps.
        
        Args:
            results: Setup results for each environment
            preferences: User preferences
        """
        print("\n" + "=" * 50)
        print("📊 Setup Summary")
        print("=" * 50)
        
        successful_envs = [env for env, success in results.items() if success]
        failed_envs = [env for env, success in results.items() if not success]
        
        if successful_envs:
            print("✅ Successfully set up environments:")
            for env in successful_envs:
                config = self.environment_manager.environments[env]
                print(f"   • {env}")
                print(f"     Virtual env: {config.venv_path}")
                print(f"     Activation: {config.activation_script}")
        
        if failed_envs:
            print("\n❌ Failed to set up environments:")
            for env in failed_envs:
                print(f"   • {env}")
        
        print(f"\n📁 Project root: {self.project_root}")
        
        # Display next steps
        print("\n🚀 Next Steps:")
        
        if successful_envs:
            recommended_env = self.platform_info.platform
            if recommended_env in successful_envs:
                config = self.environment_manager.environments[recommended_env]
                print(f"\n1. Activate your recommended environment ({recommended_env}):")
                print(f"   source {config.activation_script}")
                
            print("\n2. Test your setup:")
            print("   python -c \"import torch; print(f'PyTorch: {torch.__version__}')\"")
            
            if self.platform_info.has_mps and 'm1_native' in successful_envs:
                print("   python -c \"import torch; print(f'MPS available: {torch.backends.mps.is_available()}')\"")
            
            print("\n3. Run a demo:")
            print("   python simple_demo.py")
            
            print("\n4. Switch between environments:")
            print("   source activate_m1.sh    # For M1 environment")
            print("   source activate_vm.sh     # For VM environment")
        
        if failed_envs:
            print(f"\n⚠️  To retry failed setups:")
            print(f"   python setup_dual_env.py")
            print("   Or run individual setup scripts:")
            for env in failed_envs:
                if env == 'm1_native':
                    print("   python setup_m1_environment.py")
                elif env == 'vm_ubuntu':
                    print("   python setup_vm_environment.py")
    
    def run_interactive_setup(self):
        """Run the complete interactive setup process."""
        try:
            # Welcome and system info
            self.display_welcome()
            
            # Validate system requirements
            if not self.validate_system_requirements():
                return False
            
            # Get user preferences
            preferences = self.get_user_preferences()
            
            # Check existing environments
            existing_statuses = self.check_existing_environments()
            
            # Confirm setup
            print(f"\n📋 Setup Plan:")
            print(f"  Environments to set up: {', '.join(preferences['environments'])}")
            print(f"  Skip existing: {preferences['skip_existing']}")
            print(f"  Run tests: {preferences['run_tests']}")
            
            if not self._ask_yes_no("\nProceed with setup?", default=True):
                print("Setup cancelled by user.")
                return False
            
            # Run setup for each environment
            setup_results = {}
            
            for env_name in preferences['environments']:
                success = self.setup_environment(env_name, preferences['skip_existing'])
                setup_results[env_name] = success
                
                if not success:
                    if not self._ask_yes_no(f"\n{env_name} setup failed. Continue with other environments?"):
                        break
            
            # Create activation scripts
            if preferences['create_activation_scripts']:
                self.create_activation_scripts(preferences['environments'])
            
            # Run validation tests
            if preferences['run_tests']:
                test_results = self.run_validation_tests(preferences['environments'])
                # Update results with test outcomes
                for env_name in preferences['environments']:
                    if setup_results.get(env_name, False):
                        setup_results[env_name] = test_results.get(env_name, False)
            
            # Display summary
            self.display_setup_summary(setup_results, preferences)
            
            # Return overall success
            return any(setup_results.values())
            
        except KeyboardInterrupt:
            print("\n\nSetup cancelled by user.")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error during setup: {e}")
            return False
    
    def run_automated_setup(self, environments: Optional[List[str]] = None):
        """
        Run automated setup without user interaction.
        
        Args:
            environments: List of environments to set up (auto-detect if None)
        """
        print("🤖 Running Automated Setup...")
        
        if environments is None:
            # Auto-detect recommended environment
            recommended = self.platform_info.platform
            if recommended == 'unknown':
                recommended = 'vm_ubuntu'
            environments = [recommended]
        
        print(f"Setting up environments: {', '.join(environments)}")
        
        setup_results = {}
        
        for env_name in environments:
            print(f"\n🔧 Setting up {env_name}...")
            success = self.setup_environment(env_name, skip_existing=True)
            setup_results[env_name] = success
        
        # Run validation tests
        test_results = self.run_validation_tests(environments)
        
        # Update results with test outcomes
        for env_name in environments:
            if setup_results.get(env_name, False):
                setup_results[env_name] = test_results.get(env_name, False)
        
        # Display summary
        preferences = {
            'environments': environments,
            'skip_existing': True,
            'run_tests': True,
            'create_activation_scripts': True
        }
        self.display_setup_summary(setup_results, preferences)
        
        return any(setup_results.values())


def main():
    """Main entry point for dual environment setup."""
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Set up dual M1/VM development environments for SDR AI project"
    )
    parser.add_argument(
        '--auto', action='store_true',
        help='Run automated setup without user interaction'
    )
    parser.add_argument(
        '--env', choices=['m1_native', 'vm_ubuntu'], action='append',
        help='Specific environment(s) to set up (can be used multiple times)'
    )
    parser.add_argument(
        '--validate-only', action='store_true',
        help='Only run validation tests on existing environments'
    )
    
    args = parser.parse_args()
    
    # Initialize setup system
    setup = DualEnvironmentSetup()
    
    try:
        if args.validate_only:
            # Only run validation
            print("🧪 Running Validation Only...")
            environments = args.env or ['m1_native', 'vm_ubuntu']
            results = setup.run_validation_tests(environments)
            
            all_healthy = all(results.values())
            if all_healthy:
                print("\n✅ All environments are healthy!")
                return 0
            else:
                print("\n❌ Some environments have issues.")
                return 1
                
        elif args.auto:
            # Run automated setup
            success = setup.run_automated_setup(args.env)
            return 0 if success else 1
        else:
            # Run interactive setup
            success = setup.run_interactive_setup()
            return 0 if success else 1
            
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())