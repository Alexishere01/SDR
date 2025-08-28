#!/usr/bin/env python3
"""
Demo script showing environment switching capabilities.

This script demonstrates the key features of the dual environment setup system.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display its output."""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    """Demonstrate environment switching capabilities."""
    
    print("🚀 Dual Environment Setup Demo")
    print("This demo shows the key features of the environment switching system.")
    print("Note: This is a demonstration - no actual environments will be created.")
    
    # 1. Show help for setup script
    run_command("python setup_dual_env.py --help", 
                "Master Setup Script Help")
    
    # 2. Show help for switching script
    run_command("python switch_environment.py --help", 
                "Environment Switcher Help")
    
    # 3. Show current environment status
    run_command("python switch_environment.py status", 
                "Current Environment Status")
    
    # 4. Show detailed status
    run_command("python switch_environment.py status --detailed", 
                "Detailed Environment Status")
    
    # 5. Show environment comparison
    run_command("python switch_environment.py compare", 
                "Environment Comparison")
    
    # 6. Show validation-only mode
    run_command("python setup_dual_env.py --validate-only", 
                "Validation-Only Mode")
    
    # 7. Show health check
    run_command("python switch_environment.py health", 
                "Environment Health Check")
    
    print(f"\n{'='*60}")
    print("✅ Demo Complete!")
    print("='*60")
    print("\nKey Features Demonstrated:")
    print("• Master setup script with platform detection")
    print("• Interactive setup wizard")
    print("• Environment status checking")
    print("• Health validation and diagnostics")
    print("• Environment comparison")
    print("• Performance benchmarking capabilities")
    print("• Automated and manual setup modes")
    print("\nTo actually set up environments:")
    print("  python setup_dual_env.py")
    print("\nTo switch between environments:")
    print("  python switch_environment.py interactive")


if __name__ == "__main__":
    main()