#!/usr/bin/env python3
"""
Complete Intelligent Receiver Pipeline Runner

This script provides a simple interface to run the complete intelligent receiver
training and testing pipeline with sensible defaults.

Usage:
    # Run full pipeline (train + test)
    python scripts/run_intelligent_receiver_pipeline.py

    # Quick training (fewer episodes)
    python scripts/run_intelligent_receiver_pipeline.py --quick

    # Test existing model
    python scripts/run_intelligent_receiver_pipeline.py --test-only --model models/intelligent_receiver_best.pth

    # Interactive mode
    python scripts/run_intelligent_receiver_pipeline.py --interactive
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_dependencies():
    """Check if required files exist."""
    required_files = [
        "scripts/train_intelligent_receiver.py",
        "scripts/test_intelligent_receiver.py",
        "conf/intelligent_receiver_training.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ All required files found")
    return True

def main():
    parser = argparse.ArgumentParser(description="Intelligent Receiver Complete Pipeline")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick training with fewer episodes (200 train, 50 test)")
    parser.add_argument("--test-only", action="store_true",
                       help="Only run testing (requires --model)")
    parser.add_argument("--model", type=str,
                       help="Path to model file for testing")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive testing mode")
    parser.add_argument("--config", type=str, default=None,
                       help="Configuration file path (uses default if not specified)")
    parser.add_argument("--episodes", type=int,
                       help="Number of training episodes (overrides quick mode)")
    parser.add_argument("--test-episodes", type=int,
                       help="Number of test episodes")
    
    args = parser.parse_args()
    
    print("🚀 Intelligent Receiver Pipeline Runner")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    success = True
    
    if args.interactive:
        # Interactive testing mode
        if not args.model:
            # Look for existing models
            model_files = list(Path("models").glob("intelligent_receiver*.pth"))
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                print(f"Using latest model: {latest_model}")
                args.model = str(latest_model)
            else:
                print("❌ No model found for interactive testing")
                print("Run training first or specify --model")
                sys.exit(1)
        
        cmd = [
            sys.executable, "scripts/test_intelligent_receiver.py",
            "--model", args.model,
            "--mode", "interactive"
        ]
        
        if args.config:
            cmd.extend(["--config", args.config])
        
        success = run_command(cmd, "Interactive Testing")
    
    elif args.test_only:
        # Test only mode
        if not args.model:
            print("❌ --model required for test-only mode")
            sys.exit(1)
        
        if not os.path.exists(args.model):
            print(f"❌ Model file not found: {args.model}")
            sys.exit(1)
        
        cmd = [
            sys.executable, "scripts/test_intelligent_receiver.py",
            "--model", args.model,
            "--mode", "comprehensive"
        ]
        
        if args.test_episodes:
            cmd.extend(["--episodes", str(args.test_episodes)])
        
        if args.config:
            cmd.extend(["--config", args.config])
        
        success = run_command(cmd, "Model Testing")
    
    else:
        # Full pipeline: training + testing
        
        # Step 1: Training
        train_cmd = [
            sys.executable, "scripts/train_intelligent_receiver.py",
            "--mode", "full"
        ]
        
        # Only add config if explicitly specified
        if args.config:
            train_cmd.extend(["--config", args.config])
        
        if args.episodes:
            train_cmd.extend(["--episodes", str(args.episodes)])
        elif args.quick:
            train_cmd.extend(["--episodes", "200"])
        
        if args.test_episodes:
            train_cmd.extend(["--test_episodes", str(args.test_episodes)])
        elif args.quick:
            train_cmd.extend(["--test_episodes", "50"])
        
        success = run_command(train_cmd, "Training Pipeline")
        
        if success:
            # Step 2: Additional comprehensive testing
            print("\n🔍 Running additional comprehensive testing...")
            
            # Find the best model
            best_model = Path("models/intelligent_receiver_best.pth")
            if not best_model.exists():
                # Fallback to final model
                best_model = Path("models/intelligent_receiver_final.pth")
            
            if best_model.exists():
                test_cmd = [
                    sys.executable, "scripts/test_intelligent_receiver.py",
                    "--model", str(best_model),
                    "--mode", "comprehensive"
                ]
                
                if args.config:
                    test_cmd.extend(["--config", args.config])
                
                success = run_command(test_cmd, "Comprehensive Testing")
            else:
                print("⚠️ No trained model found for testing")
                success = False
    
    # Final summary
    print("\n" + "="*60)
    if success:
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("\nGenerated files:")
        
        # List generated models
        model_files = list(Path("models").glob("intelligent_receiver*.pth"))
        if model_files:
            print("\nTrained models:")
            for model_file in sorted(model_files):
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"  📦 {model_file} ({size_mb:.1f} MB)")
        
        # List output files
        output_files = list(Path("outputs").glob("intelligent_receiver*/*"))
        if output_files:
            print(f"\nOutput files: {len(output_files)} files in outputs/intelligent_receiver*/")
        
        print("\nNext steps:")
        print("  • Review training plots in outputs/intelligent_receiver/")
        print("  • Test model interactively:")
        print(f"    python scripts/run_intelligent_receiver_pipeline.py --interactive")
        print("  • Use model in your applications:")
        print("    from ml.intelligent_receiver import IntelligentReceiverML")
        
    else:
        print("❌ PIPELINE FAILED")
        print("\nTroubleshooting:")
        print("  • Check log files in logs/")
        print("  • Verify all dependencies are installed")
        print("  • Try running with --quick for faster testing")
        print("  • Check available memory and disk space")
    
    print("="*60)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()