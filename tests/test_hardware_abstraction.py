#!/usr/bin/env python3
"""
Test script for hardware abstraction integration.
"""

import sys
import os
import torch



def test_hardware_abstraction():
    """Test hardware abstraction functionality."""
    print("=== Testing Hardware Abstraction ===")
    
    try:
        from geminisdr.environments.hardware_abstraction import HardwareAbstraction
        
        # Initialize hardware abstraction
        hw = HardwareAbstraction()
        
        # Test device detection
        device_info = hw.get_compute_device()
        print(f"✓ Detected device: {device_info.device_type.value}")
        print(f"  Name: {device_info.name}")
        print(f"  Memory: {device_info.memory_mb} MB" if device_info.memory_mb else "  Memory: Unknown")
        
        # Test device string
        device_string = hw.get_device_string()
        print(f"✓ Device string: {device_string}")
        
        # Test SDR interface
        sdr_info = hw.get_sdr_interface()
        print(f"✓ SDR mode: {sdr_info.mode.value}")
        print(f"  Devices: {sdr_info.devices}")
        
        # Test platform optimizations
        optimizations = hw.optimize_for_platform()
        print(f"✓ Platform optimizations:")
        print(f"  Mixed precision: {optimizations.get('mixed_precision', False)}")
        print(f"  Recommended batch size: {optimizations.get('recommended_batch_size', 'Unknown')}")
        print(f"  Workers: {optimizations.get('dataloader_num_workers', 'Unknown')}")
        
        # Test validation
        validation = hw.validate_configuration()
        print(f"✓ Configuration validation:")
        for check, result in validation.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hardware abstraction test failed: {e}")
        return False

def test_ml_integration():
    """Test ML module integration with hardware abstraction."""
    print("\n=== Testing ML Integration ===")
    
    try:
        from ml.neural_amr import NeuralAMR
        from geminisdr.environments.hardware_abstraction import HardwareAbstraction
        
        # Initialize hardware abstraction
        hw = HardwareAbstraction()
        
        # Test neural AMR with hardware abstraction
        neural_amr = NeuralAMR(hardware_abstraction=hw)
        print(f"✓ Neural AMR initialized with device: {neural_amr.device}")
        
        # Test intelligent receiver
        from geminisdr.ml.intelligent_receiver import IntelligentReceiverML, SimulatedSDREnvironment
        sdr_env = SimulatedSDREnvironment()
        intelligent_receiver = IntelligentReceiverML(sdr_interface=sdr_env, hardware_abstraction=hw)
        print(f"✓ Intelligent receiver initialized with device: {intelligent_receiver.device}")
        
        return True
        
    except Exception as e:
        print(f"✗ ML integration test failed: {e}")
        return False

def test_torch_compatibility():
    """Test PyTorch compatibility with detected device."""
    print("\n=== Testing PyTorch Compatibility ===")
    
    try:
        from geminisdr.environments.hardware_abstraction import HardwareAbstraction
        
        hw = HardwareAbstraction()
        device_string = hw.get_device_string()
        device = torch.device(device_string)
        
        # Test tensor creation and operations
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.matmul(x, y)
        
        print(f"✓ Tensor operations work on {device}")
        print(f"  Result shape: {z.shape}")
        print(f"  Result device: {z.device}")
        
        # Test MPS-specific functionality if available
        if device.type == 'mps':
            # Test MPS fallback
            try:
                # Some operations that might need fallback
                result = torch.linalg.svd(x)
                print("✓ MPS advanced operations work")
            except Exception as e:
                print(f"⚠️  MPS fallback needed for some operations: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Hardware Abstraction Integration Test")
    print("=" * 50)
    
    tests = [
        test_hardware_abstraction,
        test_ml_integration,
        test_torch_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())