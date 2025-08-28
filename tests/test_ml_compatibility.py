#!/usr/bin/env python3
"""
Test script for ML cross-platform compatibility.

This script tests the updated ML modules with hardware abstraction
to ensure they work correctly across M1 and VM environments.
"""

import numpy as np
import torch
import sys
import os



def test_hardware_abstraction():
    """Test hardware abstraction layer."""
    print("=== Testing Hardware Abstraction ===")
    
    try:
        from geminisdr.environments.hardware_abstraction import HardwareAbstraction
        
        hw = HardwareAbstraction()
        
        # Test device detection
        device_info = hw.get_compute_device()
        print(f"✓ Detected compute device: {device_info.device_type.value}")
        print(f"  Device name: {device_info.name}")
        print(f"  Device string: {hw.get_device_string()}")
        
        # Test SDR interface
        sdr_info = hw.get_sdr_interface()
        print(f"✓ SDR mode: {sdr_info.mode.value}")
        print(f"  Devices: {sdr_info.devices}")
        
        # Test optimizations
        optimizations = hw.optimize_for_platform()
        print(f"✓ Platform optimizations loaded: {len(optimizations)} settings")
        
        return hw
        
    except Exception as e:
        print(f"✗ Hardware abstraction test failed: {e}")
        return None

def test_neural_amr_compatibility(hw_abstraction=None):
    """Test NeuralAMR cross-platform compatibility."""
    print("\n=== Testing NeuralAMR Compatibility ===")
    
    try:
        from ml.neural_amr import NeuralAMR
        
        # Create NeuralAMR with hardware abstraction
        amr = NeuralAMR(hardware_abstraction=hw_abstraction)
        print(f"✓ NeuralAMR initialized on device: {amr.device}")
        
        # Test with dummy data
        print("Creating dummy training data...")
        num_samples = 100
        signal_length = 1024
        
        # Generate dummy I/Q signals
        signals = []
        labels = []
        modulations = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
        
        for i in range(num_samples):
            # Create random complex signal
            signal = np.random.randn(signal_length) + 1j * np.random.randn(signal_length)
            signals.append(signal)
            labels.append(modulations[i % len(modulations)])
        
        print(f"✓ Generated {num_samples} dummy signals")
        
        # Test data preparation
        train_loader, val_loader, num_classes = amr.prepare_data(signals, labels)
        print(f"✓ Data preparation successful: {num_classes} classes")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Test model initialization (without full training)
        signal_length = signals[0].shape[0]
        from ml.neural_amr import CNNModulationClassifier
        model = CNNModulationClassifier(num_classes, signal_length).to(amr.device)
        print(f"✓ Model created and moved to device: {amr.device}")
        
        # Test forward pass
        dummy_batch = torch.randn(4, 2, signal_length).to(amr.device)
        with torch.no_grad():
            output = model(dummy_batch)
        print(f"✓ Forward pass successful: output shape {output.shape}")
        
        # Test prediction interface
        amr.model = model
        amr.label_map = {label: idx for idx, label in enumerate(modulations)}
        amr.inverse_label_map = {v: k for k, v in amr.label_map.items()}
        
        test_signal = signals[0]
        prediction, probs = amr.predict(test_signal)
        print(f"✓ Prediction successful: {prediction} (confidence: {max(probs):.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ NeuralAMR compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_intelligent_receiver_compatibility(hw_abstraction=None):
    """Test IntelligentReceiver cross-platform compatibility."""
    print("\n=== Testing IntelligentReceiver Compatibility ===")
    
    try:
        from geminisdr.ml.intelligent_receiver import IntelligentReceiverML, SimulatedSDREnvironment
        
        # Create mock SDR interface
        class MockSDR:
            def configure(self, **kwargs):
                pass
            def capture_batch(self, duration):
                return np.random.randn(int(2e6 * duration)) + 1j * np.random.randn(int(2e6 * duration))
        
        mock_sdr = MockSDR()
        
        # Create IntelligentReceiver with hardware abstraction
        receiver = IntelligentReceiverML(mock_sdr, hardware_abstraction=hw_abstraction)
        print(f"✓ IntelligentReceiver initialized on device: {receiver.device}")
        
        # Test Q-network initialization
        print(f"✓ Q-network created: {sum(p.numel() for p in receiver.q_network.parameters())} parameters")
        
        # Test environment
        env = SimulatedSDREnvironment()
        state, _ = env.reset()
        print(f"✓ Simulation environment working: state shape {state.shape}")
        
        # Test action selection
        action = receiver._choose_action(state)
        print(f"✓ Action selection working: {action}")
        
        # Test single training step (without full training)
        # Add some dummy experiences
        for _ in range(50):  # Need enough for batch_size
            next_state, reward, terminated, truncated, info = env.step(action)
            receiver.memory.append((state, action, reward, next_state, terminated or truncated))
            state = next_state
            action = receiver._choose_action(state)
        
        # Test training step
        receiver._train_step()
        print("✓ Training step successful")
        
        return True
        
    except Exception as e:
        print(f"✗ IntelligentReceiver compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_specific_features():
    """Test device-specific features and optimizations."""
    print("\n=== Testing Device-Specific Features ===")
    
    device_type = None
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_type = "mps"
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device_type = "cuda"
        device = torch.device("cuda")
    else:
        device_type = "cpu"
        device = torch.device("cpu")
    
    print(f"Testing on device: {device_type}")
    
    try:
        # Test tensor operations
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.mm(x, y)
        print(f"✓ Basic tensor operations work on {device_type}")
        
        # Test neural network operations
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        ).to(device)
        
        output = model(x)
        print(f"✓ Neural network operations work on {device_type}")
        
        # Test backward pass
        loss = torch.nn.functional.mse_loss(output, torch.randn(100, 10).to(device))
        loss.backward()
        print(f"✓ Backward pass works on {device_type}")
        
        # Device-specific tests
        if device_type == "mps":
            # Test MPS fallback
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            print("✓ MPS fallback enabled")
            
        elif device_type == "cuda":
            print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
            print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
            
        elif device_type == "cpu":
            print(f"✓ CPU threads: {torch.get_num_threads()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Device-specific test failed: {e}")
        return False

def main():
    """Run all compatibility tests."""
    print("SDR AI ML Cross-Platform Compatibility Test")
    print("=" * 50)
    
    # Test hardware abstraction
    hw_abstraction = test_hardware_abstraction()
    
    # Test device-specific features
    device_test_passed = test_device_specific_features()
    
    # Test ML modules
    neural_amr_passed = test_neural_amr_compatibility(hw_abstraction)
    intelligent_receiver_passed = test_intelligent_receiver_compatibility(hw_abstraction)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Hardware Abstraction", hw_abstraction is not None),
        ("Device-Specific Features", device_test_passed),
        ("NeuralAMR Compatibility", neural_amr_passed),
        ("IntelligentReceiver Compatibility", intelligent_receiver_passed)
    ]
    
    passed = 0
    for test_name, result in tests:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All compatibility tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())