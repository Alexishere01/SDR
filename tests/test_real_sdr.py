#!/usr/bin/env python3
# test_real_sdr.py - Quick test with real PlutoSDR
import numpy as np
from geminisdr.core.sdr_interface import PlutoSDRInterface
from ml.neural_amr import NeuralAMR
from geminisdr.ml.intelligent_receiver import IntelligentReceiverML
import os

def test_real_sdr():
    """Test the system with real PlutoSDR."""
    print("🔌 Testing Real SDR Connection...")
    
    # Connect to SDR
    sdr = PlutoSDRInterface()
    if not sdr.connect():
        print("❌ No PlutoSDR detected - using simulation")
        return
    
    print("✅ PlutoSDR connected!")
    
    # Test 1: Basic capture
    print("\n📡 Test 1: Capturing real RF data...")
    sdr.configure(center_freq=100e6, sample_rate=2e6, gain=30)
    samples = sdr.capture_batch(0.1)  # 100ms
    
    if samples is not None:
        print(f"✅ Captured {len(samples)} samples")
        print(f"   Signal power: {np.mean(np.abs(samples)**2):.6f}")
        print(f"   Peak amplitude: {np.max(np.abs(samples)):.3f}")
    else:
        print("❌ Failed to capture samples")
        return
    
    # Test 2: Classification with trained model
    if os.path.exists('models/neural_amr.pth'):
        print("\n🧠 Test 2: Classifying real signal...")
        try:
            amr = NeuralAMR()
            amr.load_model('models/neural_amr.pth')
            
            # Classify the captured signal
            prediction, confidence = amr.predict(samples[:1024])
            print(f"✅ Classification: {prediction} ({confidence[0]:.1%} confidence)")
            
        except Exception as e:
            print(f"❌ Classification failed: {e}")
    else:
        print("⚠️  No trained model found. Run 'python train_models.py --neural' first")
    
    # Test 3: Intelligent signal finding
    if os.path.exists('models/intelligent_receiver.pth'):
        print("\n🎯 Test 3: Intelligent signal finding...")
        try:
            receiver = IntelligentReceiverML(sdr)
            receiver.load_model('models/intelligent_receiver.pth')
            
            # Set poor initial conditions
            print("   Setting suboptimal parameters...")
            sdr.configure(center_freq=105e6, sample_rate=2e6, gain=15)
            
            # Let AI optimize
            print("   AI optimizing parameters...")
            result = receiver.find_signal_intelligently(search_time=15)
            
            print(f"✅ AI Results:")
            print(f"   Best frequency: {result['freq']/1e6:.3f} MHz")
            print(f"   Optimal gain: {result['gain']:.0f} dB")
            print(f"   Achieved SNR: {result['snr']:.1f} dB")
            
        except Exception as e:
            print(f"❌ Intelligent receiver failed: {e}")
    else:
        print("⚠️  No intelligent receiver model found. Run 'python train_models.py --intelligent' first")
    
    # Test 4: Frequency scan
    print("\n🔍 Test 4: Quick frequency scan...")
    frequencies = [88e6, 100e6, 162e6, 915e6]  # FM, Aircraft, NOAA, ISM
    
    for freq in frequencies:
        sdr.configure(center_freq=freq, sample_rate=2e6, gain=40)
        samples = sdr.capture_batch(0.05)  # 50ms
        
        if samples is not None:
            power = 10 * np.log10(np.mean(np.abs(samples)**2))
            print(f"   {freq/1e6:6.1f} MHz: {power:6.1f} dBm")
    
    print("\n🎉 Real SDR testing complete!")
    print("\nNext steps:")
    print("1. Try: python demo.py --demo 4  (intelligent signal finding)")
    print("2. Try: python demo.py --demo 6  (real-time processing)")
    print("3. Connect antennas for better signal reception")

if __name__ == "__main__":
    test_real_sdr()