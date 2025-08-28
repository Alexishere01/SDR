#!/usr/bin/env python3
"""Test basic baseband signal generation."""

import numpy as np
import matplotlib.pyplot as plt
from geminisdr.core.signal_generator import SDRSignalGenerator

def test_basic_signals():
    generator = SDRSignalGenerator(sample_rate=1e6)
    
    # 1. Most basic: Complex noise
    print("Generating complex noise...")
    noise = generator.generate_complex_noise(1000, noise_power=0.1)
    print(f"Noise shape: {noise.shape}, dtype: {noise.dtype}")
    
    # 2. Simplest modulated signal: BPSK
    print("\nGenerating BPSK signal...")
    bpsk = generator.generate_bpsk(num_symbols=50, samples_per_symbol=8)
    print(f"BPSK shape: {bpsk.shape}, dtype: {bpsk.dtype}")
    
    # 3. Show signal properties
    print(f"\nBPSK signal stats:")
    print(f"  Mean power: {np.mean(np.abs(bpsk)**2):.4f}")
    print(f"  Peak amplitude: {np.max(np.abs(bpsk)):.4f}")
    print(f"  Real part range: [{np.min(bpsk.real):.3f}, {np.max(bpsk.real):.3f}]")
    print(f"  Imag part range: [{np.min(bpsk.imag):.3f}, {np.max(bpsk.imag):.3f}]")
    
    return noise, bpsk

if __name__ == "__main__":
    noise, bpsk = test_basic_signals()
    print("\nBasic baseband signals generated successfully!")