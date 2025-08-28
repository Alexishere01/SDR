# Intelligent Receiver Training Guide

This guide explains how to train, test, and use the intelligent receiver model in GeminiSDR.

## Overview

The intelligent receiver uses Deep Q-Learning (DQN) to automatically find and optimize signal reception. It learns to adjust frequency, gain, and bandwidth parameters to maximize signal quality.

## Quick Start

### 1. Run Complete Pipeline (Recommended)

```bash
# Full training and testing pipeline
python scripts/run_intelligent_receiver_pipeline.py

# Quick training (fewer episodes for testing)
python scripts/run_intelligent_receiver_pipeline.py --quick
```

### 2. Individual Components

```bash
# Training only
python scripts/train_intelligent_receiver.py --mode full --episodes 1000

# Testing existing model
python scripts/test_intelligent_receiver.py --model models/intelligent_receiver_best.pth

# Interactive testing
python scripts/run_intelligent_receiver_pipeline.py --interactive
```

## Training Process

### What the Model Learns

The intelligent receiver learns to:
- **Find signals** in noisy environments
- **Optimize frequency tuning** to center on target signals
- **Adjust gain** for optimal SNR
- **Adapt bandwidth** for signal characteristics
- **Handle interference** and poor signal conditions

### Training Environment

The model trains in a simulated SDR environment that:
- Generates realistic I/Q signal data
- Simulates various modulation types (BPSK, QPSK, 8PSK, QAM)
- Adds realistic noise and interference
- Provides frequency, SNR, and timing challenges

### Training Parameters

Key training parameters (configurable in `conf/intelligent_receiver_training.yaml`):

```yaml
ml:
  batch_size: 64
  learning_rate: 0.0001
  replay_memory_size: 10000
  
training:
  num_episodes: 1000
  eval_frequency: 50
  save_frequency: 100
```

## Model Architecture

### Deep Q-Network (DQN)

- **Input**: 256-dimensional state vector (spectrum + metadata)
- **Hidden layers**: 3 layers with 512 neurons each
- **Output**: Q-values for 3 actions × 11 discrete levels each
- **Architecture**: Dueling DQN with value and advantage streams

### State Representation

The state vector includes:
- **Spectrum data** (250 bins): Power spectral density
- **Current parameters**: Frequency, gain, bandwidth
- **Signal metrics**: SNR estimate, signal power, noise floor

### Action Space

Three continuous actions, each discretized into 11 levels:
1. **Frequency adjustment**: ±500 kHz
2. **Gain adjustment**: ±5 dB  
3. **Bandwidth factor**: ±20% of sample rate

## Training Results

### Expected Performance

After successful training, the model should achieve:
- **Success rate**: >80% on basic scenarios
- **Convergence time**: <30 steps average
- **Frequency accuracy**: <100 kHz error
- **SNR improvement**: 10-15 dB over random search

### Training Metrics

The training process tracks:
- Episode rewards and success rates
- Convergence times and frequency accuracy
- Memory usage and training stability
- Model performance across different scenarios

## Testing and Evaluation

### Test Scenarios

The testing suite includes:

1. **Basic scenarios**: Clean signals at various frequencies/SNRs
2. **Interference scenarios**: Signals with narrowband/wideband interference
3. **Frequency sweep**: Performance across 70-200 MHz range
4. **SNR performance**: Capability at different signal strengths

### Performance Metrics

- **Success rate**: Percentage of episodes where signal is found
- **Convergence time**: Average steps to find signal
- **Frequency accuracy**: How close final frequency is to target
- **SNR improvement**: Signal quality improvement achieved

## Usage Examples

### Basic Training

```python
from scripts.train_intelligent_receiver import IntelligentReceiverTrainer

# Initialize trainer
trainer = IntelligentReceiverTrainer()

# Run full pipeline
trainer.run_full_pipeline(num_episodes=1000, num_test_episodes=100)
```

### Testing Existing Model

```python
from scripts.test_intelligent_receiver import IntelligentReceiverTester

# Initialize tester
tester = IntelligentReceiverTester("models/intelligent_receiver_best.pth")

# Run comprehensive tests
results = tester.run_comprehensive_test()
print(f"Success rate: {results['basic_scenarios']['success_rate']:.1%}")
```

### Using Trained Model

```python
from ml.intelligent_receiver import IntelligentReceiverML
from geminisdr.core.sdr_interface import PlutoSDRInterface

# Initialize with real SDR
sdr = PlutoSDRInterface()
receiver = IntelligentReceiverML(sdr)

# Load trained model
receiver.load_model("models/intelligent_receiver_best.pth")

# Find signal intelligently
result = receiver.find_signal_intelligently(search_time=30)
print(f"Found signal at {result['freq']/1e6:.1f} MHz with {result['snr']:.1f} dB SNR")
```

## Configuration

### Training Configuration

Edit `conf/intelligent_receiver_training.yaml` to customize:

```yaml
# Training episodes and evaluation
training:
  num_episodes: 1000
  eval_frequency: 50
  
# Model architecture
model:
  hidden_size: 512
  dropout_rate: 0.2
  
# Environment parameters
environment:
  frequency_range: [70000000, 200000000]
  snr_range: [-10, 30]
```

### Hardware Optimization

The system automatically optimizes for your hardware:
- **Apple Silicon (MPS)**: Optimized batch sizes and memory management
- **NVIDIA GPU (CUDA)**: Mixed precision and larger batches
- **CPU**: Conservative memory usage and threading

## Troubleshooting

### Common Issues

1. **Out of memory errors**
   ```bash
   # Reduce batch size in config
   ml:
     batch_size: 32  # Reduce from 64
   ```

2. **Slow training on CPU**
   ```bash
   # Use quick mode for testing
   python scripts/run_intelligent_receiver_pipeline.py --quick
   ```

3. **Poor convergence**
   ```bash
   # Increase training episodes
   python scripts/train_intelligent_receiver.py --episodes 2000
   ```

### Performance Tips

- **Use GPU/MPS** when available for 5-10x speedup
- **Monitor memory usage** - training uses ~2-4GB RAM
- **Start with quick mode** to verify setup before full training
- **Check logs** in `logs/intelligent_receiver_training.log`

## File Structure

After training, you'll have:

```
models/
├── intelligent_receiver_best.pth      # Best performing model
├── intelligent_receiver_final.pth     # Final model
└── intelligent_receiver_ep_*.pth      # Checkpoints

outputs/intelligent_receiver/
├── training_metrics_*.json            # Training data
├── test_results_*.json               # Test results
├── training_performance_*.png        # Performance plots
└── final_report_*.json              # Summary report

logs/
└── intelligent_receiver_training.log  # Training logs
```

## Advanced Usage

### Custom Training Scenarios

```python
# Generate custom training scenarios
trainer = IntelligentReceiverTrainer()
scenarios = trainer.generate_training_scenarios(num_scenarios=2000)

# Modify scenarios for specific use cases
for scenario in scenarios:
    if scenario['target_frequency'] > 150e6:
        scenario['has_interference'] = True
```

### Curriculum Learning

Enable progressive difficulty in config:

```yaml
training:
  use_curriculum: true
  curriculum_stages:
    - episodes: 200
      difficulty: "easy"
    - episodes: 300  
      difficulty: "medium"
    - episodes: 500
      difficulty: "mixed"
```

### Real Hardware Integration

```python
# Use with real SDR hardware
from geminisdr.core.sdr_interface import PlutoSDRInterface

sdr = PlutoSDRInterface()
receiver = IntelligentReceiverML(sdr)
receiver.load_model("models/intelligent_receiver_best.pth")

# Real-time signal finding
best_signal = receiver.find_signal_intelligently(search_time=60)
```

## Next Steps

1. **Train your first model** with the quick pipeline
2. **Experiment with parameters** in the config file
3. **Test on real hardware** if available
4. **Integrate into applications** using the trained model
5. **Contribute improvements** to the training process

For more details, see the API documentation and example scripts in the `examples/` directory.