# Q-Learning Step Size Analysis

## Current Action Space Configuration

### Frequency Adjustments
```python
self.freq_actions = np.linspace(-500e3, 500e3, 11)
```

**Step Size: 100 kHz**

**Reasoning:**
- **Typical Channel Spacing**: Most digital communications use channel spacings of 25 kHz to 200 kHz
- **100 kHz steps** allow fine-tuning within a channel while being large enough to hop between adjacent channels
- **±500 kHz range** covers typical frequency uncertainty in SDR systems
- **11 steps** provide good granularity without making the action space too large

**Real-world Impact:**
- At 100 MHz carrier: 100 kHz = 0.1% frequency adjustment
- Allows precise tuning for narrow-band signals
- Can quickly escape interference in adjacent frequencies

### Gain Adjustments
```python
self.gain_actions = np.linspace(-5, 5, 11)
```

**Step Size: 1 dB**

**Reasoning:**
- **1 dB steps** are the standard granularity in RF systems
- **±5 dB range** covers typical gain adjustments needed for optimization
- Smaller than 1 dB is usually not meaningful due to noise
- Larger than 1 dB can cause overshooting in sensitive systems

**Real-world Impact:**
- 1 dB ≈ 26% power change (10^(1/10) = 1.26)
- Allows fine-tuning without causing instability
- Matches typical SDR hardware granularity

### Bandwidth Adjustments
```python
self.bw_actions = np.linspace(-0.2, 0.2, 11)
```

**Step Size: 0.04 (4% of sample rate)**

**Reasoning:**
- **4% steps** allow gradual bandwidth optimization
- **±20% range** covers most practical bandwidth adjustments
- Prevents dramatic changes that could lose the signal
- Matches typical filter bandwidth adjustments

**Real-world Impact:**
- At 2 MHz sample rate: 4% = 80 kHz bandwidth change
- Allows optimization between noise rejection and signal capture
- Prevents over-narrowing that loses signal components

## Alternative Step Size Configurations

### Fine-Tuning Configuration (More Precise)
```python
# Smaller steps for precision
self.freq_actions = np.linspace(-200e3, 200e3, 21)    # 20 kHz steps
self.gain_actions = np.linspace(-3, 3, 13)            # 0.5 dB steps
self.bw_actions = np.linspace(-0.1, 0.1, 11)          # 2% steps
```

**Pros:** More precise control, better for weak signals
**Cons:** Slower learning, more actions to explore

### Coarse-Tuning Configuration (Faster Learning)
```python
# Larger steps for speed
self.freq_actions = np.linspace(-1e6, 1e6, 11)        # 200 kHz steps
self.gain_actions = np.linspace(-10, 10, 11)          # 2 dB steps
self.bw_actions = np.linspace(-0.3, 0.3, 7)           # ~8.5% steps
```

**Pros:** Faster learning, good for strong signals
**Cons:** Less precise, might miss optimal settings

### Adversarial Configuration (Anti-Jamming)
```python
# Larger frequency hops to escape jamming
self.freq_actions = np.linspace(-2e6, 2e6, 21)        # 200 kHz steps, wider range
self.gain_actions = np.linspace(-10, 10, 21)          # 1 dB steps, wider range
self.bw_actions = np.linspace(-0.4, 0.4, 11)          # 8% steps, wider range
```

**Pros:** Better jamming evasion, more agile
**Cons:** Might overshoot optimal settings

## Step Size Impact on Learning

### Learning Speed vs Precision Trade-off

**Smaller Steps:**
- **Advantages:**
  - More precise final tuning
  - Better for weak signals
  - Smoother convergence
- **Disadvantages:**
  - Slower exploration
  - More episodes needed
  - Larger action space

**Larger Steps:**
- **Advantages:**
  - Faster initial learning
  - Better exploration
  - Smaller action space
- **Disadvantages:**
  - Less precise final tuning
  - Might overshoot optimal
  - Oscillation around optimum

### Current Choice Justification

Our current configuration balances:

1. **Learning Speed**: 11 actions per dimension = 11³ = 1,331 total combinations
2. **Precision**: Steps small enough for practical tuning
3. **Coverage**: Range large enough to handle typical scenarios
4. **Hardware Compatibility**: Matches typical SDR capabilities

## Adaptive Step Sizes (Future Enhancement)

### Curriculum Learning Approach
```python
# Start with large steps, reduce over time
if episode < 100:
    freq_step = 200e3  # Large steps for exploration
elif episode < 300:
    freq_step = 100e3  # Medium steps for refinement
else:
    freq_step = 50e3   # Small steps for precision
```

### Dynamic Step Sizing
```python
# Adjust step size based on reward gradient
if reward_improving:
    step_size *= 0.9   # Smaller steps when close to optimum
else:
    step_size *= 1.1   # Larger steps when far from optimum
```

## Practical Examples

### Example 1: Finding a 100 MHz Signal
- **Initial frequency**: 99.5 MHz (500 kHz off)
- **Q-learning actions**: [+100k, +100k, +100k, +100k, +100k] = +500 kHz
- **Result**: Perfectly tuned to 100 MHz in 5 steps

### Example 2: Gain Optimization
- **Initial gain**: 25 dB (too low, SNR = 5 dB)
- **Optimal gain**: 35 dB (SNR = 20 dB)
- **Q-learning actions**: [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1] = +10 dB
- **Result**: Optimized gain in 10 steps

### Example 3: Bandwidth Tuning
- **Initial BW**: 80% of sample rate (too wide, noisy)
- **Optimal BW**: 60% of sample rate
- **Sample rate**: 2 MHz
- **Q-learning actions**: [-0.04, -0.04, -0.04, -0.04, -0.04] = -0.2 (20% reduction)
- **Result**: BW reduced from 1.6 MHz to 1.2 MHz

## Conclusion

The current step sizes (100 kHz, 1 dB, 4%) provide a good balance between:
- **Learning efficiency** (reasonable action space size)
- **Practical precision** (meaningful adjustments)
- **Hardware compatibility** (matches SDR capabilities)
- **Robustness** (works across various scenarios)

These can be adjusted based on specific requirements:
- **Precision applications**: Reduce step sizes
- **Speed applications**: Increase step sizes  
- **Adversarial environments**: Increase ranges
- **Weak signal scenarios**: Reduce frequency steps