# SDR AI System: Complete Technical Data Flow Presentation

## Table of Contents
1. [System Overview & Data Flow](#system-overview--data-flow)
2. [Training Data Generation](#training-data-generation)
3. [Model Architecture & Training](#model-architecture--training)
4. [Reinforcement Learning Implementation](#reinforcement-learning-implementation)
5. [The "Flashlight" Concept - Technical Deep Dive](#the-flashlight-concept---technical-deep-dive)
6. [Real Transmission Testing](#real-transmission-testing)
7. [Frequency Hopping Implementation](#frequency-hopping-implementation)
8. [Adversarial Training](#adversarial-training)
9. [Performance Metrics & Results](#performance-metrics--results)

---

## System Overview & Data Flow

### Complete Data Pipeline

```
Raw RF Signals → SDR Hardware → Digital I/Q Samples → Signal Processing → Feature Extraction → ML Models → Intelligent Decisions → SDR Control
```

### Detailed Data Flow Stages

#### Stage 1: RF Signal Acquisition
**Input:** Electromagnetic waves (70 MHz - 6 GHz)
**Hardware:** PlutoSDR with 12-bit ADC
**Output:** Complex I/Q samples at 2 MHz sample rate

```
RF Signal (analog) → ADC → I/Q Stream
- Sample Rate: 2 MHz
- Resolution: 12-bit
- Format: Complex float32 (I + jQ)
- Data Rate: 16 MB/second
```

#### Stage 2: Digital Signal Processing
**Input:** Raw I/Q samples (complex float32)
**Processing:** FFT, filtering, normalization
**Output:** Processed spectrum data

```
I/Q Samples → FFT(2048 points) → Magnitude Spectrum → Downsampling → Feature Vector
- FFT Size: 2048 points
- Spectrum Bins: 1024 (positive frequencies)
- Downsampled: 250 bins for ML
- Frequency Resolution: 1.56 kHz per bin
```

#### Stage 3: Feature Engineering
**Input:** Spectrum data + SDR parameters
**Processing:** Normalization, concatenation, windowing
**Output:** 256-dimensional state vector

```
Feature Vector Composition:
- Spectrum Features: 250 values (normalized magnitude)
- Current Frequency: 1 value (normalized to 0-1)
- Current Gain: 1 value (normalized to 0-1)
- Bandwidth Factor: 1 value (0.8 default)
- SNR Estimate: 1 value
- Signal Power: 1 value
- Noise Floor: 1 value
Total: 256 dimensions
```

---

## Training Data Generation

### Synthetic Data Generation Pipeline

#### Mathematical Signal Models
**BPSK Generation:**
```
Data Bits: b[n] ∈ {0,1}
Symbols: s[n] = 2*b[n] - 1  (maps to {-1,+1})
Baseband: x(t) = Σ s[n] * p(t - nT)
where p(t) = pulse shaping filter
```

**QPSK Generation:**
```
Data Bits: [b0, b1] → Symbol Index
Constellation: {1+j, -1+j, -1-j, 1-j} / √2
Complex Signal: x(t) = Σ c[n] * p(t - nT)
```

**Channel Impairments:**
```
Received Signal: r(t) = x(t) * e^(j2πΔft) * e^(jφ) + n(t)
- Δf: Frequency offset
- φ: Phase offset  
- n(t): AWGN with specified SNR
```

#### Dataset Generation Parameters
**Training Dataset Composition:**
- **Modulations:** BPSK, QPSK, 8PSK, 16QAM, 64QAM
- **SNR Range:** -10 dB to +30 dB (5 dB steps)
- **Samples per Class:** 1000-2000
- **Signal Length:** 1024 complex samples
- **Total Dataset Size:** ~50,000 samples

**Data Augmentation:**
- Frequency offsets: ±50 kHz
- Phase rotations: 0° to 360°
- Timing offsets: ±10 samples
- Amplitude scaling: ±3 dB

---

## Model Architecture & Training

### Traditional AMR Architecture

#### Feature Extraction Pipeline
**Statistical Moments:**
```
M_pq = E[|r(t)|^p * (r(t))^q * (r*(t))^(p-q)]
- M20: Second-order moment
- M21: Magnitude squared
- M40, M41, M42: Fourth-order cumulants
```

**Spectral Features:**
```
Spectral Centroid: fc = Σ(f * |X(f)|) / Σ|X(f)|
Spectral Bandwidth: BW = √(Σ((f-fc)² * |X(f)|) / Σ|X(f)|)
PAPR: Peak-to-Average Power Ratio
```

**Cyclostationary Features:**
```
Cyclic Autocorrelation: R_α(τ) = E[r(t) * r*(t-τ) * e^(-j2παt)]
For α ∈ {0, 0.5, 1.0, 2.0} (cycle frequencies)
```

#### Random Forest Classifier
- **Trees:** 100 estimators
- **Features:** 23 hand-crafted features
- **Depth:** Unlimited (with min_samples_split=2)
- **Training:** Scikit-learn RandomForestClassifier

### Neural AMR Architecture

#### CNN Architecture Details
```
Input: I/Q samples (2 channels × 1024 samples)
├── Conv1D(64 filters, kernel=7, padding=3)
├── BatchNorm1D(64)
├── ReLU()
├── MaxPool1D(2)
├── Conv1D(128 filters, kernel=5, padding=2)
├── BatchNorm1D(128)
├── ReLU()
├── MaxPool1D(2)
├── Conv1D(256 filters, kernel=3, padding=1)
├── BatchNorm1D(256)
├── ReLU()
├── MaxPool1D(2)
├── Flatten()
├── Linear(256 * 128, 256)
├── Dropout(0.5)
├── ReLU()
├── Linear(256, 128)
├── Dropout(0.5)
├── ReLU()
└── Linear(128, num_classes)
```

#### Training Hyperparameters
- **Optimizer:** Adam (lr=0.001, β1=0.9, β2=0.999)
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 32
- **Epochs:** 50
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Regularization:** Dropout(0.5), BatchNorm

---

## Reinforcement Learning Implementation

### MDP Formulation

#### State Space (S)
**Dimension:** 256
**Composition:**
```
s_t = [spectrum_bins(250), freq_norm(1), gain_norm(1), bw_factor(1), snr_est(1), power(1), noise(1)]
```

**State Preprocessing:**
- Spectrum normalization: s_spectrum = spectrum / max(spectrum)
- Frequency normalization: s_freq = (freq - 70e6) / (6e9 - 70e6)
- Gain normalization: s_gain = gain / 70

#### Action Space (A)
**Dimension:** 3 (continuous actions discretized)
**Actions:**
1. **Frequency Adjustment:** Δf ∈ [-500kHz, +500kHz] (11 discrete steps)
2. **Gain Adjustment:** Δg ∈ [-5dB, +5dB] (11 discrete steps)
3. **Bandwidth Factor:** Δbw ∈ [-0.2, +0.2] (11 discrete steps)

**Action Discretization:**
```
freq_actions = linspace(-500e3, 500e3, 11)  # 100 kHz steps
gain_actions = linspace(-5, 5, 11)          # 1 dB steps
bw_actions = linspace(-0.2, 0.2, 11)       # 0.04 factor steps
```

**Total Action Space:** 11³ = 1,331 discrete actions

#### Reward Function (R)
**Primary Reward:** Achieved SNR
```
R_primary = SNR_achieved
```

**Bonus Rewards:**
```
R_frequency = {
    +30 if |freq_error| < 100kHz
    +10 if |freq_error| < 500kHz
    -20 if |freq_error| > 2MHz
    0   otherwise
}

R_snr_bonus = {
    +10 if SNR > 15dB
    0   otherwise
}
```

**Total Reward:**
```
R_total = R_primary + R_frequency + R_snr_bonus
```

#### Why These Choices?

**State Space Rationale:**
- **Spectrum (250 bins):** Captures frequency domain characteristics
- **Current Settings (3 values):** Provides context for relative adjustments
- **Performance Metrics (3 values):** Feedback on current tuning quality

**Action Space Rationale:**
- **100 kHz frequency steps:** Match typical channel spacing
- **1 dB gain steps:** Standard RF engineering granularity
- **4% bandwidth steps:** Gradual optimization without signal loss

**Reward Function Rationale:**
- **SNR-based:** Direct optimization of signal quality
- **Frequency accuracy bonus:** Encourages precise tuning
- **Threshold-based bonuses:** Clear success criteria

### Deep Q-Network Architecture

#### Dueling DQN Structure
```
Input State (256) → Shared Network → Value/Advantage Split
├── Shared Layers:
│   ├── Linear(256, 512) + ReLU
│   ├── Linear(512, 512) + ReLU
│   └── Linear(512, 256) + ReLU
├── Value Stream:
│   └── Linear(256, 1)
└── Advantage Stream:
    └── Linear(256, 3*11) → reshape(3, 11)

Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

#### Training Algorithm
**Experience Replay:**
- Buffer Size: 10,000 transitions
- Batch Size: 32
- Sampling: Uniform random

**Target Network:**
- Update Frequency: Every 10 episodes
- Soft Update: τ = 1.0 (hard update)

**Exploration Strategy:**
- ε-greedy with decay
- Initial ε: 1.0
- Decay Rate: 0.995 per episode
- Minimum ε: 0.01

---

## The "Flashlight" Concept - Technical Deep Dive

### Instantaneous Bandwidth Analysis

#### Hardware Constraints
**PlutoSDR Specifications:**
- Maximum Sample Rate: 61.44 MHz
- Usable Bandwidth: 80% of sample rate
- Current Configuration: 2 MHz sample rate → 1.6 MHz bandwidth

#### Frequency Domain Resolution
**FFT Analysis:**
```
FFT Size: 2048 points
Sample Rate: 2 MHz
Frequency Resolution: 2MHz / 2048 = 976.56 Hz per bin
Usable Bins: 1024 (positive frequencies)
Bandwidth per Bin: 1.95 kHz
```

#### Spatial-Frequency Analogy
**"Flashlight" Parameters:**
- **Beam Width:** 1.6 MHz (what we can see at once)
- **Search Space:** 5.93 GHz (where we can point it)
- **Resolution:** 1.95 kHz (finest detail we can resolve)
- **Movement Speed:** 100 kHz per Q-learning action

**Coverage Analysis:**
```
Total Search Positions: (6e9 - 70e6) / 100e3 = 59,300 positions
Positions per Bandwidth: 1.6e6 / 100e3 = 16 positions
Non-overlapping Coverage: 59,300 / 16 = 3,706 unique windows
```

#### Multi-Scale Search Strategy
**Coarse Search:**
- Step Size: 1 MHz (no overlap)
- Coverage Time: 5,930 steps
- Purpose: Rapid spectrum survey

**Fine Search:**
- Step Size: 100 kHz (overlap for continuity)
- Coverage Time: 59,300 steps
- Purpose: Precise signal acquisition

**Intelligent Search:**
- AI-guided movement
- Typical Convergence: 10-50 steps
- Speedup Factor: 1,000-6,000x over brute force

---

## Real Transmission Testing

### Test Signal Generation

#### Hardware Setup
**Transmitter:** Second PlutoSDR or signal generator
**Receiver:** Primary PlutoSDR with AI system
**Environment:** Controlled RF chamber or open air

#### Test Signal Parameters
**Signal Types:**
- Continuous Wave (CW) tones
- Modulated signals (BPSK, QPSK, etc.)
- Frequency hopping sequences
- Spread spectrum signals

**Test Scenarios:**
1. **Static Signal:** Fixed frequency, known location
2. **Weak Signal:** Low power, high noise environment
3. **Interfered Signal:** Multiple signals, interference
4. **Mobile Signal:** Frequency drifting over time

### Performance Validation

#### Metrics Collection
**Acquisition Metrics:**
- Time to first detection
- Frequency accuracy at detection
- SNR at detection
- False alarm rate

**Tracking Metrics:**
- Lock maintenance duration
- Re-acquisition time after loss
- Frequency tracking accuracy
- SNR optimization performance

#### Comparison Baselines
**Traditional Scanning:**
- Sequential frequency sweep
- Fixed step size
- No intelligence

**Energy Detection:**
- Threshold-based detection
- No modulation recognition
- No optimization

**AI System:**
- Intelligent search
- Adaptive optimization
- Pattern learning

---

## Frequency Hopping Implementation

### Transmitter Design

#### Pseudo-Random Sequence Generation
**Linear Congruential Generator:**
```
X_{n+1} = (a * X_n + c) mod m
where:
- a = 1664525 (multiplier)
- c = 1013904223 (increment)  
- m = 2^32 (modulus)
- X_0 = seed
```

**Frequency Mapping:**
```
freq_index = X_n mod num_channels
frequency = freq_min + freq_index * channel_spacing
```

#### Hopping Parameters
**Slow Hopping (WiFi-style):**
- Hop Rate: 2.5 hops/second
- Dwell Time: 400 ms
- Channels: 79 (2.4 GHz ISM band)

**Fast Hopping (Bluetooth-style):**
- Hop Rate: 1600 hops/second
- Dwell Time: 625 μs
- Channels: 79 (2.4 GHz ISM band)

**Military-style:**
- Hop Rate: 100 hops/second
- Dwell Time: 10 ms
- Channels: 1000 (VHF/UHF bands)

### Receiver Synchronization Approaches

#### Approach 1: Pattern Learning
**LSTM-based Predictor:**
```
Input: Frequency history [f_{t-n}, ..., f_{t-1}]
Hidden State: h_t = LSTM(f_{t-1}, h_{t-1})
Prediction: f̂_t = Linear(h_t)
```

**Training Data:**
- Observed hop sequences
- Multiple hopping patterns
- Various hop rates

#### Approach 2: Wideband Capture
**Requirements:**
- Sample Rate: 61.44 MHz (maximum)
- Bandwidth: 49 MHz instantaneous
- Processing: Real-time hop detection

**Hop Detection:**
```
Spectrogram Analysis:
- Window Size: 1024 samples
- Overlap: 50%
- Hop Detection: Energy threshold + timing
```

#### Approach 3: Multi-Hypothesis Tracking
**Parallel Tracking:**
- Track multiple potential patterns simultaneously
- Bayesian model selection
- Confidence-weighted predictions

### Training for Frequency Hopping

#### Simulation Environment
**Hopping Transmitter Simulator:**
- Configurable hop rates
- Multiple PRNG algorithms
- Realistic channel models
- Interference simulation

**Training Scenarios:**
1. **Known Pattern:** Learn specific hopping sequence
2. **Unknown Pattern:** Discover pattern from observations
3. **Changing Pattern:** Adapt to pattern modifications
4. **Jammed Pattern:** Maintain sync under jamming

#### Curriculum Learning
**Phase 1:** Slow, predictable hopping
**Phase 2:** Faster hopping with pattern changes
**Phase 3:** Adversarial hopping with anti-learning

---

## Adversarial Training

### Game-Theoretic Framework

#### Two-Player Zero-Sum Game
**Players:**
- Receiver (Maximizer): Wants to maximize SNR and maintain communication
- Jammer (Minimizer): Wants to minimize receiver performance

**Payoff Matrix:**
```
U_receiver = SNR_achieved + Communication_success
U_jammer = -U_receiver
```

#### Nash Equilibrium
**Receiver Strategy:** Optimal frequency/gain/bandwidth selection
**Jammer Strategy:** Optimal jamming power allocation across strategies

### Jammer Intelligence Levels

#### Level 1: Random Jammer
**Behavior:** Random strategy selection
**Parameters:** Uniform distribution over jamming types
**Receiver Challenge:** Learn to ignore random interference

#### Level 2: Adaptive Jammer
**Behavior:** Multi-armed bandit strategy selection
**Learning:** Track success rate of each jamming strategy
**Parameters:** UCB (Upper Confidence Bound) selection

```
UCB Score = success_rate + √(2 * ln(total_attempts) / strategy_attempts)
```

#### Level 3: Intelligent Jammer
**Behavior:** Pattern recognition and prediction
**Learning:** LSTM-based receiver behavior prediction
**Strategy:** Predictive jamming at anticipated frequencies

### Jamming Strategies

#### Narrowband Jamming
**Mechanism:** Concentrate power at receiver frequency
**Power Allocation:** 80% of budget at single frequency
**Effectiveness:** High against static receivers, low against agile receivers

#### Wideband Jamming
**Mechanism:** Spread power across wide bandwidth
**Power Allocation:** Uniform across 10-20 MHz
**Effectiveness:** Moderate against all receivers

#### Sweep Jamming
**Mechanism:** Frequency sweep around target
**Parameters:** Sweep rate, sweep bandwidth
**Effectiveness:** Good against slow-adapting receivers

#### Pulse Jamming
**Mechanism:** High-power intermittent bursts
**Parameters:** Pulse width, pulse repetition rate
**Effectiveness:** Disrupts timing-sensitive protocols

#### Deceptive Jamming
**Mechanism:** Generate fake signals
**Implementation:** Modulated interference signals
**Effectiveness:** Confuses signal detection algorithms

### Co-evolutionary Training

#### Training Loop
```
For each episode:
1. Receiver selects action based on current policy
2. Jammer observes receiver action
3. Jammer selects jamming strategy
4. Environment computes rewards
5. Both agents update their policies
6. Repeat until convergence
```

#### Curriculum Progression
**Stage 1:** Receiver vs Random Jammer
- Build basic anti-jamming skills
- Learn frequency agility

**Stage 2:** Receiver vs Adaptive Jammer
- Develop pattern randomization
- Learn counter-adaptation

**Stage 3:** Receiver vs Intelligent Jammer
- Advanced evasion techniques
- Predictive counter-measures

### Performance Metrics

#### Receiver Metrics
- **Acquisition Time:** Time to find signal under jamming
- **Maintenance Rate:** Percentage of time maintaining good SNR
- **Anti-Jam Effectiveness:** Performance degradation under jamming
- **Adaptation Speed:** Time to counter new jamming strategies

#### Jammer Metrics
- **Disruption Rate:** Percentage of successful communication blocks
- **Power Efficiency:** Jamming effectiveness per watt
- **Learning Speed:** Time to adapt to receiver changes
- **Strategy Diversity:** Number of effective jamming modes

#### System Metrics
- **Nash Equilibrium Convergence:** Stability of final strategies
- **Robustness:** Performance under strategy variations
- **Scalability:** Performance with multiple jammers/receivers

---

## Performance Metrics & Results

### Traditional AMR Results
**Accuracy by SNR:**
- SNR > 20 dB: 95-98% accuracy
- SNR 10-20 dB: 85-95% accuracy
- SNR 0-10 dB: 70-85% accuracy
- SNR < 0 dB: 50-70% accuracy

**Processing Speed:** 1-5 ms per classification

### Neural AMR Results
**Accuracy by SNR:**
- SNR > 20 dB: 98-99% accuracy
- SNR 10-20 dB: 90-98% accuracy
- SNR 0-10 dB: 80-90% accuracy
- SNR < 0 dB: 60-80% accuracy

**Processing Speed:** 0.5-2 ms per classification

### Intelligent Receiver Results
**Signal Acquisition:**
- Average Time: 15-45 seconds
- Success Rate: 90-95% (clean environment)
- Frequency Accuracy: ±50 kHz typical

**Adversarial Performance:**
- vs Random Jammer: 85% success rate
- vs Adaptive Jammer: 70% success rate
- vs Intelligent Jammer: 60% success rate

### Frequency Hopping Results
**Synchronization Performance:**
- Slow Hoppers (<10 hops/sec): 95% sync rate
- Medium Hoppers (10-100 hops/sec): 80% sync rate
- Fast Hoppers (>1000 hops/sec): 60% sync rate (wideband approach)

**Learning Time:**
- Pattern Recognition: 10-50 hops
- Synchronization: 5-20 hops after pattern learned
- Re-sync after Loss: 2-10 hops

This comprehensive technical presentation covers the complete data flow from RF signals to intelligent decisions, providing the technical depth needed for a professional presentation on the SDR AI system.