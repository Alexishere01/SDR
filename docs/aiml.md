# AI/ML Architecture for SDR System

## Overview

This document explains the AI/ML architecture of our Software Defined Radio (SDR) system, which combines traditional signal processing with modern machine learning techniques to create intelligent, adaptive radio systems.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SDR AI/ML SYSTEM                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Signal Gen    │  │   SDR Interface │  │   Visualizer    │  │
│  │                 │  │                 │  │                 │  │
│  │ • Modulations   │  │ • PlutoSDR      │  │ • Real-time     │  │
│  │ • Channel Fx    │  │ • Simulation    │  │ • Spectrograms  │  │
│  │ • Noise/Jamming │  │ • Streaming     │  │ • Constellations│  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        ML LAYER                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Traditional AMR │  │   Neural AMR    │  │ Intelligent RX  │  │
│  │                 │  │                 │  │                 │  │
│  │ • Hand-crafted  │  │ • CNN-based     │  │ • Deep Q-Net    │  │
│  │ • Features      │  │ • End-to-end    │  │ • Reinforcement │  │
│  │ • Random Forest │  │ • PyTorch       │  │ • Signal Search │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    ADVERSARIAL LAYER                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Smart Jammer    │  │ Adversarial Env │  │ Battle-hardened │  │
│  │                 │  │                 │  │ Receiver        │  │
│  │ • Multi-strategy│  │ • Game Theory   │  │ • Anti-jamming  │  │
│  │ • Learning      │  │ • Co-evolution  │  │ • Frequency Hop │  │
│  │ • Adaptation    │  │ • Real-time     │  │ • Robust        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Signal Generation & Processing

**Purpose**: Create realistic training data and test scenarios

**Key Features**:
- Multiple modulation schemes (BPSK, QPSK, 8PSK, 16QAM, 64QAM)
- Realistic channel impairments (noise, fading, frequency offset)
- Pulse shaping and filtering
- Configurable SNR and interference

**Implementation**: `core/signal_generator.py`

### 2. SDR Interface Layer

**Purpose**: Abstract hardware/simulation interface

**Key Features**:
- PlutoSDR hardware support
- Simulation fallback mode
- Real-time streaming
- Batch capture modes
- Automatic retry and error handling

**Implementation**: `core/sdr_interface.py`

### 3. Traditional AMR (Automatic Modulation Recognition)

**Purpose**: Baseline modulation classification using classical methods

**Architecture**:
```
Signal → Feature Extraction → Classification → Prediction
         ├─ Statistical moments
         ├─ Higher-order cumulants  
         ├─ Spectral features
         ├─ Cyclostationary features
         └─ Instantaneous features
```

**Features Extracted**:
- **Statistical Moments**: 2nd-4th order moments of amplitude and phase
- **Cumulants**: C20, C21, C40, C41, C42 for modulation discrimination
- **Spectral**: Centroid, bandwidth, rolloff, PAPR
- **Cyclostationary**: Cyclic spectral density at key frequencies
- **Instantaneous**: Amplitude, phase, frequency variations

**Classifier**: Random Forest (100 trees)

**Implementation**: `ml/traditional_amr.py`

### 4. Neural AMR

**Purpose**: Deep learning approach to modulation recognition

**Architecture**:
```
I/Q Signal → CNN Layers → Dense Layers → Softmax → Classification
             ├─ Conv1D(64, kernel=7)
             ├─ Conv1D(128, kernel=5)  
             ├─ Conv1D(256, kernel=3)
             ├─ MaxPool + BatchNorm
             ├─ Dense(256)
             ├─ Dense(128)
             └─ Dense(num_classes)
```

**Key Features**:
- End-to-end learning from raw I/Q samples
- Convolutional layers for feature extraction
- Batch normalization for training stability
- Dropout for regularization
- Multi-class softmax output

**Training**: Adam optimizer, cross-entropy loss, learning rate scheduling

**Implementation**: `ml/neural_amr.py`

### 5. Intelligent Receiver (Deep Q-Learning)

**Purpose**: Autonomous signal acquisition and optimization

**Architecture**:
```
Environment State → DQN → Action Selection → SDR Control
                    ├─ Dueling Architecture
                    ├─ Experience Replay
                    ├─ Target Network
                    └─ Epsilon-Greedy
```

**State Space**: 256-dimensional vector containing:
- Spectrum analysis (250 bins)
- Current frequency, gain, bandwidth
- SNR estimate, signal power, noise floor

**Action Space**: 3-dimensional continuous control:
- Frequency adjustment (-1MHz to +1MHz)
- Gain adjustment (-10dB to +10dB)  
- Bandwidth factor (-0.5 to +0.5)

**Reward Function**:
- Primary: Achieved SNR
- Bonus: Frequency accuracy, signal acquisition
- Penalty: Poor performance, excessive adjustments

**Implementation**: `ml/intelligent_receiver.py`

## Adversarial AI System

### Smart Jammer Architecture

**Purpose**: Intelligent opposition to test receiver robustness

**Jamming Strategies**:

1. **Narrowband**: Concentrated power at receiver frequency
2. **Wideband**: Spread spectrum noise jamming
3. **Sweep**: Frequency sweeping around target
4. **Pulse**: High-power intermittent jamming
5. **Adaptive**: Learns receiver patterns
6. **Deceptive**: Fake signals to confuse
7. **Smart Follow**: Predicts receiver movement

**Learning Algorithm**:
```
Receiver Observation → Strategy Selection → Jamming → Effectiveness → Learning Update
                       ├─ Multi-armed Bandit
                       ├─ Upper Confidence Bound
                       ├─ Success Rate Tracking
                       └─ Power Allocation
```

**Implementation**: `ml/adversarial_jamming.py`

### Adversarial Training Environment

**Purpose**: Co-evolutionary training of receiver vs jammer

**Game Theory Approach**:
- Receiver tries to maximize SNR and signal acquisition
- Jammer tries to minimize receiver performance
- Both agents learn and adapt in real-time
- Nash equilibrium emerges through training

**Training Phases**:
1. **Random Jammer**: Baseline performance
2. **Adaptive Jammer**: Pattern learning
3. **Intelligent Jammer**: Predictive strategies

**Metrics**:
- Win/loss ratios
- SNR degradation
- Strategy effectiveness
- Adaptation speed

## Parameter Selection & Optimization

### Why These Parameters?

**DQN Architecture**:
- **State size (256)**: Balances information richness with computational efficiency
- **Hidden layers (512)**: Sufficient capacity for complex signal patterns
- **Dueling architecture**: Separates value and advantage for better learning
- **Experience replay (10k-20k)**: Breaks correlation, improves stability

**Training Hyperparameters**:
- **Learning rate (1e-4 to 5e-5)**: Conservative for stable convergence
- **Epsilon decay (0.995-0.9998)**: Gradual shift from exploration to exploitation
- **Batch size (32-64)**: Balance between gradient noise and memory efficiency
- **Target network update (10-20 episodes)**: Stability vs adaptation speed

**Action Discretization**:
- **Frequency steps (11-21)**: Fine enough for precise tuning
- **Gain steps (11-21)**: Covers typical dynamic range
- **Bandwidth steps (11)**: Sufficient for most scenarios

### Feature Engineering Rationale

**Traditional AMR Features**:
- **Moments**: Capture amplitude/phase statistics
- **Cumulants**: Modulation-specific signatures
- **Spectral**: Frequency domain characteristics
- **Cyclostationary**: Exploit periodic properties of digital modulations

**Neural AMR Input**:
- **I/Q samples**: Raw complex baseband preserves all information
- **Sequence length (1024)**: Compromise between context and computation
- **Normalization**: Prevents saturation, improves convergence

## Training Methodology

### Progressive Difficulty Training

**Intelligent Receiver**:
1. **Simulation Environment**: Learn basic signal acquisition
2. **Noise Robustness**: Handle various SNR conditions
3. **Adversarial Training**: Battle against intelligent jamming

**Adversarial Training**:
1. **Weak Jammer (20W)**: Basic anti-jamming skills
2. **Moderate Jammer (40W)**: Intermediate challenges
3. **Strong Jammer (60W)**: Advanced techniques
4. **Powerful Jammer (80W)**: Extreme scenarios
5. **Elite Jammer (100W)**: Ultimate test

### Curriculum Learning Benefits

- **Gradual Complexity**: Prevents overwhelming the learning agent
- **Skill Building**: Each level builds on previous capabilities
- **Robust Performance**: Handles wide range of scenarios
- **Transfer Learning**: Skills transfer across difficulty levels

## Performance Metrics

### Traditional AMR
- **Accuracy**: Overall classification rate
- **Confusion Matrix**: Per-class performance
- **SNR Sensitivity**: Performance vs noise level
- **Feature Importance**: Which features matter most

### Neural AMR
- **Validation Accuracy**: Generalization performance
- **Training Loss**: Convergence monitoring
- **Per-class Precision/Recall**: Balanced performance
- **Computational Efficiency**: Inference speed

### Intelligent Receiver
- **Signal Acquisition Rate**: How often it finds signals
- **Convergence Speed**: Time to optimal tuning
- **SNR Achievement**: Quality of acquired signals
- **Robustness**: Performance under interference

### Adversarial System
- **Win/Loss Ratios**: Receiver vs jammer success
- **SNR Degradation**: Jamming effectiveness
- **Strategy Adaptation**: Learning speed
- **Anti-jamming Success**: Overcoming interference

## Real-World Applications

### Military/Defense
- **Electronic Warfare**: Robust communications under jamming
- **Spectrum Surveillance**: Automatic signal identification
- **Cognitive Radio**: Adaptive spectrum usage

### Commercial
- **5G/6G Networks**: Intelligent interference mitigation
- **IoT Communications**: Adaptive low-power protocols
- **Satellite Communications**: Robust uplink/downlink

### Research
- **Spectrum Management**: Dynamic allocation algorithms
- **Protocol Development**: AI-assisted design
- **Security**: Anti-jamming and spoofing resistance

## Future Enhancements

### Planned Improvements
1. **Multi-agent Systems**: Cooperative receiver networks
2. **Transfer Learning**: Pre-trained models for new scenarios
3. **Federated Learning**: Distributed training across devices
4. **Explainable AI**: Understanding decision processes
5. **Real-time Optimization**: Hardware-accelerated inference

### Advanced Techniques
- **Graph Neural Networks**: For network topology awareness
- **Transformer Models**: For sequence modeling
- **Meta-learning**: Quick adaptation to new environments
- **Adversarial Training**: More sophisticated opponents

This AI/ML architecture represents a comprehensive approach to intelligent radio systems, combining classical signal processing wisdom with cutting-edge machine learning techniques to create robust, adaptive, and intelligent SDR systems.