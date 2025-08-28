# SDR AI System: Complete Technical Explanation

## High-Level Overview

This system creates **intelligent software-defined radios** that can automatically find, identify, and optimize radio signals using artificial intelligence. Think of it as giving a radio "eyes and a brain" - it can see the radio spectrum, understand what signals are present, and automatically tune itself for optimal reception.

### What Makes This Special?

1. **Automatic Signal Finding**: Instead of manually tuning frequencies, the AI searches and finds signals automatically
2. **Intelligent Modulation Recognition**: Automatically identifies what type of signal it's receiving (BPSK, QPSK, etc.)
3. **Adversarial Robustness**: Can fight against intelligent jamming and interference
4. **Real-time Adaptation**: Continuously learns and adapts to changing conditions

---

## Level 1: System Components (High Level)

### Core Architecture

```
Radio Waves → SDR Hardware → Signal Processing → AI/ML → Intelligent Decisions
```

**Four Main Pillars**:

1. **Signal Generation & Interface**: Creates test signals and talks to radio hardware
2. **Traditional Analysis**: Classical signal processing and feature extraction
3. **Neural Networks**: Deep learning for pattern recognition
4. **Intelligent Agents**: AI that makes decisions and learns from experience

### Key Capabilities

- **Automatic Modulation Recognition (AMR)**: "What type of signal is this?"
- **Intelligent Signal Search**: "Where are the interesting signals?"
- **Anti-jamming**: "How do I maintain communication when someone tries to block me?"
- **Real-time Adaptation**: "How do I get better over time?"

---

## Level 2: Technical Deep Dive

### 1. Signal Generation System (`core/signal_generator.py`)

**Purpose**: Create realistic training data and test scenarios

**How It Works**:
```python
# Generate a QPSK signal with noise
signal = generator.generate_modulated_signal('QPSK', num_symbols=256, snr_db=15)
```

**Key Features**:
- **Modulation Types**: BPSK, QPSK, 8PSK, 16QAM, 64QAM
- **Channel Impairments**: Adds realistic noise, frequency offsets, phase errors
- **Pulse Shaping**: Makes signals look like real-world transmissions

**Why These Parameters?**:
- **256 symbols**: Good balance between training speed and signal diversity
- **SNR range (0-30 dB)**: Covers poor to excellent signal conditions
- **Sample rate (2 MHz)**: Typical for many digital communications

### 2. SDR Interface (`core/sdr_interface.py`)

**Purpose**: Bridge between software and radio hardware

**Dual Mode Operation**:
```python
# Try real hardware first, fall back to simulation
if PLUTO_AVAILABLE:
    sdr = adi.Pluto('ip:192.168.4.1')  # Real PlutoSDR
else:
    # Use mathematical simulation
    signal = simulate_received_signal(freq, gain, snr)
```

**Smart Features**:
- **Automatic Retry**: Handles connection failures gracefully
- **Simulation Fallback**: Works without hardware for development
- **Real-time Streaming**: Continuous data capture
- **Error Recovery**: Robust against hardware glitches

### 3. Traditional AMR (`ml/traditional_amr.py`)

**Purpose**: Classical approach to modulation recognition

**Feature Extraction Pipeline**:
```
Raw I/Q Signal → Feature Extraction → Random Forest → Classification
```

**Features Computed**:

1. **Statistical Moments** (6 features):
   ```python
   # 2nd, 3rd, 4th order moments of amplitude and phase
   mom_amp = np.mean(np.abs(signal_norm)**order)
   mom_phase = np.mean(phase**order)
   ```

2. **Higher-Order Cumulants** (5 features):
   ```python
   # C20, C21, C40, C41, C42 - modulation-specific signatures
   c20 = np.mean(signal_norm**2)
   c40 = np.mean(signal_norm**4) - 3 * c20**2
   ```

3. **Spectral Features** (4 features):
   ```python
   # Centroid, bandwidth, rolloff, peak-to-average power ratio
   centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
   ```

4. **Cyclostationary Features** (4 features):
   ```python
   # Exploit periodic properties of digital modulations
   for alpha in [0, 0.5, 1.0, 2.0]:  # Cycle frequencies
       R_alpha = compute_cyclic_autocorrelation(signal, alpha)
   ```

5. **Instantaneous Features** (4 features):
   ```python
   # Amplitude, phase, frequency variations
   inst_amp = np.abs(signal)
   inst_phase = np.unwrap(np.angle(signal))
   inst_freq = np.diff(inst_phase)
   ```

**Why Random Forest?**:
- **Robust**: Handles noisy features well
- **Interpretable**: Can see which features matter most
- **Fast**: Quick training and inference
- **No Overfitting**: Ensemble method reduces variance

### 4. Neural AMR (`ml/neural_amr.py`)

**Purpose**: Deep learning approach to modulation recognition

**CNN Architecture**:
```python
Input: I/Q samples (2 channels × 1024 samples)
    ↓
Conv1D(64 filters, kernel=7) + BatchNorm + ReLU + MaxPool
    ↓
Conv1D(128 filters, kernel=5) + BatchNorm + ReLU + MaxPool
    ↓
Conv1D(256 filters, kernel=3) + BatchNorm + ReLU + MaxPool
    ↓
Flatten → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.5)
    ↓
Dense(num_classes) → Softmax → Classification
```

**Why This Architecture?**:
- **Conv1D**: Captures temporal patterns in I/Q data
- **Increasing Filters**: Learn hierarchical features (simple → complex)
- **BatchNorm**: Stabilizes training, allows higher learning rates
- **Dropout**: Prevents overfitting to training data
- **Progressive Kernel Sizes**: 7→5→3 captures different time scales

**Training Process**:
```python
# Data preparation
train_loader, val_loader = prepare_data(signals, labels)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### 5. Intelligent Receiver (`ml/intelligent_receiver.py`)

**Purpose**: AI agent that automatically finds and optimizes signals

**Deep Q-Learning Architecture**:
```python
State (256D) → DQN → Q-values → Action Selection → SDR Control
```

**State Representation**:
- **Spectrum (250 values)**: FFT of received signal
- **Current Settings (6 values)**: Frequency, gain, bandwidth, SNR, power, noise

**Action Space**:
- **Frequency Adjustment**: -1MHz to +1MHz (21 discrete steps)
- **Gain Adjustment**: -10dB to +10dB (21 discrete steps)
- **Bandwidth Factor**: -0.3 to +0.3 (11 discrete steps)

**DQN Network**:
```python
class DeepQLearningReceiver(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(256, 512)  # Input layer
        self.fc2 = nn.Linear(512, 512)  # Hidden layer
        self.fc3 = nn.Linear(512, 256)  # Feature layer
        
        # Dueling architecture
        self.value_head = nn.Linear(256, 1)           # State value
        self.advantage_head = nn.Linear(256, 3*11)    # Action advantages
```

**Why Dueling DQN?**:
- **Value Stream**: "How good is this state?"
- **Advantage Stream**: "How much better is each action?"
- **Combined**: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
- **Benefit**: Better learning when many actions have similar values

**Training Algorithm**:
```python
# Experience replay
memory = deque(maxlen=10000)

# Training step
batch = random.sample(memory, batch_size)
current_q = q_network(states)
next_q = target_network(next_states)
target_q = rewards + gamma * next_q * (1 - dones)
loss = MSE(current_q, target_q)
```

**Reward Function**:
```python
def calculate_reward(snr, freq_error):
    reward = snr  # Base reward on signal quality
    
    # Bonus for frequency accuracy
    if freq_error < 0.1e6:  # Within 100 kHz
        reward += 30
    elif freq_error < 0.5e6:  # Within 500 kHz
        reward += 10
    
    # Penalty for being far off
    if freq_error > 2e6:  # More than 2 MHz off
        reward -= 20
    
    return reward
```

---

## Level 3: Adversarial AI System (Advanced)

### The Problem: Intelligent Jamming

Traditional jammers are "dumb" - they just blast noise. But what if the jammer is also intelligent and learns your patterns?

### Smart Jammer Architecture (`ml/adversarial_jamming.py`)

**Multi-Strategy Jammer**:

1. **Narrowband Jamming**:
   ```python
   # Concentrate power at receiver frequency
   jamming = jam_power * np.exp(1j * 2 * np.pi * rx_freq * t)
   ```

2. **Wideband Jamming**:
   ```python
   # Spread noise across wide bandwidth
   noise = jam_power * (np.random.randn(N) + 1j * np.random.randn(N))
   ```

3. **Sweep Jamming**:
   ```python
   # Frequency sweep around target
   sweep_freq = rx_freq + sweep_range * np.sin(2 * np.pi * sweep_rate * t)
   jamming = jam_power * np.exp(1j * 2 * np.pi * sweep_freq * t)
   ```

4. **Pulse Jamming**:
   ```python
   # High-power intermittent bursts
   for i in range(0, len(signal), pulse_period):
       signal[i:i+pulse_width] *= (1 + jam_power * 10)
   ```

5. **Adaptive Jamming**:
   ```python
   # Learn receiver patterns and predict next frequency
   freq_changes = np.diff(frequency_history[-10:])
   predicted_freq = current_freq + np.mean(freq_changes)
   # Jam both current and predicted locations
   ```

6. **Deceptive Jamming**:
   ```python
   # Generate fake signals that look real
   fake_signal = generate_modulated_signal('QPSK', snr_db=15)
   fake_signal *= np.exp(1j * 2 * np.pi * fake_freq * t)
   ```

**Learning Algorithm (Multi-Armed Bandit)**:
```python
# Track success rate of each strategy
success_rates = {'narrowband': 0.3, 'wideband': 0.7, ...}

# Upper Confidence Bound selection
for strategy in strategies:
    confidence = sqrt(2 * log(total_usage) / strategy_usage)
    ucb_score = success_rate + confidence

# Select strategy with highest UCB score
best_strategy = max(strategies, key=lambda s: ucb_scores[s])
```

### Adversarial Training Environment

**Game Theory Setup**:
- **Receiver Goal**: Maximize SNR and signal acquisition
- **Jammer Goal**: Minimize receiver performance
- **Nash Equilibrium**: Both agents reach optimal strategies

**Co-evolutionary Training**:
```python
for episode in range(num_episodes):
    # Receiver takes action
    rx_action = receiver.choose_action(state)
    
    # Jammer observes and responds
    jam_strategy = jammer.select_strategy(rx_freq, rx_gain, rx_snr)
    
    # Environment applies both
    jammed_signal = jammer.apply_jamming(clean_signal, jam_strategy)
    final_snr = calculate_snr(jammed_signal)
    
    # Both agents learn
    receiver.learn(state, rx_action, reward, next_state)
    jammer.update_effectiveness(jam_strategy, jamming_success)
```

**Progressive Difficulty**:
1. **Random Jammer**: Baseline performance
2. **Adaptive Jammer**: Learns patterns
3. **Intelligent Jammer**: Predictive strategies

### Anti-Jamming Techniques

**Enhanced Receiver Training**:
```python
def choose_anti_jam_action(self, state, env):
    # Detect jamming presence
    jamming_detected = np.any(state[256:259] > 0.5)
    
    if jamming_detected:
        # Bias toward aggressive frequency changes (frequency hopping)
        freq_idx = np.random.choice(extreme_freq_indices)
    else:
        # Normal operation
        action = self.q_network(state).argmax()
```

**Anti-Jamming Reward**:
```python
def calculate_anti_jam_reward(self, base_reward, info):
    enhanced_reward = base_reward
    
    # Big bonus for maintaining SNR under jamming
    if info['jamming_effectiveness'] > 5:  # Significant jamming
        if info['snr'] > 10:  # But still good SNR
            enhanced_reward += 20  # Anti-jam success bonus
    
    return enhanced_reward
```

---

## Level 4: Implementation Details & Parameter Choices

### Why These Specific Parameters?

**DQN Hyperparameters**:
- **Learning Rate (1e-4)**: Conservative to prevent instability
- **Epsilon Decay (0.995)**: Gradual shift from exploration to exploitation
- **Memory Size (10k-20k)**: Balance between diversity and memory usage
- **Batch Size (32-64)**: Compromise between gradient noise and efficiency

**Network Architecture**:
- **Hidden Size (512)**: Sufficient capacity without overfitting
- **State Size (256)**: Rich enough for complex patterns, manageable computation
- **Action Discretization (11-21 steps)**: Fine enough for precision, coarse enough for learning

**Training Schedule**:
- **Target Network Update (10-20 episodes)**: Balance stability vs adaptation
- **Episodes (300-1500)**: Enough for convergence without excessive training time

### Feature Engineering Rationale

**Traditional AMR Features**:
- **Moments**: Capture basic statistical properties
- **Cumulants**: Exploit modulation-specific signatures
- **Spectral**: Frequency domain characteristics
- **Cyclostationary**: Periodic properties unique to digital modulations

**Neural Network Input**:
- **I/Q Samples**: Preserve all signal information
- **Sequence Length (1024)**: Good compromise between context and computation
- **Two Channels**: Real and imaginary parts of complex signal

### Training Methodology

**Curriculum Learning**:
```python
difficulty_levels = [
    {'power': 20, 'name': 'Weak Jammer'},
    {'power': 40, 'name': 'Moderate Jammer'},
    {'power': 60, 'name': 'Strong Jammer'},
    {'power': 80, 'name': 'Powerful Jammer'},
    {'power': 100, 'name': 'Elite Jammer'}
]
```

**Benefits**:
- **Gradual Complexity**: Prevents overwhelming the agent
- **Skill Transfer**: Each level builds on previous capabilities
- **Robust Performance**: Handles wide range of scenarios

---

## Level 5: How Everything Works Together

### Complete System Flow

1. **Signal Generation**:
   ```python
   # Create training data
   signals, labels = generate_dataset(modulations=['BPSK', 'QPSK', '8PSK'])
   ```

2. **Traditional AMR Training**:
   ```python
   # Extract features and train classifier
   features = [extract_features(sig) for sig in signals]
   classifier.fit(features, labels)
   ```

3. **Neural AMR Training**:
   ```python
   # End-to-end learning
   model.train(signals, labels, epochs=50)
   ```

4. **Intelligent Receiver Training**:
   ```python
   # Reinforcement learning in simulation
   for episode in range(500):
       state = env.reset()
       while not done:
           action = agent.choose_action(state)
           next_state, reward, done = env.step(action)
           agent.learn(state, action, reward, next_state)
   ```

5. **Adversarial Training**:
   ```python
   # Co-evolutionary training
   for level in difficulty_levels:
       env = AdversarialEnvironment(jammer_power=level['power'])
       agent.train_against_jammer(env, episodes=300)
   ```

6. **Real-World Deployment**:
   ```python
   # Use trained models with real SDR
   sdr.connect()
   while True:
       samples = sdr.capture_batch()
       modulation = neural_amr.predict(samples)
       optimal_settings = intelligent_rx.optimize(samples)
       sdr.configure(**optimal_settings)
   ```

### Demo System (`demo_adversarial.py`)

**Four Main Demonstrations**:

1. **Jamming Strategies**: Show how different jamming techniques work
2. **AI vs AI Battle**: Receiver vs jammer in real-time combat
3. **Receiver Comparison**: Regular vs adversarial-trained performance
4. **Escalating Jamming**: Performance against increasing jammer power

### Training Scripts

**`train_models.py`**: Complete training pipeline
- Traditional AMR → Neural AMR → Intelligent Receiver
- Automatic dataset generation
- Model comparison and evaluation

**`train_adversarial.py`**: Adversarial training
- Progressive difficulty levels
- Battle statistics tracking
- Anti-jamming skill development

---

## What's Left to Be Done

### Immediate Improvements

1. **Real Hardware Integration**:
   - Better PlutoSDR error handling
   - Support for other SDR platforms (USRP, RTL-SDR)
   - Hardware-in-the-loop testing

2. **Performance Optimization**:
   - GPU acceleration for real-time inference
   - Model quantization for embedded deployment
   - Parallel processing for multiple channels

3. **Advanced Features**:
   - Multi-signal detection and tracking
   - Cooperative receiver networks
   - Spectrum database integration

### Research Directions

1. **Advanced AI Techniques**:
   - Transformer models for sequence processing
   - Graph neural networks for network topology
   - Meta-learning for quick adaptation
   - Federated learning for distributed training

2. **Enhanced Adversarial Systems**:
   - Multi-agent jammer coordination
   - Sophisticated deception techniques
   - Game-theoretic analysis
   - Evolutionary strategies

3. **Real-World Applications**:
   - 5G/6G integration
   - IoT protocol optimization
   - Satellite communication robustness
   - Military/defense applications

### System Extensions

1. **Protocol Stack Integration**:
   - MAC layer adaptation
   - Network layer optimization
   - Cross-layer design

2. **Security Enhancements**:
   - Cryptographic integration
   - Authentication mechanisms
   - Anti-spoofing techniques

3. **Cognitive Radio Features**:
   - Dynamic spectrum access
   - Interference avoidance
   - Regulatory compliance

This system represents a comprehensive approach to intelligent radio communications, combining classical signal processing with cutting-edge AI to create robust, adaptive, and intelligent SDR systems that can operate effectively in challenging and adversarial environments.