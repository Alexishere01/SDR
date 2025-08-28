# ml/traditional_amr.py
from geminisdr.config.config_manager import get_config_manager, SystemConfig
from geminisdr.core.error_handling import (
    ErrorHandler, ModelError, ConfigurationError, ErrorSeverity, ErrorContext
)
from geminisdr.core.logging_manager import StructuredLogger
import numpy as np
from scipy import signal as sp_signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

class TraditionalAMR:
    """Traditional Automatic Modulation Recognition methods with configuration management."""
    
    def __init__(self, config: SystemConfig = None):
        # Load configuration
        if config is None:
            try:
                config_manager = get_config_manager()
                self.config = config_manager.get_config()
                if self.config is None:
                    self.config = config_manager.load_config()
            except Exception as e:
                # Use fallback configuration
                from geminisdr.config.config_models import SystemConfig, MLConfig, LoggingConfig
                self.config = SystemConfig(
                    ml=MLConfig(),
                    logging=LoggingConfig()
                )
        else:
            self.config = config
        
        # Initialize error handling and logging
        self.logger = StructuredLogger(__name__, self.config.logging)
        self.error_handler = ErrorHandler(self.logger.logger)
        
        # Initialize components with configuration
        self.scaler = StandardScaler()
        
        # Use configuration for classifier parameters
        n_estimators = getattr(self.config.ml, 'n_estimators', 100)
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        
        self.feature_names = [
            'mom2_amp', 'mom2_phase', 'mom3_amp', 'mom3_phase', 'mom4_amp', 'mom4_phase',
            'c20', 'c21', 'c40', 'c41', 'c42',
            'spec_centroid', 'spec_bandwidth', 'spec_rolloff', 'papr',
            'cyclo_0', 'cyclo_0.5', 'cyclo_1', 'cyclo_2',
            'std_amp', 'std_phase', 'std_freq', 'peak_avg_amp'
        ]
        
        self.logger.logger.info(f"Initialized TraditionalAMR with {n_estimators} estimators")
    
    def extract_features(self, signal):
        """Extract hand-crafted features from signal."""
        features = []
        
        # 1. Statistical moments
        features.extend(self._compute_moments(signal))
        
        # 2. Higher-order cumulants
        features.extend(self._compute_cumulants(signal))
        
        # 3. Spectral features
        features.extend(self._compute_spectral_features(signal))
        
        # 4. Cyclostationary features
        features.extend(self._compute_cyclostationary_features(signal))
        
        # 5. Instantaneous features
        features.extend(self._compute_instantaneous_features(signal))
        
        return np.array(features)
    
    def _compute_moments(self, signal):
        """Compute statistical moments of the signal."""
        # Normalize signal
        signal_norm = signal / (np.mean(np.abs(signal)) + 1e-12)
        
        # Compute moments up to 4th order
        moments = []
        for order in range(2, 5):
            # Moments of amplitude
            mom_amp = np.mean(np.abs(signal_norm)**order)
            moments.append(mom_amp)
            
            # Moments of phase
            phase = np.angle(signal_norm)
            mom_phase = np.mean(phase**order)
            moments.append(mom_phase)
        
        return moments
    
    def _compute_cumulants(self, signal):
        """Compute higher-order cumulants."""
        # Normalize
        signal_norm = signal / (np.std(signal) + 1e-12)
        
        # Second-order cumulants
        c20 = np.mean(signal_norm**2)
        c21 = np.mean(np.abs(signal_norm)**2)
        
        # Fourth-order cumulants
        c40 = np.mean(signal_norm**4) - 3 * c20**2
        c41 = np.mean(signal_norm**3 * np.conj(signal_norm)) - 3 * c20 * c21
        c42 = np.mean(np.abs(signal_norm)**4) - 2 * c21**2 - np.abs(c20)**2
        
        return [np.abs(c20), np.abs(c21), np.abs(c40), np.abs(c41), np.abs(c42)]
    
    def _compute_spectral_features(self, signal):
        """Compute frequency domain features."""
        # FFT
        fft = np.fft.fft(signal)
        fft_mag = np.abs(fft)
        fft_centered = np.fft.fftshift(fft_mag)
        
        # Spectral centroid
        freqs = np.fft.fftfreq(len(signal))
        freqs_shifted = np.fft.fftshift(freqs)
        centroid = np.sum(freqs_shifted * fft_centered) / (np.sum(fft_centered) + 1e-12)
        
        # Spectral bandwidth
        variance = np.sum((freqs_shifted - centroid)**2 * fft_centered) / (np.sum(fft_centered) + 1e-12)
        bandwidth = np.sqrt(variance)
        
        # Spectral rolloff
        cumsum = np.cumsum(fft_centered)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        if len(rolloff_idx) > 0:
            rolloff = freqs_shifted[rolloff_idx[0]]
        else:
            rolloff = 0
        
        # Peak to average power ratio
        peak_power = np.max(fft_mag**2)
        avg_power = np.mean(fft_mag**2)
        papr = peak_power / (avg_power + 1e-12)
        
        return [centroid, bandwidth, rolloff, papr]
    
    def _compute_cyclostationary_features(self, signal):
        """Compute cyclostationary features."""
        # Simplified cyclic spectral density at specific cycle frequencies
        features = []
        
        # Common cycle frequencies for digital modulations
        alpha_values = [0, 0.5, 1.0, 2.0]
        
        for alpha in alpha_values:
            # Compute cyclic autocorrelation
            if alpha == 0:
                # Standard autocorrelation
                R_alpha = np.correlate(signal, signal, mode='same')
            else:
                # Cyclic autocorrelation
                shifted = signal * np.exp(-1j * 2 * np.pi * alpha * np.arange(len(signal)))
                R_alpha = np.correlate(shifted, signal, mode='same')
            
            # Take magnitude at zero lag
            features.append(np.abs(R_alpha[len(R_alpha)//2]))
        
        return features
    
    def _compute_instantaneous_features(self, signal):
        """Compute instantaneous amplitude, phase, and frequency features."""
        # Instantaneous amplitude
        inst_amp = np.abs(signal)
        
        # Instantaneous phase
        inst_phase = np.unwrap(np.angle(signal))
        
        # Instantaneous frequency (derivative of phase)
        inst_freq = np.diff(inst_phase)
        
        # Features
        features = [
            np.std(inst_amp),                    # Amplitude variation
            np.std(inst_phase),                   # Phase variation
            np.std(inst_freq),                    # Frequency variation
            np.max(inst_amp) / (np.mean(inst_amp) + 1e-12) # Peak to average amplitude
        ]
        
        return features
    
    def train(self, signals, labels):
        """Train the traditional AMR classifier."""
        print("Extracting features from training data...")
        
        # Extract features for all signals
        features_list = []
        for i, signal in enumerate(signals):
            if i % 100 == 0:
                print(f"  Processing signal {i}/{len(signals)}")
            features = self.extract_features(signal)
            features_list.append(features)
        
        X = np.array(features_list)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train classifier
        print("Training Random Forest classifier...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_val)
        print("\nValidation Results:")
        print(classification_report(y_val, y_pred))
        
        # Feature importance
        importances = self.classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 10 Most Important Features:")
        for i in range(min(10, len(self.feature_names))):
            print(f"  {self.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return self.classifier.score(X_val, y_val)
    
    def predict(self, signal):
        """Predict modulation type of a signal."""
        features = self.extract_features(signal)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        return prediction, probabilities
    
    def save_model(self, filepath):
        """Save trained model and scaler."""
        joblib.dump({
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and scaler."""
        data = joblib.load(filepath)
        self.classifier = data['classifier']
        self.scaler = data['scaler']
        if 'feature_names' in data:
            self.feature_names = data['feature_names']
        print(f"Model loaded from {filepath}")