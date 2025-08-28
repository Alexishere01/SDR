#!/usr/bin/env python3
# test_system.py - System integration tests
import torch
import numpy as np
import os
import sys
import time
import unittest
from geminisdr.core.signal_generator import SDRSignalGenerator, SDRDatasetManager
from core.sdr_interface import PlutoSDRInterface
from geminisdr.core.visualizer import StaticVisualizer
from ml.traditional_amr import TraditionalAMR
from ml.neural_amr import NeuralAMR
from ml.intelligent_receiver import IntelligentReceiverML

class TestSignalGeneration(unittest.TestCase):
    """Test signal generation functionality."""
    
    def setUp(self):
        self.generator = SDRSignalGenerator()
    
    def test_bpsk_generation(self):
        """Test BPSK signal generation."""
        signal = self.generator.generate_modulated_signal('BPSK', num_symbols=100, snr_db=20)
        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(signal.dtype, complex)
        self.assertGreater(len(signal), 0)
    
    def test_qpsk_generation(self):
        """Test QPSK signal generation."""
        signal = self.generator.generate_modulated_signal('QPSK', num_symbols=100, snr_db=20)
        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(signal.dtype, complex)
        self.assertGreater(len(signal), 0)
    
    def test_qam_generation(self):
        """Test QAM signal generation."""
        signal16 = self.generator.generate_modulated_signal('16QAM', num_symbols=100, snr_db=20)
        signal64 = self.generator.generate_modulated_signal('64QAM', num_symbols=100, snr_db=20)
        
        self.assertIsInstance(signal16, np.ndarray)
        self.assertIsInstance(signal64, np.ndarray)
        self.assertEqual(signal16.dtype, complex)
        self.assertEqual(signal64.dtype, complex)
    
    def test_snr_effects(self):
        """Test SNR effects on signals."""
        high_snr = self.generator.generate_modulated_signal('QPSK', num_symbols=100, snr_db=30)
        low_snr = self.generator.generate_modulated_signal('QPSK', num_symbols=100, snr_db=0)
        
        # High SNR should have less noise variance
        high_power = np.var(np.abs(high_snr))
        low_power = np.var(np.abs(low_snr))
        
        # This is a rough test - low SNR should have more variation
        self.assertGreater(low_power, high_power * 0.5)

class TestSDRInterface(unittest.TestCase):
    """Test SDR interface functionality."""
    
    def setUp(self):
        self.sdr = PlutoSDRInterface()
    
    def test_connection(self):
        """Test SDR connection (will use simulation if no hardware)."""
        result = self.sdr.connect()
        self.assertTrue(result)
    
    def test_configuration(self):
        """Test SDR configuration."""
        self.sdr.connect()
        result = self.sdr.configure(
            center_freq=100e6,
            sample_rate=2e6,
            gain=30
        )
        self.assertTrue(result)
    
    def test_batch_capture(self):
        """Test batch sample capture."""
        self.sdr.connect()
        self.sdr.configure(center_freq=100e6, sample_rate=2e6, gain=30)
        
        samples = self.sdr.capture_batch(0.01)  # 10ms
        
        self.assertIsNotNone(samples)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.dtype, complex)
        self.assertGreater(len(samples), 0)

class TestDatasetManager(unittest.TestCase):
    """Test dataset management functionality."""
    
    def setUp(self):
        self.dm = SDRDatasetManager(base_path='./test_datasets')
    
    def tearDown(self):
        # Clean up test datasets
        import shutil
        if os.path.exists('./test_datasets'):
            shutil.rmtree('./test_datasets')
    
    def test_dataset_generation(self):
        """Test dataset generation and loading."""
        # Generate small test dataset
        filename = self.dm.generate_dataset(
            num_samples_per_class=10,
            snr_range=(10, 20),
            signal_length=512,
            modulations=['BPSK', 'QPSK']
        )
        
        self.assertTrue(os.path.exists(filename))
        
        # Load and verify
        signals, labels, snrs = self.dm.load_dataset(filename)
        
        self.assertIsInstance(signals, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertIsInstance(snrs, np.ndarray)
        
        self.assertEqual(len(signals), len(labels))
        self.assertEqual(len(signals), len(snrs))
        
        # Check modulations
        unique_labels = set(labels)
        self.assertEqual(unique_labels, {'BPSK', 'QPSK'})

class TestTraditionalAMR(unittest.TestCase):
    """Test traditional AMR functionality."""
    
    def setUp(self):
        self.amr = TraditionalAMR()
        self.generator = SDRSignalGenerator()
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        signal = self.generator.generate_modulated_signal('QPSK', num_symbols=100, snr_db=20)
        features = self.amr.extract_features(signal)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(self.amr.feature_names))
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_training_and_prediction(self):
        """Test training and prediction."""
        # Generate small training set
        signals = []
        labels = []
        
        for mod in ['BPSK', 'QPSK']:
            for _ in range(20):
                signal = self.generator.generate_modulated_signal(mod, num_symbols=100, snr_db=20)
                signals.append(signal)
                labels.append(mod)
        
        # Train
        accuracy = self.amr.train(signals, labels)
        self.assertGreater(accuracy, 0.5)  # Should be better than random
        
        # Test prediction
        test_signal = self.generator.generate_modulated_signal('BPSK', num_symbols=100, snr_db=20)
        prediction, probabilities = self.amr.predict(test_signal)
        
        self.assertIn(prediction, ['BPSK', 'QPSK'])
        self.assertEqual(len(probabilities), 2)
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=5)

class TestNeuralAMR(unittest.TestCase):
    """Test neural network AMR functionality."""
    
    def setUp(self):
        self.amr = NeuralAMR()
        self.generator = SDRSignalGenerator()
    
    def test_dataset_preparation(self):
        """Test dataset preparation."""
        # Generate small dataset
        signals = []
        labels = []
        
        for mod in ['BPSK', 'QPSK']:
            for _ in range(20):
                signal = self.generator.generate_modulated_signal(mod, num_symbols=128, snr_db=20)
                signals.append(signal)
                labels.append(mod)
        
        train_loader, val_loader, num_classes = self.amr.prepare_data(signals, labels, batch_size=8)
        
        self.assertEqual(num_classes, 2)
        self.assertIsNotNone(self.amr.label_map)
        self.assertIsNotNone(self.amr.inverse_label_map)
    
    def test_model_creation(self):
        """Test model creation."""
        from ml.neural_amr import CNNModulationClassifier
        
        model = CNNModulationClassifier(num_classes=5, signal_length=1024)
        
        # Test forward pass
        dummy_input = torch.randn(1, 2, 1024)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 5))

class TestIntelligentReceiver(unittest.TestCase):
    """Test intelligent receiver functionality."""
    
    def setUp(self):
        # Create dummy SDR for testing
        class DummySDR:
            def configure(self, **kwargs):
                return True
            def capture_batch(self, duration):
                return np.random.randn(1000) + 1j * np.random.randn(1000)
        
        self.dummy_sdr = DummySDR()
        self.receiver = IntelligentReceiverML(self.dummy_sdr)
    
    def test_environment_creation(self):
        """Test simulated environment creation."""
        from geminisdr.ml.intelligent_receiver import SimulatedSDREnvironment
        
        env = SimulatedSDREnvironment()
        
        # Test reset
        observation, info = env.reset()
        self.assertEqual(len(observation), 256)
        
        # Test step
        action = np.array([1000, 1, 0.1])  # Small adjustments
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(len(next_obs), 256)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
    
    def test_action_selection(self):
        """Test action selection."""
        # Create dummy state
        state = np.random.randn(256).astype(np.float32)
        
        # Test random action (high epsilon)
        self.receiver.epsilon = 1.0
        action = self.receiver._choose_action(state)
        
        self.assertEqual(len(action), 3)
        self.assertTrue(np.all(np.isfinite(action)))

class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from signal generation to classification."""
        # Generate signal
        generator = SDRSignalGenerator()
        signal = generator.generate_modulated_signal('QPSK', num_symbols=128, snr_db=15)
        
        # Test traditional AMR
        trad_amr = TraditionalAMR()
        
        # Quick training with minimal data
        signals = [signal] * 10
        labels = ['QPSK'] * 10
        
        # Add some BPSK for binary classification
        for _ in range(10):
            bpsk_signal = generator.generate_modulated_signal('BPSK', num_symbols=128, snr_db=15)
            signals.append(bpsk_signal)
            labels.append('BPSK')
        
        accuracy = trad_amr.train(signals, labels)
        self.assertGreater(accuracy, 0.4)  # Should be better than random
        
        # Test prediction
        test_signal = generator.generate_modulated_signal('QPSK', num_symbols=128, snr_db=15)
        prediction, probabilities = trad_amr.predict(test_signal)
        
        self.assertIn(prediction, ['BPSK', 'QPSK'])
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Create and train a simple model
        generator = SDRSignalGenerator()
        trad_amr = TraditionalAMR()
        
        # Minimal training data
        signals = []
        labels = []
        for mod in ['BPSK', 'QPSK']:
            for _ in range(15):
                signal = generator.generate_modulated_signal(mod, num_symbols=100, snr_db=20)
                signals.append(signal)
                labels.append(mod)
        
        trad_amr.train(signals, labels)
        
        # Save model
        test_model_path = 'test_model.pkl'
        trad_amr.save_model(test_model_path)
        self.assertTrue(os.path.exists(test_model_path))
        
        # Load model
        new_amr = TraditionalAMR()
        new_amr.load_model(test_model_path)
        
        # Test prediction with loaded model
        test_signal = generator.generate_modulated_signal('BPSK', num_symbols=100, snr_db=20)
        prediction, probabilities = new_amr.predict(test_signal)
        
        self.assertIn(prediction, ['BPSK', 'QPSK'])
        
        # Clean up
        os.remove(test_model_path)

def run_performance_tests():
    """Run performance benchmarks."""
    print("\n" + "="*60)
    print("PERFORMANCE TESTS")
    print("="*60)
    
    generator = SDRSignalGenerator()
    
    # Test signal generation speed
    print("\n1. Signal Generation Speed:")
    start_time = time.time()
    for _ in range(100):
        signal = generator.generate_modulated_signal('QPSK', num_symbols=100, snr_db=20)
    gen_time = time.time() - start_time
    print(f"   Generated 100 signals in {gen_time:.2f}s ({100/gen_time:.1f} signals/sec)")
    
    # Test feature extraction speed
    print("\n2. Feature Extraction Speed:")
    amr = TraditionalAMR()
    test_signal = generator.generate_modulated_signal('QPSK', num_symbols=100, snr_db=20)
    
    start_time = time.time()
    for _ in range(100):
        features = amr.extract_features(test_signal)
    feat_time = time.time() - start_time
    print(f"   Extracted features 100 times in {feat_time:.2f}s ({100/feat_time:.1f} extractions/sec)")
    
    # Test memory usage
    print("\n3. Memory Usage:")
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"   Current memory usage: {memory_mb:.1f} MB")

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SDR AI System Tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not any([args.unit, args.performance, args.all]):
        args.all = True
    
    print("="*60)
    print("SDR AI SYSTEM TESTS")
    print("="*60)
    
    if args.all or args.unit:
        print("\nRunning unit tests...")
        
        # Configure test verbosity
        verbosity = 2 if args.verbose else 1
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test classes
        test_classes = [
            TestSignalGeneration,
            TestSDRInterface,
            TestDatasetManager,
            TestTraditionalAMR,
            TestNeuralAMR,
            TestIntelligentReceiver,
            TestSystemIntegration
        ]
        
        for test_class in test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        # Print summary
        print(f"\nUnit Test Summary:")
        print(f"  Tests run: {result.testsRun}")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        
        if result.failures:
            print(f"\nFailures:")
            for test, traceback in result.failures:
                print(f"  {test}: {traceback}")
        
        if result.errors:
            print(f"\nErrors:")
            for test, traceback in result.errors:
                print(f"  {test}: {traceback}")
    
    if args.all or args.performance:
        run_performance_tests()
    
    print(f"\n{'='*60}")
    print("TESTS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()