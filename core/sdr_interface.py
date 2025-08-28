# core/sdr_interface.py
from geminisdr.core.error_handling import (
    ErrorHandler, GeminiSDRError, HardwareError, ConfigurationError,
    ErrorSeverity, ErrorContext, retry_with_backoff, fallback_to_simulation
)
from geminisdr.core.logging_manager import StructuredLogger
from geminisdr.config.config_manager import get_config_manager, SystemConfig
import numpy as np
import time
import threading
import queue
from collections import deque
import logging

try:
    import adi
    PLUTO_AVAILABLE = True
except ImportError:
    PLUTO_AVAILABLE = False
    print("Warning: pyadi-iio not available. Using simulation mode.")

class PlutoSDRInterface:
    """Robust PlutoSDR interface with real-time and batch modes, error handling, and configuration management."""
    
    def __init__(self, retry_attempts=3, config: SystemConfig = None):
        # Load configuration
        if config is None:
            try:
                config_manager = get_config_manager()
                self.config = config_manager.get_config()
                if self.config is None:
                    self.config = config_manager.load_config()
            except Exception as e:
                # Use fallback configuration
                from geminisdr.config.config_models import SystemConfig, HardwareConfig, LoggingConfig
                self.config = SystemConfig(
                    hardware=HardwareConfig(),
                    logging=LoggingConfig()
                )
        else:
            self.config = config
        
        # Initialize error handling and logging
        self.logger = StructuredLogger(__name__, self.config.logging)
        self.error_handler = ErrorHandler(self.logger.logger)
        self._register_recovery_strategies()
        
        self.sdr = None
        self.retry_attempts = retry_attempts
        self.is_streaming = False
        self.sample_queue = queue.Queue(maxsize=100)
        self.capture_thread = None
        self.simulation_mode = not PLUTO_AVAILABLE
        
        # Use configuration for simulation parameters
        self.sim_center_freq = 100e6
        self.sim_sample_rate = 2e6
        self.sim_gain = 30
        
        if self.simulation_mode:
            self.logger.logger.info("PlutoSDR not available, using simulation mode")
    
    def _register_recovery_strategies(self):
        """Register error recovery strategies."""
        # Hardware connection recovery - fallback to simulation
        def connection_fallback_strategy(error: Exception, context: ErrorContext) -> bool:
            try:
                self.logger.logger.warning(f"Hardware connection failed, switching to simulation: {error}")
                self.simulation_mode = True
                self.logger.logger.info("Successfully switched to simulation mode")
                return True
            except Exception as e:
                self.logger.logger.error(f"Connection fallback strategy failed: {e}")
                return False
        
        # Configuration error recovery - use defaults
        def config_recovery_strategy(error: Exception, context: ErrorContext) -> bool:
            try:
                self.logger.logger.warning(f"Configuration error, using defaults: {error}")
                # Reset to default simulation parameters
                self.sim_center_freq = 100e6
                self.sim_sample_rate = 2e6
                self.sim_gain = 30
                return True
            except Exception as e:
                self.logger.logger.error(f"Configuration recovery strategy failed: {e}")
                return False
        
        # Register strategies
        self.error_handler.register_recovery_strategy(HardwareError, connection_fallback_strategy)
        self.error_handler.register_recovery_strategy(ConfigurationError, config_recovery_strategy)
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(HardwareError,))
    def connect(self):
        """Connect to PlutoSDR with retry logic and error handling."""
        try:
            with self.error_handler.error_context("sdr_connection", component="PlutoSDRInterface"):
                if not PLUTO_AVAILABLE:
                    self.logger.logger.info("✓ Using simulation mode (PlutoSDR not available)")
                    self.simulation_mode = True
                    return True
                    
                for attempt in range(self.retry_attempts):
                    try:
                        # Close any existing connection first
                        if hasattr(self, 'sdr') and self.sdr is not None:
                            try:
                                del self.sdr
                            except:
                                pass
                        
                        self.sdr = adi.Pluto('ip:192.168.4.1')
                        self.logger.logger.info(f"✓ Connected to PlutoSDR on attempt {attempt + 1}")
                        
                        # Test connection
                        self.sdr.rx_lo = int(100e6)
                        _ = self.sdr.rx_lo
                        
                        self.simulation_mode = False
                        return True
                    
                    except Exception as e:
                        error_msg = str(e)
                        self.logger.logger.warning(f"✗ Connection attempt {attempt + 1} failed: {error_msg}")
                        
                        # If device is busy, suggest solution
                        if "Device or resource busy" in error_msg:
                            self.logger.logger.info("Device appears to be in use. Try closing other SDR applications or reconnecting device")
                        
                        if attempt < self.retry_attempts - 1:
                            time.sleep(3)  # Longer wait for device busy issues
                
                # All attempts failed, raise error to trigger recovery
                raise HardwareError(
                    "Failed to connect to PlutoSDR after all attempts",
                    device_type="PlutoSDR",
                    device_id="ip:192.168.4.1",
                    severity=ErrorSeverity.HIGH
                )
                
        except HardwareError:
            # Let the error handler try recovery strategies
            raise
        except Exception as e:
            raise HardwareError(
                f"Unexpected error during SDR connection: {str(e)}",
                device_type="PlutoSDR",
                severity=ErrorSeverity.HIGH,
                cause=e
            )
    
    def configure(self, center_freq, sample_rate, bandwidth=None, gain=30):
        """Configure SDR parameters with error handling."""
        try:
            with self.error_handler.error_context(
                "sdr_configuration", 
                component="PlutoSDRInterface",
                center_freq=center_freq,
                sample_rate=sample_rate,
                gain=gain
            ):
                if self.simulation_mode:
                    self.sim_center_freq = center_freq
                    self.sim_sample_rate = sample_rate
                    self.sim_gain = gain
                    self.logger.logger.info(f"✓ Simulated Config: {center_freq/1e6:.1f} MHz, "
                          f"{sample_rate/1e6:.1f} MSps, Gain: {gain} dB")
                    return True
                    
                if self.sdr is None:
                    raise HardwareError(
                        "SDR not connected",
                        device_type="PlutoSDR",
                        severity=ErrorSeverity.HIGH
                    )
                
                # Validate configuration parameters
                if not (70e6 <= center_freq <= 6e9):
                    raise ConfigurationError(
                        f"Center frequency {center_freq/1e6:.1f} MHz out of range (70-6000 MHz)",
                        config_key="center_freq",
                        severity=ErrorSeverity.MEDIUM
                    )
                
                if not (520833 <= sample_rate <= 61440000):
                    raise ConfigurationError(
                        f"Sample rate {sample_rate/1e6:.1f} MSps out of range (0.52-61.44 MSps)",
                        config_key="sample_rate",
                        severity=ErrorSeverity.MEDIUM
                    )
                
                self.sdr.rx_lo = int(center_freq)
                self.sdr.sample_rate = int(sample_rate)
                
                if bandwidth is None:
                    bandwidth = sample_rate * 0.8
                self.sdr.rx_rf_bandwidth = int(bandwidth)
                
                self.sdr.rx_hardwaregain_chan0 = gain
                self.sdr.rx_buffer_size = int(sample_rate * 0.001)  # 1ms buffer
                
                self.logger.logger.info(f"✓ Configured: {center_freq/1e6:.1f} MHz, "
                      f"{sample_rate/1e6:.1f} MSps, Gain: {gain} dB")
                
                return True
        
        except (HardwareError, ConfigurationError):
            # Let error handler try recovery strategies
            raise
        except Exception as e:
            raise HardwareError(
                f"Configuration failed: {str(e)}",
                device_type="PlutoSDR",
                severity=ErrorSeverity.HIGH,
                cause=e
            )
    
    def capture_batch(self, duration_seconds):
        """Capture samples in batch mode."""
        if self.simulation_mode:
            return self._simulate_capture(duration_seconds)
            
        if self.sdr is None:
            raise RuntimeError("SDR not connected")
        
        samples_needed = int(self.sdr.sample_rate * duration_seconds)
        samples_per_buffer = self.sdr.rx_buffer_size
        num_buffers = int(np.ceil(samples_needed / samples_per_buffer))
        
        all_samples = []
        print(f"Capturing {num_buffers} buffers...")
        
        for i in range(num_buffers):
            try:
                buffer = self.sdr.rx()
                all_samples.append(buffer)
                
                if (i + 1) % 10 == 0:
                    progress = (i + 1) / num_buffers * 100
                    print(f"Progress: {progress:.0f}%")
            
            except Exception as e:
                print(f"✗ Buffer capture failed: {e}")
                break
        
        # Combine and trim to exact length
        if all_samples:
            combined = np.concatenate(all_samples)[:samples_needed]
            print(f"✓ Captured {len(combined)} samples")
            return combined
        else:
            return None
    
    def _simulate_capture(self, duration_seconds):
        """Simulate signal capture for testing."""
        from core.signal_generator import SDRSignalGenerator
        
        samples_needed = int(self.sim_sample_rate * duration_seconds)
        
        # Generate a mix of signals and noise
        generator = SDRSignalGenerator(self.sim_sample_rate)
        
        # Simulate signal based on frequency tuning accuracy
        target_freq = 100e6  # Assume signal at 100 MHz
        freq_error = abs(self.sim_center_freq - target_freq)
        
        # SNR decreases with frequency error and increases with gain
        base_snr = 20 - (freq_error / 1e6)  # Lose 1 dB per MHz off
        snr_with_gain = base_snr + (self.sim_gain - 30) / 2
        actual_snr = np.clip(snr_with_gain + np.random.normal(0, 3), -10, 40)
        
        # Generate signal
        num_symbols = samples_needed // 8
        modulation = np.random.choice(['BPSK', 'QPSK', '8PSK', '16QAM'])
        
        signal = generator.generate_modulated_signal(
            modulation,
            num_symbols=num_symbols,
            snr_db=actual_snr
        )
        
        # Pad or trim to exact length
        if len(signal) < samples_needed:
            signal = np.pad(signal, (0, samples_needed - len(signal)), mode='constant')
        else:
            signal = signal[:samples_needed]
        
        print(f"✓ Simulated {len(signal)} samples (SNR: {actual_snr:.1f} dB)")
        return signal
    
    def start_streaming(self, callback_func=None):
        """Start real-time streaming mode."""
        if self.is_streaming:
            print("Already streaming")
            return
        
        self.is_streaming = True
        self.capture_thread = threading.Thread(
            target=self._streaming_worker,
            args=(callback_func,),
            daemon=True
        )
        self.capture_thread.start()
        print("✓ Started streaming mode")
    
    def stop_streaming(self):
        """Stop real-time streaming."""
        self.is_streaming = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        print("✓ Stopped streaming")
    
    def _streaming_worker(self, callback_func):
        """Worker thread for continuous streaming."""
        error_count = 0
        max_errors = 10
        
        while self.is_streaming and error_count < max_errors:
            try:
                if self.simulation_mode:
                    samples = self._simulate_capture(0.01)  # 10ms chunks
                else:
                    samples = self.sdr.rx()
                
                # Put in queue for processing
                if not self.sample_queue.full():
                    self.sample_queue.put(samples)
                
                # Call callback if provided
                if callback_func:
                    callback_func(samples)
                
                error_count = 0  # Reset on success
                time.sleep(0.01)  # Small delay for simulation
            
            except Exception as e:
                error_count += 1
                print(f"✗ Streaming error ({error_count}/{max_errors}): {e}")
                time.sleep(0.1)
        
        if error_count >= max_errors:
            print("✗ Too many errors, stopping stream")
            self.is_streaming = False
    
    def get_stream_samples(self, timeout=1.0):
        """Get samples from streaming queue."""
        try:
            return self.sample_queue.get(timeout=timeout)
        except queue.Empty:
            return None