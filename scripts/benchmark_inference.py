#!/usr/bin/env python3
"""
Inference performance benchmark script.

This script benchmarks inference performance across different models,
batch sizes, and devices to detect performance regressions.
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class InferenceBenchmark:
    """Inference performance benchmark suite."""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize benchmark suite.
        
        Args:
            device: Device to run benchmarks on ('cpu', 'cuda', 'mps')
        """
        self.device = torch.device(device)
        self.results = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validate device
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        
        self.logger.info(f"Initialized inference benchmark on {self.device}")
    
    def create_neural_amr_model(self, input_size: int, num_classes: int) -> nn.Module:
        """Create a model similar to NeuralAMR for benchmarking."""
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        ).to(self.device)
    
    def create_intelligent_receiver_model(self, state_size: int, action_size: int) -> nn.Module:
        """Create a model similar to IntelligentReceiver for benchmarking."""
        return nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        ).to(self.device)
    
    def create_test_data(self, num_samples: int, input_size: int, batch_size: int) -> DataLoader:
        """Create test data for inference benchmarking."""
        X = torch.randn(num_samples, input_size)
        # Create dummy labels (not used in inference)
        y = torch.zeros(num_samples, dtype=torch.long)
        
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def benchmark_inference_speed(self, model: nn.Module, test_loader: DataLoader,
                                 warmup_batches: int = 10) -> Dict[str, float]:
        """Benchmark inference speed for a model."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader):
                if batch_idx >= warmup_batches:
                    break
                data = data.to(self.device)
                _ = model(data)
        
        # Synchronize for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Actual benchmark
        start_time = time.time()
        total_samples = 0
        total_batches = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                _ = model(data)
                
                total_samples += data.size(0)
                total_batches += 1
        
        # Synchronize for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'duration_seconds': duration,
            'total_samples': total_samples,
            'total_batches': total_batches,
            'samples_per_second': total_samples / duration,
            'batches_per_second': total_batches / duration,
            'avg_batch_time_ms': (duration / total_batches) * 1000 if total_batches > 0 else 0,
            'avg_sample_time_ms': (duration / total_samples) * 1000 if total_samples > 0 else 0
        }
    
    def benchmark_batch_sizes(self, model_factory, input_size: int, num_classes: int,
                             batch_sizes: List[int], num_samples: int = 1000) -> Dict[str, Dict[str, float]]:
        """Benchmark different batch sizes for inference."""
        results = {}
        
        for batch_size in batch_sizes:
            self.logger.info(f"Benchmarking inference batch size {batch_size}")
            
            try:
                # Create model and test data
                model = model_factory(input_size, num_classes)
                test_loader = self.create_test_data(num_samples, input_size, batch_size)
                
                # Benchmark
                metrics = self.benchmark_inference_speed(model, test_loader)
                metrics['batch_size'] = batch_size
                metrics['input_size'] = input_size
                metrics['num_classes'] = num_classes
                
                results[f'batch_size_{batch_size}'] = metrics
                
                # Clean up
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning(f"OOM for batch size {batch_size}: {e}")
                    results[f'batch_size_{batch_size}'] = {
                        'error': 'out_of_memory',
                        'batch_size': batch_size
                    }
                else:
                    raise e
        
        return results
    
    def benchmark_model_types(self, batch_sizes: List[int]) -> Dict[str, Dict[str, float]]:
        """Benchmark different model types."""
        results = {}
        
        model_configs = [
            ('neural_amr_small', 256, 4, self.create_neural_amr_model),
            ('neural_amr_medium', 512, 8, self.create_neural_amr_model),
            ('neural_amr_large', 1024, 16, self.create_neural_amr_model),
            ('intelligent_receiver_small', 128, 8, self.create_intelligent_receiver_model),
            ('intelligent_receiver_medium', 256, 16, self.create_intelligent_receiver_model),
            ('intelligent_receiver_large', 512, 32, self.create_intelligent_receiver_model)
        ]
        
        for model_name, input_size, output_size, model_factory in model_configs:
            self.logger.info(f"Benchmarking {model_name} model")
            
            model_results = {}
            
            for batch_size in batch_sizes:
                try:
                    model = model_factory(input_size, output_size)
                    test_loader = self.create_test_data(1000, input_size, batch_size)
                    
                    metrics = self.benchmark_inference_speed(model, test_loader)
                    metrics['batch_size'] = batch_size
                    metrics['model_name'] = model_name
                    metrics['input_size'] = input_size
                    metrics['output_size'] = output_size
                    metrics['num_parameters'] = sum(p.numel() for p in model.parameters())
                    
                    model_results[f'batch_{batch_size}'] = metrics
                    
                    # Clean up
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    self.logger.warning(f"Error benchmarking {model_name} with batch size {batch_size}: {e}")
                    model_results[f'batch_{batch_size}'] = {
                        'error': str(e),
                        'batch_size': batch_size,
                        'model_name': model_name
                    }
            
            results[model_name] = model_results
        
        return results
    
    def benchmark_precision_modes(self) -> Dict[str, Dict[str, float]]:
        """Benchmark different precision modes (fp32, fp16)."""
        results = {}
        
        if not torch.cuda.is_available():
            self.logger.info("Skipping precision benchmarks (CUDA not available)")
            return results
        
        input_size, num_classes = 512, 8
        batch_size = 32
        
        precision_modes = [
            ('fp32', torch.float32),
            ('fp16', torch.float16)
        ]
        
        for mode_name, dtype in precision_modes:
            self.logger.info(f"Benchmarking {mode_name} precision")
            
            try:
                model = self.create_neural_amr_model(input_size, num_classes)
                model = model.to(dtype=dtype)
                
                # Create test data with appropriate dtype
                X = torch.randn(1000, input_size, dtype=dtype)
                y = torch.zeros(1000, dtype=torch.long)
                dataset = TensorDataset(X, y)
                test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                metrics = self.benchmark_inference_speed(model, test_loader)
                metrics['precision'] = mode_name
                metrics['dtype'] = str(dtype)
                
                results[f'precision_{mode_name}'] = metrics
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"Error benchmarking {mode_name}: {e}")
                results[f'precision_{mode_name}'] = {
                    'error': str(e),
                    'precision': mode_name
                }
        
        return results
    
    def benchmark_sequence_lengths(self) -> Dict[str, Dict[str, float]]:
        """Benchmark different input sequence lengths."""
        results = {}
        
        batch_size = 16
        base_features = 64
        num_classes = 8
        
        sequence_lengths = [128, 256, 512, 1024, 2048]
        
        for seq_len in sequence_lengths:
            self.logger.info(f"Benchmarking sequence length {seq_len}")
            
            try:
                input_size = seq_len * base_features
                model = self.create_neural_amr_model(input_size, num_classes)
                test_loader = self.create_test_data(500, input_size, batch_size)
                
                metrics = self.benchmark_inference_speed(model, test_loader)
                metrics['sequence_length'] = seq_len
                metrics['base_features'] = base_features
                metrics['total_input_size'] = input_size
                
                results[f'seq_len_{seq_len}'] = metrics
                
                # Clean up
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning(f"OOM for sequence length {seq_len}: {e}")
                    results[f'seq_len_{seq_len}'] = {
                        'error': 'out_of_memory',
                        'sequence_length': seq_len
                    }
                else:
                    self.logger.error(f"Error benchmarking sequence length {seq_len}: {e}")
                    results[f'seq_len_{seq_len}'] = {
                        'error': str(e),
                        'sequence_length': seq_len
                    }
        
        return results
    
    def run_all_benchmarks(self, batch_sizes: List[int], models: List[str]) -> Dict[str, Any]:
        """Run all inference benchmarks."""
        self.logger.info("Starting comprehensive inference benchmarks")
        
        all_results = {
            'device': str(self.device),
            'timestamp': time.time(),
            'batch_sizes': batch_sizes,
            'models': models,
            'benchmarks': {}
        }
        
        # Benchmark batch sizes for simple model
        if 'neural_amr' in models:
            self.logger.info("Running neural AMR batch size benchmarks")
            batch_size_results = self.benchmark_batch_sizes(
                self.create_neural_amr_model, 512, 8, batch_sizes
            )
            all_results['benchmarks']['neural_amr_batch_sizes'] = batch_size_results
        
        # Benchmark different model types
        self.logger.info("Running model type benchmarks")
        model_type_results = self.benchmark_model_types(batch_sizes)
        all_results['benchmarks']['model_types'] = model_type_results
        
        # Benchmark precision modes (CUDA only)
        if torch.cuda.is_available():
            self.logger.info("Running precision mode benchmarks")
            precision_results = self.benchmark_precision_modes()
            all_results['benchmarks']['precision_modes'] = precision_results
        
        # Benchmark sequence lengths
        self.logger.info("Running sequence length benchmarks")
        sequence_results = self.benchmark_sequence_lengths()
        all_results['benchmarks']['sequence_lengths'] = sequence_results
        
        # Add system info
        all_results['system_info'] = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 1,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
        
        self.logger.info("Inference benchmarks completed")
        return all_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Inference performance benchmarks")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                       help="Device to run benchmarks on")
    parser.add_argument("--batch-sizes", default="1,8,16,32",
                       help="Comma-separated list of batch sizes to test")
    parser.add_argument("--models", default="neural_amr,intelligent_receiver",
                       help="Comma-separated list of models to test")
    parser.add_argument("--output", required=True,
                       help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse arguments
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
    models = [x.strip() for x in args.models.split(',')]
    
    try:
        # Run benchmarks
        benchmark = InferenceBenchmark(device=args.device)
        results = benchmark.run_all_benchmarks(batch_sizes, models)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Inference benchmarks completed successfully")
        print(f"Results saved to {args.output}")
        
        # Print summary
        print("\nSummary:")
        for category, benchmarks in results['benchmarks'].items():
            print(f"  {category}: {len(benchmarks)} benchmarks")
        
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()