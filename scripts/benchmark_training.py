#!/usr/bin/env python3
"""
Training performance benchmark script.

This script benchmarks training performance across different models,
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
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geminisdr.config.config_manager import ConfigManager
from tests.conftest import generate_test_signals


class TrainingBenchmark:
    """Training performance benchmark suite."""
    
    def __init__(self, device: str = 'cpu', duration_minutes: int = 30):
        """
        Initialize benchmark suite.
        
        Args:
            device: Device to run benchmarks on ('cpu', 'cuda', 'mps')
            duration_minutes: Maximum duration for benchmarks
        """
        self.device = torch.device(device)
        self.duration_seconds = duration_minutes * 60
        self.results = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validate device
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        
        self.logger.info(f"Initialized training benchmark on {self.device}")
    
    def create_simple_model(self, input_size: int, num_classes: int) -> nn.Module:
        """Create a simple CNN model for benchmarking."""
        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        ).to(self.device)
    
    def create_complex_model(self, input_size: int, num_classes: int) -> nn.Module:
        """Create a more complex model for benchmarking."""
        return nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        ).to(self.device)
    
    def create_test_dataset(self, num_samples: int, input_size: int, num_classes: int) -> DataLoader:
        """Create synthetic test dataset."""
        # Generate random data
        X = torch.randn(num_samples, input_size)
        y = torch.randint(0, num_classes, (num_samples,))
        
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=32, shuffle=True)
    
    def benchmark_training_speed(self, model: nn.Module, train_loader: DataLoader, 
                                num_epochs: int = 10) -> Dict[str, float]:
        """Benchmark training speed for a model."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Warmup
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 5:  # 5 warmup batches
                break
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Actual benchmark
        start_time = time.time()
        total_samples = 0
        total_batches = 0
        total_loss = 0.0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_samples += data.size(0)
                total_batches += 1
                total_loss += loss.item()
                
                # Check time limit
                if time.time() - start_time > self.duration_seconds:
                    break
            
            if time.time() - start_time > self.duration_seconds:
                break
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'duration_seconds': duration,
            'total_samples': total_samples,
            'total_batches': total_batches,
            'samples_per_second': total_samples / duration,
            'batches_per_second': total_batches / duration,
            'average_loss': total_loss / total_batches if total_batches > 0 else 0,
            'epochs_completed': epoch + 1
        }
    
    def benchmark_batch_sizes(self, model_factory, input_size: int, num_classes: int,
                             batch_sizes: List[int]) -> Dict[str, Dict[str, float]]:
        """Benchmark different batch sizes."""
        results = {}
        
        for batch_size in batch_sizes:
            self.logger.info(f"Benchmarking batch size {batch_size}")
            
            try:
                # Create model and dataset
                model = model_factory(input_size, num_classes)
                
                # Create dataset with this batch size
                X = torch.randn(1000, input_size)
                y = torch.randint(0, num_classes, (1000,))
                dataset = TensorDataset(X, y)
                train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Benchmark
                metrics = self.benchmark_training_speed(model, train_loader, num_epochs=5)
                metrics['batch_size'] = batch_size
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
    
    def benchmark_model_sizes(self) -> Dict[str, Dict[str, float]]:
        """Benchmark different model sizes."""
        results = {}
        
        model_configs = [
            ('small', 256, 4, self.create_simple_model),
            ('medium', 512, 8, self.create_complex_model),
            ('large', 1024, 16, self.create_complex_model)
        ]
        
        for name, input_size, num_classes, model_factory in model_configs:
            self.logger.info(f"Benchmarking {name} model")
            
            try:
                model = model_factory(input_size, num_classes)
                train_loader = self.create_test_dataset(2000, input_size, num_classes)
                
                metrics = self.benchmark_training_speed(model, train_loader, num_epochs=5)
                metrics['model_size'] = name
                metrics['input_size'] = input_size
                metrics['num_classes'] = num_classes
                metrics['num_parameters'] = sum(p.numel() for p in model.parameters())
                
                results[f'model_{name}'] = metrics
                
                # Clean up
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                self.logger.error(f"Error benchmarking {name} model: {e}")
                results[f'model_{name}'] = {
                    'error': str(e),
                    'model_size': name
                }
        
        return results
    
    def benchmark_optimizers(self) -> Dict[str, Dict[str, float]]:
        """Benchmark different optimizers."""
        results = {}
        
        input_size, num_classes = 512, 8
        
        optimizers = [
            ('adam', lambda params: optim.Adam(params, lr=1e-3)),
            ('sgd', lambda params: optim.SGD(params, lr=1e-2, momentum=0.9)),
            ('rmsprop', lambda params: optim.RMSprop(params, lr=1e-3)),
            ('adamw', lambda params: optim.AdamW(params, lr=1e-3))
        ]
        
        for opt_name, opt_factory in optimizers:
            self.logger.info(f"Benchmarking {opt_name} optimizer")
            
            try:
                model = self.create_simple_model(input_size, num_classes)
                train_loader = self.create_test_dataset(1000, input_size, num_classes)
                
                # Custom training loop for this optimizer
                model.train()
                optimizer = opt_factory(model.parameters())
                criterion = nn.CrossEntropyLoss()
                
                start_time = time.time()
                total_samples = 0
                
                for epoch in range(10):
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(self.device), target.to(self.device)
                        
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        
                        total_samples += data.size(0)
                        
                        if time.time() - start_time > 60:  # 1 minute max per optimizer
                            break
                    
                    if time.time() - start_time > 60:
                        break
                
                duration = time.time() - start_time
                
                results[f'optimizer_{opt_name}'] = {
                    'duration_seconds': duration,
                    'total_samples': total_samples,
                    'samples_per_second': total_samples / duration,
                    'optimizer': opt_name
                }
                
                # Clean up
                del model, optimizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                self.logger.error(f"Error benchmarking {opt_name}: {e}")
                results[f'optimizer_{opt_name}'] = {
                    'error': str(e),
                    'optimizer': opt_name
                }
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all training benchmarks."""
        self.logger.info("Starting comprehensive training benchmarks")
        
        all_results = {
            'device': str(self.device),
            'timestamp': time.time(),
            'duration_limit_seconds': self.duration_seconds,
            'benchmarks': {}
        }
        
        # Benchmark batch sizes
        self.logger.info("Running batch size benchmarks")
        batch_size_results = self.benchmark_batch_sizes(
            self.create_simple_model, 256, 4, [8, 16, 32, 64, 128]
        )
        all_results['benchmarks']['batch_sizes'] = batch_size_results
        
        # Benchmark model sizes
        self.logger.info("Running model size benchmarks")
        model_size_results = self.benchmark_model_sizes()
        all_results['benchmarks']['model_sizes'] = model_size_results
        
        # Benchmark optimizers
        self.logger.info("Running optimizer benchmarks")
        optimizer_results = self.benchmark_optimizers()
        all_results['benchmarks']['optimizers'] = optimizer_results
        
        # Add system info
        all_results['system_info'] = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 1,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
        
        self.logger.info("Training benchmarks completed")
        return all_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Training performance benchmarks")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                       help="Device to run benchmarks on")
    parser.add_argument("--duration", type=int, default=30,
                       help="Maximum duration in minutes")
    parser.add_argument("--output", required=True,
                       help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run benchmarks
        benchmark = TrainingBenchmark(device=args.device, duration_minutes=args.duration)
        results = benchmark.run_all_benchmarks()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training benchmarks completed successfully")
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