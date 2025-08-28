#!/usr/bin/env python3
"""
Compare benchmark results between two runs and detect performance regressions.

This script compares benchmark results from two JSON files and generates
a report highlighting significant performance changes.
"""

import json
import argparse
import sys
from typing import Dict, List, Any, Tuple
from pathlib import Path
import statistics


class BenchmarkComparator:
    """Compare benchmark results and detect regressions."""
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize comparator.
        
        Args:
            threshold: Regression threshold (e.g., 0.1 for 10%)
        """
        self.threshold = threshold
        self.results = {}
    
    def load_benchmark_file(self, filepath: str) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Handle different benchmark file formats
            if 'benchmarks' in data:
                return data['benchmarks']
            elif isinstance(data, list):
                return {item['name']: item for item in data}
            else:
                return data
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {filepath}: {e}")
            return {}
    
    def extract_benchmark_metrics(self, benchmarks: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from benchmark data."""
        metrics = {}
        
        for name, data in benchmarks.items():
            if isinstance(data, dict):
                # Try different metric names
                metric_value = None
                for metric_key in ['mean', 'avg', 'median', 'time', 'duration']:
                    if metric_key in data:
                        metric_value = data[metric_key]
                        break
                
                if metric_value is not None:
                    metrics[name] = float(metric_value)
                elif 'stats' in data and 'mean' in data['stats']:
                    metrics[name] = float(data['stats']['mean'])
        
        return metrics
    
    def compare_benchmarks(self, baseline_file: str, current_file: str) -> Dict[str, Any]:
        """Compare two benchmark files and return analysis."""
        baseline_data = self.load_benchmark_file(baseline_file)
        current_data = self.load_benchmark_file(current_file)
        
        baseline_metrics = self.extract_benchmark_metrics(baseline_data)
        current_metrics = self.extract_benchmark_metrics(current_data)
        
        if not baseline_metrics:
            print(f"Warning: No baseline metrics found in {baseline_file}")
            return {'status': 'no_baseline', 'message': 'No baseline data available'}
        
        if not current_metrics:
            print(f"Warning: No current metrics found in {current_file}")
            return {'status': 'no_current', 'message': 'No current data available'}
        
        # Find common benchmarks
        common_benchmarks = set(baseline_metrics.keys()) & set(current_metrics.keys())
        
        if not common_benchmarks:
            print("Warning: No common benchmarks found between files")
            return {'status': 'no_common', 'message': 'No common benchmarks found'}
        
        # Compare metrics
        comparisons = []
        regressions = []
        improvements = []
        
        for benchmark in common_benchmarks:
            baseline_value = baseline_metrics[benchmark]
            current_value = current_metrics[benchmark]
            
            if baseline_value == 0:
                continue  # Skip to avoid division by zero
            
            percent_change = (current_value - baseline_value) / baseline_value
            
            comparison = {
                'name': benchmark,
                'baseline': baseline_value,
                'current': current_value,
                'absolute_change': current_value - baseline_value,
                'percent_change': percent_change,
                'is_regression': percent_change > self.threshold,
                'is_improvement': percent_change < -self.threshold
            }
            
            comparisons.append(comparison)
            
            if comparison['is_regression']:
                regressions.append(comparison)
            elif comparison['is_improvement']:
                improvements.append(comparison)
        
        # Sort by impact
        comparisons.sort(key=lambda x: abs(x['percent_change']), reverse=True)
        regressions.sort(key=lambda x: x['percent_change'], reverse=True)
        improvements.sort(key=lambda x: x['percent_change'])
        
        analysis = {
            'status': 'success',
            'threshold': self.threshold,
            'total_benchmarks': len(comparisons),
            'regressions_count': len(regressions),
            'improvements_count': len(improvements),
            'comparisons': comparisons,
            'regressions': regressions,
            'improvements': improvements,
            'summary': {
                'has_regressions': len(regressions) > 0,
                'max_regression': max([r['percent_change'] for r in regressions], default=0),
                'max_improvement': min([i['percent_change'] for i in improvements], default=0),
                'avg_change': statistics.mean([c['percent_change'] for c in comparisons]) if comparisons else 0
            }
        }
        
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any], output_file: str = None) -> str:
        """Generate human-readable report."""
        if analysis['status'] != 'success':
            return f"Analysis failed: {analysis.get('message', 'Unknown error')}"
        
        report_lines = []
        report_lines.append("# Benchmark Comparison Report")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        report_lines.append(f"- Total benchmarks compared: {analysis['total_benchmarks']}")
        report_lines.append(f"- Regressions detected: {analysis['regressions_count']}")
        report_lines.append(f"- Improvements detected: {analysis['improvements_count']}")
        report_lines.append(f"- Regression threshold: {analysis['threshold']*100:.1f}%")
        report_lines.append("")
        
        if analysis['summary']['has_regressions']:
            max_reg = analysis['summary']['max_regression'] * 100
            report_lines.append(f"⚠️  **Maximum regression: {max_reg:.1f}%**")
        else:
            report_lines.append("✅ **No significant regressions detected**")
        
        if analysis['improvements']:
            max_imp = abs(analysis['summary']['max_improvement']) * 100
            report_lines.append(f"🚀 **Maximum improvement: {max_imp:.1f}%**")
        
        report_lines.append("")
        
        # Regressions
        if analysis['regressions']:
            report_lines.append("## 🔴 Performance Regressions")
            report_lines.append("")
            report_lines.append("| Benchmark | Baseline | Current | Change | % Change |")
            report_lines.append("|-----------|----------|---------|--------|----------|")
            
            for reg in analysis['regressions'][:10]:  # Top 10
                change_pct = reg['percent_change'] * 100
                report_lines.append(
                    f"| {reg['name']} | {reg['baseline']:.4f} | {reg['current']:.4f} | "
                    f"{reg['absolute_change']:+.4f} | {change_pct:+.1f}% |"
                )
            
            if len(analysis['regressions']) > 10:
                report_lines.append(f"| ... | ... | ... | ... | *{len(analysis['regressions']) - 10} more* |")
            
            report_lines.append("")
        
        # Improvements
        if analysis['improvements']:
            report_lines.append("## 🟢 Performance Improvements")
            report_lines.append("")
            report_lines.append("| Benchmark | Baseline | Current | Change | % Change |")
            report_lines.append("|-----------|----------|---------|--------|----------|")
            
            for imp in analysis['improvements'][:10]:  # Top 10
                change_pct = imp['percent_change'] * 100
                report_lines.append(
                    f"| {imp['name']} | {imp['baseline']:.4f} | {imp['current']:.4f} | "
                    f"{imp['absolute_change']:+.4f} | {change_pct:+.1f}% |"
                )
            
            if len(analysis['improvements']) > 10:
                report_lines.append(f"| ... | ... | ... | ... | *{len(analysis['improvements']) - 10} more* |")
            
            report_lines.append("")
        
        # All changes (if not too many)
        if analysis['total_benchmarks'] <= 20:
            report_lines.append("## All Benchmark Changes")
            report_lines.append("")
            report_lines.append("| Benchmark | Baseline | Current | Change | % Change | Status |")
            report_lines.append("|-----------|----------|---------|--------|----------|--------|")
            
            for comp in analysis['comparisons']:
                change_pct = comp['percent_change'] * 100
                if comp['is_regression']:
                    status = "🔴 Regression"
                elif comp['is_improvement']:
                    status = "🟢 Improvement"
                else:
                    status = "⚪ No change"
                
                report_lines.append(
                    f"| {comp['name']} | {comp['baseline']:.4f} | {comp['current']:.4f} | "
                    f"{comp['absolute_change']:+.4f} | {change_pct:+.1f}% | {status} |"
                )
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("baseline", help="Baseline benchmark JSON file")
    parser.add_argument("current", help="Current benchmark JSON file")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Regression threshold (default: 0.1 for 10%)")
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--json-output", help="Output analysis as JSON")
    parser.add_argument("--fail-on-regression", action="store_true",
                       help="Exit with error code if regressions detected")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.baseline).exists():
        print(f"Error: Baseline file {args.baseline} not found")
        sys.exit(1)
    
    if not Path(args.current).exists():
        print(f"Error: Current file {args.current} not found")
        sys.exit(1)
    
    # Compare benchmarks
    comparator = BenchmarkComparator(threshold=args.threshold)
    analysis = comparator.compare_benchmarks(args.baseline, args.current)
    
    # Save JSON analysis if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {args.json_output}")
    
    # Generate and display report
    report = comparator.generate_report(analysis, args.output)
    print(report)
    
    # Exit with error if regressions detected and flag is set
    if args.fail_on_regression and analysis.get('summary', {}).get('has_regressions', False):
        print("\n❌ Performance regressions detected!")
        sys.exit(1)
    
    print("\n✅ Benchmark comparison completed successfully")


if __name__ == "__main__":
    main()