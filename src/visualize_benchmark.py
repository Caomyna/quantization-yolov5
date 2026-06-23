"""
Visualization module for benchmark results.
Generates comparison plots from benchmark_results.json.
Can be run standalone to visualize existing results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

from config import BENCHMARK_RESULTS_PATH, BENCHMARK_PLOT_PATH


@dataclass
class InferenceMetrics:
    """Data class to store inference metrics."""
    model_name: str
    model_path: str
    model_size_mb: float
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Throughput metrics
    throughput_fps: float
    total_time_sec: float
    
    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float
    
    # Additional info
    num_iterations: int
    num_images: int
    warmup_iterations: int


def load_benchmark_results(results_path: Path) -> Dict:
    """
    Load benchmark results from JSON file.
    
    Args:
        results_path: Path to benchmark_results.json
        
    Returns:
        Dict: Benchmark results
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Benchmark results not found at: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def dict_to_metrics(data: Dict) -> InferenceMetrics:
    """
    Convert dictionary to InferenceMetrics dataclass.
    
    Args:
        data: Dictionary with metrics data
        
    Returns:
        InferenceMetrics: Metrics object
    """
    return InferenceMetrics(**data)


def plot_comparison(
    fp32_metrics: InferenceMetrics,
    fp16_metrics: InferenceMetrics,
    comparison: Dict,
    output_path: Path
):
    """
    Generate comparison plots.
    
    Args:
        fp32_metrics: FP32 model metrics
        fp16_metrics: FP16 model metrics
        comparison: Comparison results
        output_path: Path to save plot
    """
    print(f"[INFO] Generating comparison plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('FP32 vs FP16 Benchmark Comparison', fontsize=16, fontweight='bold')
    
    # Color palette
    colors = {'fp32': '#3498db', 'fp16': '#e74c3c'}
    
    # 1. Latency Comparison (Bar)
    ax1 = axes[0, 0]
    categories = ['Avg', 'Min', 'Max', 'P95', 'P99']
    fp32_values = [
        fp32_metrics.avg_latency_ms,
        fp32_metrics.min_latency_ms,
        fp32_metrics.max_latency_ms,
        fp32_metrics.p95_latency_ms,
        fp32_metrics.p99_latency_ms
    ]
    fp16_values = [
        fp16_metrics.avg_latency_ms,
        fp16_metrics.min_latency_ms,
        fp16_metrics.max_latency_ms,
        fp16_metrics.p95_latency_ms,
        fp16_metrics.p99_latency_ms
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    ax1.bar(x - width/2, fp32_values, width, label='FP32', color=colors['fp32'])
    ax1.bar(x + width/2, fp16_values, width, label='FP16', color=colors['fp16'])
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Throughput Comparison (Bar)
    ax2 = axes[0, 1]
    throughputs = [fp32_metrics.throughput_fps, fp16_metrics.throughput_fps]
    bars = ax2.bar(['FP32', 'FP16'], throughputs, color=[colors['fp32'], colors['fp16']])
    ax2.set_ylabel('Throughput (FPS)')
    ax2.set_title('Throughput Comparison')
    ax2.grid(True, alpha=0.3)
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 3. Model Size Comparison (Bar)
    ax3 = axes[0, 2]
    sizes = [fp32_metrics.model_size_mb, fp16_metrics.model_size_mb]
    bars = ax3.bar(['FP32', 'FP16'], sizes, color=[colors['fp32'], colors['fp16']])
    ax3.set_ylabel('Size (MB)')
    ax3.set_title('Model Size Comparison')
    ax3.grid(True, alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 4. Latency Distribution (Box plot)
    ax4 = axes[1, 0]
    # We don't have raw latency data here, so we'll use summary stats
    latency_data = [
        [fp32_metrics.min_latency_ms, fp32_metrics.avg_latency_ms, 
         fp32_metrics.p95_latency_ms, fp32_metrics.p99_latency_ms, fp32_metrics.max_latency_ms],
        [fp16_metrics.min_latency_ms, fp16_metrics.avg_latency_ms,
         fp16_metrics.p95_latency_ms, fp16_metrics.p99_latency_ms, fp16_metrics.max_latency_ms]
    ]
    bp = ax4.boxplot(latency_data, labels=['FP32', 'FP16'], patch_artist=True)
    bp['boxes'][0].set_facecolor(colors['fp32'])
    bp['boxes'][1].set_facecolor(colors['fp16'])
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title('Latency Distribution')
    ax4.grid(True, alpha=0.3)
    
    # 5. Improvement Metrics (Horizontal bar)
    ax5 = axes[1, 1]
    improvements = [
        comparison['latency_speedup'],
        comparison['throughput_improvement'],
        comparison['size_reduction'] / 100  # Convert to ratio
    ]
    categories = ['Latency\nSpeedup', 'Throughput\nImprovement', 'Size\nReduction']
    colors_imp = ['#2ecc71', '#2ecc71', '#2ecc71']
    bars = ax5.barh(categories, improvements, color=colors_imp)
    ax5.set_xlabel('Improvement Factor')
    ax5.set_title('Performance Improvements')
    ax5.grid(True, alpha=0.3)
    ax5.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
    ax5.legend()
    for bar in bars:
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}x', ha='left', va='center')
    
    # 6. Summary Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    table_data = [
        ['Metric', 'FP32', 'FP16', 'Improvement'],
        ['Avg Latency', f"{fp32_metrics.avg_latency_ms:.2f} ms", 
         f"{fp16_metrics.avg_latency_ms:.2f} ms", f"{comparison['latency_reduction']:.1f}%"],
        ['Throughput', f"{fp32_metrics.throughput_fps:.2f} FPS",
         f"{fp16_metrics.throughput_fps:.2f} FPS", f"{comparison['throughput_improvement']:.2f}x"],
        ['Model Size', f"{fp32_metrics.model_size_mb:.2f} MB",
         f"{fp16_metrics.model_size_mb:.2f} MB", f"{comparison['size_reduction']:.1f}%"],
        ['P95 Latency', f"{fp32_metrics.p95_latency_ms:.2f} ms",
         f"{fp16_metrics.p95_latency_ms:.2f} ms", '-'],
        ['P99 Latency', f"{fp32_metrics.p99_latency_ms:.2f} ms",
         f"{fp16_metrics.p99_latency_ms:.2f} ms", '-'],
    ]
    table = ax6.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.25, 0.25, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to: {output_path}")
    
    # Show plot (optional, comment out for headless environments)
    # plt.show()
    plt.close()


def main():
    """Main execution function - visualize existing benchmark results."""
    print("=" * 70)
    print("Benchmark Visualization Module")
    print("=" * 70)
    
    try:
        # Load benchmark results
        print(f"[INFO] Loading benchmark results from: {BENCHMARK_RESULTS_PATH}")
        results = load_benchmark_results(BENCHMARK_RESULTS_PATH)
        
        # Convert to metrics objects
        fp32_metrics = dict_to_metrics(results['fp32'])
        fp16_metrics = dict_to_metrics(results['fp16'])
        comparison = results['comparison']
        
        print(f"[INFO] FP32 Model: {fp32_metrics.model_name}")
        print(f"[INFO] FP16 Model: {fp16_metrics.model_name}")
        
        # Generate plots
        plot_comparison(fp32_metrics, fp16_metrics, comparison, BENCHMARK_PLOT_PATH)
        
        print("\n" + "=" * 70)
        print("Visualization completed successfully!")
        print(f"Plot saved to: {BENCHMARK_PLOT_PATH}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] Visualization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()