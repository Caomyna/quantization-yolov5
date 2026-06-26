"""
Benchmark a single model pair.
Usage: python benchmark_single.py --model model_name
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Dict, Any

from benchmark.benchmark_core import (
    ONNXInferenceBenchmark,
    load_test_images,
    compare_benchmarks,
    save_results,
)
from quantize.config import MODELS_DIR, DATASET_DIR, BENCHMARK_CONFIG


def benchmark_single_model(model_name: str) -> Dict[str, Any]:
    """
    Benchmark a single model pair.
    
    Args:
        model_name: Model name (without extension), e.g., 'magnitude_0.3_decoded'
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    # Find model paths
    fp32_path = MODELS_DIR / f"{model_name}.onnx"
    fp16_path = MODELS_DIR / f"{model_name}_fp16.onnx"
    
    if not fp32_path.exists():
        raise FileNotFoundError(f"FP32 model not found: {fp32_path}")
    
    print(f"FP32 model: {fp32_path}")
    print(f"FP16 model: {fp16_path if fp16_path.exists() else 'Not found'}")
    
    # Load dataset
    images = load_test_images(DATASET_DIR)
    
    # Benchmark FP32
    print(f"\nRunning FP32 benchmark...")
    fp32_benchmark = ONNXInferenceBenchmark(fp32_path, f"{model_name}_FP32")
    fp32_metrics = fp32_benchmark.benchmark(
        images,
        warmup=BENCHMARK_CONFIG["warmup_iterations"],
        iterations=BENCHMARK_CONFIG["num_iterations"]
    )
    
    # Benchmark FP16
    print(f"\nRunning FP16 benchmark...")
    fp16_metrics = None
    if fp16_path.exists():
        try:
            fp16_benchmark = ONNXInferenceBenchmark(fp16_path, f"{model_name}_FP16")
            fp16_metrics = fp16_benchmark.benchmark(
                images,
                warmup=BENCHMARK_CONFIG["warmup_iterations"],
                iterations=BENCHMARK_CONFIG["num_iterations"]
            )
        except RuntimeError as e:
            print(f"Warning: FP16 benchmark skipped - {e}")
    else:
        print(f"Warning: FP16 model not found, skipping FP16 benchmark")
    
    # Compare
    comparison = compare_benchmarks(fp32_metrics, fp16_metrics)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"FP32 avg latency: {fp32_metrics.avg_latency_ms:.2f} ms")
    print(f"FP32 throughput: {fp32_metrics.throughput_fps:.2f} FPS")
    print(f"FP32 model size: {fp32_metrics.model_size_mb:.2f} MB")
    
    if fp16_metrics:
        print(f"\nFP16 avg latency: {fp16_metrics.avg_latency_ms:.2f} ms")
        print(f"FP16 throughput: {fp16_metrics.throughput_fps:.2f} FPS")
        print(f"FP16 model size: {fp16_metrics.model_size_mb:.2f} MB")
        print(f"\nSpeedup: {comparison['latency_speedup']:.2f}x")
        print(f"Size reduction: {comparison['size_reduction_pct']:.1f}%")
    else:
        print("\nFP16: Skipped (model not found or requires CUDA)")
    
    # Save results
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_benchmark_results.json"
    
    save_results(fp32_metrics, fp16_metrics, comparison, output_path)
    
    print(f"\nResults saved: {output_path}")
    
    return {
        "model_name": model_name,
        "fp32": fp32_metrics,
        "fp16": fp16_metrics,
        "comparison": comparison,
    }


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Benchmark a single model pair")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (without extension), e.g., 'magnitude_0.3_decoded'"
    )
    
    args = parser.parse_args()
    
    try:
        benchmark_single_model(args.model)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())