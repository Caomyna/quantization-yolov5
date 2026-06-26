"""
Benchmark all models in weights/ directory and generate comparison reports.
Scans for FP32/FP16 model pairs, runs benchmarks, and exports results.
Each row in the Excel represents one model with both FP32 and FP16 metrics.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from typing import Dict, List, Any
import pandas as pd
from dataclasses import asdict

from benchmark.benchmark_core import (
    ONNXInferenceBenchmark,
    load_test_images,
    compare_benchmarks,
    save_results,
)
from quantize.config import MODELS_DIR, DATASET_DIR, ANNOTATION_FILE, BENCHMARK_CONFIG


def find_model_pairs(models_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan models directory for FP32/FP16 model pairs.
    Looks for files matching pattern: {name}.onnx and {name}_fp16.onnx
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Find all FP32 models
    fp32_models = list(models_dir.glob("*.onnx"))
    
    pairs = []
    for fp32_path in fp32_models:
        # Skip FP16 models
        if "_fp16" in fp32_path.stem:
            continue
        
        # Look for corresponding FP16 model
        fp16_path = models_dir / f"{fp32_path.stem}_fp16.onnx"
        
        model_name = fp32_path.stem
        
        pairs.append({
            "name": model_name,
            "fp32": fp32_path,
            "fp16": fp16_path if fp16_path.exists() else None,
        })
    
    return sorted(pairs, key=lambda x: x["name"])


def benchmark_model_pair(
    fp32_path: Path,
    fp16_path: Path,
    model_name: str,
    dataset_dir: Path,
    config: dict,
) -> Dict[str, Any]:
    """Benchmark a single FP32 vs FP16 model pair."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    warmup = config["warmup_iterations"]
    iterations = config["num_iterations"]
    
    images = load_test_images(dataset_dir, ANNOTATION_FILE, max_images=1000)
    
    # Benchmark FP32
    print(f"Running FP32 benchmark ({iterations} iterations)...")
    fp32_benchmark = ONNXInferenceBenchmark(fp32_path, f"{model_name}_FP32")
    fp32_metrics = fp32_benchmark.benchmark(images, warmup, iterations)
    
    # Benchmark FP16
    print(f"Running FP16 benchmark ({iterations} iterations)...")
    fp16_metrics = None
    if fp16_path and fp16_path.exists():
        try:
            fp16_benchmark = ONNXInferenceBenchmark(fp16_path, f"{model_name}_FP16")
            fp16_metrics = fp16_benchmark.benchmark(images, warmup, iterations)
        except RuntimeError as e:
            print(f"Warning: FP16 benchmark skipped - {e}")
    else:
        print(f"Warning: FP16 model not found: {fp16_path}")
    
    # Compare
    comparison = compare_benchmarks(fp32_metrics, fp16_metrics)
    
    # Print summary
    print(f"\nResults for {model_name}:")
    print(f"  FP32 avg latency: {fp32_metrics.avg_latency_ms:.2f} ms")
    if fp16_metrics:
        print(f"  FP16 avg latency: {fp16_metrics.avg_latency_ms:.2f} ms")
        print(f"  Speedup: {comparison['latency_speedup']:.2f}x")
    print(f"  Size reduction: {comparison['size_reduction_pct']:.1f}%")
    
    return {
        "model_name": model_name,
        "fp32": asdict(fp32_metrics),
        "fp16": asdict(fp16_metrics) if fp16_metrics else None,
        "comparison": comparison,
    }


def save_all_results(benchmark_results: List[Dict[str, Any]], output_dir: Path):
    """Save individual JSON files and summary Excel file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual benchmark JSON files
    for result in benchmark_results:
        model_name = result["model_name"]
        json_path = output_dir / f"{model_name}_benchmark_results.json"
        
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "fp32": result["fp32"],
            "fp16": result["fp16"],
            "comparison": result["comparison"],
        }
        
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {json_path}")
    
    # Create summary Excel file — one row per model variant (FP32 row, FP16 row)
    excel_path = output_dir / "benchmark_summary.xlsx"
    
    flat_data = []
    for result in benchmark_results:
        model_name = result["model_name"]
        fp32 = result["fp32"]
        fp16 = result["fp16"]
        comp = result["comparison"]
        
        # FP32 row
        flat_data.append({
            "Model": f"{model_name}.onnx",
            "Precision": "FP32",
            "Size (MB)": fp32["model_size_mb"],
            "Avg Latency (ms)": fp32["avg_latency_ms"],
            "Min Latency (ms)": fp32["min_latency_ms"],
            "Max Latency (ms)": fp32["max_latency_ms"],
            "Std Latency (ms)": fp32["std_latency_ms"],
            "P95 Latency (ms)": fp32["p95_latency_ms"],
            "P99 Latency (ms)": fp32["p99_latency_ms"],
            "FPS": fp32["throughput_fps"],
            "Peak Memory (MB)": fp32["peak_memory_mb"],
            "Avg Memory (MB)": fp32["avg_memory_mb"],
            "Num Iterations": fp32["num_iterations"],
            "Num Images": fp32["num_images"],
            "Warmup Iterations": fp32["warmup_iterations"],
            "Notes": "",
        })
        
        # FP16 row
        if fp16:
            flat_data.append({
                "Model": f"{model_name}_fp16.onnx",
                "Precision": "FP16",
                "Size (MB)": fp16["model_size_mb"],
                "Avg Latency (ms)": fp16["avg_latency_ms"],
                "Min Latency (ms)": fp16["min_latency_ms"],
                "Max Latency (ms)": fp16["max_latency_ms"],
                "Std Latency (ms)": fp16["std_latency_ms"],
                "P95 Latency (ms)": fp16["p95_latency_ms"],
                "P99 Latency (ms)": fp16["p99_latency_ms"],
                "FPS": fp16["throughput_fps"],
                "Peak Memory (MB)": fp16["peak_memory_mb"],
                "Avg Memory (MB)": fp16["avg_memory_mb"],
                "Num Iterations": fp16["num_iterations"],
                "Num Images": fp16["num_images"],
                "Warmup Iterations": fp16["warmup_iterations"],
                "Notes": f"Size reduction: {comp['size_reduction_pct']:.1f}%, "
                         f"Latency reduction: {comp['latency_reduction_pct']:.1f}%, "
                         f"Speedup: {comp['latency_speedup']:.2f}x",
            })
    
    df_flat = pd.DataFrame(flat_data)
    
    # Write to Excel with formatting
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_flat.to_excel(writer, sheet_name='Benchmark Results', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Benchmark Results']
        
        # Format headers (bold)
        from openpyxl.styles import Font, Alignment
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            adjusted_width = min(max(max_length + 2, 12), 30)
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Format cell alignment (no number formatting to preserve original precision)
        for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
            for cell in row:
                cell.alignment = Alignment(horizontal='center', vertical='center')
    
    print(f"\nSaved Excel summary: {excel_path}")
    print(f"  - Sheet 1: Benchmark Results ({len(df_flat)} rows)")


def main():
    """Main function to benchmark all model pairs."""
    print("="*60)
    print("BENCHMARKING ALL MODELS")
    print("="*60)
    
    # Find all model pairs
    print("\nScanning for model pairs...")
    model_pairs = find_model_pairs(MODELS_DIR)
    
    if not model_pairs:
        print(f"No model pairs found in {MODELS_DIR}")
        print("Expected format: model.onnx and model_fp16.onnx")
        return
    
    print(f"\nFound {len(model_pairs)} model pair(s):")
    for pair in model_pairs:
        fp16_status = "✓" if pair["fp16"] else "✗ (FP16 not found)"
        print(f"  - {pair['name']}: FP32 ✓, FP16 {fp16_status}")
    
    # Check dataset
    dataset_exists = DATASET_DIR.exists()
    print(f"\nDataset: {'✓' if dataset_exists else '✗'} {DATASET_DIR}")
    if not dataset_exists:
        print("ERROR: Dataset not found!")
        return
    
    # Run benchmarks
    benchmark_results = []
    
    for pair in model_pairs:
        try:
            result = benchmark_model_pair(
                fp32_path=pair["fp32"],
                fp16_path=pair["fp16"],
                model_name=pair["name"],
                dataset_dir=DATASET_DIR,
                config=BENCHMARK_CONFIG,
            )
            benchmark_results.append(result)
        except Exception as e:
            print(f"\nERROR processing {pair['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    if benchmark_results:
        output_dir = Path("reports")
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")
        save_all_results(benchmark_results, output_dir)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed {len(benchmark_results)} model pair(s)")
        print(f"\nOutput files:")
        for result in benchmark_results:
            json_file = f"reports/{result['model_name']}_benchmark_results.json"
            print(f"  - {json_file}")
        print(f"  - reports/benchmark_summary.xlsx")
    else:
        print("\nNo successful benchmarks completed!")


if __name__ == "__main__":
    main()