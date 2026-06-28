"""
Benchmark reporter - Save benchmark results to JSON and Excel.
"""

import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict

from .benchmark import InferenceMetrics


def save_benchmark_results(
    fp32: InferenceMetrics,
    fp16: InferenceMetrics,
    comparison: Dict[str, float],
    path: Path
):
    """
    Save benchmark results as JSON.
    
    Args:
        fp32: FP32 metrics
        fp16: FP16 metrics (can be None)
        comparison: Comparison metrics
        path: Output JSON path
    """
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fp32": asdict(fp32),
        "fp16": asdict(fp16) if fp16 else None,
        "comparison": comparison,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Results saved: {path}")


def save_benchmark_excel(
    benchmark_results: List[Dict[str, Any]],
    output_dir: Path
):
    """
    Save benchmark results as Excel file.
    
    Args:
        benchmark_results: List of benchmark result dictionaries
        output_dir: Output directory
    """
    import pandas as pd
    from openpyxl.styles import Font, Alignment
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            "Size (MB)": fp32.model_size_mb,
            "Avg Latency (ms)": fp32.avg_latency_ms,
            "Min Latency (ms)": fp32.min_latency_ms,
            "Max Latency (ms)": fp32.max_latency_ms,
            "Std Latency (ms)": fp32.std_latency_ms,
            "P95 Latency (ms)": fp32.p95_latency_ms,
            "P99 Latency (ms)": fp32.p99_latency_ms,
            "FPS": fp32.throughput_fps,
            "Peak Memory (MB)": fp32.peak_memory_mb,
            "Avg Memory (MB)": fp32.avg_memory_mb,
            "Num Iterations": fp32.num_iterations,
            "Num Images": fp32.num_images,
            "Warmup Iterations": fp32.warmup_iterations,
            "Notes": "",
        })
        
        # FP16 row
        if fp16:
            flat_data.append({
                "Model": f"{model_name}_fp16.onnx",
                "Precision": "FP16",
                "Size (MB)": fp16.model_size_mb,
                "Avg Latency (ms)": fp16.avg_latency_ms,
                "Min Latency (ms)": fp16.min_latency_ms,
                "Max Latency (ms)": fp16.max_latency_ms,
                "Std Latency (ms)": fp16.std_latency_ms,
                "P95 Latency (ms)": fp16.p95_latency_ms,
                "P99 Latency (ms)": fp16.p99_latency_ms,
                "FPS": fp16.throughput_fps,
                "Peak Memory (MB)": fp16.peak_memory_mb,
                "Avg Memory (MB)": fp16.avg_memory_mb,
                "Num Iterations": fp16.num_iterations,
                "Num Images": fp16.num_images,
                "Warmup Iterations": fp16.warmup_iterations,
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
        
        # Format cell alignment
        for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
            for cell in row:
                cell.alignment = Alignment(horizontal='center', vertical='center')
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Saved Excel summary: {excel_path}")
    logger.info(f"Benchmark Results ({len(df_flat)} rows)")