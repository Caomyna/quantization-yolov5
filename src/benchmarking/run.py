"""
Benchmark runner - High-level benchmarking workflows and CLI entry point.

Usage:
    python src/benchmarking/run.py --model best_decoded
    python src/benchmarking/run.py --all
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Setup Python path so src/ is importable
exec(open(Path(__file__).resolve().parent.parent / 'core' / 'path_setup.py').read())

from src.benchmarking.benchmark import BenchmarkEngine, InferenceMetrics
from src.benchmarking.reporter import save_benchmark_results, save_benchmark_excel
from src.core.config import MODELS_DIR, DATASET_DIR, ANNOTATION_FILE, BENCHMARK_CONFIG, REPORTS_DIR

logger = logging.getLogger(__name__)


def find_model_pairs(models_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan models directory for FP32/FP16 model pairs.
    
    Args:
        models_dir: Directory containing models
        
    Returns:
        List of model pair dictionaries
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


def load_test_images(dataset_dir: Path, annotation_file: Path = None, max_images: int = 1000) -> List[Path]:
    """
    Find image files in dataset directory.
    
    Args:
        dataset_dir: Directory containing images
        annotation_file: Optional COCO annotation file
        max_images: Maximum number of images to load
        
    Returns:
        List of image paths
    """
    from pycocotools.coco import COCO
    
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    # If annotation file provided, use COCO-based selection
    if annotation_file and Path(annotation_file).exists():
        coco = COCO(str(annotation_file))
        
        # Get all image IDs sorted
        img_ids = sorted(coco.getImgIds())
        
        # Limit to max_images
        if max_images > 0:
            img_ids = img_ids[:max_images]
        
        # Load image paths in the same order as COCO image IDs
        images = []
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = dataset_dir / img_info["file_name"]
            if img_path.exists():
                images.append(img_path)
            else:
                logger.warning(f"Image not found: {img_path}")
        return images
    
    images = sorted(
        p for e in exts for p in list(dataset_dir.glob(f"*{e}")) + list(dataset_dir.glob(f"*{e.upper()}"))
    )
    if not images:
        raise ValueError(f"No images in {dataset_dir}")
    if len(images) > max_images:
        images = images[:max_images]
    
    return images


def compare_benchmarks(fp32: InferenceMetrics, fp16: Optional[InferenceMetrics]) -> Dict[str, float]:
    """
    Compare FP32 and FP16 benchmark results.
    
    Args:
        fp32: FP32 metrics
        fp16: FP16 metrics (optional)
        
    Returns:
        Dictionary with comparison metrics
    """
    if fp32 is None or fp16 is None:
        return {
            "latency_speedup": 0,
            "latency_reduction_pct": 0,
            "throughput_improvement": 0,
            "size_reduction_pct": 0,
        }
    
    return {
        "latency_speedup": round(fp32.avg_latency_ms / fp16.avg_latency_ms, 4),
        "latency_reduction_pct": round(
            (fp32.avg_latency_ms - fp16.avg_latency_ms) / fp32.avg_latency_ms * 100, 2
        ),
        "throughput_improvement": round(
            fp16.throughput_fps / fp32.throughput_fps, 4
        ),
        "size_reduction_pct": round(
            (fp32.model_size_mb - fp16.model_size_mb) / fp32.model_size_mb * 100, 2
        ),
    }


def benchmark_model_pair(
    fp32_path: Path,
    fp16_path: Path,
    model_name: str,
    dataset_dir: Path,
    config: dict,
) -> Dict[str, Any]:
    """
    Benchmark a single FP32 vs FP16 model pair.
    
    Args:
        fp32_path: Path to FP32 model
        fp16_path: Path to FP16 model
        model_name: Model name
        dataset_dir: Directory with test images
        config: Benchmark configuration
        
    Returns:
        Dictionary with benchmark results
    """
    warmup = config["warmup_iterations"]
    iterations = config["num_iterations"]
    
    images = load_test_images(dataset_dir, ANNOTATION_FILE, max_images=1000)
    
    # Benchmark FP32
    logger.info(f"Running FP32 benchmark ({iterations} iterations)...")
    fp32_benchmark = BenchmarkEngine(fp32_path, f"{model_name}_FP32")
    fp32_metrics = fp32_benchmark.benchmark(images, warmup, iterations)
    
    # Benchmark FP16
    logger.info(f"Running FP16 benchmark ({iterations} iterations)...")
    fp16_metrics = None
    if fp16_path and fp16_path.exists():
        try:
            fp16_benchmark = BenchmarkEngine(fp16_path, f"{model_name}_FP16")
            fp16_metrics = fp16_benchmark.benchmark(images, warmup, iterations)
        except RuntimeError as e:
            logger.warning(f"FP16 benchmark skipped - {e}")
    else:
        logger.warning(f"FP16 model not found: {fp16_path}")
    
    # Compare
    comparison = compare_benchmarks(fp32_metrics, fp16_metrics)
    
    return {
        "model_name": model_name,
        "fp32": fp32_metrics,
        "fp16": fp16_metrics,
        "comparison": comparison,
    }


def benchmark_single(model_name, output_dir, config):
    """Benchmark a single model pair."""
    # Support both model name (best_decoded) and full path (weights/best_decoded.onnx)
    model_path = Path(model_name)
    if model_path.suffix == '.onnx' or len(model_path.parts) > 1:
        fp32_path = model_path.resolve()
        fp16_path = fp32_path.parent / f"{fp32_path.stem}_fp16.onnx"
        display_name = fp32_path.stem
    else:
        fp32_path = MODELS_DIR / f"{model_name}.onnx"
        fp16_path = MODELS_DIR / f"{model_name}_fp16.onnx"
        display_name = model_name

    if not fp32_path.exists():
        logger.error(f"FP32 model not found: {fp32_path}")
        return None

    logger.info(f"\nBenchmarking: {display_name}")
    images = load_test_images(DATASET_DIR, ANNOTATION_FILE, max_images=1000)

    # FP32
    logger.info(f"  Running FP32 benchmark ({config['num_iterations']} iterations)...")
    fp32_eng = BenchmarkEngine(fp32_path, f"{model_name}_FP32")
    fp32_metrics = fp32_eng.benchmark(images, config["warmup_iterations"], config["num_iterations"])

    # FP16
    fp16_metrics = None
    if fp16_path.exists():
        logger.info(f"  Running FP16 benchmark ({config['num_iterations']} iterations)...")
        try:
            fp16_eng = BenchmarkEngine(fp16_path, f"{model_name}_FP16")
            fp16_metrics = fp16_eng.benchmark(images, config["warmup_iterations"], config["num_iterations"])
        except RuntimeError as e:
            logger.warning(f"  FP16 benchmark skipped - {e}")
    else:
        logger.info(f"  FP16 model not found, skipping")

    comparison = compare_benchmarks(fp32_metrics, fp16_metrics)

    # Print summary
    logger.info(f"\n  Results for {model_name}:")
    logger.info(f"    FP32 avg latency: {fp32_metrics.avg_latency_ms:.2f} ms")
    logger.info(f"    FP32 throughput:  {fp32_metrics.throughput_fps:.2f} FPS")
    if fp16_metrics:
        logger.info(f"    FP16 avg latency: {fp16_metrics.avg_latency_ms:.2f} ms")
        logger.info(f"    FP16 throughput:  {fp16_metrics.throughput_fps:.2f} FPS")
        logger.info(f"    Speedup:          {comparison['latency_speedup']:.2f}x")
    logger.info(f"    Size reduction:   {comparison['size_reduction_pct']:.1f}%")

    # Save individual result
    result_path = output_dir / f"{model_name}_benchmark_results.json"
    save_benchmark_results(fp32_metrics, fp16_metrics, comparison, result_path)
    logger.info(f"  Results saved: {result_path}")

    return {"model_name": model_name, "fp32": fp32_metrics, "fp16": fp16_metrics, "comparison": comparison}


def benchmark_all(output_dir, config):
    """Benchmark all model pairs."""
    model_pairs = find_model_pairs(MODELS_DIR)
    if not model_pairs:
        logger.error(f"No model pairs found in {MODELS_DIR}")
        return

    logger.info(f"Found {len(model_pairs)} model pair(s)")
    results = []
    for pair in model_pairs:
        try:
            result = benchmark_model_pair(
                fp32_path=pair["fp32"], fp16_path=pair["fp16"],
                model_name=pair["name"], dataset_dir=DATASET_DIR, config=config,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {pair['name']}: {e}")
            continue

    if results:
        save_benchmark_excel(results, output_dir)
        logger.info(f"Saved summary: {output_dir / 'benchmark_summary.xlsx'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ONNX models")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g. 'best_decoded') or path to ONNX model (e.g. 'weights/best_decoded.onnx')")
    parser.add_argument("--all", action="store_true", help="Benchmark all model pairs in weights/")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR,
                        help="Output directory for reports (default: reports/)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of benchmark iterations (default: 100)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Number of warmup iterations (default: 10)")
    return parser.parse_args()


def main():
    """Main entry point for benchmarking."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])
    
    args = parse_args()
    config = dict(BENCHMARK_CONFIG)
    if args.iterations:
        config["num_iterations"] = args.iterations
    if args.warmup:
        config["warmup_iterations"] = args.warmup

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        benchmark_all(output_dir, config)
    elif args.model:
        benchmark_single(args.model, output_dir, config)
    else:
        logger.error("Provide --model <name> or --all")
        return 1

    logger.info("\nBenchmark complete.")
    return 0


if __name__ == "__main__":
    exit(main())