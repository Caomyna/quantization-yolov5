"""
Evaluation runner - High-level evaluation workflows and CLI entry point.

Usage:
    python src/evaluation/run.py --model best_decoded
    python src/evaluation/run.py --all
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup Python path so src/ is importable
exec(open(Path(__file__).resolve().parent.parent / 'core' / 'path_setup.py').read())

from src.evaluation.evaluator import EvaluationEngine
from src.evaluation.reporter import save_evaluation_results, save_evaluation_excel
from src.core.config import MODELS_DIR, DATASET_DIR, ANNOTATION_FILE, MODEL_CONFIG, REPORTS_DIR
from src.benchmarking.run import find_model_pairs

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: Path,
    model_name: str,
    dataset_dir: Path,
    annotation_file: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_images: int = 500,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a single model.
    
    Args:
        model_path: Path to ONNX model
        model_name: Name for logging
        dataset_dir: Directory with test images
        annotation_file: COCO annotation JSON file
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        max_images: Maximum images to evaluate
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = EvaluationEngine(
        model_path=model_path,
        model_name=model_name,
        dataset_dir=dataset_dir,
        annotation_file=annotation_file,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_images=max_images,
    )
    
    return evaluator.evaluate()


def evaluate_model_pair(
    fp32_path: Path,
    fp16_path: Path,
    model_name: str,
    dataset_dir: Path,
    annotation_file: Path,
    config: dict,
    max_images: int = 500,
) -> Dict[str, Any]:
    """
    Evaluate a single FP32 vs FP16 model pair.
    
    Args:
        fp32_path: Path to FP32 model
        fp16_path: Path to FP16 model
        model_name: Model name
        dataset_dir: Directory with test images
        annotation_file: COCO annotation JSON file
        config: Configuration dictionary
        max_images: Maximum images to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    conf_threshold = config.get("conf_threshold", 0.25)
    iou_threshold = config.get("iou_threshold", 0.45)
    
    # Evaluate FP32
    logger.info(f"Running FP32 evaluation (max {max_images} images)...")
    fp32_eval = evaluate_model(
        fp32_path,
        f"{model_name}_FP32",
        dataset_dir,
        annotation_file,
        conf_threshold,
        iou_threshold,
        max_images
    )
    
    # Evaluate FP16
    logger.info(f"Running FP16 evaluation (max {max_images} images)...")
    fp16_eval = None
    if fp16_path and fp16_path.exists():
        try:
            fp16_eval = evaluate_model(
                fp16_path,
                f"{model_name}_FP16",
                dataset_dir,
                annotation_file,
                conf_threshold,
                iou_threshold,
                max_images
            )
        except Exception as e:
            logger.warning(f"FP16 evaluation skipped - {e}")
    else:
        logger.warning(f"FP16 model not found: {fp16_path}")
    
    return {
        "model_name": model_name,
        "fp32_eval": fp32_eval,
        "fp16_eval": fp16_eval,
    }


def evaluate_single(model_name, output_dir, config, max_images):
    """Evaluate a single model pair."""
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
    if not ANNOTATION_FILE.exists():
        logger.error(f"COCO annotations not found: {ANNOTATION_FILE}")
        return None

    conf_threshold = config.get("conf_threshold", 0.25)
    iou_threshold = config.get("iou_threshold", 0.45)

    logger.info(f"\nEvaluating: {display_name}")

    # FP32
    logger.info(f"  Running FP32 evaluation (max {max_images} images)...")
    fp32_result = evaluate_model(
        fp32_path, f"{model_name}_FP32", DATASET_DIR, ANNOTATION_FILE,
        conf_threshold, iou_threshold, max_images
    )

    # FP16
    fp16_result = None
    if fp16_path.exists():
        logger.info(f"  Running FP16 evaluation (max {max_images} images)...")
        try:
            fp16_result = evaluate_model(
                fp16_path, f"{model_name}_FP16", DATASET_DIR, ANNOTATION_FILE,
                conf_threshold, iou_threshold, max_images
            )
        except Exception as e:
            logger.warning(f"  FP16 evaluation skipped - {e}")
    else:
        logger.info(f"  FP16 model not found, skipping")

    # Summary
    if fp32_result:
        m = fp32_result.get("metrics", {})
        logger.info(f"\n  FP32 Results:")
        logger.info(f"    Precision:    {m.get('precision', 0):.4f}")
        logger.info(f"    Recall:       {m.get('recall', 0):.4f}")
        logger.info(f"    mAP@0.50:     {m.get('map50', 0):.4f}")
        logger.info(f"    mAP@0.50:0.95:{m.get('map50_95', 0):.4f}")

    if fp16_result:
        m = fp16_result.get("metrics", {})
        logger.info(f"\n  FP16 Results:")
        logger.info(f"    Precision:    {m.get('precision', 0):.4f}")
        logger.info(f"    Recall:       {m.get('recall', 0):.4f}")
        logger.info(f"    mAP@0.50:     {m.get('map50', 0):.4f}")
        logger.info(f"    mAP@0.50:0.95:{m.get('map50_95', 0):.4f}")
        map50_diff = fp32_result["metrics"]["map50"] - fp16_result["metrics"]["map50"]
        logger.info(f"\n  mAP50 difference: {map50_diff:.4f}")

    # Save individual result
    result_data = {
        "model_name": model_name,
        "fp32_eval": fp32_result,
        "fp16_eval": fp16_result,
    }
    result_path = output_dir / f"{model_name}_evaluation_results.json"
    save_evaluation_results(result_data, result_path)
    logger.info(f"  Results saved: {result_path}")
    return result_data


def evaluate_all(output_dir, config, max_images):
    """Evaluate all model pairs."""
    model_pairs = find_model_pairs(MODELS_DIR)
    if not model_pairs:
        logger.error(f"No model pairs found in {MODELS_DIR}")
        return

    logger.info(f"Found {len(model_pairs)} model pair(s)")
    results = []
    for pair in model_pairs:
        if not pair["fp32"].exists():
            continue
        try:
            result = evaluate_model_pair(
                fp32_path=pair["fp32"], fp16_path=pair["fp16"],
                model_name=pair["name"], dataset_dir=DATASET_DIR,
                annotation_file=ANNOTATION_FILE, config=config, max_images=max_images,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {pair['name']}: {e}")
            continue

    if results:
        save_evaluation_excel(results, output_dir)
        logger.info(f"Saved summary: {output_dir / 'evaluation_summary.xlsx'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ONNX models with COCO metrics")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g. 'best_decoded') or path to ONNX model (e.g. 'weights/best_decoded.onnx')")
    parser.add_argument("--all", action="store_true", help="Evaluate all model pairs in weights/")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR,
                        help="Output directory for reports (default: reports/)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum images to evaluate (default: 500)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=None,
                        help="IoU threshold for NMS (default: 0.45)")
    return parser.parse_args()


def main():
    """Main entry point for evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])
    
    args = parse_args()
    config = dict(MODEL_CONFIG)
    if args.conf is not None:
        config["conf_threshold"] = args.conf
    if args.iou is not None:
        config["iou_threshold"] = args.iou

    max_images = args.max_images or MODEL_CONFIG.get("max_eval_images", 1000)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        evaluate_all(output_dir, config, max_images)
    elif args.model:
        evaluate_single(args.model, output_dir, config, max_images)
    else:
        logger.error("Provide --model <name> or --all")
        return 1

    logger.info("\nEvaluation complete.")
    return 0


if __name__ == "__main__":
    exit(main())