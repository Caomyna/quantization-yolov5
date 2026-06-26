"""
Evaluate a single model.
Usage: python evaluate_single.py --model model_name
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from typing import Dict, Any

from evaluation.evaluation_core import evaluate_model
from quantize.config import MODELS_DIR, DATASET_DIR, MODEL_CONFIG


def evaluate_single_model(model_name: str) -> Dict[str, Any]:
    """
    Evaluate a single model.
    
    Args:
        model_name: Model name (without extension), e.g., 'magnitude_0.3_decoded'
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    # Find model paths
    fp32_path = MODELS_DIR / f"{model_name}.onnx"
    fp16_path = MODELS_DIR / f"{model_name}_fp16.onnx"
    
    if not fp32_path.exists():
        raise FileNotFoundError(f"FP32 model not found: {fp32_path}")
    
    print(f"FP32 model: {fp32_path}")
    print(f"FP16 model: {fp16_path if fp16_path.exists() else 'Not found'}")
    
    # Find annotation file
    annotation_file = DATASET_DIR.parent / "annotations" / "instances_val2017.json"
    if not annotation_file.exists():
        raise FileNotFoundError(f"COCO annotations not found: {annotation_file}")
    
    # Get config
    conf_threshold = MODEL_CONFIG.get("conf_threshold", 0.25)
    iou_threshold = MODEL_CONFIG.get("iou_threshold", 0.45)
    max_images = MODEL_CONFIG.get("max_eval_images", 500)
    
    # Evaluate FP32
    print(f"\nRunning FP32 evaluation (max {max_images} images)...")
    fp32_result = evaluate_model(
        fp32_path,
        f"{model_name}_FP32",
        DATASET_DIR,
        annotation_file,
        conf_threshold,
        iou_threshold,
        max_images
    )
    
    # Evaluate FP16
    print(f"\nRunning FP16 evaluation (max {max_images} images)...")
    fp16_result = None
    if fp16_path.exists():
        try:
            fp16_result = evaluate_model(
                fp16_path,
                f"{model_name}_FP16",
                DATASET_DIR,
                annotation_file,
                conf_threshold,
                iou_threshold,
                max_images
            )
        except RuntimeError as e:
            print(f"Warning: FP16 evaluation skipped - {e}")
    else:
        print(f"Warning: FP16 model not found, skipping FP16 evaluation")
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    if fp32_result:
        print(f"\nFP32 Results:")
        print(f"  Precision: {fp32_result['metrics']['precision']:.4f}")
        print(f"  Recall: {fp32_result['metrics']['recall']:.4f}")
        print(f"  mAP@0.50: {fp32_result['metrics']['map50']:.4f}")
        print(f"  mAP@0.50:0.95: {fp32_result['metrics']['map50_95']:.4f}")
    
    if fp16_result:
        print(f"\nFP16 Results:")
        print(f"  Precision: {fp16_result['metrics']['precision']:.4f}")
        print(f"  Recall: {fp16_result['metrics']['recall']:.4f}")
        print(f"  mAP@0.50: {fp16_result['metrics']['map50']:.4f}")
        print(f"  mAP@0.50:0.95: {fp16_result['metrics']['map50_95']:.4f}")
        
        # Compare
        print(f"\nComparison:")
        map50_diff = fp32_result['metrics']['map50'] - fp16_result['metrics']['map50']
        print(f"  mAP50 difference: {map50_diff:.4f}")
    else:
        print("\nFP16: Skipped (model not found or requires CUDA)")
    
    # Save results
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_evaluation_results.json"
    
    result = {
        "model_name": model_name,
        "fp32_evaluation": fp32_result,
        "fp16_evaluation": fp16_result,
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved: {output_path}")
    
    return result


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a single model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (without extension), e.g., 'magnitude_0.3_decoded'"
    )
    
    args = parser.parse_args()
    
    try:
        evaluate_single_model(args.model)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())