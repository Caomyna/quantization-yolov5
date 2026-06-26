"""
Evaluate all models in weights/ directory and generate comparison reports.
Scans for FP32/FP16 model pairs, runs proper COCO evaluation, and exports results.
Each row in the Excel represents one model variant with standard COCO metrics.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from typing import Dict, List, Any
import pandas as pd

from evaluation.evaluation_core import evaluate_model
from quantize.config import MODELS_DIR, DATASET_DIR, ANNOTATION_FILE, MODEL_CONFIG


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
    Evaluate a single FP32 vs FP16 model pair using proper COCO evaluation.
    
    Uses the same evaluation pipeline for both precision types, ensuring
    identical preprocessing, NMS, and COCO metric computation.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    conf_threshold = config.get("conf_threshold", 0.25)
    iou_threshold = config.get("iou_threshold", 0.45)
    
    # Evaluate FP32 using the proper COCO pipeline
    print(f"Running FP32 evaluation (max {max_images} images)...")
    fp32_eval = evaluate_model(
        fp32_path,
        f"{model_name}_FP32",
        dataset_dir,
        annotation_file,
        conf_threshold,
        iou_threshold,
        max_images
    )
    
    # Evaluate FP16 using the same COCO pipeline
    print(f"Running FP16 evaluation (max {max_images} images)...")
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
            print(f"Warning: FP16 evaluation skipped - {e}")
    else:
        print(f"Warning: FP16 model not found: {fp16_path}")
    
    # Print summary
    print(f"\nEvaluation results for {model_name}:")
    if fp32_eval:
        m = fp32_eval.get('metrics', {})
        print(f"  FP32 - Precision: {m.get('precision', 0):.4f}, Recall: {m.get('recall', 0):.4f}, "
              f"F1: {m.get('f1_score', 0):.4f}, mAP50: {m.get('map50', 0):.4f}, "
              f"mAP50-95: {m.get('map50_95', 0):.4f}")
    if fp16_eval:
        m = fp16_eval.get('metrics', {})
        print(f"  FP16 - Precision: {m.get('precision', 0):.4f}, Recall: {m.get('recall', 0):.4f}, "
              f"F1: {m.get('f1_score', 0):.4f}, mAP50: {m.get('map50', 0):.4f}, "
              f"mAP50-95: {m.get('map50_95', 0):.4f}")
    
    return {
        "model_name": model_name,
        "fp32_eval": fp32_eval,
        "fp16_eval": fp16_eval,
    }


def save_all_results(eval_results: List[Dict[str, Any]], output_dir: Path):
    """Save individual JSON files and summary Excel file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual evaluation JSON files
    for result in eval_results:
        model_name = result["model_name"]
        json_path = output_dir / f"{model_name}_evaluation_results.json"
        
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "fp32_evaluation": result["fp32_eval"],
            "fp16_evaluation": result["fp16_eval"],
        }
        
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {json_path}")
    
    # Create summary Excel file — one row per model variant with standard COCO metrics
    excel_path = output_dir / "evaluation_summary.xlsx"
    
    flat_data = []
    for result in eval_results:
        model_name = result["model_name"]
        fp32_eval = result["fp32_eval"]
        fp16_eval = result["fp16_eval"]
        
        # FP32 row
        if fp32_eval:
            fp32_metrics = fp32_eval.get("metrics", {})
            flat_data.append({
                "Model": f"{model_name}.onnx",
                "Precision Type": "FP32",
                "Precision": fp32_metrics.get("precision", 0),
                "Recall": fp32_metrics.get("recall", 0),
                "F1-score": fp32_metrics.get("f1_score", 0),
                "mAP@0.50": fp32_metrics.get("map50", 0),
                "mAP@0.50:0.95": fp32_metrics.get("map50_95", 0),
                "Num Images": fp32_eval.get("num_images", 0),
                "Num Predictions": fp32_eval.get("num_predictions", 0),
                "Eval Time (s)": fp32_eval.get("elapsed_time_sec", 0),
                "Notes": "",
            })
        
        # FP16 row
        if fp16_eval:
            fp16_metrics = fp16_eval.get("metrics", {})
            fp32_metrics = fp32_eval.get("metrics", {}) if fp32_eval else {}
            
            map50_diff = fp32_metrics.get("map50", 0) - fp16_metrics.get("map50", 0)
            map50_95_diff = fp32_metrics.get("map50_95", 0) - fp16_metrics.get("map50_95", 0)
            
            flat_data.append({
                "Model": f"{model_name}_fp16.onnx",
                "Precision Type": "FP16",
                "Precision": fp16_metrics.get("precision", 0),
                "Recall": fp16_metrics.get("recall", 0),
                "F1-score": fp16_metrics.get("f1_score", 0),
                "mAP@0.50": fp16_metrics.get("map50", 0),
                "mAP@0.50:0.95": fp16_metrics.get("map50_95", 0),
                "Num Images": fp16_eval.get("num_images", 0),
                "Num Predictions": fp16_eval.get("num_predictions", 0),
                "Eval Time (s)": fp16_eval.get("elapsed_time_sec", 0),
                "Notes": f"mAP50 diff: {map50_diff:.4f}, mAP50-95 diff: {map50_95_diff:.4f}",
            })
    
    df_flat = pd.DataFrame(flat_data)
    
    # Write to Excel with formatting
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_flat.to_excel(writer, sheet_name='Evaluation Results', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Evaluation Results']
        
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
    print(f"  - Sheet 1: Evaluation Results ({len(df_flat)} rows)")


def main():
    """Main function to evaluate all model pairs."""
    print("="*60)
    print("EVALUATING ALL MODELS")
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
    
    # Find COCO annotations (for proper COCO evaluation)
    annotation_file = ANNOTATION_FILE
    if not annotation_file.exists():
        annotation_file = ANNOTATION_FILE
    
    annotations_exist = annotation_file.exists()
    print(f"COCO annotations: {'✓' if annotations_exist else '✗'} {annotation_file}")
    if not annotations_exist:
        print("ERROR: COCO annotations required for evaluation!")
        return
    
    # Run evaluations
    eval_results = []
    
    for pair in model_pairs:
        if not pair["fp32"].exists():
            print(f"\nSkipping {pair['name']} - FP32 model not found")
            continue
        
        try:
            result = evaluate_model_pair(
                fp32_path=pair["fp32"],
                fp16_path=pair["fp16"],
                model_name=pair["name"],
                dataset_dir=DATASET_DIR,
                annotation_file=annotation_file,
                config=MODEL_CONFIG,
                max_images=1000,
            )
            eval_results.append(result)
        except Exception as e:
            print(f"\nERROR processing {pair['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    if eval_results:
        output_dir = Path("reports")
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")
        save_all_results(eval_results, output_dir)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed {len(eval_results)} model pair(s)")
        print(f"\nOutput files:")
        for result in eval_results:
            json_file = f"reports/{result['model_name']}_evaluation_results.json"
            print(f"  - {json_file}")
        print(f"  - reports/evaluation_summary.xlsx")
    else:
        print("\nNo successful evaluations completed!")


if __name__ == "__main__":
    main()