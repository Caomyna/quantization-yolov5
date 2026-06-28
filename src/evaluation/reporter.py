"""
Evaluation reporter - Save evaluation results to JSON and Excel.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any


def save_evaluation_results(
    eval_results: Dict[str, Any],
    output_path: Path
):
    """
    Save evaluation results as JSON.
    
    Args:
        eval_results: Evaluation results dictionary
        output_path: Output JSON path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": eval_results.get("model_name"),
        "fp32_evaluation": eval_results.get("fp32_eval"),
        "fp16_evaluation": eval_results.get("fp16_eval"),
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Results saved: {output_path}")


def save_evaluation_excel(
    eval_results: List[Dict[str, Any]],
    output_dir: Path
):
    """
    Save evaluation results as Excel file.
    
    Args:
        eval_results: List of evaluation result dictionaries
        output_dir: Output directory
    """
    import pandas as pd
    from openpyxl.styles import Font, Alignment
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary Excel file
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
    logger.info(f"  - Sheet 1: Evaluation Results ({len(df_flat)} rows)")