"""
Evaluation module for YOLOv5 model accuracy assessment.
Computes COCO metrics: Precision, Recall, mAP50, mAP50-95.
"""

from .evaluation_core import (
    COCOEvaluator,
    evaluate_model,
    compute_coco_metrics,
)

__all__ = [
    "COCOEvaluator",
    "evaluate_model",
    "compute_coco_metrics",
]