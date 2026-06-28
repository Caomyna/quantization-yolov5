"""
Evaluation module - COCO metrics computation for YOLO models.
"""

from .evaluator import EvaluationEngine
from .run import evaluate_model
from .reporter import save_evaluation_results, save_evaluation_excel

__all__ = ['EvaluationEngine', 'evaluate_model', 'save_evaluation_results', 'save_evaluation_excel']