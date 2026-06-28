"""
Non-Maximum Suppression implementation.
Single implementation used by all modules.
"""

import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


def compute_iou(box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between one box and multiple boxes.
    
    Args:
        box1: Single box [4] in format [x1, y1, x2, y2]
        boxes2: Multiple boxes [N, 4] in format [x1, y1, x2, y2]
        
    Returns:
        IoU values [N]
    """
    # Area of box1
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    
    # Areas of boxes2
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection coordinates
    x1 = np.maximum(box1[0], boxes2[:, 0])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y2 = np.minimum(box1[3], boxes2[:, 3])
    
    # Intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # IoU
    iou = intersection / (area1 + areas2 - intersection + 1e-6)
    
    return iou


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Apply Non-Maximum Suppression (NMS) to detections.
    
    Args:
        boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
        scores: Confidence scores [N]
        class_ids: Class IDs [N]
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score (descending)
    indices = np.argsort(-scores)
    
    keep = []
    while len(indices) > 0:
        # Pick the box with highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Get IoU with remaining boxes
        ious = compute_iou(boxes[current], boxes[indices[1:]])
        
        # Keep boxes with IoU below threshold (and same class)
        same_class = class_ids[indices[1:]] == class_ids[current]
        keep_indices = np.where((ious < iou_threshold) | ~same_class)[0]
        
        # Update indices (skip first element which we kept)
        indices = indices[keep_indices + 1]
    
    return keep