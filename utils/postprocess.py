"""
Post-processing utilities for YOLO detections.
Handles NMS, filtering, and detection formatting.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


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


def postprocess_detections(
    detections: np.ndarray,
    original_shape: Tuple[int, int],
    input_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> List[Dict[str, Any]]:
    """
    Post-process raw model detections.
    
    Args:
        detections: Raw model output [N, 85] (x, y, w, h, conf, class_probs...)
        original_shape: Original image shape (h, w)
        input_size: Model input size
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of detection dictionaries
    """
    h, w = original_shape
    scale = min(input_size / h, input_size / w)
    pad_h = (input_size - int(h * scale)) // 2
    pad_w = (input_size - int(w * scale)) // 2
    
    results = []
    
    if len(detections) == 0:
        return results
    
    # Parse detections
    boxes = []
    scores = []
    class_ids = []
    
    for det in detections:
        if len(det) < 6:
            continue
        
        x, y, w_box, h_box = det[0:4]
        conf = det[4]
        class_probs = det[5:]
        
        # Get class with highest probability
        class_id = int(np.argmax(class_probs))
        final_conf = float(conf * class_probs[class_id])
        
        if final_conf < conf_threshold:
            continue
        
        # Convert to original image coordinates (corner format for NMS)
        x1 = (x - pad_w) / scale
        y1 = (y - pad_h) / scale
        x2 = (x + w_box - pad_w) / scale
        y2 = (y + h_box - pad_h) / scale
        
        # Clamp to image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        boxes.append([x1, y1, x2, y2])
        scores.append(final_conf)
        class_ids.append(class_id)
    
    if len(boxes) == 0:
        return results
    
    # Apply NMS
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    keep_indices = apply_nms(boxes, scores, class_ids, iou_threshold)
    
    # Format results
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes[idx]
        
        # Convert to [x, y, w, h] format
        bbox = [x1, y1, x2 - x1, y2 - y1]
        
        results.append({
            "bbox": bbox,
            "confidence": float(scores[idx]),
            "class_id": int(class_ids[idx]),
        })
    
    return results


def filter_vehicles(
    detections: List[Dict[str, Any]],
    vehicle_classes: List[int] = None
) -> List[Dict[str, Any]]:
    """
    Filter detections to vehicles only.
    
    Args:
        detections: List of detection dictionaries
        vehicle_classes: List of vehicle class IDs (default: COCO vehicle classes)
        
    Returns:
        Filtered list of vehicle detections
    """
    if vehicle_classes is None:
        vehicle_classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
    
    vehicle_detections = [
        det for det in detections
        if det.get("class_id") in vehicle_classes
    ]
    
    return vehicle_detections


def filter_by_class(
    detections: List[Dict[str, Any]],
    class_ids: List[int]
) -> List[Dict[str, Any]]:
    """
    Filter detections by specific class IDs.
    
    Args:
        detections: List of detection dictionaries
        class_ids: List of class IDs to keep
        
    Returns:
        Filtered list of detections
    """
    filtered = [
        det for det in detections
        if det.get("class_id") in class_ids
    ]
    
    return filtered


def format_detections(
    detections: List[Dict[str, Any]],
    class_names: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Format detections with class names.
    
    Args:
        detections: List of detection dictionaries
        class_names: List of class names (COCO format)
        
    Returns:
        Formatted detections with class_name field
    """
    if class_names is None:
        class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
    
    formatted = []
    for det in detections:
        class_id = det.get("class_id", 0)
        formatted.append({
            "bbox": det["bbox"],
            "confidence": det["confidence"],
            "class_id": class_id,
            "class_name": class_names[class_id] if class_id < len(class_names) else "unknown",
        })
    
    return formatted