"""
Detection post-processing utilities.
Handles coordinate transformation, filtering, and formatting.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from .nms import apply_nms
from ..core.config import COCO_CLASS_NAMES, VEHICLE_CLASS_IDS

logger = logging.getLogger(__name__)


def postprocess_detections(
    detections: np.ndarray,
    original_shape: Tuple[int, int],
    input_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    cat_ids: List[int] = None
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
        
        cat_id = int(class_ids[idx]) if cat_ids is None else cat_ids[int(class_ids[idx])]
        results.append({
            "bbox": bbox,
            "score": float(scores[idx]),
            "class_id": int(class_ids[idx]),
            "category_id": cat_id,
        })
    
    return results


def convert_3output_to_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    original_shape: Tuple[int, int],
    input_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    cat_ids: List[int] = None
) -> List[Dict[str, Any]]:
    """
    Convert 3-output format (boxes, scores, class_ids) to detection list.
    
    This handles models that output 3 separate tensors instead of [N, 85].
    Boxes are in corner format (x1, y1, x2, y2) normalized to [0, 1].
    
    Args:
        boxes: Bounding boxes [N, 4] in corner format, normalized [0, 1]
        scores: Confidence scores [N]
        class_ids: Class IDs [N]
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
    
    all_boxes = []
    all_scores = []
    all_class_ids = []
    
    # Process each detection
    for i in range(len(scores)):
        score = float(scores[i])
        if score < conf_threshold:
            continue
        
        class_id = int(class_ids[i])
        
        # Convert corner format to center format, then to pixel coords
        x1_norm, y1_norm, x2_norm, y2_norm = boxes[i]
        
        # Scale to input_size (640)
        x1_scaled = x1_norm * input_size
        y1_scaled = y1_norm * input_size
        x2_scaled = x2_norm * input_size
        y2_scaled = y2_norm * input_size
        
        # Convert to center format
        cx = (x1_scaled + x2_scaled) / 2
        cy = (y1_scaled + y2_scaled) / 2
        bw = x2_scaled - x1_scaled
        bh = y2_scaled - y1_scaled
        
        # Adjust to original image space
        original_x1 = (cx - bw / 2 - pad_w) / scale
        original_y1 = (cy - bh / 2 - pad_h) / scale
        original_x2 = (cx + bw / 2 - pad_w) / scale
        original_y2 = (cy + bh / 2 - pad_h) / scale
        
        # Clamp to image boundaries
        original_x1 = max(0.0, min(original_x1, float(w)))
        original_y1 = max(0.0, min(original_y1, float(h)))
        original_x2 = max(0.0, min(original_x2, float(w)))
        original_y2 = max(0.0, min(original_y2, float(h)))
        
        # Calculate width and height
        box_width = original_x2 - original_x1
        box_height = original_y2 - original_y1
        
        # Filter out extremely small boxes
        if box_width < 1.0 or box_height < 1.0:
            continue
        
        all_boxes.append([original_x1, original_y1, original_x2, original_y2])
        all_scores.append(score)
        all_class_ids.append(class_id)
    
    if not all_boxes:
        return []
    
    # Apply NMS
    bboxes_np = np.array(all_boxes, dtype=np.float32)
    scores_np = np.array(all_scores, dtype=np.float32)
    class_ids_np = np.array(all_class_ids, dtype=np.int32)
    
    keep_indices = apply_nms(bboxes_np, scores_np, class_ids_np, iou_threshold)
    
    # Format results
    nms_results = []
    for i in keep_indices:
        x1, y1, x2, y2 = bboxes_np[i]
        cat_id = int(class_ids_np[i]) if cat_ids is None else cat_ids[int(class_ids_np[i])]
        nms_results.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(scores_np[i]),
            "class_id": int(class_ids_np[i]),
            "category_id": cat_id,
        })
    
    return nms_results


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
        vehicle_classes = VEHICLE_CLASS_IDS
    
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
        class_names = COCO_CLASS_NAMES
    
    formatted = []
    for det in detections:
        class_id = det.get("class_id", 0)
        formatted.append({
            "bbox": det["bbox"],
            "score": det.get("score", det.get("confidence", 0.0)),
            "class_id": class_id,
            "class_name": class_names[class_id] if class_id < len(class_names) else "unknown",
        })
    
    return formatted