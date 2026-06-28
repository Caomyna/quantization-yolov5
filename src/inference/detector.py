"""
YOLO detector for object detection.
Uses BaseONNXModel and shared preprocessing/postprocessing modules.
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from ..core.base import BaseONNXModel
from ..preprocessing.preprocessor import preprocess_image
from ..postprocessing.detections import (
    postprocess_detections,
    convert_3output_to_detections,
    filter_vehicles,
    format_detections,
)
from ..core.config import COCO_CLASS_NAMES, VEHICLE_CLASS_IDS

logger = logging.getLogger(__name__)


class YOLODetector(BaseONNXModel):
    """
    YOLOv5/YOLOv8 detector wrapper for ONNX models.
    Supports vehicle detection and general object detection.
    """
    
    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        vehicle_classes: List[int] = None,
        providers: List[str] = None,
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to ONNX model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            vehicle_classes: List of vehicle class IDs (COCO format)
            providers: ONNX Runtime providers
        """
        super().__init__(model_path, providers, conf_threshold, iou_threshold)
        self.vehicle_classes = vehicle_classes or VEHICLE_CLASS_IDS
        self.class_names = COCO_CLASS_NAMES
    
    def preprocess(self, image: np.ndarray, input_size: int = 640) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image (BGR format from OpenCV)
            input_size: Model input size (default: 640)
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        dtype = self.get_input_dtype()
        return preprocess_image(image, input_size=input_size, dtype=dtype)
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a single image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detection dictionaries with keys: bbox, confidence, class_id, class_name
        """
        # Preprocess
        tensor = self.preprocess(image)
        
        # Inference - get ALL output tensors
        outputs = self.run(tensor)
        
        # Post-process
        results = self.postprocess(outputs, image.shape[:2])
        
        # Format with class names
        results = format_detections(results, self.class_names)
        
        return results
    
    def postprocess(
        self,
        outputs: List[np.ndarray],
        original_shape: Tuple[int, int],
        input_size: int = 640
    ) -> List[Dict[str, Any]]:
        """
        Post-process model outputs.
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (h, w)
            input_size: Model input size
            
        Returns:
            List of detection dictionaries
        """
        # Handle 3-output format (boxes, scores, class_ids)
        if len(outputs) == 3:
            boxes, scores, class_ids = outputs[0], outputs[1], outputs[2]
            
            # Squeeze batch dimension
            if len(boxes.shape) == 3:
                boxes = boxes[0]
            if len(scores.shape) > 1:
                scores = scores[0]
            if len(class_ids.shape) > 1:
                class_ids = class_ids[0]
            
            return convert_3output_to_detections(
                boxes, scores, class_ids,
                original_shape, input_size,
                self.conf_threshold, self.iou_threshold
            )
        
        # Single output format [N, 85]
        else:
            detections = outputs[0]
            if len(detections.shape) == 3:
                detections = detections[0]
            
            return postprocess_detections(
                detections, original_shape, input_size,
                self.conf_threshold, self.iou_threshold
            )
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect only vehicles in the image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of vehicle detection dictionaries
        """
        all_detections = self.detect(image)
        
        # Filter to vehicles only
        vehicle_detections = filter_vehicles(all_detections, self.vehicle_classes)
        
        return vehicle_detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info["vehicle_classes"] = self.vehicle_classes
        return info