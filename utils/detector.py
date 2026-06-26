"""
YOLO model wrapper for object detection.
Handles model loading, inference, and detection formatting.
"""

import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from quantize.config import ONNX_PROVIDERS
from utils.postprocess import apply_nms


logger = logging.getLogger(__name__)


class YOLODetector:
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
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to ONNX model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            vehicle_classes: List of vehicle class IDs (COCO format)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.vehicle_classes = vehicle_classes or [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
        
        # Load model
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=ONNX_PROVIDERS
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.input_type = self.session.get_inputs()[0].type
            self.output_name = self.session.get_outputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            logger.info(f"Loaded model: {self.model_path.name}")
        except Exception as e:
            raise RuntimeError(f"Cannot load model {self.model_path}: {e}")
        
        # COCO class names
        self.class_names = [
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

    def preprocess(self, image: np.ndarray, input_size: int = 640) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image (BGR format from OpenCV)
            input_size: Model input size (default: 640)
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Letterbox resize
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to input_size
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_h, pad_w = (input_size - new_h) // 2, (input_size - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img
        
        # Normalize and transpose
        tensor = padded.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :]
        
        # Match model dtype
        if "float16" in self.input_type.lower():
            tensor = tensor.astype(np.float16)
        
        return tensor

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
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        
        # Decoded models output 3 tensors: [boxes, scores, class_ids]
        # boxes: [1, N, 4], scores: [1, N], class_ids: [1, N]
        if len(outputs) == 3:
            boxes, scores, class_ids = outputs[0], outputs[1], outputs[2]
            
            # Squeeze batch dimension
            if len(boxes.shape) == 3:
                boxes = boxes[0]
            if len(scores.shape) > 1:
                scores = scores[0]
            if len(class_ids.shape) > 1:
                class_ids = class_ids[0]
            
            # Convert to [N, 85] format for postprocessing
            # boxes are in corner format (x1, y1, x2, y2) normalized to [0, 1]
            # Need to convert to center format (cx, cy, w, h) in pixel coords
            N = boxes.shape[0]
            detections = np.zeros((N, 85), dtype=np.float32)
            
            # Corner to center conversion, scale to input_size (640)
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            detections[:, 0] = cx * 640.0
            detections[:, 1] = cy * 640.0
            detections[:, 2] = bw * 640.0
            detections[:, 3] = bh * 640.0
            detections[:, 4] = scores  # objectness = score
            
            # Set class probability: 1-hot at class_id index
            cls_indices = class_ids.astype(int)
            cls_indices = np.clip(cls_indices, 0, 79)
            detections[np.arange(N), 5 + cls_indices] = 1.0
            
            logger.debug(f"Converted 3-output format to {detections.shape}")
        else:
            # Single output format [N, 85] or [1, N, 85]
            detections = outputs[0]
            if len(detections.shape) == 3:
                detections = detections[0]
        
        # Debug logging
        logger.debug(f"Raw output shape: {detections.shape}")
        logger.debug(f"Raw output dtype: {detections.dtype}")
        if len(detections) > 0:
            logger.debug(f"Raw output sample: {detections[0][:10]}")
        
        # Post-process
        results = self.postprocess(detections, image.shape[:2])
        
        return results

    def postprocess(
        self,
        detections: np.ndarray,
        original_shape: Tuple[int, int],
        input_size: int = 640
    ) -> List[Dict[str, Any]]:
        """
        Post-process model outputs.
        
        Args:
            detections: Raw model output [N, 85] (x, y, w, h, conf, class_probs...)
            original_shape: Original image shape (h, w)
            input_size: Model input size
            
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
        
        # Log detection shape for debugging
        logger.debug(f"Detection shape: {detections.shape}")
        logger.debug(f"Detection dtype: {detections.dtype}")
        
        # Handle different output shapes
        # Format: [batch, N, 4] where 4 = [x, y, w, h]
        if len(detections.shape) == 3:
            detections = detections[0]  # Remove batch dimension, now [N, 4]
        
        # All models should output in [N, 85] format (x, y, w, h, conf, class_probs...) after decoding
        # If it's a raw YOLO output, it needs to be decoded first. Assume decoded.
        
        for det in detections:
            # Ensure detection has enough elements
            if len(det) < 6:
                continue
            
            # Extract bounding box, confidence, and class probabilities
            x_center, y_center, width, height = det[0:4]
            confidence = det[4]  # Objectness score
            class_probs = det[5:] # Class probabilities
            
            # Determine the class_id with the highest probability
            if class_probs.size == 0:
                # If no class probabilities, skip or assign a default/unknown class
                continue
            class_id = int(np.argmax(class_probs))
            
            # Calculate final confidence score (objectness * class confidence)
            final_conf = float(confidence * class_probs[class_id])
            
            # Apply confidence threshold
            if final_conf < self.conf_threshold:
                continue
            
            # Convert normalized box coordinates to absolute pixel coordinates
            # The model outputs coordinates relative to the input_size (640x640)
            x1_scaled = (x_center - width / 2) # * input_size
            y1_scaled = (y_center - height / 2) # * input_size
            x2_scaled = (x_center + width / 2) # * input_size
            y2_scaled = (y_center + height / 2) # * input_size
            
            # Adjust coordinates to original image size (undo letterbox padding and scaling)
            # x = (x_model - pad_w) / scale
            # y = (y_model - pad_h) / scale
            
            # Note: The decoded ONNX models output absolute pixel coordinates relative to the input image size (640x640)
            # We need to scale them back to the original image dimensions.
            
            # Calculate coordinates in the original image space
            original_x1 = (x1_scaled - pad_w) / scale
            original_y1 = (y1_scaled - pad_h) / scale
            original_x2 = (x2_scaled - pad_w) / scale
            original_y2 = (y2_scaled - pad_h) / scale
            
            # Clamp coordinates to image boundaries
            original_x1 = max(0, min(original_x1, w))
            original_y1 = max(0, min(original_y1, h))
            original_x2 = max(0, min(original_x2, w))
            original_y2 = max(0, min(original_y2, h))
            
            # Calculate width and height in original image space
            box_width = original_x2 - original_x1
            box_height = original_y2 - original_y1
            
            # Filter out extremely small or invalid boxes
            if box_width < 1 or box_height < 1:
                continue
            
            results.append({
                "bbox": [original_x1, original_y1, box_width, box_height], # COCO format: [x, y, w, h]
                "confidence": float(final_conf),
                "class_id": class_id, # Class ID is 0-indexed
            })
        
        # Apply Non-Maximum Suppression (NMS) to filter redundant boxes
        # We need to convert results to numpy arrays for NMS
        if not results:
            return []

        bboxes = np.array([res["bbox"] for res in results], dtype=np.float32)
        scores = np.array([res["confidence"] for res in results], dtype=np.float32)
        class_ids = np.array([res["class_id"] for res in results], dtype=np.int32)

        # Convert [x, y, w, h] to [x1, y1, x2, y2] for NMS
        boxes_for_nms = bboxes.copy()
        boxes_for_nms[:, 2] = boxes_for_nms[:, 0] + boxes_for_nms[:, 2]  # x2 = x1 + w
        boxes_for_nms[:, 3] = boxes_for_nms[:, 1] + boxes_for_nms[:, 3]  # y2 = y1 + h
        
        keep_indices = apply_nms(boxes_for_nms, scores, class_ids, self.iou_threshold)

        # Filter results using NMS indices
        nms_results = []
        for i in keep_indices:
            result = results[i]
            result["class_name"] = self.class_names[result["class_id"]] # Add class name
            nms_results.append(result)

        return nms_results

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
        vehicle_detections = [
            det for det in all_detections
            if det["class_id"] in self.vehicle_classes
        ]
        
        return vehicle_detections

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "input_type": self.input_type,
            "output_name": self.output_name,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "vehicle_classes": self.vehicle_classes,
        }