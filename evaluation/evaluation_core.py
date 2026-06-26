"""
Core evaluation functionality for YOLOv5 models.
Computes COCO metrics: Precision, Recall, mAP50, mAP50-95.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import logging
import onnxruntime as ort
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from quantize.config import ONNX_PROVIDERS
from utils.postprocess import apply_nms
from utils.postprocess import apply_nms


logger = logging.getLogger(__name__)


class COCOEvaluator:
    """
    Evaluates YOLOv5 models on COCO dataset.
    Computes standard COCO metrics including mAP.
    """

    def __init__(
        self,
        model_path: Path,
        model_name: str,
        dataset_dir: Path,
        annotation_file: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_images: int = 500,
    ):
        """
        Initialize COCO evaluator.
        
        Args:
            model_path: Path to ONNX model
            model_name: Name for logging
            dataset_dir: Directory with test images
            annotation_file: COCO annotation JSON file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_images: Maximum number of images to evaluate
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.dataset_dir = Path(dataset_dir)
        self.annotation_file = Path(annotation_file)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_images = max_images
        
        # Load ONNX model
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=ONNX_PROVIDERS
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.input_type = self.session.get_inputs()[0].type
            self.output_names = [o.name for o in self.session.get_outputs()]
            logger.info(f"Loaded model: {model_name}, outputs: {self.output_names}")
        except Exception as e:
            raise RuntimeError(f"Cannot load model {model_name}: {e}")
        
        # Load COCO annotations
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"COCO annotations not found: {self.annotation_file}")
        self.coco_gt = COCO(str(self.annotation_file))
        self.cat_ids = sorted(self.coco_gt.getCatIds())
        
        # Storage for predictions
        self.predictions = []
        self.image_ids = []

    def preprocess_image(self, image_path: Path, input_size: int = 640) -> np.ndarray:
        """Preprocess image for model inference."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Letterbox padding
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

    def run_inference(self, input_tensor: np.ndarray) -> list:
        """Run model inference and return ALL output tensors."""
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return outputs

    def postprocess_detections(
        self,
        detections: np.ndarray,
        original_shape: Tuple[int, int],
        input_size: int = 640
    ) -> List[Dict[str, Any]]:
        """
        Post-process model outputs to COCO format.
        
        Args:
            detections: Raw model output. Supports formats:
                - [N, 85]: (x, y, w, h, conf, class_probs...)
                - [1, N, 4]: (batch, anchors, bbox_coords) - box only
            original_shape: Original image shape (h, w)
            input_size: Model input size
            
        Returns:
            List of detection dictionaries in COCO format
        """
        h, w = original_shape
        scale = min(input_size / h, input_size / w)
        pad_h = (input_size - int(h * scale)) // 2
        pad_w = (input_size - int(w * scale)) // 2
        
        results = []
        
        if len(detections) == 0:
            return results
        
        # Handle batch dimension
        if len(detections.shape) == 3:
            detections = detections[0]  # Remove batch dim
            
            # Initialize lists for bounding boxes, scores, and class IDs
            all_boxes = []
            all_scores = []
            all_class_ids = []

            # Process each detection
            for det in detections:
                if len(det) < 6: # Ensure detection has enough elements (x,y,w,h,conf,class_probs)
                    continue
                
                # Extract bounding box, objectness score, and class probabilities
                x_center, y_center, width, height = det[0:4] # Box coordinates (center_x, center_y, width, height)
                obj_conf = det[4] # Objectness confidence
                class_probs = det[5:] # Class probabilities

                # Determine the class_id with the highest probability
                if class_probs.size == 0:
                    continue # Skip if no class probabilities
                class_id = int(np.argmax(class_probs))

                # Calculate the final confidence score (objectness * class confidence)
                final_conf = float(obj_conf * class_probs[class_id])

                # Apply confidence threshold
                if final_conf < self.conf_threshold:
                    continue

                # Convert normalized box coordinates to absolute pixel coordinates relative to input_size (e.g., 640x640)
                x1_scaled = (x_center - width / 2) 
                y1_scaled = (y_center - height / 2) 
                x2_scaled = (x_center + width / 2) 
                y2_scaled = (y_center + height / 2) 

                # Scale coordinates back to original image size and adjust for letterbox padding
                original_x1 = (x1_scaled - pad_w) / scale
                original_y1 = (y1_scaled - pad_h) / scale
                original_x2 = (x2_scaled - pad_w) / scale
                original_y2 = (y2_scaled - pad_h) / scale

                # Clamp coordinates to image boundaries
                original_x1 = max(0.0, min(original_x1, float(w)))
                original_y1 = max(0.0, min(original_y1, float(h)))
                original_x2 = max(0.0, min(original_x2, float(w)))
                original_y2 = max(0.0, min(original_y2, float(h)))

                # Calculate width and height for the original image
                box_width = original_x2 - original_x1
                box_height = original_y2 - original_y1

                # Filter out extremely small or invalid boxes
                if box_width < 1.0 or box_height < 1.0:
                    continue

                all_boxes.append([original_x1, original_y1, box_width, box_height])
                all_scores.append(final_conf)
                all_class_ids.append(self.cat_ids[class_id])  # Map to actual COCO cat ID

            # Apply Non-Maximum Suppression (NMS) to filter redundant boxes
            if not all_boxes:
                return []

            # Convert lists to numpy arrays for NMS
            bboxes_np = np.array(all_boxes, dtype=np.float32)
            scores_np = np.array(all_scores, dtype=np.float32)
            class_ids_np = np.array(all_class_ids, dtype=np.int32)

            # Convert [x, y, w, h] to [x1, y1, x2, y2] format for NMS
            boxes_for_nms = bboxes_np.copy()
            boxes_for_nms[:, 2] = boxes_for_nms[:, 0] + boxes_for_nms[:, 2]  # x2 = x1 + w
            boxes_for_nms[:, 3] = boxes_for_nms[:, 1] + boxes_for_nms[:, 3]  # y2 = y1 + h
            
            keep_indices = apply_nms(boxes_for_nms, scores_np, class_ids_np, self.iou_threshold)

            # Filter results using the indices from NMS
            nms_results = []
            for i in keep_indices:
                nms_results.append({
                    "bbox": all_boxes[i],
                    "score": all_scores[i],
                    "category_id": all_class_ids[i],
                })
            return nms_results
        
        # If not 3D, assume already flat detections [N, 85]
        else:
            # This path is for models that directly output processed detections without batch dim
            # and with class probabilities (e.g., [x, y, w, h, obj_conf, class_prob1, ...])
            all_boxes = []
            all_scores = []
            all_class_ids = []

            for det in detections:
                if len(det) < 6:
                    continue

                x_center, y_center, width, height = det[0:4]
                obj_conf = det[4]
                class_probs = det[5:]

                if class_probs.size == 0:
                    continue
                class_id = int(np.argmax(class_probs))
                final_conf = float(obj_conf * class_probs[class_id])

                if final_conf < self.conf_threshold:
                    continue

                x1_scaled = (x_center - width / 2) 
                y1_scaled = (y_center - height / 2) 
                x2_scaled = (x_center + width / 2) 
                y2_scaled = (y_center + height / 2) 

                original_x1 = (x1_scaled - pad_w) / scale
                original_y1 = (y1_scaled - pad_h) / scale
                original_x2 = (x2_scaled - pad_w) / scale
                original_y2 = (y2_scaled - pad_h) / scale

                original_x1 = max(0.0, min(original_x1, float(w)))
                original_y1 = max(0.0, min(original_y1, float(h)))
                original_x2 = max(0.0, min(original_x2, float(w)))
                original_y2 = max(0.0, min(original_y2, float(h)))

                box_width = original_x2 - original_x1
                box_height = original_y2 - original_y1

                if box_width < 1.0 or box_height < 1.0:
                    continue

                all_boxes.append([original_x1, original_y1, box_width, box_height])
                all_scores.append(final_conf)
                all_class_ids.append(self.cat_ids[class_id])  # Map to actual COCO cat ID

            if not all_boxes:
                return []

            bboxes_np = np.array(all_boxes, dtype=np.float32)
            scores_np = np.array(all_scores, dtype=np.float32)
            class_ids_np = np.array(all_class_ids, dtype=np.int32)

            boxes_for_nms = bboxes_np.copy()
            boxes_for_nms[:, 2] = boxes_for_nms[:, 0] + boxes_for_nms[:, 2]  # x2 = x1 + w
            boxes_for_nms[:, 3] = boxes_for_nms[:, 1] + boxes_for_nms[:, 3]  # y2 = y1 + h

            keep_indices = apply_nms(boxes_for_nms, scores_np, class_ids_np, self.iou_threshold)

            nms_results = []
            for i in keep_indices:
                nms_results.append({
                    "bbox": all_boxes[i],
                    "score": all_scores[i],
                    "category_id": all_class_ids[i],
                })
            return nms_results



    def evaluate_image(self, image_path: Path, image_id: int) -> List[Dict[str, Any]]:
        """Evaluate a single image."""
        # Preprocess
        tensor = self.preprocess_image(image_path)
        
        # Inference – get ALL output tensors (boxes, scores, class_ids)
        outputs = self.run_inference(tensor)
        
        # Decoded models output 3 tensors: [boxes, scores, class_ids]
        # boxes: [1, N, 4], scores: [1, N], class_ids: [1, N]
        boxes, scores, class_ids = outputs[0], outputs[1], outputs[2]
        
        # Squeeze batch dim
        if len(boxes.shape) == 3:
            boxes = boxes[0]
        if len(scores.shape) > 1:
            scores = scores[0]
        if len(class_ids.shape) > 1:
            class_ids = class_ids[0]
        
        # Convert to a [N, 85] detection format for the postprocessor
        # The decoded models output boxes in corner format (x1, y1, x2, y2) normalized to [0, 1]
        # The postprocessor expects center format (cx, cy, w, h) in pixel coords relative to input_size (640)
        # So we need to:
        # 1. Convert corner -> center format
        # 2. Multiply by input_size to get pixel coords
        N = boxes.shape[0]
        detections = np.zeros((N, 85), dtype=np.float32)
        # Corner to center conversion
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        detections[:, 0] = cx * 640.0  # center_x in pixel coords
        detections[:, 1] = cy * 640.0  # center_y in pixel coords
        detections[:, 2] = bw * 640.0  # width in pixel coords
        detections[:, 3] = bh * 640.0  # height in pixel coords
        detections[:, 4] = scores  # objectness = score
        # Set class probability: use 1-hot at the class_id index PER ROW
        cls_indices = class_ids.astype(int)
        cls_indices = np.clip(cls_indices, 0, 79)
        detections[np.arange(N), 5 + cls_indices] = 1.0
        
        # Post-process with original image shape for correct scaling
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            logger.error(f"Could not read image for post-processing: {image_path}")
            return []
        original_shape = original_image.shape[:2]
        
        results = self.postprocess_detections(detections, original_shape)
        
        # Add image_id to results
        for result in results:
            result["image_id"] = image_id
        
        return results

    def evaluate_dataset(self) -> Tuple[List[Dict], List[int]]:
        """
        Evaluate entire dataset.
        
        Returns:
            Tuple of (predictions, image_ids)
        """
        # Get all image IDs from COCO dataset (sorted for consistency)
        img_ids = sorted(self.coco_gt.getImgIds())
        
        # Limit to max_images
        if self.max_images > 0:
            img_ids = img_ids[:self.max_images]
        
        logger.info(f"Evaluating {len(img_ids)} images...")
        
        all_predictions = []
        self.image_ids = img_ids
        
        for idx, img_id in enumerate(img_ids):
            # Get image info
            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = self.dataset_dir / img_info["file_name"]
            
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            
            # Evaluate image
            try:
                predictions = self.evaluate_image(img_path, img_id)
                all_predictions.extend(predictions)
            except Exception as e:
                logger.error(f"Error evaluating {img_path}: {e}")
                continue
            
            if (idx + 1) % 50 == 0:
                logger.info(f"  Processed {idx + 1}/{len(img_ids)} images")
        
        self.predictions = all_predictions
        logger.info(f"Total predictions: {len(all_predictions)}")
        
        return all_predictions, img_ids
    
    def get_image_ids_from_coco(self, max_images: int = 1000) -> List[int]:
        """
        Get image IDs from COCO annotations in sorted order.
        
        This ensures the same images are used for both benchmarking and evaluation.
        
        Args:
            max_images: Maximum number of images to select
            
        Returns:
            List of image IDs in sorted order
        """
        # Get all image IDs sorted (same order as COCO)
        img_ids = sorted(self.coco_gt.getImgIds())
        
        # Limit to max_images
        if max_images > 0:
            img_ids = img_ids[:max_images]
        
        logger.info(f"Selected {len(img_ids)} images from COCO annotations")
        return img_ids

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute COCO evaluation metrics.
        
        Returns:
            Dictionary with metrics: precision, recall, mAP50, mAP50-95
        """
        if not self.predictions:
            raise RuntimeError("No predictions available. Run evaluate_dataset() first.")
        
        logger.info("Computing COCO metrics...")
        
        # Create COCO results object
        coco_dt = self.coco_gt.loadRes(self.predictions)
        
        # Initialize COCO evaluator
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
        
        # Set evaluation parameters
        coco_eval.params.imgIds = self.image_ids
        coco_eval.params.catIds = self.cat_ids
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        precision = float(coco_eval.stats[0])  # AP at IoU=.50:.95
        recall = float(coco_eval.stats[8])  # AR at IoU=.50:.95
        map50 = float(coco_eval.stats[1])  # AP at IoU=.50
        map50_95 = float(coco_eval.stats[0])  # AP at IoU=.50:.95
        
        # Compute F1-score from precision and recall
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "map50": map50,
            "map50_95": map50_95,
        }
        
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-score: {metrics['f1_score']:.4f}")
        logger.info(f"mAP@0.50: {metrics['map50']:.4f}")
        logger.info(f"mAP@0.50:0.95: {metrics['map50_95']:.4f}")
        
        return metrics

    def evaluate(self) -> Dict[str, Any]:
        """
        Run full evaluation pipeline.
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"{'='*60}")
        logger.info(f"Evaluating: {self.model_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # Run evaluation
        predictions, image_ids = self.evaluate_dataset()
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        elapsed_time = time.time() - start_time
        
        result = {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": str(self.dataset_dir),
            "num_images": len(image_ids),
            "num_predictions": len(predictions),
            "elapsed_time_sec": elapsed_time,
            "config": {
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "max_images": self.max_images,
            },
            "metrics": metrics,
        }
        
        logger.info(f"Evaluation completed in {elapsed_time:.2f}s")
        
        return result


def evaluate_model(
    model_path: Path,
    model_name: str,
    dataset_dir: Path,
    annotation_file: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_images: int = 500,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a single model.
    
    Args:
        model_path: Path to ONNX model
        model_name: Name for logging
        dataset_dir: Directory with test images
        annotation_file: COCO annotation JSON file
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        max_images: Maximum images to evaluate
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = COCOEvaluator(
        model_path=model_path,
        model_name=model_name,
        dataset_dir=dataset_dir,
        annotation_file=annotation_file,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_images=max_images,
    )
    
    return evaluator.evaluate()


def compute_coco_metrics(predictions: List[Dict], ground_truth: COCO) -> Dict[str, float]:
    """
    Compute COCO metrics from predictions and ground truth.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: COCO ground truth object
        
    Returns:
        Dictionary with metrics
    """
    coco_dt = ground_truth.loadRes(predictions)
    coco_eval = COCOeval(ground_truth, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return {
        "precision": float(coco_eval.stats[0]),
        "recall": float(coco_eval.stats[8]),
        "map50": float(coco_eval.stats[1]),
        "map50_95": float(coco_eval.stats[0]),
    }