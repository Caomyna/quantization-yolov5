"""
Evaluation engine - COCO evaluation for YOLO models.
Uses BaseONNXModel to eliminate duplicated session creation.
"""

import time
import json
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..core.base import BaseONNXModel
from ..preprocessing.preprocessor import preprocess_image
from ..postprocessing.detections import postprocess_detections, convert_3output_to_detections
from ..core.config import COCO_CLASS_NAMES

logger = logging.getLogger(__name__)


class EvaluationEngine(BaseONNXModel):
    """
    Evaluation engine for YOLO models.
    Extends BaseONNXModel to add COCO evaluation capabilities.
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
        providers: List[str] = None,
    ):
        """
        Initialize evaluation engine.
        
        Args:
            model_path: Path to ONNX model
            model_name: Name for logging
            dataset_dir: Directory with test images
            annotation_file: COCO annotation JSON file
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_images: Maximum images to evaluate
            providers: ONNX Runtime providers
        """
        super().__init__(model_path, providers, conf_threshold, iou_threshold)
        self.model_name = model_name
        self.dataset_dir = Path(dataset_dir)
        self.annotation_file = Path(annotation_file)
        self.max_images = max_images
        
        # Load COCO annotations
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"COCO annotations not found: {self.annotation_file}")
        self.coco_gt = COCO(str(self.annotation_file))
        self.cat_ids = sorted(self.coco_gt.getCatIds())
        
        # Storage for predictions
        self.predictions = []
        self.image_ids = []
    
    def preprocess_image(self, image_path: Path, input_size: int = 640) -> np.ndarray:
        """
        Preprocess image for evaluation.
        
        Args:
            image_path: Path to image
            input_size: Model input size
            
        Returns:
            Preprocessed tensor
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        dtype = self.get_input_dtype()
        return preprocess_image(img, input_size=input_size, dtype=dtype)
    
    def run_inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """
        Run model inference and return ALL output tensors.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            List of output tensors
        """
        return self.run(input_tensor)
    
    def postprocess_detections(
        self,
        outputs: List[np.ndarray],
        original_shape: Tuple[int, int],
        input_size: int = 640
    ) -> List[Dict[str, Any]]:
        """
        Post-process model outputs to COCO format.
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (h, w)
            input_size: Model input size
            
        Returns:
            List of detection dictionaries in COCO format
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
                self.conf_threshold, self.iou_threshold,
                self.cat_ids
            )
        
        # Single output format [N, 85]
        else:
            detections = outputs[0]
            if len(detections.shape) == 3:
                detections = detections[0]
            
            return postprocess_detections(
                detections, original_shape, input_size,
                self.conf_threshold, self.iou_threshold,
                self.cat_ids
            )
    
    def evaluate_image(self, image_path: Path, image_id: int) -> List[Dict[str, Any]]:
        """
        Evaluate a single image.
        
        Args:
            image_path: Path to image
            image_id: COCO image ID
            
        Returns:
            List of detection dictionaries
        """
        # Preprocess
        tensor = self.preprocess_image(image_path)
        
        # Inference
        outputs = self.run_inference(tensor)
        
        # Read original image for shape
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            logger.error(f"Could not read image for post-processing: {image_path}")
            return []
        original_shape = original_image.shape[:2]
        
        # Post-process
        results = self.postprocess_detections(outputs, original_shape)
        
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
        
        # Compute F1-score
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