"""
Evaluate models with custom YOLO-format labels.
Supports datasets with images/ and labels/ folders in YOLO format.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import logging
from typing import Dict, List, Any
import numpy as np
from utils.detector import YOLODetector
import cv2

logging.basicConfig(level=logging.INFO)

from evaluation.evaluation_core import evaluate_model
from quantize.config import MODELS_DIR, DATASET_DIR, MODEL_CONFIG


def parse_yolo_label(label_path: Path, img_width: int, img_height: int) -> List[Dict[str, Any]]:
    """
    Parse YOLO format label file.
    
    Format: class_id x_center y_center width height (all normalized)
    
    Args:
        label_path: Path to label file
        img_width: Image width
        img_height: Image height
        
    Returns:
        List of ground truth annotations
    """
    if not label_path.exists():
        return []
    
    annotations = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert normalized to pixel coordinates
            x1 = (x_center - width / 2) * img_width
            y1 = (y_center - height / 2) * img_height
            x2 = (x_center + width / 2) * img_width
            y2 = (y_center + height / 2) * img_height
            
            annotations.append({
                "class_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # [x, y, w, h]
                "area": (x2 - x1) * (y2 - y1),
            })
    
    return annotations


def evaluate_model_custom(
    model_path: Path,
    model_name: str,
    dataset_dir: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_images: int = 100,
) -> Dict[str, Any]:
    """
    Evaluate model with custom YOLO-format dataset.
    
    Args:
        model_path: Path to ONNX model
        model_name: Name for logging
        dataset_dir: Directory with images/ and labels/ subdirs
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        max_images: Maximum images to evaluate
        
    Returns:
        Evaluation results
    """
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Initialize detector with lower threshold for debugging
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf_threshold, 
        iou_threshold=iou_threshold,
    )
    
    # Get image files
    image_files = sorted(list(images_dir.glob("*.jpg")) + 
                       list(images_dir.glob("*.jpeg")) + 
                       list(images_dir.glob("*.png")))
    
    if max_images > 0:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images")
    print(f"Running evaluation (max {max_images} images)...")
    
    # Metrics
    total_predictions = 0
    total_ground_truth = 0
    correct_predictions = 0
    
    start_time = time.time()
    
    for idx, img_path in enumerate(image_files):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Run detection
        detections = detector.detect(image)
        
        # Debug logging
        if idx < 5:  # Log first 5 images
            print(f"  Image {img_path.name}: {len(detections)} detections")
            if len(detections) > 0:
                print(f"    Sample detection: {detections[0]}")
        
        # Load ground truth
        label_path = labels_dir / f"{img_path.stem}.txt"
        ground_truth = parse_yolo_label(label_path, img_width, img_height)
        
        # Count
        total_predictions += len(detections)
        total_ground_truth += len(ground_truth)
        
        # Simple matching: check if any detection matches any ground truth
        for gt in ground_truth:
            gt_bbox = np.array(gt["bbox"])
            gt_class = gt["class_id"]
            
            for det in detections:
                det_bbox = np.array(det["bbox"])
                det_class = det["class_id"]
                
                # Check class match
                if gt_class != det_class:
                    continue
                
                # Check IoU
                iou = compute_iou_simple(gt_bbox, det_bbox)
                if iou >= 0.5:
                    correct_predictions += 1
                    break
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(image_files)} images")
    
    elapsed_time = time.time() - start_time
    
    # Compute metrics
    precision = correct_predictions / total_predictions if total_predictions > 0 else 0
    recall = correct_predictions / total_ground_truth if total_ground_truth > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_predictions": total_predictions,
        "total_ground_truth": total_ground_truth,
        "correct_predictions": correct_predictions,
    }
    
    print(f"\nResults:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Total ground truth: {total_ground_truth}")
    print(f"  Correct matches: {correct_predictions}")
    
    result = {
        "model_name": model_name,
        "model_path": str(model_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(dataset_dir),
        "num_images": len(image_files),
        "elapsed_time_sec": round(elapsed_time, 2),
        "config": {
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "max_images": max_images,
        },
        "metrics": metrics,
    }
    
    return result


def compute_iou_simple(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def main():
    """Main function."""
    print("="*60)
    print("CUSTOM DATASET EVALUATION")
    print("="*60)
    
    dataset_dir = DATASET_DIR
    print(f"\nDataset: {dataset_dir}")
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset not found: {dataset_dir}")
        return
    
    # Find models
    fp32_path = MODELS_DIR / "magnitude_0.3_decoded.onnx"
    fp16_path = MODELS_DIR / "magnitude_0.3_decoded_fp16.onnx"
    
    if not fp32_path.exists():
        print(f"ERROR: FP32 model not found: {fp32_path}")
        return
    
    # Get config
    conf_threshold = MODEL_CONFIG.get("conf_threshold", 0.25)
    iou_threshold = MODEL_CONFIG.get("iou_threshold", 0.45)
    max_images = 100  # Limit for testing
    
    # Evaluate FP32
    print(f"\nEvaluating FP32 model...")
    fp32_result = evaluate_model_custom(
        fp32_path,
        "magnitude_0.3_decoded_FP32",
        dataset_dir,
        conf_threshold,
        iou_threshold,
        max_images
    )
    
    # Evaluate FP16
    fp16_result = None
    if fp16_path.exists():
        print(f"\nEvaluating FP16 model...")
        fp16_result = evaluate_model_custom(
            fp16_path,
            "magnitude_0.3_decoded_FP16",
            dataset_dir,
            conf_threshold,
            iou_threshold,
            max_images
        )
    
    # Save results
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "custom_evaluation_results.json"
    
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(dataset_dir),
        "fp32_evaluation": fp32_result,
        "fp16_evaluation": fp16_result,
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved: {output_path}")


if __name__ == "__main__":
    main()