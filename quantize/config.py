"""
Configuration file for YOLOv5 FP32 to FP16 quantization pipeline.
Centralized settings for paths, quantization parameters, and benchmarking.
"""

import os
from pathlib import Path
import onnxruntime as ort

# ============================================================================
# PROJECT PATHS
# ============================================================================
# quantize/ is one level below project root
BASE_DIR = Path(__file__).parent.parent.resolve()

# Model weights directory
MODELS_DIR = BASE_DIR / "weights"  # Directory to store models and weights
# YOLOV5S_PT_PATH = MODELS_DIR / "magnitude_0.3_recovered.pt"  # Original PyTorch model
ONNX_FP32_PATH = MODELS_DIR / "best_decoded.onnx"  # Exported ONNX FP32 model
ONNX_FP16_PATH = MODELS_DIR / "best_decoded_fp16.onnx"  # Quantized ONNX FP16 model

# Dataset paths
DATASET_DIR = BASE_DIR / "dataset" / "coco2017" /"val2017"  # Folder containing 1000 test images
ANNOTATION_FILE = BASE_DIR / "dataset" / "coco2017" / "annotations" / "instances_val2017.json"

# Logging paths
LOGS_DIR = BASE_DIR / "logs"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline_log.txt"

# Reports directory
REPORTS_DIR = BASE_DIR / "reports"

# Output directory (for video demo)
OUTPUT_DIR = BASE_DIR / "output"

# ============================================================================
# QUANTIZATION PARAMETERS (FP16)
# ============================================================================
# These parameters are used for FP16 quantization via onnxconverter_common
# Can be easily modified for future INT8 quantization tasks

QUANTIZATION_CONFIG = {
    # Minimum positive value threshold (prevents underflow to zero)
    "min_positive_val": 1e-7,
    
    # Maximum finite value threshold (prevents overflow to inf)
    "max_finite_val": 3.4e+38,
    
    # True: Keep model inputs/outputs as FP32, only convert internal weights/activations to FP16
    "keep_io_types": True,  
    
    # Enable shape inference during quantization
    "disable_shape_infer": False,
}

# ============================================================================
# BENCHMARKING PARAMETERS
# ============================================================================
BENCHMARK_CONFIG = {
    # Number of warmup iterations (to stabilize performance measurements)
    "warmup_iterations": 10,
    
    # Number of measured inference iterations
    "num_iterations": 100,
    
    # Batch size for inference (1 for real-time, higher for throughput testing)
    "batch_size": 1,
    
    # Confidence threshold for YOLO inference
    "conf_threshold": 0.25,
    
    # IoU threshold for NMS
    "iou_threshold": 0.45,
}

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
MODEL_CONFIG = {
    # Input image size (YOLOv5s standard)
    "input_size": 640,
    
    # Number of classes (COCO dataset)
    "num_classes": 80,
    
    # Class names (COCO 80 classes)
    "class_names": [
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
    ],
    
    # Traffic analysis: Vehicle class IDs (COCO dataset)
    "vehicle_classes": {
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    },
    
    # Traffic analysis: Vehicle class names list
    "vehicle_class_names": ["bicycle", "car", "motorcycle", "bus", "truck"],
    
    # Detection thresholds
    "conf_threshold": 0.25,  # Confidence threshold for detections
    "iou_threshold": 0.45,   # IoU threshold for NMS
}

# ============================================================================
# TRAFFIC ANALYSIS CONFIGURATION
# ============================================================================
TRAFFIC_CONFIG = {
    # Vehicle detection classes (COCO IDs)
    "vehicle_class_ids": [1, 2, 3, 5, 7],
    
    # Detection parameters
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    
    # Visualization colors (BGR format for OpenCV)
    "colors": {
        "car": (0, 255, 0),        # Green
        "truck": (0, 165, 255),    # Orange
        "bus": (0, 0, 255),        # Red
        "motorcycle": (255, 0, 0), # Blue
        "bicycle": (255, 255, 0),  # Cyan
    },
    
    # Default input/output paths
    "default_input_dir": "dataset/coco2017/val2017",
    "default_output_dir": "output",
}

# ============================================================================
# ONNX EXPORT PARAMETERS
# ============================================================================
ONNX_EXPORT_CONFIG = {
    "opset_version": 14,  # ONNX opset version (14+ has better FP16/quantization support)
    "dynamic_axes": True,  # Support dynamic batch size
    "simplify": False,  # Simplify ONNX model after export (requires onnx-simplifier)
}

# ============================================================================
# ONNX RUNTIME PROVIDER CONFIGURATION
# ============================================================================
# For CPU-only environments, use CPUExecutionProvider
# For GPU environments, use CUDAExecutionProvider
def _get_providers() -> list:
    """Select providers: prefer CUDA, fallback to CPU."""
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

ONNX_PROVIDERS = _get_providers()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [MODELS_DIR, LOGS_DIR, REPORTS_DIR, OUTPUT_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_model_paths(model_name: str = "magnitude_0.3_decoded"):
    """Return dictionary of model paths for a given model name."""
    return {
        "pt": str(MODELS_DIR / f"{model_name}.pt"),
        "onnx_fp32": str(MODELS_DIR / f"{model_name}.onnx"),
        "onnx_fp16": str(MODELS_DIR / f"{model_name}_fp16.onnx"),
    }

# def get_model_paths():
#     """Return dictionary of all model paths for easy access."""
#     return {
#         "pt": str(YOLOV5S_PT_PATH),
#         "onnx_fp32": str(ONNX_FP32_PATH),
#         "onnx_fp16": str(ONNX_FP16_PATH),
#     }


def get_quantization_params():
    """Return quantization parameters for easy modification."""
    return QUANTIZATION_CONFIG


def get_benchmark_params():
    """Return benchmark parameters for easy modification."""
    return BENCHMARK_CONFIG


# ============================================================================
# INITIALIZATION
# ============================================================================
# Ensure directories exist on import
ensure_directories()