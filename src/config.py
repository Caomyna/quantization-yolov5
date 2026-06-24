"""
Configuration file for YOLOv5 FP32 to FP16 quantization pipeline.
Centralized settings for paths, quantization parameters, and benchmarking.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
# src/ is one level below project root
BASE_DIR = Path(__file__).parent.parent.resolve()

# Model paths
MODELS_DIR = BASE_DIR / "weights"  # Directory to store models and weights
YOLOV5S_PT_PATH = MODELS_DIR / "magnitude_0.3_recovered.pt"  # Original PyTorch model
ONNX_FP32_PATH = MODELS_DIR / "magnitude_0.3_recovered_fp32.onnx"  # Exported ONNX FP32 model
ONNX_FP16_PATH = MODELS_DIR / "magnitude_0.3_recovered_fp16.onnx"  # Quantized ONNX FP16 model

# Dataset paths
DATASET_DIR = BASE_DIR / "dataset"  # Folder containing ~1000 test images
BENCHMARK_DIR = BASE_DIR / "benchmark"
BENCHMARK_RESULTS_PATH = BENCHMARK_DIR / "magnitude_0.3_recovered_benchmark_results.json"
BENCHMARK_PLOT_PATH = BENCHMARK_DIR / "magnitude_0.3_recovered_benchmark_comparison.png"

# Logging paths
LOGS_DIR = BASE_DIR / "logs"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline_log.txt"

# Weights/output directory
WEIGHTS_DIR = BASE_DIR / "weights"

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
    
    # Convert both input and output tensors to FP16
    "keep_io_types": False,  # Keep weights as FP32, only activations FP16
    
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
}

# ============================================================================
# ONNX EXPORT PARAMETERS
# ============================================================================
ONNX_EXPORT_CONFIG = {
    "opset_version": 12,  # ONNX opset version
    "dynamic_axes": True,  # Support dynamic batch size
    "simplify": True,  # Simplify ONNX model after export
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [MODELS_DIR, BENCHMARK_DIR, WEIGHTS_DIR, LOGS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_model_paths():
    """Return dictionary of all model paths for easy access."""
    return {
        "pt": str(YOLOV5S_PT_PATH),
        "onnx_fp32": str(ONNX_FP32_PATH),
        "onnx_fp16": str(ONNX_FP16_PATH),
    }


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