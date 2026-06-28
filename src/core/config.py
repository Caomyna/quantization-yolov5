"""
Centralized configuration for YOLOv5 quantization pipeline.
No ONNX Runtime import - pure configuration.
"""

from pathlib import Path
from typing import Dict, List, Any

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.parent.resolve()

MODELS_DIR = BASE_DIR / "weights"
DATASET_DIR = BASE_DIR / "dataset" / "coco2017" / "val2017"
ANNOTATION_FILE = BASE_DIR / "dataset" / "coco2017" / "annotations" / "instances_val2017.json"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_DIR = BASE_DIR / "output"

# ============================================================================
# QUANTIZATION PARAMETERS
# ============================================================================
QUANTIZATION_CONFIG: Dict[str, Any] = {
    "min_positive_val": 1e-7,
    "max_finite_val": 3.4e+38,
    "keep_io_types": False,
    "disable_shape_infer": False,
}

# ============================================================================
# BENCHMARKING PARAMETERS
# ============================================================================
BENCHMARK_CONFIG: Dict[str, Any] = {
    "warmup_iterations": 10,
    "num_iterations": 100,
    "batch_size": 1,
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
}

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
MODEL_CONFIG: Dict[str, Any] = {
    "input_size": 640,
    "num_classes": 80,
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
}

# ============================================================================
# COCO CLASS NAMES (80 classes)
# ============================================================================
COCO_CLASS_NAMES: List[str] = [
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

# ============================================================================
# VEHICLE CLASSES (COCO IDs)
# ============================================================================
VEHICLE_CLASS_IDS: List[int] = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
VEHICLE_CLASS_NAMES: List[str] = ["bicycle", "car", "motorcycle", "bus", "truck"]

# ============================================================================
# TRAFFIC ANALYSIS CONFIGURATION
# ============================================================================
TRAFFIC_CONFIG: Dict[str, Any] = {
    "vehicle_class_ids": VEHICLE_CLASS_IDS,
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "colors": {
        "car": (0, 255, 0),
        "truck": (0, 165, 255),
        "bus": (0, 0, 255),
        "motorcycle": (255, 0, 0),
        "bicycle": (255, 255, 0),
    },
}

# ============================================================================
# ONNX EXPORT PARAMETERS
# ============================================================================
ONNX_EXPORT_CONFIG: Dict[str, Any] = {
    "opset_version": 14,
    "dynamic_axes": True,
    "simplify": False,
}


class Config:
    """Configuration singleton with helper methods."""
    
    @staticmethod
    def get_onnx_providers() -> List[str]:
        """Get ONNX Runtime providers (lazy import to avoid circular deps)."""
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    
    @staticmethod
    def ensure_directories():
        """Create necessary directories if they don't exist."""
        for directory in [MODELS_DIR, LOGS_DIR, REPORTS_DIR, OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_model_paths(model_name: str) -> Dict[str, Path]:
        """Return dictionary of model paths for a given model name."""
        return {
            "onnx_fp32": MODELS_DIR / f"{model_name}.onnx",
            "onnx_fp16": MODELS_DIR / f"{model_name}_fp16.onnx",
        }


# Initialize directories on import
Config.ensure_directories()