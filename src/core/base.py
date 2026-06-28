"""
Base class for ONNX model wrappers.
Eliminates duplicated session creation and metadata extraction.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import onnxruntime as ort
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """ONNX model metadata."""
    input_name: str
    input_shape: List[Any]
    input_type: str
    output_names: List[str]
    model_path: Path
    providers: List[str]


class BaseONNXModel:
    """
    Base class for ONNX model wrappers.
    Handles session creation, metadata extraction, and common operations.
    """
    
    def __init__(
        self,
        model_path: Path,
        providers: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize ONNX model wrapper.
        
        Args:
            model_path: Path to ONNX model file
            providers: ONNX Runtime providers (None for auto-detect)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Auto-detect providers if not provided
        if providers is None:
            providers = self._get_default_providers()
        
        # Create ONNX Runtime session
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot load model {self.model_path.name}\n"
                f"Path: {self.model_path}\n"
                f"Error: {e}"
            )
        
        # Extract metadata
        self.metadata = self._extract_metadata()
        
        logger.info(
            f"Loaded {self.model_path.name}: "
            f"input={self.metadata.input_name} {self.metadata.input_shape} "
            f"type={self.metadata.input_type}"
        )
    
    def _get_default_providers(self) -> List[str]:
        """Get default ONNX Runtime providers."""
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    
    def _extract_metadata(self) -> ModelMetadata:
        """Extract model metadata from ONNX session."""
        input_info = self.session.get_inputs()[0]
        output_infos = self.session.get_outputs()
        
        return ModelMetadata(
            input_name=input_info.name,
            input_shape=input_info.shape,
            input_type=input_info.type,
            output_names=[o.name for o in output_infos],
            model_path=self.model_path,
            providers=self.session.get_providers(),
        )
    
    def get_input_dtype(self) -> np.dtype:
        """Get numpy dtype for model input."""
        if "float16" in self.metadata.input_type.lower():
            return np.float16
        return np.float32
    
    def run(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on input tensor.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            List of output tensors
        """
        return self.session.run(
            self.metadata.output_names,
            {self.metadata.input_name: input_tensor}
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information dictionary."""
        return {
            "model_path": str(self.metadata.model_path),
            "model_name": self.metadata.model_path.name,
            "input_name": self.metadata.input_name,
            "input_shape": self.metadata.input_shape,
            "input_type": self.metadata.input_type,
            "output_names": self.metadata.output_names,
            "providers": self.metadata.providers,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
        }
    
    def warmup(self, input_tensor: np.ndarray, iterations: int = 5):
        """
        Warmup model with dummy inferences.
        
        Args:
            input_tensor: Dummy input tensor
            iterations: Number of warmup iterations
        """
        for _ in range(iterations):
            self.run(input_tensor)
        logger.debug(f"Warmup completed: {iterations} iterations")