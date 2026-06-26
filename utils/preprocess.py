"""
Image preprocessing utilities for YOLO models.
Handles resizing, padding, normalization, and batching.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def preprocess_image(
    image: np.ndarray,
    input_size: int = 640,
    normalize: bool = True,
    to_rgb: bool = True
) -> np.ndarray:
    """
    Preprocess a single image for YOLO inference.
    
    Args:
        image: Input image (BGR or RGB format)
        input_size: Target size (square)
        normalize: Whether to normalize to [0, 1]
        to_rgb: Whether to convert BGR to RGB
        
    Returns:
        Preprocessed tensor [1, 3, input_size, input_size]
    """
    # Convert BGR to RGB if needed
    if to_rgb and len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()
    
    h, w = img.shape[:2]
    
    # Letterbox resize (maintain aspect ratio)
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to input_size
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_h = (input_size - new_h) // 2
    pad_w = (input_size - new_w) // 2
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img
    
    # Normalize
    if normalize:
        padded = padded.astype(np.float32) / 255.0
    
    # Transpose to CHW format and add batch dimension
    tensor = np.transpose(padded, (2, 0, 1))[None, :]
    
    return tensor


def preprocess_image_path(
    image_path: Union[str, Path],
    input_size: int = 640,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image from file path.
    
    Args:
        image_path: Path to image file
        input_size: Target size
        normalize: Whether to normalize
        
    Returns:
        Tuple of (preprocessed tensor, original image)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Preprocess
    tensor = preprocess_image(image, input_size, normalize, to_rgb=True)
    
    return tensor, image


def batch_preprocess(
    images: List[np.ndarray],
    input_size: int = 640,
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a batch of images.
    
    Args:
        images: List of input images
        input_size: Target size
        normalize: Whether to normalize
        
    Returns:
        Batched tensor [batch_size, 3, input_size, input_size]
    """
    tensors = []
    for img in images:
        tensor = preprocess_image(img, input_size, normalize, to_rgb=True)
        tensors.append(tensor)
    
    # Concatenate along batch dimension
    batch = np.concatenate(tensors, axis=0)
    
    return batch


def get_preprocess_params(
    original_shape: Tuple[int, int],
    input_size: int = 640
) -> Tuple[float, int, int]:
    """
    Get preprocessing parameters for coordinate transformation.
    
    Args:
        original_shape: Original image shape (h, w)
        input_size: Model input size
        
    Returns:
        Tuple of (scale, pad_h, pad_w)
    """
    h, w = original_shape
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    pad_h = (input_size - new_h) // 2
    pad_w = (input_size - new_w) // 2
    
    return scale, pad_h, pad_w


def denormalize_coordinates(
    bbox: List[float],
    scale: float,
    pad_h: int,
    pad_w: int
) -> List[float]:
    """
    Convert bbox coordinates from model space to original image space.
    
    Args:
        bbox: Bounding box [x, y, w, h] in model space
        scale: Scale factor from preprocessing
        pad_h: Height padding
        pad_w: Width padding
        
    Returns:
        Bounding box [x, y, w, h] in original image space
    """
    x, y, w, h = bbox
    
    # Convert from center format to corner format
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    
    # Remove padding and scale
    x1 = (x1 - pad_w) / scale
    y1 = (y1 - pad_h) / scale
    x2 = (x2 - pad_w) / scale
    y2 = (y2 - pad_h) / scale
    
    # Convert back to center format
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    return [x, y, w, h]