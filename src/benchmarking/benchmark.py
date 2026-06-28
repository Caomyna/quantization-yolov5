"""
Benchmark engine - Core benchmarking functionality.
Uses BaseONNXModel to eliminate duplicated session creation.
"""

import time
import tracemalloc
import numpy as np
import logging
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from ..core.base import BaseONNXModel
from ..preprocessing.preprocessor import preprocess_image
from ..core.config import BENCHMARK_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Data class to store inference metrics."""
    model_name: str
    model_path: str
    model_size_mb: float
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Throughput metrics
    throughput_fps: float
    total_time_sec: float
    
    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float
    
    # Additional info
    num_iterations: int
    num_images: int
    warmup_iterations: int


class BenchmarkEngine(BaseONNXModel):
    """
    Benchmark engine for ONNX models.
    Extends BaseONNXModel to add benchmarking capabilities.
    """
    
    def __init__(
        self,
        model_path: Path,
        model_name: str,
        providers: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize benchmark engine.
        
        Args:
            model_path: Path to ONNX model
            model_name: Name for logging
            providers: ONNX Runtime providers
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
        """
        super().__init__(model_path, providers, conf_threshold, iou_threshold)
        self.model_name = model_name
    
    def preprocess_image(self, image_path: Path, input_size: int = 640) -> np.ndarray:
        """
        Preprocess image for benchmarking.
        
        Args:
            image_path: Path to image
            input_size: Model input size
            
        Returns:
            Preprocessed tensor
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")
        
        dtype = self.get_input_dtype()
        return preprocess_image(img, input_size=input_size, dtype=dtype)
    
    def run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference and return primary output.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            Primary output tensor
        """
        outputs = self.run(input_tensor)
        return outputs[0]
    
    def benchmark(
        self,
        image_paths: List[Path],
        warmup: int = 10,
        iterations: int = 100,
        input_size: int = 640
    ) -> InferenceMetrics:
        """
        Run benchmark on model.
        
        Args:
            image_paths: List of image paths
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations
            input_size: Model input size
            
        Returns:
            InferenceMetrics with benchmark results
        """
        logger.info(f"Benchmarking {self.model_name}: {len(image_paths)} images, {iterations} iters")
        
        # Warmup
        warmup_img = self.preprocess_image(image_paths[0], input_size)
        for _ in range(warmup):
            self.run_inference(warmup_img)
        
        # Metrics storage
        latencies = []
        memory_peaks = []
        
        tracemalloc.start()
        
        # Benchmark loop
        for idx in range(iterations):
            img_path = image_paths[idx % len(image_paths)]
            tensor = self.preprocess_image(img_path, input_size)
            
            t0 = time.perf_counter()
            self.run_inference(tensor)
            t1 = time.perf_counter()
            
            latencies.append((t1 - t0) * 1000)
            
            # Track memory
            current, peak = tracemalloc.get_traced_memory()
            memory_peaks.append(peak / 1024**2)
        
        tracemalloc.stop()
        
        lat = np.array(latencies)
        mem = np.array(memory_peaks)
        
        return InferenceMetrics(
            model_name=self.model_name,
            model_path=str(self.model_path),
            model_size_mb=self.model_path.stat().st_size / 1024**2,
            
            avg_latency_ms=float(np.mean(lat)),
            min_latency_ms=float(np.min(lat)),
            max_latency_ms=float(np.max(lat)),
            std_latency_ms=float(np.std(lat)),
            p95_latency_ms=float(np.percentile(lat, 95)),
            p99_latency_ms=float(np.percentile(lat, 99)),
            
            throughput_fps=1000.0 / float(np.mean(lat)),
            total_time_sec=float(np.sum(lat) / 1000),
            
            peak_memory_mb=float(np.max(mem)),
            avg_memory_mb=float(np.mean(mem)),
            
            num_iterations=iterations,
            num_images=len(image_paths),
            warmup_iterations=warmup,
        )