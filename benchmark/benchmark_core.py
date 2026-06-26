"""
Core benchmarking functionality for ONNX models.
Measures latency, throughput, memory usage, and model size.
"""

import time
import json
import csv
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import onnxruntime as ort
import cv2
import psutil
from pycocotools.coco import COCO
from quantize.config import ONNX_PROVIDERS


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


class ONNXInferenceBenchmark:
    """Benchmark a single ONNX model."""

    def __init__(self, model_path: Path, model_name: str):
        self.model_path = Path(model_path)
        self.model_name = model_name
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=ONNX_PROVIDERS
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot load model {model_name}\n"
                f"Path: {model_path}\n"
                f"Error: {e}"
            )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type
        self.output_name = self.session.get_outputs()[0].name
        logger.info(f"Loaded {model_name}: input={self.input_name} {self.input_shape} type={self.input_type}")

    def preprocess_image(self, image_path: Path, input_size: int = 640) -> np.ndarray:
        """Resize, pad, normalize image to [1,3,H,W]."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_h, pad_w = (input_size - new_h) // 2, (input_size - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img
        tensor = padded.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :]
        # Match model input dtype (FP16 models need FP16 input)
        if "float16" in self.input_type.lower():
            tensor = tensor.astype(np.float16)
        return tensor

    def run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run ORT inference."""
        return self.session.run([self.output_name], {self.input_name: input_tensor})[0]

    def benchmark(
        self, image_paths: List[Path], warmup: int = 10, iterations: int = 100
    ) -> InferenceMetrics:

        logger.info(f"Benchmarking {self.model_name}: {len(image_paths)} images, {iterations} iters")

        # ------------------------
        # Warmup
        # ------------------------
        warmup_img = self.preprocess_image(image_paths[0])
        for _ in range(warmup):
            self.run_inference(warmup_img)

        # ------------------------
        # Metrics storage
        # ------------------------
        latencies = []
        memory_peaks = []

        import tracemalloc
        tracemalloc.start()

        # ------------------------
        # Benchmark loop
        # ------------------------
        for idx in range(iterations):
            img_path = image_paths[idx % len(image_paths)]
            tensor = self.preprocess_image(img_path)

            t0 = time.perf_counter()
            self.run_inference(tensor)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)

            # memory fix (REAL peak memory)
            current, peak = tracemalloc.get_traced_memory()
            memory_peaks.append(peak / 1024**2)

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

            # FIX: rename correct meaning
            throughput_fps=1000.0 / float(np.mean(lat)),

            total_time_sec=float(np.sum(lat) / 1000),

            peak_memory_mb=float(np.max(mem)),
            avg_memory_mb=float(np.mean(mem)),

            num_iterations=iterations,
            num_images=len(image_paths),
            warmup_iterations=warmup,
        )


def load_test_images(dataset_dir: Path, annotation_file: Path = None, max_images: int = 1000) -> List[Path]:
    """
    Find image files in dataset directory.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    # If annotation file provided, use COCO-based selection for consistency with evaluation
    if annotation_file and Path(annotation_file).exists():
        logger.info(f"Using COCO annotations for image selection: {annotation_file}")
        coco = COCO(str(annotation_file))
        
        # Get all image IDs sorted (same order as evaluation)
        img_ids = sorted(coco.getImgIds())
        
        # Limit to max_images
        if max_images > 0:
            img_ids = img_ids[:max_images]
        
        # Load image paths in the same order as COCO image IDs
        images = []
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = dataset_dir / img_info["file_name"]
            if img_path.exists():
                images.append(img_path)
            else:
                logger.warning(f"Image not found: {img_path}")
        
        logger.info(f"Loaded {len(images)} images from COCO annotations (max_images={max_images})")
        return images
    
    # Fallback: load alphabetically
    images = sorted(
        p for e in exts for p in list(dataset_dir.glob(f"*{e}")) + list(dataset_dir.glob(f"*{e.upper()}"))
    )
    if not images:
        raise ValueError(f"No images in {dataset_dir}")
    if len(images) > max_images:
        images = images[:max_images]
    logger.info(f"Loaded {len(images)} images from {dataset_dir}")
    return images


def compare_benchmarks(fp32: InferenceMetrics, fp16: InferenceMetrics) -> Dict[str, float]:

    if fp32 is None or fp16 is None:
        return {
            "latency_speedup": 0,
            "latency_reduction_pct": 0,
            "throughput_improvement": 0,
            "size_reduction_pct": 0,
        }

    return {
        "latency_speedup": round(fp32.avg_latency_ms / fp16.avg_latency_ms, 4),
        "latency_reduction_pct": round(
            (fp32.avg_latency_ms - fp16.avg_latency_ms) / fp32.avg_latency_ms * 100, 2
        ),
        "throughput_improvement": round(
            fp16.throughput_fps / fp32.throughput_fps, 4
        ),
        "size_reduction_pct": round(
            (fp32.model_size_mb - fp16.model_size_mb) / fp32.model_size_mb * 100, 2
        ),
    }


def save_results(fp32: InferenceMetrics, fp16: InferenceMetrics, comparison: Dict, path: Path):
    """Save benchmark results as JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fp32": asdict(fp32),
        "fp16": asdict(fp16) if fp16 else None,
        "comparison": comparison,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved: {path}")


def save_csv(fp32: InferenceMetrics, fp16: InferenceMetrics, comparison: Dict, path: Path):
    """Save benchmark results as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fp16_size = f"{fp16.model_size_mb:.2f}" if fp16 else "N/A"
    fp16_lat = f"{fp16.avg_latency_ms:.2f}" if fp16 else "N/A"
    fp16_fps = f"{fp16.throughput_fps:.2f}" if fp16 else "N/A"
    fp16_p95 = f"{fp16.p95_latency_ms:.2f}" if fp16 else "N/A"
    fp16_p99 = f"{fp16.p99_latency_ms:.2f}" if fp16 else "N/A"
    fp16_mem = f"{fp16.peak_memory_mb:.2f}" if fp16 else "N/A"
    rows = [
        ["metric", "fp32", "fp16", "change"],
        ["model_size_mb", f"{fp32.model_size_mb:.2f}", fp16_size, f"{comparison['size_reduction_pct']:.2f}%"],
        ["avg_latency_ms", f"{fp32.avg_latency_ms:.2f}", fp16_lat, f"{comparison['latency_speedup']:.2f}x"],
        ["throughput_fps", f"{fp32.throughput_fps:.2f}", fp16_fps, f"{comparison['throughput_improvement']:.2f}x"],
        ["p95_latency_ms", f"{fp32.p95_latency_ms:.2f}", fp16_p95, ""],
        ["p99_latency_ms", f"{fp32.p99_latency_ms:.2f}", fp16_p99, ""],
        ["peak_memory_mb", f"{fp32.peak_memory_mb:.2f}", fp16_mem, ""],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    logger.info(f"CSV saved: {path}")