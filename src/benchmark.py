"""
Benchmarking module for comparing FP32 and FP16 ONNX model inference.
Measures latency, throughput, and memory usage on test dataset.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import onnxruntime as ort
import cv2
import psutil

from config import (
    ONNX_FP32_PATH, ONNX_FP16_PATH, DATASET_DIR,
    BENCHMARK_CONFIG, BENCHMARK_RESULTS_PATH,
    MODEL_CONFIG
)


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
    """Benchmark class for ONNX model inference."""
    
    def __init__(
        self,
        model_path: Path,
        model_name: str,
        providers: List[str] = None
    ):
        """
        Initialize benchmark for a specific ONNX model.
        
        Args:
            model_path: Path to ONNX model
            model_name: Name identifier for the model
            providers: ONNX Runtime execution providers (e.g., ['CPUExecutionProvider'])
        """
        self.model_path = model_path
        self.model_name = model_name
        self.providers = providers or ['CPUExecutionProvider']
        
        # Load model
        print(f"\n[INFO] Loading {model_name} model from: {model_path}")
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                providers=self.providers
            )
            print(f"[SUCCESS] {model_name} model loaded successfully")
            
            # Get model info
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"[INFO] Input: {self.input_name}, shape: {self.input_shape}")
            print(f"[INFO] Output: {self.output_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_name} model: {str(e)}")
        
        # Metrics storage
        self.latencies = []
        self.memory_usage = []
    
    def preprocess_image(self, image_path: Path, input_size: int = 640) -> np.ndarray:
        """
        Preprocess image for YOLOv5 inference.
        
        Args:
            image_path: Path to input image
            input_size: Target input size (default: 640)
            
        Returns:
            np.ndarray: Preprocessed image tensor [1, 3, H, W]
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to input size (maintain aspect ratio with padding)
        h, w = img.shape[:2]
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        img_padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_h = (input_size - new_h) // 2
        pad_w = (input_size - new_w) // 2
        img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized
        
        # Normalize to [0, 1] and convert to float32
        img_normalized = img_padded.astype(np.float32) / 255.0
        
        # HWC to CHW and add batch dimension
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch
    
    def run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on input tensor.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            np.ndarray: Model output
        """
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        return outputs[0]
    
    def benchmark(
        self,
        image_paths: List[Path],
        warmup_iterations: int = 10,
        num_iterations: int = 100
    ) -> InferenceMetrics:
        """
        Run benchmark on list of images.
        
        Args:
            image_paths: List of image paths to test
            warmup_iterations: Number of warmup runs
            num_iterations: Number of measured iterations
            
        Returns:
            InferenceMetrics: Benchmark results
        """
        print(f"\n[INFO] Starting benchmark: {self.model_name}")
        print(f"[INFO] Images: {len(image_paths)}, Warmup: {warmup_iterations}, Iterations: {num_iterations}")
        
        # Get process for memory tracking
        process = psutil.Process()
        
        # Warmup phase
        print(f"[INFO] Warmup phase ({warmup_iterations} iterations)...")
        warmup_img = self.preprocess_image(image_paths[0])
        for i in range(warmup_iterations):
            _ = self.run_inference(warmup_img)
        print("[SUCCESS] Warmup completed")
        
        # Benchmark phase
        print(f"[INFO] Benchmark phase ({num_iterations} iterations)...")
        self.latencies = []
        self.memory_usage = []
        
        for idx in range(num_iterations):
            # Cycle through images
            img_path = image_paths[idx % len(image_paths)]
            
            # Preprocess
            input_tensor = self.preprocess_image(img_path)
            
            # Measure memory before inference
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Run inference and measure time
            start_time = time.perf_counter()
            _ = self.run_inference(input_tensor)
            end_time = time.perf_counter()
            
            # Measure memory after inference
            mem_after = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Record metrics
            latency_ms = (end_time - start_time) * 1000
            self.latencies.append(latency_ms)
            self.memory_usage.append(mem_after - mem_before)
            
            # Progress indicator
            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx + 1}/{num_iterations} iterations")
        
        print("[SUCCESS] Benchmark completed")
        
        # Calculate metrics
        latencies = np.array(self.latencies)
        memory_usage = np.array(self.memory_usage)
        
        metrics = InferenceMetrics(
            model_name=self.model_name,
            model_path=str(self.model_path),
            model_size_mb=self.model_path.stat().st_size / (1024 * 1024),
            
            # Latency
            avg_latency_ms=float(np.mean(latencies)),
            min_latency_ms=float(np.min(latencies)),
            max_latency_ms=float(np.max(latencies)),
            std_latency_ms=float(np.std(latencies)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            
            # Throughput
            throughput_fps=1000.0 / float(np.mean(latencies)),
            total_time_sec=float(np.sum(latencies) / 1000.0),
            
            # Memory
            peak_memory_mb=float(np.max(memory_usage)),
            avg_memory_mb=float(np.mean(memory_usage)),
            
            # Additional info
            num_iterations=num_iterations,
            num_images=len(image_paths),
            warmup_iterations=warmup_iterations
        )
        
        return metrics
    
    def print_metrics(self, metrics: InferenceMetrics):
        """Print formatted metrics."""
        print("\n" + "=" * 70)
        print(f"Benchmark Results: {metrics.model_name}")
        print("=" * 70)
        print(f"Model: {metrics.model_name}")
        print(f"Model Size: {metrics.model_size_mb:.2f} MB")
        print(f"\nLatency Statistics:")
        print(f"  - Average: {metrics.avg_latency_ms:.2f} ms")
        print(f"  - Min: {metrics.min_latency_ms:.2f} ms")
        print(f"  - Max: {metrics.max_latency_ms:.2f} ms")
        print(f"  - Std Dev: {metrics.std_latency_ms:.2f} ms")
        print(f"  - P95: {metrics.p95_latency_ms:.2f} ms")
        print(f"  - P99: {metrics.p99_latency_ms:.2f} ms")
        print(f"\nThroughput:")
        print(f"  - FPS: {metrics.throughput_fps:.2f}")
        print(f"  - Total Time: {metrics.total_time_sec:.2f} s")
        print(f"\nMemory Usage:")
        print(f"  - Peak: {metrics.peak_memory_mb:.2f} MB")
        print(f"  - Average: {metrics.avg_memory_mb:.2f} MB")
        print("=" * 70)


def load_test_images(dataset_dir: Path, max_images: int = 1000) -> List[Path]:
    """
    Load test images from dataset directory.
    
    Args:
        dataset_dir: Directory containing test images
        max_images: Maximum number of images to load
        
    Returns:
        List[Path]: List of image paths
    """
    print(f"\n[INFO] Loading test images from: {dataset_dir}")
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Find all images
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(dataset_dir.glob(f"*{ext}"))
        image_paths.extend(dataset_dir.glob(f"*{ext.upper()}"))
    
    # Sort for reproducibility
    image_paths = sorted(image_paths)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {dataset_dir}")
    
    # Limit to max_images
    if len(image_paths) > max_images:
        print(f"[INFO] Found {len(image_paths)} images, using first {max_images}")
        image_paths = image_paths[:max_images]
    else:
        print(f"[INFO] Found {len(image_paths)} images")
    
    return image_paths


def compare_benchmarks(
    fp32_metrics: InferenceMetrics,
    fp16_metrics: InferenceMetrics
) -> Dict:
    """
    Compare FP32 and FP16 benchmark results.
    
    Args:
        fp32_metrics: FP32 model metrics
        fp16_metrics: FP16 model metrics
        
    Returns:
        Dict: Comparison results
    """
    print("\n" + "=" * 70)
    print("FP32 vs FP16 Comparison")
    print("=" * 70)
    
    comparison = {
        'latency_speedup': fp32_metrics.avg_latency_ms / fp16_metrics.avg_latency_ms,
        'throughput_improvement': fp16_metrics.throughput_fps / fp32_metrics.throughput_fps,
        'size_reduction': (
            (fp32_metrics.model_size_mb - fp16_metrics.model_size_mb) / fp32_metrics.model_size_mb
        ) * 100,
        'latency_reduction': (
            (fp32_metrics.avg_latency_ms - fp16_metrics.avg_latency_ms) / fp32_metrics.avg_latency_ms
        ) * 100,
    }
    
    print(f"\nPerformance Comparison:")
    print(f"  - Latency Speedup: {comparison['latency_speedup']:.2f}x")
    print(f"  - Throughput Improvement: {comparison['throughput_improvement']:.2f}x")
    print(f"  - Latency Reduction: {comparison['latency_reduction']:.2f}%")
    print(f"  - Model Size Reduction: {comparison['size_reduction']:.2f}%")
    
    print(f"\nDetailed Metrics:")
    print(f"{'Metric':<30} {'FP32':<20} {'FP16':<20} {'Change':<15}")
    print("-" * 85)
    print(f"{'Avg Latency (ms)':<30} {fp32_metrics.avg_latency_ms:<20.2f} {fp16_metrics.avg_latency_ms:<20.2f} {comparison['latency_reduction']:>+.2f}%")
    print(f"{'Throughput (FPS)':<30} {fp32_metrics.throughput_fps:<20.2f} {fp16_metrics.throughput_fps:<20.2f} {comparison['throughput_improvement']:>+.2f}x")
    print(f"{'Model Size (MB)':<30} {fp32_metrics.model_size_mb:<20.2f} {fp16_metrics.model_size_mb:<20.2f} {comparison['size_reduction']:>+.2f}%")
    print(f"{'P95 Latency (ms)':<30} {fp32_metrics.p95_latency_ms:<20.2f} {fp16_metrics.p95_latency_ms:<20.2f}")
    print(f"{'P99 Latency (ms)':<30} {fp32_metrics.p99_latency_ms:<20.2f} {fp16_metrics.p99_latency_ms:<20.2f}")
    print("=" * 70)
    
    return comparison


def save_results(
    fp32_metrics: InferenceMetrics,
    fp16_metrics: InferenceMetrics,
    comparison: Dict,
    output_path: Path
):
    """
    Save benchmark results to JSON file.
    
    Args:
        fp32_metrics: FP32 model metrics
        fp16_metrics: FP16 model metrics
        comparison: Comparison results
        output_path: Path to save JSON file
    """
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'fp32': asdict(fp32_metrics),
        'fp16': asdict(fp16_metrics),
        'comparison': comparison
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SUCCESS] Results saved to: {output_path}")




def main():
    """Main execution function."""
    print("=" * 70)
    print("YOLOv5s FP32 vs FP16 Benchmarking")
    print("=" * 70)
    
    # Load test images
    try:
        image_paths = load_test_images(DATASET_DIR, max_images=1000)
    except Exception as e:
        print(f"[ERROR] Failed to load test images: {str(e)}")
        print("[INFO] Creating dummy benchmark with synthetic data...")
        # Create dummy images for testing
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for i in range(10):
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            dummy_path = DATASET_DIR / f"dummy_{i}.jpg"
            cv2.imwrite(str(dummy_path), dummy_img)
            image_paths.append(dummy_path)
        print(f"[INFO] Created {len(image_paths)} dummy test images")
    
    # Get benchmark config
    warmup = BENCHMARK_CONFIG['warmup_iterations']
    iterations = BENCHMARK_CONFIG['num_iterations']
    
    # Benchmark FP32 model
    print("\n" + "=" * 70)
    print("Benchmarking FP32 Model")
    print("=" * 70)
    fp32_benchmark = ONNXInferenceBenchmark(
        ONNX_FP32_PATH,
        "FP32"
    )
    fp32_metrics = fp32_benchmark.benchmark(
        image_paths,
        warmup_iterations=warmup,
        num_iterations=iterations
    )
    fp32_benchmark.print_metrics(fp32_metrics)
    
    # Benchmark FP16 model
    print("\n" + "=" * 70)
    print("Benchmarking FP16 Model")
    print("=" * 70)
    fp16_benchmark = ONNXInferenceBenchmark(
        ONNX_FP16_PATH,
        "FP16"
    )
    fp16_metrics = fp16_benchmark.benchmark(
        image_paths,
        warmup_iterations=warmup,
        num_iterations=iterations
    )
    fp16_benchmark.print_metrics(fp16_metrics)
    
    # Compare results
    comparison = compare_benchmarks(fp32_metrics, fp16_metrics)
    
    # Save results
    save_results(fp32_metrics, fp16_metrics, comparison, BENCHMARK_RESULTS_PATH)
    
    
    print("\n" + "=" * 70)
    print("Benchmarking completed successfully!")
    print(f"Results: {BENCHMARK_RESULTS_PATH}")
    print("=" * 70)
    
    return fp32_metrics, fp16_metrics, comparison


if __name__ == "__main__":
    main()