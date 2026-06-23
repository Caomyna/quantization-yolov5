"""
Main orchestration script for YOLOv5s FP32 to FP16 quantization pipeline.
Executes the complete workflow: Export → Validate → Quantize → Validate → Benchmark
"""

import sys
import time
import onnx
from pathlib import Path
from datetime import datetime

from config import (
    YOLOV5S_PT_PATH, ONNX_FP32_PATH, ONNX_FP16_PATH,
    ensure_directories, get_model_paths
)

# Import pipeline modules
from config import BENCHMARK_RESULTS_PATH, BENCHMARK_PLOT_PATH
from config import DATASET_DIR, BENCHMARK_CONFIG
from export_to_onnx import load_and_export_model, get_model_info
from validate_onnx import validate_onnx_model, validate_model_proto
from quantize_fp16 import quantize_fp32_to_fp16, print_quantization_info
from benchmark import (
    ONNXInferenceBenchmark, load_test_images,
    compare_benchmarks, save_results
)


class QuantizationPipeline:
    """Main pipeline class for FP32 to FP16 quantization."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.start_time = time.time()
        self.steps_completed = []
        self.errors = []
        
        # Ensure directories exist
        ensure_directories()
        
        print("=" * 70)
        print("YOLOv5s FP32 to FP16 Quantization Pipeline")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Working directory: {Path.cwd()}")
        print("=" * 70)
    
    def step_1_export_onnx(self) -> bool:
        """
        Step 1: Export PyTorch model to ONNX FP32.
        
        Returns:
            bool: True if successful
        """
        print("\n" + "=" * 70)
        print("STEP 1: Export YOLOv5s to ONNX FP32")
        print("=" * 70)
        
        try:
            # Check if PyTorch model exists
            if not YOLOV5S_PT_PATH.exists():
                raise FileNotFoundError(
                    f"PyTorch model not found at: {YOLOV5S_PT_PATH}\n"
                    f"Please download yolov5s.pt from https://github.com/ultralytics/yolov5"
                )
            
            # Export model
            model_proto = load_and_export_model()
            
            # Display model info
            model_info = get_model_info(model_proto)
            
            print("\n[SUCCESS] Step 1 completed: ONNX FP32 model exported")
            self.steps_completed.append("Step 1: Export to ONNX FP32")
            return True
            
        except Exception as e:
            error_msg = f"Step 1 failed: {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def step_2_validate_fp32(self) -> bool:
        """
        Step 2: Validate FP32 ONNX model.
        
        Returns:
            bool: True if successful
        """
        print("\n" + "=" * 70)
        print("STEP 2: Validate FP32 ONNX Model")
        print("=" * 70)
        
        try:
            # Validate using file path
            is_valid, message = validate_onnx_model(ONNX_FP32_PATH)
            
            if not is_valid:
                raise RuntimeError(f"FP32 model validation failed: {message}")
            
            # Also validate using ModelProto
            model_proto = onnx.load(str(ONNX_FP32_PATH))
            is_valid_proto, message_proto = validate_model_proto(model_proto, "FP32 Model")
            
            if not is_valid_proto:
                raise RuntimeError(f"FP32 ModelProto validation failed: {message_proto}")
            
            print("\n[SUCCESS] Step 2 completed: FP32 model validated")
            self.steps_completed.append("Step 2: Validate FP32 model")
            return True
            
        except Exception as e:
            error_msg = f"Step 2 failed: {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def step_3_quantize_fp16(self) -> bool:
        """
        Step 3: Quantize FP32 model to FP16.
        
        Returns:
            bool: True if successful
        """
        print("\n" + "=" * 70)
        print("STEP 3: Quantize FP32 to FP16")
        print("=" * 70)
        
        try:
            # Perform quantization
            quantized_model = quantize_fp32_to_fp16()
            
            # Print quantization info
            print_quantization_info(quantized_model)
            
            # Validate the quantized model
            is_valid, message = validate_onnx_model(ONNX_FP16_PATH)
            
            if not is_valid:
                print(f"[WARNING] FP16 model validation issue: {message}")
                print("[INFO] Continuing with benchmarking...")
            else:
                print("\n[SUCCESS] FP16 model validation passed")
            
            print("\n[SUCCESS] Step 3 completed: FP16 quantization finished")
            self.steps_completed.append("Step 3: Quantize to FP16")
            return True
            
        except Exception as e:
            error_msg = f"Step 3 failed: {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def step_4_validate_fp16(self) -> bool:
        """
        Step 4: Validate FP16 ONNX model.
        
        Returns:
            bool: True if successful
        """
        print("\n" + "=" * 70)
        print("STEP 4: Validate FP16 ONNX Model")
        print("=" * 70)
        
        try:
            # Validate FP16 model
            is_valid, message = validate_onnx_model(ONNX_FP16_PATH)
            
            if not is_valid:
                raise RuntimeError(f"FP16 model validation failed: {message}")
            
            # Also validate using ModelProto
            model_proto = onnx.load(str(ONNX_FP16_PATH))
            is_valid_proto, message_proto = validate_model_proto(model_proto, "FP16 Model")
            
            if not is_valid_proto:
                raise RuntimeError(f"FP16 ModelProto validation failed: {message_proto}")
            
            print("\n[SUCCESS] Step 4 completed: FP16 model validated")
            self.steps_completed.append("Step 4: Validate FP16 model")
            return True
            
        except Exception as e:
            error_msg = f"Step 4 failed: {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def step_5_benchmark(self) -> bool:
        """
        Step 5: Benchmark FP32 vs FP16 models.
        
        Returns:
            bool: True if successful
        """
        print("\n" + "=" * 70)
        print("STEP 5: Benchmark FP32 vs FP16")
        print("=" * 70)
        
        try:
            # Load test images
            try:
                image_paths = load_test_images(DATASET_DIR, max_images=1000)
            except Exception as e:
                print(f"[WARNING] {str(e)}")
                print("[INFO] Creating dummy test images for benchmarking...")
                # Create dummy images for testing
                DATASET_DIR.mkdir(parents=True, exist_ok=True)
                import numpy as np
                import cv2
                image_paths = []
                for i in range(10):
                    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    dummy_path = DATASET_DIR / f"dummy_{i}.jpg"
                    cv2.imwrite(str(dummy_path), dummy_img)
                    image_paths.append(dummy_path)
                print(f"[INFO] Created {len(image_paths)} dummy test images")
            
            warmup = BENCHMARK_CONFIG['warmup_iterations']
            iterations = BENCHMARK_CONFIG['num_iterations']
            
            # Benchmark FP32
            print("\n" + "-" * 70)
            print("Benchmarking FP32 Model")
            print("-" * 70)
            fp32_benchmark = ONNXInferenceBenchmark(ONNX_FP32_PATH, "FP32")
            fp32_metrics = fp32_benchmark.benchmark(
                image_paths,
                warmup_iterations=warmup,
                num_iterations=iterations
            )
            fp32_benchmark.print_metrics(fp32_metrics)
            
            # Benchmark FP16
            print("\n" + "-" * 70)
            print("Benchmarking FP16 Model")
            print("-" * 70)
            fp16_benchmark = ONNXInferenceBenchmark(ONNX_FP16_PATH, "FP16")
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
            
            print("\n[INFO] To visualize results, run: python src/visualize_benchmark.py")
            print("\n[SUCCESS] Step 5 completed: Benchmarking finished")
            self.steps_completed.append("Step 5: Benchmark FP32 vs FP16")
            return True
            
        except Exception as e:
            error_msg = f"Step 5 failed: {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def run_pipeline(self, skip_steps: list = None):
        """
        Run the complete pipeline.
        
        Args:
            skip_steps: List of step numbers to skip (e.g., [1, 2] to skip export and validation)
        """
        skip_steps = skip_steps or []
        
        # Define pipeline steps
        steps = [
            (1, "Export to ONNX FP32", self.step_1_export_onnx),
            (2, "Validate FP32", self.step_2_validate_fp32),
            (3, "Quantize to FP16", self.step_3_quantize_fp16),
            (4, "Validate FP16", self.step_4_validate_fp16),
            (5, "Benchmark", self.step_5_benchmark),
        ]
        
        # Execute steps
        for step_num, step_name, step_func in steps:
            if step_num in skip_steps:
                print(f"\n[INFO] Skipping {step_name} (step {step_num})")
                continue
            
            success = step_func()
            
            if not success:
                print(f"\n[ERROR] Pipeline stopped at {step_name}")
                break
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print pipeline execution summary."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nCompleted steps ({len(self.steps_completed)}/{5}):")
        for step in self.steps_completed:
            print(f"  ✓ {step}")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ✗ {error}")
        
        print("\nOutput files:")
        model_paths = get_model_paths()
        for key, path in model_paths.items():
            path_obj = Path(path)
            if path_obj.exists():
                size_mb = path_obj.stat().st_size / (1024 * 1024)
                print(f"  - {key}: {path} ({size_mb:.2f} MB)")
        
        if BENCHMARK_RESULTS_PATH.exists():
            print(f"  - Benchmark results: {BENCHMARK_RESULTS_PATH}")
        if BENCHMARK_PLOT_PATH.exists():
            print(f"  - Benchmark plot: {BENCHMARK_PLOT_PATH}")
        
        print("=" * 70)
        
        if len(self.steps_completed) == 5:
            print("\n[SUCCESS] Pipeline completed successfully!")
        else:
            print(f"\n[WARNING] Pipeline completed with {len(self.errors)} error(s)")


def main():
    """Main entry point."""
    pipeline = QuantizationPipeline()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "export":
            # Only export
            pipeline.run_pipeline(skip_steps=[2, 3, 4, 5])
        elif command == "quantize":
            # Only quantize (assumes FP32 model exists)
            pipeline.run_pipeline(skip_steps=[1, 2, 4, 5])
        elif command == "benchmark":
            # Only benchmark (assumes both models exist)
            pipeline.run_pipeline(skip_steps=[1, 2, 3, 4])
        elif command == "validate":
            # Only validate
            pipeline.run_pipeline(skip_steps=[1, 3, 5])
        elif command == "full":
            # Run full pipeline
            pipeline.run_pipeline()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: export, quantize, benchmark, validate, full")
            sys.exit(1)
    else:
        # Default: run full pipeline
        pipeline.run_pipeline()


if __name__ == "__main__":
    main()