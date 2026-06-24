# YOLOv5s FP32 to FP16 Quantization Pipeline

A comprehensive quantization pipeline for converting YOLOv5s from FP32 to FP16 precision using ONNX Runtime, with benchmarking and visualization capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Configuration](#configuration)
- [Extending to INT8](#extending-to-int8)

## 🎯 Overview

This project implements a complete quantization workflow for YOLOv5s model:
1. **Inspect**: Analyze checkpoint (sparsity, parameters, pruning metadata)
2. **Export**: Convert PyTorch model (.pt) to ONNX FP32 format
3. **Validate FP32**: Verify ONNX standard compliance and runtime inference
4. **Quantize**: Convert FP32 to FP16 (50% size reduction)
5. **Validate FP16**: Verify quantized model (requires CUDA for runtime)
6. **Benchmark**: Compare FP32 vs FP16 performance (latency, throughput, memory)



## 📁 Project Structure

```
quantization-yolov5/
├── config.py                          # Configuration settings
├── models/
│   ├── yolov5s.pt                     # Original PyTorch model
│   ├── yolov5s_fp32.onnx              # Exported FP32 model
│   └── yolov5s_fp16.onnx              # Quantized FP16 model
├── benchmark/
│   ├── benchmark_results.json         # Benchmark metrics
│   └── benchmark_comparison.png       # Visualization plots
├── src/
│   ├── __init__.py
│   ├── config.py                      # Path and parameter configuration
│   ├── export_to_onnx.py              # PyTorch → ONNX FP32 export
│   ├── validate_onnx.py               # ONNX model validation
│   ├── quantize_fp16.py               # FP32 → FP16 quantization
│   ├── benchmark.py                   # Performance benchmarking
│   ├── visualize_benchmark.py         # Standalone visualization module
│   └── main.py                        # Pipeline orchestration
├── yolov5/                            # YOLOv5 repository (for export)
├── dataset/                           # Test images (optional)
└── README.md
```


## 🔧 Installation

### Prerequisites

- Python 3.9+
- Conda environment (recommended)
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Caomyna/quantization-yolov5.git
cd quantization-yolov5
```

2. **Create and activate conda environment**:
```bash
conda create -n quant python=3.9
conda activate quant
```

3. **Install dependencies**:
```bash
pip install -r requirement.yml
```

Key dependencies:
- `torch` - PyTorch for model loading
- `onnx` - ONNX model format
- `onnxconverter-common` - FP16 quantization
- `onnxruntime` - Inference engine
- `opencv-python` - Image preprocessing
- `matplotlib`, `seaborn` - Visualization
- `psutil` - Memory monitoring

4. **Download YOLOv5s model**:
```bash
# Download from Ultralytics
# Place yolov5s.pt in models/ directory
```

## 🚀 Usage

### Run Complete Pipeline

Execute the full quantization workflow:

```bash
python src/main.py full
```

Or run individual stages:

```bash
# Inspect checkpoint only
python src/main.py inspect

# Export only
python src/main.py export

# Validate FP32 only
python src/main.py validate

# Quantize only (assumes FP32 model exists)
python src/main.py quantize

# Benchmark only (assumes both models exist)
python src/main.py benchmark

# Run full pipeline
python src/main.py full
```






## 📊 Results

### Model Comparison

| Metric | FP32 | FP16 | Improvement |
|--------|------|------|-------------|
| **Model Size** | 27.60 MB | 13.87 MB | **-49.8%** ✓ |
| **Avg Latency (CPU)** | ~125 ms | N/A | Requires CUDA |
| **Throughput (CPU)** | ~8 FPS | N/A | Requires CUDA |



### Output Files

- `weights/*_fp32.onnx` - Exported FP32 model
- `weights/*_fp16.onnx` - Quantized FP16 model (50% smaller)
- `logs/checkpoint_report.json` - Checkpoint inspection results
- `logs/validation_fp32_report.json` - FP32 validation results
- `logs/validation_fp16_report.json` - FP16 validation results
- `logs/benchmark_results.json` - Detailed benchmark metrics
- `logs/benchmark_results.csv` - Benchmark metrics (CSV)
- `logs/summary.json` - Unified pipeline summary
- `logs/summary.md` - Human-readable summary

### Output Files

- `models/yolov5s_fp32.onnx` - Original FP32 model (28 MB)
- `models/yolov5s_fp16.onnx` - Quantized FP16 model (14 MB)
- `benchmark/benchmark_results.json` - Detailed metrics in JSON format

## ⚙️ Configuration

All settings are centralized in `src/config.py`:

### Model Paths
```python
YOLOV5S_PT_PATH = "models/yolov5s.pt"
ONNX_FP32_PATH = "models/yolov5s_fp32.onnx"
ONNX_FP16_PATH = "models/yolov5s_fp16.onnx"
```

### Quantization Parameters (FP16)
```python
QUANTIZATION_CONFIG = {
    "min_positive_val": 1e-7,      # Minimum positive value
    "max_finite_val": 3.4e+38,     # Maximum finite value
    "keep_io_types": True,         # Keep input/output as FP32
    "disable_shape_infer": False,  # Enable shape inference
}
```

### Benchmark Parameters
```python
BENCHMARK_CONFIG = {
    "warmup_iterations": 10,
    "num_iterations": 100,
    "batch_size": 1,
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
}
```


```

## 🛠️ Technical Details

### Dynamic Provider Selection

The pipeline automatically selects the best available ONNX Runtime provider:

```python
def _get_providers():
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
```

- **With CUDA**: FP16 runs on GPU (2-8x speedup)
- **Without CUDA**: FP16 validation passes, runtime is skipped gracefully

### Quantization Strategy

- Uses `onnxconverter_common.float16.convert_float_to_float16`
- Converts backbone + neck to FP16
- Keeps Detect head outputs as FP32 for post-processing compatibility
- 50% model size reduction achieved

### Benchmarking

- ONNX Runtime for inference
- Measures latency (min, max, avg, P95, P99)
- Calculates throughput (FPS)
- Monitors memory usage via `tracemalloc`
- Warmup iterations for stable measurements

## 📝 Implementation Notes

### Why FP16 on CPU is Not Supported for Runtime
- ONNX Runtime CPU does not support FP16 Conv operations
- The `convert_float_to_float16` converter inserts internal Cast nodes that are incompatible with CPU
- **Solution**: FP16 models are validated (checker + shape inference) but runtime inference requires CUDA
