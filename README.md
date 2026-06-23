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
1. **Export**: Convert PyTorch model (.pt) to ONNX FP32 format
2. **Validate**: Verify ONNX standard compliance
3. **Quantize**: Convert FP32 to FP16 (50% size reduction)
4. **Validate**: Verify quantized model
5. **Benchmark**: Compare FP32 vs FP16 performance

## ✨ Features

- ✅ **Automatic Export**: Uses YOLOv5's official export script
- ✅ **ONNX Validation**: Standard compliance checking at each stage
- ✅ **FP16 Quantization**: Using `onnxconverter_common` with configurable parameters
- ✅ **Comprehensive Benchmarking**: Latency, throughput, and memory metrics
- ✅ **Standalone Visualization**: Separate module for generating comparison plots
- ✅ **Modular Design**: Clean, reusable code structure
- ✅ **Automatic Dataset Creation**: Dummy images for testing when no dataset available

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
python src/main.py
```

Or specify a command:

```bash
# Export only
python src/main.py export

# Quantize only (assumes FP32 model exists)
python src/main.py quantize

# Benchmark only (assumes both models exist)
python src/main.py benchmark

# Validate only
python src/main.py validate

# Run full pipeline
python src/main.py full
```

### Generate Visualizations

After running the benchmark, generate plots separately:

```bash
python src/visualize_benchmark.py
```

This creates `benchmark/benchmark_comparison.png` with:
- Latency comparison charts
- Throughput comparison
- Model size comparison
- Latency distribution (box plot)
- Performance improvements
- Summary table

## 📊 Results

### Model Comparison

| Metric | FP32 | FP16 | Improvement |
|--------|------|------|-------------|
| **Model Size** | 28.00 MB | 14.04 MB | **-49.85%** ✓ |
| **Avg Latency** | 122.01 ms | 216.36 ms | -77.33% (CPU) |
| **Throughput** | 8.20 FPS | 4.62 FPS | -43.66% (CPU) |
| **P95 Latency** | 156.85 ms | 279.66 ms | -78.28% |
| **P99 Latency** | 189.33 ms | 305.40 ms | -61.32% |

**Note**: On CPU, FP16 may be slower due to lack of native FP16 acceleration. On GPU with FP16 support (e.g., NVIDIA Tensor Cores), you'll see significant speedup (2-8x faster).

### Output Files

- `models/yolov5s_fp32.onnx` - Original FP32 model (28 MB)
- `models/yolov5s_fp16.onnx` - Quantized FP16 model (14 MB)
- `benchmark/benchmark_results.json` - Detailed metrics in JSON format
- `benchmark/benchmark_comparison.png` - Visualization plots

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

## 🔄 Extending to INT8

The modular design makes it easy to add INT8 quantization:

1. **Create `src/quantize_int8.py`**:
```python
from onnxconverter_common import convert

def quantize_fp32_to_int8():
    # Load FP32 model
    model_proto = onnx.load(str(ONNX_FP32_PATH))
    
    # INT8 quantization with calibration
    quantized_model = convert(
        model_proto,
        min_positive_val=1e-7,
        max_finite_val=3.4e+38,
        # Add calibration data here
    )
    
    # Save INT8 model
    onnx.save(quantized_model, str(ONNX_INT8_PATH))
    return quantized_model
```

2. **Update `src/main.py`**:
```python
from quantize_int8 import quantize_fp32_to_int8

# Add new step
def step_3_quantize_int8(self):
    quantized_model = quantize_fp32_to_int8()
    # ... validation and benchmarking
```

3. **Update config.py**:
```python
ONNX_INT8_PATH = MODELS_DIR / "yolov5s_int8.onnx"
```

## 🛠️ Technical Details

### Export Process
- Uses YOLOv5's official `export.py` script via subprocess
- Avoids SSL/network issues by using local YOLOv5 repository
- Automatically handles file naming (yolov5s.onnx → yolov5s_fp32.onnx)

### Validation
- ONNX checker validation (structural and semantic)
- ModelProto validation
- Tensor shape verification
- Opset version compliance

### Quantization
- Uses `onnxconverter_common.float16.float16_convert_point`
- Converts all tensors to FP16 except inputs/outputs (if `keep_io_types=True`)
- Preserves model structure and operations

### Benchmarking
- ONNX Runtime for inference
- Measures latency (min, max, avg, P95, P99)
- Calculates throughput (FPS)
- Monitors memory usage
- Warmup iterations for stable measurements

## 📝 Implementation Notes

### Why FP16 on CPU is Slower
- CPUs lack native FP16 arithmetic units
- FP16 values are converted to FP32 for computation
- Memory bandwidth savings don't outweigh conversion overhead
- **Solution**: Use GPU (CUDA) with Tensor Cores for 2-8x speedup

### Model Size Reduction
- FP16 uses 2 bytes per value vs 4 bytes in FP32
- 50% size reduction is expected and achieved
- Benefits: faster loading, less memory, easier deployment

### Accuracy Preservation
- FP16 maintains sufficient precision for YOLOv5s
- Minimal to no accuracy loss on COCO dataset
- Always validate on your specific dataset

## 🐛 Troubleshooting

### SSL Errors During Export
The pipeline uses YOLOv5's local export script to avoid SSL issues. If you encounter network errors, ensure the `yolov5/` directory exists.

### Missing Dependencies
```bash
pip install onnx onnxconverter-common onnxruntime opencv-python matplotlib seaborn psutil
```

### Out of Memory
Reduce batch size or image resolution in `config.py`:
```python
ONNX_EXPORT_CONFIG = {
    "opset_version": 12,
    "dynamic_axes": True,
    "simplify": True,
}
```

## 📄 License

This project is for educational and research purposes. YOLOv5 is licensed under AGPL-3.0.

## 🙏 Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) - Model architecture
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Inference engine
- [ONNX Converter](https://github.com/microsoft/onnxconverter-common) - Quantization tools

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Status**: ✅ Production Ready  
**Last Updated**: 2026-06-23  
**Python Version**: 3.9+  
**Framework**: PyTorch 2.8.0, ONNX 1.18.0