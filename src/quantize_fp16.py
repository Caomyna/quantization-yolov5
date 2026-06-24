"""
Stage 4: Quantize ONNX FP32 model to FP16.
Uses convert_float_to_float16, then fixes Detect output to FP32 for post-processing.
"""

import onnx
from onnxconverter_common.float16 import convert_float_to_float16
import logging
from pathlib import Path

from config import ONNX_FP32_PATH, ONNX_FP16_PATH, QUANTIZATION_CONFIG

logger = logging.getLogger(__name__)


def load_model(path: Path) -> onnx.ModelProto:
    """Load ONNX model from file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    logger.info(f"Loading: {path} ({path.stat().st_size / 1024**2:.2f} MB)")
    return onnx.load(str(path))


def convert_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert FP32 to FP16, then set Detect output back to FP32."""
    cfg = QUANTIZATION_CONFIG
    fp16_model = convert_float_to_float16(
        model,
        min_positive_val=cfg["min_positive_val"],
        max_finite_val=cfg["max_finite_val"],
        keep_io_types=cfg["keep_io_types"],
        disable_shape_infer=cfg["disable_shape_infer"],
    )
    logger.info("Converted FP32 → FP16")

    # Fix: set Detect output back to FP32 for post-processing
    for node in fp16_model.graph.node:
        if node.op_type == "Detect":
            for out in node.output:
                for vi in fp16_model.graph.value_info:
                    if vi.name == out:
                        vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
                        logger.info(f"Set Detect output '{out}' to FP32")

    onnx.checker.check_model(fp16_model)
    logger.info("ONNX checker: passed")
    return fp16_model


def save_model(model: onnx.ModelProto, path: Path):
    """Save ONNX model to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    logger.info(f"Saved: {path} ({path.stat().st_size / 1024**2:.2f} MB)")


def quantize_model(
    input_path: Path = ONNX_FP32_PATH,
    output_path: Path = ONNX_FP16_PATH,
) -> onnx.ModelProto:
    """Load FP32 → convert to FP16 → save. Returns converted model."""
    logger.info("=" * 50)
    logger.info("Quantize FP32 → FP16")
    logger.info("=" * 50)

    model = load_model(input_path)
    fp16_model = convert_fp16(model)
    save_model(fp16_model, output_path)

    fp32_mb = input_path.stat().st_size / 1024**2
    fp16_mb = output_path.stat().st_size / 1024**2
    reduction = (1 - fp16_mb / fp32_mb) * 100
    logger.info(f"Size reduction: {reduction:.1f}% ({fp32_mb:.2f} → {fp16_mb:.2f} MB)")

    logger.info("Quantization done")
    return fp16_model


def main():
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])

    quantize_model()
    logger.info("Done")


if __name__ == "__main__":
    main()