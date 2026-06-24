"""
Export PyTorch checkpoint to ONNX FP32.
Loads checkpoint, builds model, loads weights, exports to ONNX.
"""

import sys
import torch
import onnx
import logging
from pathlib import Path

from config import YOLOV5S_PT_PATH, ONNX_FP32_PATH, ONNX_EXPORT_CONFIG
from inspect_checkpoint import extract_state_dict

ROOT = Path(__file__).resolve().parents[1]   # project root
YOLOV5_DIR = ROOT / "yolov5"

sys.path.insert(0, str(ROOT))        # để import src/config
sys.path.insert(0, str(YOLOV5_DIR))  # để import yolov5 models

logger = logging.getLogger(__name__)


def load_checkpoint(pt_path: Path) -> dict:
    """Load PyTorch checkpoint on CPU."""
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")
    logger.info(f"Loading: {pt_path} ({pt_path.stat().st_size / 1024**2:.2f} MB)")
    return torch.load(pt_path, map_location="cpu")


def build_model(yaml_path: str = "yolov5/models/yolov5s.yaml") -> torch.nn.Module:
    """Rebuild YOLOv5 architecture from YAML."""
    from yolov5.models.yolo import Model
    logger.info(f"Building model from: {yaml_path}")
    return Model(yaml_path)


def load_weights(model: torch.nn.Module, checkpoint: dict) -> torch.nn.Module:
    """Extract state_dict and load into model (strict=False)."""
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Weights loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path = ONNX_FP32_PATH,
    opset: int = None,
    dynamic: bool = True,
) -> onnx.ModelProto:
    """Export model to ONNX FP32 and return ModelProto."""
    opset = opset or ONNX_EXPORT_CONFIG["opset_version"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(1, 3, 640, 640)
    dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}} if dynamic else None

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    logger.info(f"ONNX exported: {output_path} ({output_path.stat().st_size / 1024**2:.2f} MB)")
    return onnx.load(str(output_path))


def export_model(
    pt_path: Path = None,
    output_path: Path = None,
    yaml_path: str = "yolov5/models/yolov5s.yaml",
) -> onnx.ModelProto:
    """Orchestrate full export: load → build → load_weights → export."""
    pt_path = pt_path or YOLOV5S_PT_PATH
    output_path = output_path or ONNX_FP32_PATH

    logger.info("=" * 50)
    logger.info("Export to ONNX FP32")
    logger.info("=" * 50)

    checkpoint = load_checkpoint(pt_path)
    model = build_model(yaml_path)
    model = load_weights(model, checkpoint)
    model_proto = export_to_onnx(model, output_path)

    logger.info("Export done")
    return model_proto


def main():
    """Entry point."""
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])

    export_model()
    logger.info("Done")


if __name__ == "__main__":
    main()