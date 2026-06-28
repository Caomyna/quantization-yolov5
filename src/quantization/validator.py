"""
ONNX model validation.
Runs checker, shape inference, ONNX Runtime inference, extracts graph info.
"""

import json
import onnx
import onnxruntime as ort
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

from ..core.config import Config

logger = logging.getLogger(__name__)


def load_model(path: Path) -> onnx.ModelProto:
    """Load ONNX model from file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    logger.info(f"Loading: {path} ({path.stat().st_size / 1024**2:.2f} MB)")
    return onnx.load(str(path))


def check_model(model: onnx.ModelProto) -> bool:
    """Run ONNX checker. Returns True if valid."""
    try:
        onnx.checker.check_model(model)
        logger.info("ONNX checker: passed")
        return True
    except onnx.checker.ValidationError as e:
        logger.error(f"ONNX checker: failed ({e})")
        return False


def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run shape inference on the model."""
    inferred = onnx.shape_inference.infer_shapes(model)
    logger.info("Shape inference: done")
    return inferred


def runtime_inference(model: onnx.ModelProto) -> Dict[str, Any]:
    """Run dummy inference with ONNX Runtime. Return output info."""
    providers = Config.get_onnx_providers()
    try:
        session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    except Exception as e:
        raise RuntimeError(f"Failed to create ONNX Runtime session with providers {providers}: {e}")
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    output_type = session.get_outputs()[0].type

    # Determine dtype based on model input type
    if "float16" in input_type:
        np_dtype = np.float16
    else:
        np_dtype = np.float32

    # Build dummy input matching expected shape
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
    dummy = np.random.randn(*shape).astype(np_dtype)

    outputs = session.run([output_name], {input_name: dummy})
    actual_output = outputs[0]

    result = {
        "input": {"name": input_name, "shape": list(input_shape), "type": input_type},
        "output": {"name": output_name, "shape": list(output_shape), "type": output_type},
        "actual_output_shape": list(actual_output.shape),
        "actual_output_dtype": str(actual_output.dtype),
        "output_min": round(float(actual_output.min()), 4),
        "output_max": round(float(actual_output.max()), 4),
        "output_mean": round(float(actual_output.mean()), 4),
        "inference_successful": True,
    }

    logger.info(f"Runtime inference: {result['actual_output_shape']} ({result['actual_output_dtype']})")
    return result


def get_graph_info(model: onnx.ModelProto) -> Dict[str, Any]:
    """Extract graph metadata."""
    graph = model.graph
    info = {
        "graph_name": graph.name,
        "ir_version": model.ir_version,
        "opset_version": model.opset_import[0].version if model.opset_import else None,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "num_inputs": len(graph.input),
        "num_outputs": len(graph.output),
        "num_nodes": len(graph.node),
        "num_initializers": len(graph.initializer),
        "inputs": [
            {"name": inp.name, "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim]}
            for inp in graph.input
        ],
        "outputs": [
            {"name": out.name, "shape": [d.dim_value for d in out.type.tensor_type.shape.dim]}
            for out in graph.output
        ],
    }
    logger.info(f"Graph: {info['graph_name']}, {info['num_nodes']} nodes, {info['num_inputs']} inputs, {info['num_outputs']} outputs")
    return info


def validate_onnx(path: Path) -> Dict[str, Any]:
    """Run full ONNX validation pipeline."""
    logger.info("=" * 50)
    logger.info("Validate ONNX")
    logger.info("=" * 50)

    model = load_model(path)
    checker_ok = check_model(model)
    model_inferred = infer_shapes(model)

    runtime = None
    try:
        runtime = runtime_inference(model_inferred)
    except RuntimeError as e:
        if "CUDAExecutionProvider" in str(e) or "Type Error" in str(e):
            logger.warning(f"FP16 inference skipped: requires CUDA (CPU ONNX Runtime does not support FP16 Conv). Error: {e}")
            runtime = {
                "inference_successful": False,
                "error": "FP16 requires CUDAExecutionProvider",
                "details": str(e),
            }
        else:
            raise

    graph_info = get_graph_info(model_inferred)

    report = {
        "model_path": str(path),
        "checker_passed": checker_ok,
        "runtime": runtime,
        "graph": graph_info,
    }

    logger.info("Validation done")
    return report


def save_report(report: Dict[str, Any], output_path: Path):
    """Save validation report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved: {output_path}")