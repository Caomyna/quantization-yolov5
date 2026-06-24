"""
Stage 1: Inspect pruned PyTorch checkpoint.
Analyzes checkpoint type, state_dict, parameters, tensor names, dtypes,
pruning metadata, and sparsity. Generates checkpoint_report.json.
"""

import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any

from config import YOLOV5S_PT_PATH

logger = logging.getLogger(__name__)


def load_checkpoint(pt_path: Path) -> Dict[str, Any]:
    """Load PyTorch checkpoint from disk."""
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {pt_path}")
    logger.info(f"Loading checkpoint: {pt_path} ({pt_path.stat().st_size / 1024**2:.2f} MB)")
    return torch.load(pt_path, map_location="cpu")


def extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Extract state_dict from various checkpoint formats."""
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected dict, got {type(checkpoint)}")

    if "model" in checkpoint:
        m = checkpoint["model"]
        if hasattr(m, "state_dict"):
            return m.state_dict()
        if isinstance(m, dict):
            return m
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    tensor_count = sum(1 for v in checkpoint.values() if isinstance(v, torch.Tensor))
    if tensor_count > len(checkpoint) * 0.5:
        return checkpoint

    raise ValueError(f"Cannot extract state_dict from keys: {list(checkpoint.keys())}")


def detect_checkpoint_type(checkpoint: Dict[str, Any]) -> str:
    """Detect checkpoint structure type."""
    if not isinstance(checkpoint, dict):
        return type(checkpoint).__name__
    if "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
        return "YOLOv5 model wrapper"
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return "state_dict under 'model' key"
    if "state_dict" in checkpoint:
        return "state_dict wrapper"
    if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return "raw state_dict"
    return "other dict"


def inspect_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Inspect state_dict contents: parameters, dtypes, layer counts."""
    names = list(state_dict.keys())
    total_params = sum(t.numel() for t in state_dict.values())
    dtypes: Dict[str, int] = {}
    shapes = {}

    for name, tensor in state_dict.items():
        d = str(tensor.dtype)
        dtypes[d] = dtypes.get(d, 0) + 1
        if any(k in name for k in [".weight", ".bias", ".running_mean", ".running_var"]):
            shapes[name] = {"shape": list(tensor.shape), "numel": tensor.numel()}

    conv = sum(1 for n in names if "conv" in n.lower() and ".weight" in n)
    bn = sum(1 for n in names if "bn" in n.lower() and ".weight" in n)

    logger.info(f"Parameters: {total_params:,} | Tensors: {len(names)} | Conv: {conv} | BN: {bn}")
    return {
        "total_parameters": total_params,
        "total_tensors": len(names),
        "dtype_distribution": dtypes,
        "layer_counts": {"conv_layers": conv, "bn_layers": bn},
        "sample_shapes": dict(list(shapes.items())[:10]),
    }


def detect_pruning_metadata(
    checkpoint: Dict[str, Any], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    """Detect pruning artifacts in checkpoint and state_dict."""
    pruning_keys = [
        k for k in state_dict if "mask" in k.lower()
    ]
    weight_orig = [k for k in state_dict if "weight_orig" in k.lower()]

    meta_keys = []
    sparsity_val = None
    if isinstance(checkpoint, dict):
        meta_keys = [
            k for k in checkpoint
            if any(w in k.lower() for w in ["prune", "sparse", "mask"])
        ]
        for k in meta_keys:
            if isinstance(checkpoint[k], (int, float)):
                sparsity_val = float(checkpoint[k])

    detected = bool(pruning_keys or weight_orig or meta_keys)
    if detected:
        logger.info(f"Pruning: masks={len(pruning_keys)}, weight_orig={len(weight_orig)}")

    return {
        "pruning_detected": detected,
        "pruning_masks": len(pruning_keys),
        "weight_orig_tensors": len(weight_orig),
        "metadata_keys": meta_keys,
        "declared_sparsity": sparsity_val,
    }


def compute_sparsity(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Compute actual weight sparsity from state_dict."""
    total = 0
    zero = 0
    layer_sparsity = {}

    for name, tensor in state_dict.items():
        if "weight" not in name.lower():
            continue
        n = tensor.numel()
        z = (torch.abs(tensor) < 1e-10).sum().item()
        total += n
        zero += z
        if n > 0:
            layer_sparsity[name] = round(z / n * 100, 2)

    global_sp = round(zero / total * 100, 2) if total > 0 else 0.0
    sorted_layers = sorted(layer_sparsity.items(), key=lambda x: x[1])
    bottom5 = sorted_layers[:5]
    top5 = sorted_layers[-5:] if len(sorted_layers) >= 5 else sorted_layers

    logger.info(f"Sparsity: {global_sp}% ({zero:,}/{total:,} zero weights)")
    return {
        "total_weight_elements": total,
        "zero_weight_elements": zero,
        "global_sparsity_pct": global_sp,
        "least_sparse": [{"name": n, "sparsity_pct": s} for n, s in bottom5],
        "most_sparse": [{"name": n, "sparsity_pct": s} for n, s in reversed(top5)],
    }


def inspect_checkpoint(pt_path: Path = None) -> Dict[str, Any]:
    """Run full checkpoint inspection pipeline."""
    pt_path = pt_path or YOLOV5S_PT_PATH

    logger.info("=" * 50)
    logger.info("Checkpoint Inspection")
    logger.info("=" * 50)

    checkpoint = load_checkpoint(pt_path)
    ckpt_type = detect_checkpoint_type(checkpoint)
    state_dict = extract_state_dict(checkpoint)
    sd_info = inspect_state_dict(state_dict)
    pruning = detect_pruning_metadata(checkpoint, state_dict)
    sparsity = compute_sparsity(state_dict)

    logger.info("Checkpoint inspection done")

    return {
        "checkpoint_path": str(pt_path),
        "file_size_mb": round(pt_path.stat().st_size / 1024**2, 2),
        "checkpoint_type": ckpt_type,
        "top_level_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
        "state_dict": sd_info,
        "pruning": pruning,
        "sparsity": sparsity,
    }


def save_report(report: Dict[str, Any], output_path: Path):
    """Save report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved: {output_path}")


def main():
    """Entry point."""
    from config import LOGS_DIR

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])

    report = inspect_checkpoint()
    save_report(report, LOGS_DIR / "checkpoint_report.json")

    logger.info("Done")


if __name__ == "__main__":
    main()