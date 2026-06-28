"""
Quantize ONNX Model - FP32 to FP16 conversion with optional validation.

Usage:
    python src/quantization/run.py --validate-only weights/best_decoded.onnx

    python src/quantization/run.py --model best_decoded
    python src/quantization/run.py --model best_decoded --validate
    python src/quantization/run.py --input model.onnx --output model_fp16.onnx
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup Python path so src/ is importable
exec(open(Path(__file__).resolve().parent.parent / 'core' / 'path_setup.py').read())

from src.core.config import Config, LOGS_DIR
from src.quantization.fp16 import quantize_fp16
from src.quantization.validator import validate_onnx, save_report

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize FP32 ONNX model to FP16")
    parser.add_argument("--model", type=str, default="best_decoded",
                        help="Model name (e.g. 'best_decoded') or path to ONNX model (default: best_decoded)")
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to FP32 ONNX model (overrides --model lookup)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path for output FP16 ONNX model (overrides --model lookup)")
    parser.add_argument("--validate", action="store_true",
                        help="Run ONNX validation after conversion")
    parser.add_argument("--validate-only", type=Path, default=None,
                        help="Validate an existing ONNX model without quantizing")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])
    
    args = parse_args()

    # Only validation mode
    if args.validate_only:
        model_path = args.validate_only
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return 1

        logger.info("=" * 54)
        logger.info("          ONNX MODEL VALIDATION")
        logger.info("=" * 54)
        logger.info(f"  Model  : {model_path}")
        logger.info(f"  Size   : {model_path.stat().st_size / 1024**2:.2f} MB")

        report = validate_onnx(model_path)
        report_path = LOGS_DIR / f"{model_path.stem}_validation_report.json"
        save_report(report, report_path)

        logger.info(f"\n  Checker  : {'PASSED' if report['checker_passed'] else 'FAILED'}")
        if report.get("runtime") and report["runtime"].get("inference_successful"):
            logger.info(f"  Runtime  : OK")
        else:
            logger.info(f"  Runtime  : skipped (may require CUDA)")
        logger.info(f"\n  Report   : {report_path}")
        logger.info("\n" + "=" * 54)
        logger.info("          VALIDATION COMPLETE")
        logger.info("=" * 54)
        return 0

    # Quantization mode - support both model name and full path
    model_arg = args.model
    model_path = Path(model_arg)
    if model_path.suffix == '.onnx' or len(model_path.parts) > 1:
        input_path = model_path.resolve()
        output_path = args.output or input_path.parent / f"{input_path.stem}_fp16.onnx"
    else:
        paths = Config.get_model_paths(args.model)
        input_path = args.input or paths["onnx_fp32"]
        output_path = args.output or paths["onnx_fp16"]

    if not input_path.exists():
        logger.error(f"FP32 model not found: {input_path}")
        logger.error("Use --model to specify a model name from weights/, or --input for a custom path.")
        return 1

    logger.info(f"Input  : {input_path}")
    logger.info(f"Output : {output_path}")

    quantize_fp16(input_path, output_path)

    logger.info(f"Completed")

    if args.validate:
        logger.info("\n" + "-" * 50)
        logger.info("  Running ONNX validation...")
        report = validate_onnx(output_path)
        report_path = LOGS_DIR / f"{args.model}_validation_report.json"
        save_report(report, report_path)
        logger.info(f"  Validation report saved: {report_path}")
        logger.info(f"  Checker  : {'PASSED' if report['checker_passed'] else 'FAILED'}")
        if report.get("runtime") and report["runtime"].get("inference_successful"):
            logger.info(f"  Runtime  : OK")
        else:
            logger.info(f"  Runtime  : skipped (may require CUDA)")

    logger.info("\n" + "=" * 54)
    logger.info("QUANTIZATION COMPLETE")
    logger.info("=" * 54)
    return 0


if __name__ == "__main__":
    exit(main())