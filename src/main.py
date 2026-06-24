"""
Main orchestration script.
Runs: Inspect → Export → Validate FP32 → Quantize → Validate FP16 → Benchmark → Report
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

from config import (
    YOLOV5S_PT_PATH, ONNX_FP32_PATH, ONNX_FP16_PATH,
    ensure_directories, get_model_paths, PIPELINE_LOG_PATH, LOGS_DIR
)

from inspect_checkpoint import inspect_checkpoint, save_report as save_ckpt_report
from export_to_onnx import export_model
from validate_onnx import validate_onnx, save_report as save_val_report
from quantize_fp16 import quantize_model
from benchmark import benchmark_models

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PIPELINE_LOG_PATH, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class QuantizationPipeline:
    """Pipeline: Inspect → Export → Validate FP32 → Quantize → Validate FP16 → Benchmark → Report."""

    def __init__(self):
        self.start_time = time.time()
        self.steps_completed = []
        self.errors = []
        ensure_directories()

        logger.info("=" * 50)
        logger.info("Quantization Pipeline")
        logger.info("=" * 50)
        logger.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log: {PIPELINE_LOG_PATH}")
        logger.info("=" * 50)

    def step_1_inspect(self) -> bool:
        """Stage 1: Inspect checkpoint."""
        logger.info("-" * 50)
        logger.info("Stage 1: Inspect Checkpoint")
        logger.info("-" * 50)
        try:
            report = inspect_checkpoint()
            save_ckpt_report(report, LOGS_DIR / "checkpoint_report.json")
            self.steps_completed.append("Stage 1: Inspect checkpoint")
            return True
        except Exception as e:
            self.errors.append(f"Stage 1 failed: {e}")
            logger.error(f"Stage 1 failed: {e}")
            return False

    def step_2_export(self) -> bool:
        """Stage 2: Export to ONNX FP32."""
        logger.info("-" * 50)
        logger.info("Stage 2: Export to ONNX FP32")
        logger.info("-" * 50)
        try:
            export_model()
            self.steps_completed.append("Stage 2: Export to ONNX FP32")
            return True
        except Exception as e:
            self.errors.append(f"Stage 2 failed: {e}")
            logger.error(f"Stage 2 failed: {e}")
            return False

    def step_3_validate_fp32(self) -> bool:
        """Stage 3: Validate FP32 ONNX."""
        logger.info("-" * 50)
        logger.info("Stage 3: Validate FP32 ONNX")
        logger.info("-" * 50)
        try:
            report = validate_onnx(ONNX_FP32_PATH)
            save_val_report(report, LOGS_DIR / "validation_fp32_report.json")
            self.steps_completed.append("Stage 3: Validate FP32 ONNX")
            return True
        except Exception as e:
            self.errors.append(f"Stage 3 failed: {e}")
            logger.error(f"Stage 3 failed: {e}")
            return False

    def step_4_quantize(self) -> bool:
        """Stage 4: Quantize FP32 → FP16."""
        logger.info("-" * 50)
        logger.info("Stage 4: Quantize FP32 → FP16")
        logger.info("-" * 50)
        try:
            quantize_model()
            self.steps_completed.append("Stage 4: Quantize FP32 → FP16")
            return True
        except Exception as e:
            self.errors.append(f"Stage 4 failed: {e}")
            logger.error(f"Stage 4 failed: {e}")
            return False

    def step_5_validate_fp16(self) -> bool:
        """Stage 5: Validate FP16 ONNX."""
        logger.info("-" * 50)
        logger.info("Stage 5: Validate FP16 ONNX")
        logger.info("-" * 50)
        try:
            report = validate_onnx(ONNX_FP16_PATH)
            save_val_report(report, LOGS_DIR / "validation_fp16_report.json")
            self.steps_completed.append("Stage 5: Validate FP16 ONNX")
            return True
        except Exception as e:
            self.errors.append(f"Stage 5 failed: {e}")
            logger.error(f"Stage 5 failed: {e}")
            return False

    def step_6_benchmark(self) -> bool:
        """Stage 6: Benchmark FP32 vs FP16."""
        logger.info("-" * 50)
        logger.info("Stage 6: Benchmark FP32 vs FP16")
        logger.info("-" * 50)
        try:
            benchmark_models()
            self.steps_completed.append("Stage 6: Benchmark FP32 vs FP16")
            return True
        except Exception as e:
            self.errors.append(f"Stage 6 failed: {e}")
            logger.error(f"Stage 6 failed: {e}")
            return False

    def run_pipeline(self, skip_steps: list = None):
        """Run pipeline stages."""
        skip_steps = skip_steps or []
        steps = [
            (1, "Inspect Checkpoint", self.step_1_inspect),
            (2, "Export to ONNX FP32", self.step_2_export),
            (3, "Validate FP32 ONNX", self.step_3_validate_fp32),
            (4, "Quantize FP32 → FP16", self.step_4_quantize),
            (5, "Validate FP16 ONNX", self.step_5_validate_fp16),
            (6, "Benchmark FP32 vs FP16", self.step_6_benchmark),
        ]
        for num, name, func in steps:
            if num in skip_steps:
                logger.info(f"Skipping {name}")
                continue
            if not func():
                logger.error(f"Pipeline stopped at {name}")
                break
        # self.generate_final_report()
        self.print_summary()

    # def generate_final_report(self):
    #     """Stage 7: Merge all reports into summary.json and summary.md."""
    #     logger.info("-" * 50)
    #     logger.info("Stage 7: Generate Final Report")
    #     logger.info("-" * 50)

    #     reports = {}
    #     for name in ["checkpoint_report.json", "validation_fp32_report.json",
    #                  "validation_fp16_report.json", "benchmark_results.json"]:
    #         path = LOGS_DIR / name
    #         if path.exists():
    #             with open(path) as f:
    #                 reports[name.replace("_report.json", "").replace("_results.json", "")] = json.load(f)

    #     summary = {
    #         "timestamp": datetime.now().isoformat(),
    #         "pipeline": "quantization-yolov5",
    #         "stages_completed": self.steps_completed,
    #         "reports": reports,
    #     }

    #     summary_json = LOGS_DIR / "summary.json"
    #     with open(summary_json, "w") as f:
    #         json.dump(summary, f, indent=2, default=str)
    #     logger.info(f"Summary JSON: {summary_json}")

    #     summary_md = LOGS_DIR / "summary.md"
    #     with open(summary_md, "w") as f:
    #         f.write("# Quantization Pipeline Summary\n\n")
    #         f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    #         f.write("## Stages Completed\n\n")
    #         for s in self.steps_completed:
    #             f.write(f"- ✓ {s}\n")
    #         f.write("\n## Reports\n\n")
    #         for key in reports:
    #             f.write(f"- {key}: `logs/{key}_report.json`\n")
    #     logger.info(f"Summary Markdown: {summary_md}")

    def print_summary(self):
        """Print execution summary."""
        elapsed = time.time() - self.start_time
        logger.info("=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Time: {elapsed:.2f}s")
        logger.info(f"Completed: {len(self.steps_completed)}/{6}")
        for s in self.steps_completed:
            logger.info(f"  ✓ {s}")
        for e in self.errors:
            logger.info(f"  ✗ {e}")
        logger.info("=" * 50)


def main():
    """Entry point."""
    pipeline = QuantizationPipeline()

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "inspect":
            pipeline.run_pipeline(skip_steps=[2, 3, 4, 5, 6])
        elif cmd == "export":
            pipeline.run_pipeline(skip_steps=[1, 3, 4, 5, 6])
        elif cmd == "validate":
            pipeline.run_pipeline(skip_steps=[1, 2, 4, 5, 6])
        elif cmd == "quantize":
            pipeline.run_pipeline(skip_steps=[1, 2, 3, 5, 6])
        elif cmd == "benchmark":
            pipeline.run_pipeline(skip_steps=[1, 2, 3, 4, 5])
        elif cmd == "full":
            pipeline.run_pipeline()
        else:
            logger.error(f"Unknown: {cmd}")
            logger.info("Commands: inspect, export, validate, quantize, benchmark, full")
            sys.exit(1)
    else:
        pipeline.run_pipeline()


if __name__ == "__main__":
    main()