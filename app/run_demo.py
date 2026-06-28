"""
Traffic Analysis Demo - Vehicle detection and counting using YOLOv5 ONNX models.

Usage:
    python app/run_demo.py --video dataset/video.mp4
    python app/run_demo.py --video dataset/video.mp4 --model weights/best_decoded_fp16.onnx
    python app/run_demo.py --video dataset/video.mp4 --show-preview --max-frames 500
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.detector import YOLODetector
from src.inference.counter import VehicleCounter
from src.inference.video_utils import VideoProcessor, draw_detections, draw_counts
from src.core.config import TRAFFIC_CONFIG, MODELS_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Analysis Demo")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--output", type=str, default=None, help="Path to output video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--show-preview", action="store_true", help="Show preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    return parser.parse_args()


def process_frame(frame, frame_idx, detector, counter, colors):
    detections = detector.detect_vehicles(frame)
    counter.update(detections)
    annotated = draw_detections(frame, detections, colors=colors, show_labels=True, show_confidence=True)
    counts = counter.get_frame_counts()
    annotated = draw_counts(annotated, counts)
    return annotated


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    input_video = Path(args.video)
    if not input_video.exists():
        logger.error(f"Input video not found: {input_video}")
        return 1

    # Resolve model path
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = MODELS_DIR / "best_decoded_fp16.onnx"
        if not model_path.exists():
            model_path = MODELS_DIR / "best_decoded.onnx"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"traffic_analysis_{input_video.stem}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Config
    colors = TRAFFIC_CONFIG.get("colors", {
        "car": (0, 255, 0), "truck": (0, 165, 255), "bus": (0, 0, 255),
        "motorcycle": (255, 0, 0), "bicycle": (255, 255, 0),
    })

    # Initialize
    logger.info("=" * 60)
    logger.info("TRAFFIC ANALYSIS DEMO")
    logger.info("=" * 60)

    logger.info("Initializing detector...")
    detector = YOLODetector(model_path=model_path, conf_threshold=args.conf, iou_threshold=args.iou)

    logger.info("Initializing video processor...")
    processor = VideoProcessor(input_path=input_video, output_path=output_path,
                               show_preview=args.show_preview, save_output=True)

    logger.info("Initializing vehicle counter...")
    counter = VehicleCounter()

    if not processor.open():
        logger.error("Failed to open video")
        return 1

    logger.info(f"Input: {input_video}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model: {model_path.name}")
    logger.info(f"Conf: {args.conf}, IoU: {args.iou}")

    try:
        for frame_idx, original_frame, processed_frame in processor.process_frames(
            process_fn=lambda f, idx: process_frame(f, idx, detector, counter, colors),
            max_frames=args.max_frames, show_progress=True
        ):
            if (frame_idx + 1) % 100 == 0:
                stats = counter.get_statistics()
                logger.info(f"Frame {frame_idx + 1}: {stats['total_vehicles']} vehicles detected")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        processor.close()

    # Final summary
    logger.info(f"{'=' * 60}")
    logger.info("PROCESSING COMPLETE")
    logger.info(f"{'=' * 60}")
    stats = counter.get_statistics()
    logger.info(f"Total frames: {processor.frames_processed}")
    logger.info(f"Total vehicles: {stats['total_vehicles']}")
    logger.info("Breakdown:")
    for class_name in ["car", "motorcycle", "bus", "truck", "bicycle"]:
        count = stats["per_class"].get(class_name, 0)
        logger.info(f"  {class_name:15s}: {count:4d}")
    logger.info(f"Output video: {output_path}")
    print("\n" + counter.get_summary_string())
    return 0


if __name__ == "__main__":
    exit(main())