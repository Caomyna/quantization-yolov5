"""
Traffic Analysis Demo
Vehicle detection and counting using YOLOv5 ONNX models.

Usage:
    python app/main.py --video dat/video.mp4 --model weights/magnitude_0.3_decoded_fp16.onnx
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import numpy as np
from typing import Dict, Any

from utils.detector import YOLODetector
from utils.counter import VehicleCounter
from utils.video_processor import VideoProcessor, draw_detections, draw_counts
from quantize.config import TRAFFIC_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Traffic Analysis Demo")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to ONNX model (default: magnitude_0.3_decoded_fp16.onnx)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output video (default: output/traffic_analysis_output.mp4)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show preview window during processing"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all)"
    )
    
    return parser.parse_args()


def process_frame(
    frame,
    frame_idx: int,
    detector: YOLODetector,
    counter: VehicleCounter,
    colors: Dict[str, tuple]
) -> np.ndarray:
    """
    Process a single frame.
    
    Args:
        frame: Input frame
        frame_idx: Frame index
        detector: YOLO detector
        counter: Vehicle counter
        colors: Color mapping for classes
        
    Returns:
        Annotated frame
    """
    # Detect vehicles
    detections = detector.detect_vehicles(frame)
    
    # Update counter
    counter.update(detections)
    
    # Draw detections
    annotated = draw_detections(
        frame,
        detections,
        colors=colors,
        show_labels=True,
        show_confidence=True
    )
    
    # Draw counts
    counts = counter.get_frame_counts()
    annotated = draw_counts(annotated, counts)
    
    return annotated


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup paths
    input_video = Path(args.video)
    if not input_video.exists():
        logger.error(f"Input video not found: {input_video}")
        return 1
    
    # Model path
    if args.model:
        model_path = Path(args.model)
    else:
        # Default model
        model_path = Path("weights") / "magnitude_0.3_decoded_fp16.onnx"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1
    
    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("output") / "traffic_analysis_output.mp4"
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get configuration
    vehicle_classes = TRAFFIC_CONFIG.get("vehicle_class_ids", [1, 2, 3, 5, 7])
    colors = TRAFFIC_CONFIG.get("colors", {
        "car": (0, 255, 0),
        "truck": (0, 165, 255),
        "bus": (0, 0, 255),
        "motorcycle": (255, 0, 0),
        "bicycle": (255, 255, 0),
    })
    
    # Initialize components
    logger.info("="*60)
    logger.info("TRAFFIC ANALYSIS DEMO")
    logger.info("="*60)
    
    logger.info(f"\nInitializing detector...")
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        vehicle_classes=vehicle_classes
    )
    
    logger.info(f"\nInitializing video processor...")
    processor = VideoProcessor(
        input_path=input_video,
        output_path=output_path,
        show_preview=args.show_preview,
        save_output=True
    )
    
    logger.info(f"\nInitializing vehicle counter...")
    counter = VehicleCounter()
    
    # Open video
    if not processor.open():
        logger.error("Failed to open video")
        return 1
    
    # Print info
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING")
    logger.info(f"{'='*60}")
    logger.info(f"Input: {input_video}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model: {model_path.name}")
    logger.info(f"Confidence threshold: {args.conf_threshold}")
    logger.info(f"IoU threshold: {args.iou_threshold}")
    logger.info(f"Max frames: {args.max_frames or 'All'}")
    logger.info(f"{'='*60}\n")
    
    # Process frames
    try:
        for frame_idx, original_frame, processed_frame in processor.process_frames(
            process_fn=lambda f, idx: process_frame(f, idx, detector, counter, colors),
            max_frames=args.max_frames,
            show_progress=True
        ):
            # Log progress every 100 frames
            if (frame_idx + 1) % 100 == 0:
                stats = counter.get_statistics()
                logger.info(f"Frame {frame_idx + 1}: {stats['total_vehicles']} vehicles detected")
    
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Close video
        processor.close()
    
    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    
    stats = counter.get_statistics()
    logger.info(f"\nTotal frames processed: {processor.frames_processed}")
    logger.info(f"Total vehicles detected: {stats['total_vehicles']}")
    logger.info(f"\nBreakdown by class:")
    for class_name in ["car", "motorcycle", "bus", "truck", "bicycle"]:
        count = stats["per_class"].get(class_name, 0)
        logger.info(f"  {class_name:15s}: {count:4d}")
    
    logger.info(f"\nOutput video: {output_path}")
    logger.info(f"{'='*60}\n")
    
    # Print summary to console
    print("\n" + counter.get_summary_string())
    
    return 0


if __name__ == "__main__":
    exit(main())



# python app/main.py --video dataset/video_test.mp4 --model weights/best_decoded.onnx --show-preview --max-frames 500