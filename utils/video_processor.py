"""
Video processing utilities for traffic analysis.
Handles video I/O, frame processing, and output video generation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Generator
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processor for traffic analysis.
    Handles video reading, frame processing, and video writing.
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        show_preview: bool = False,
        save_output: bool = True,
    ):
        """
        Initialize video processor.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video (optional)
            show_preview: Whether to show preview window
            save_output: Whether to save output video
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else None
        self.show_preview = show_preview
        self.save_output = save_output
        
        # Video capture
        self.cap = None
        self.writer = None
        
        # Video properties
        self.fps = 0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        
        # Statistics
        self.frames_processed = 0

    def open(self) -> bool:
        """
        Open video file.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.input_path.exists():
            logger.error(f"Input video not found: {self.input_path}")
            return False
        
        # Open video capture
        self.cap = cv2.VideoCapture(str(self.input_path))
        if not self.cap.isOpened():
            logger.error(f"Cannot open video: {self.input_path}")
            return False
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Opened video: {self.input_path.name}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps}")
        logger.info(f"  Total frames: {self.total_frames}")
        
        # Initialize video writer if needed
        if self.save_output and self.output_path:
            self._init_writer()
        
        return True

    def _init_writer(self):
        """Initialize video writer."""
        if self.output_path is None:
            return
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create writer
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        logger.info(f"Output video: {self.output_path}")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame

    def write_frame(self, frame: np.ndarray):
        """
        Write frame to output video.
        
        Args:
            frame: Frame to write
        """
        if self.writer is not None:
            self.writer.write(frame)

    def process_frames(
        self,
        process_fn,
        max_frames: Optional[int] = None,
        show_progress: bool = True,
    ) -> Generator[Tuple[int, np.ndarray, Any], None, None]:
        """
        Process video frames with a custom function.
        
        Args:
            process_fn: Function to process each frame (frame, frame_idx) -> processed_frame
            max_frames: Maximum number of frames to process (None for all)
            show_progress: Whether to show progress bar
            
        Yields:
            Tuple of (frame_index, original_frame, processed_frame)
        """
        if not self.cap or not self.cap.isOpened():
            logger.error("Video not opened. Call open() first.")
            return
        
        # Determine number of frames to process
        num_frames = min(self.total_frames, max_frames) if max_frames else self.total_frames
        
        # Progress bar
        pbar = tqdm(total=num_frames, desc="Processing") if show_progress else None
        
        frame_idx = 0
        while frame_idx < num_frames:
            ret, frame = self.read_frame()
            if not ret:
                break
            
            # Process frame
            processed_frame = process_fn(frame, frame_idx)
            
            # Write to output
            if self.save_output and self.writer:
                self.write_frame(processed_frame)
            
            # Show preview
            if self.show_preview:
                cv2.imshow("Traffic Analysis", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Preview stopped by user")
                    break
            
            # Update progress
            if pbar:
                pbar.update(1)
            
            self.frames_processed += 1
            
            yield frame_idx, frame, processed_frame
            
            frame_idx += 1
        
        if pbar:
            pbar.close()

    def close(self):
        """Release video resources."""
        if self.cap:
            self.cap.release()
        
        if self.writer:
            self.writer.release()
        
        if self.show_preview:
            cv2.destroyAllWindows()
        
        logger.info(f"Processed {self.frames_processed} frames")

    def get_info(self) -> Dict[str, Any]:
        """
        Get video information.
        
        Returns:
            Dictionary with video properties
        """
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path) if self.output_path else None,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "frames_processed": self.frames_processed,
        }


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    class_names: Dict[int, str] = None,
    colors: Dict[str, Tuple[int, int, int]] = None,
    show_labels: bool = True,
    show_confidence: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw detections on frame.
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries
        class_names: Dictionary mapping class_id to class_name
        colors: Dictionary mapping class_name to BGR color
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores
        line_thickness: Thickness of bounding box lines
        
    Returns:
        Frame with drawn detections
    """
    annotated = frame.copy()
    
    # Default colors (BGR format)
    if colors is None:
        colors = {
            "car": (0, 255, 0),
            "truck": (0, 165, 255),
            "bus": (0, 0, 255),
            "motorcycle": (255, 0, 0),
            "bicycle": (255, 255, 0),
        }
    
    # Default class names
    if class_names is None:
        class_names = {
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
    
    for det in detections:
        bbox = det["bbox"]
        class_id = det.get("class_id", 0)
        confidence = det.get("confidence", 0.0)
        
        # Get class name and color
        class_name = class_names.get(class_id, f"class_{class_id}")
        color = colors.get(class_name, (0, 255, 0))
        
        # Convert bbox to integers
        x, y, w, h = [int(v) for v in bbox]
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label
        if show_labels:
            label = class_name
            if show_confidence:
                label += f" {confidence:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    return annotated


def draw_counts(
    frame: np.ndarray,
    counts: Dict[str, int],
    position: Tuple[int, int] = (20, 40),
    font_scale: float = 0.8,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw vehicle counts on frame.
    
    Args:
        frame: Input frame
        counts: Dictionary with class counts
        position: Top-left position for text
        font_scale: Font size
        thickness: Text thickness
        
    Returns:
        Frame with drawn counts
    """
    annotated = frame.copy()
    x, y = position
    
    # Draw background
    text_lines = []
    text_lines.append("Vehicle Counts:")
    for class_name in ["car", "motorcycle", "bus", "truck", "bicycle"]:
        count = counts.get(class_name, 0)
        text_lines.append(f"  {class_name}: {count}")
    
    # Calculate background size
    max_width = 0
    for line in text_lines:
        (width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_width = max(max_width, width)
    
    line_height = int(30 * font_scale)
    bg_height = len(text_lines) * line_height + 20
    
    # Draw semi-transparent background
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (x - 10, y - 30),
        (x + max_width + 20, y + bg_height),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    
    # Draw text
    for i, line in enumerate(text_lines):
        y_pos = y + i * line_height
        cv2.putText(
            annotated,
            line,
            (x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
    
    return annotated