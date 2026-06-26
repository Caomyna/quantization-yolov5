"""
Vehicle counting utilities for traffic analysis.
Handles counting, statistics tracking, and per-class counting.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class VehicleCounter:
    """
    Vehicle counter for traffic analysis.
    Tracks vehicle counts by class and provides statistics.
    """

    # COCO vehicle class IDs
    VEHICLE_CLASSES = {
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    def __init__(self):
        """Initialize vehicle counter."""
        # Total counts per class
        self.total_counts = defaultdict(int)
        
        # Frame-level counts (for current frame)
        self.frame_counts = defaultdict(int)
        
        # Overall total
        self.total_vehicles = 0
        
        # Class names mapping
        self.class_names = {
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

    def update(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Update counts based on current detections.
        
        Args:
            detections: List of detection dictionaries with 'class_id' field
            
        Returns:
            Dictionary with current frame counts per class
        """
        # Reset frame counts
        self.frame_counts = defaultdict(int)
        
        # Count vehicles in current frame
        for det in detections:
            class_id = det.get("class_id")
            if class_id in self.VEHICLE_CLASSES:
                self.frame_counts[class_id] += 1
                self.total_counts[class_id] += 1
                self.total_vehicles += 1
        
        return dict(self.frame_counts)

    def get_counts(self) -> Dict[str, int]:
        """
        Get total counts per vehicle class.
        
        Returns:
            Dictionary with class names as keys and counts as values
        """
        result = {}
        for class_id, count in self.total_counts.items():
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            result[class_name] = count
        
        return result

    def get_frame_counts(self) -> Dict[str, int]:
        """
        Get counts for current frame.
        
        Returns:
            Dictionary with class names as keys and counts as values
        """
        result = {}
        for class_id, count in self.frame_counts.items():
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            result[class_name] = count
        
        return result

    def get_total_count(self) -> int:
        """Get total vehicle count across all classes."""
        return self.total_vehicles

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics.
        
        Returns:
            Dictionary with all statistics
        """
        stats = {
            "total_vehicles": self.total_vehicles,
            "per_class": self.get_counts(),
            "per_class_frame": self.get_frame_counts(),
            "num_classes_detected": len(self.total_counts),
        }
        
        return stats

    def reset(self):
        """Reset all counters."""
        self.total_counts = defaultdict(int)
        self.frame_counts = defaultdict(int)
        self.total_vehicles = 0

    def print_summary(self):
        """Print formatted summary of counts."""
        print("\n" + "="*60)
        print("VEHICLE COUNT SUMMARY")
        print("="*60)
        
        counts = self.get_counts()
        for class_name in ["car", "motorcycle", "bus", "truck", "bicycle"]:
            count = counts.get(class_name, 0)
            print(f"{class_name:15s}: {count:4d}")
        
        print("-"*60)
        print(f"{'TOTAL':15s}: {self.total_vehicles:4d}")
        print("="*60 + "\n")

    def get_summary_string(self) -> str:
        """
        Get summary as formatted string.
        
        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("="*60)
        lines.append("VEHICLE COUNT SUMMARY")
        lines.append("="*60)
        
        counts = self.get_counts()
        for class_name in ["car", "motorcycle", "bus", "truck", "bicycle"]:
            count = counts.get(class_name, 0)
            lines.append(f"{class_name:15s}: {count:4d}")
        
        lines.append("-"*60)
        lines.append(f"{'TOTAL':15s}: {self.total_vehicles:4d}")
        lines.append("="*60)
        
        return "\n".join(lines)


class SimpleCounter:
    """
    Simple vehicle counter without tracking.
    Just counts detections per frame.
    """

    def __init__(self):
        """Initialize simple counter."""
        self.counts = defaultdict(int)
        self.class_names = {
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

    def count_frame(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Count vehicles in a single frame.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with counts for this frame
        """
        frame_counts = defaultdict(int)
        
        for det in detections:
            class_id = det.get("class_id")
            if class_id in self.class_names:
                frame_counts[class_id] += 1
                self.counts[class_id] += 1
        
        return {self.class_names[k]: v for k, v in frame_counts.items()}

    def get_total_counts(self) -> Dict[str, int]:
        """Get total counts across all frames."""
        return {self.class_names[k]: v for k, v in self.counts.items()}

    def get_grand_total(self) -> int:
        """Get grand total of all vehicles."""
        return sum(self.counts.values())

    def reset(self):
        """Reset all counts."""
        self.counts = defaultdict(int)