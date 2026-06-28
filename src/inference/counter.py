"""
Vehicle counting utilities for traffic analysis.
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

    def __init__(self):
        """Initialize vehicle counter."""
        # Total counts per class
        self.total_counts = defaultdict(int)
        
        # Frame-level counts (for current frame)
        self.frame_counts = defaultdict(int)
        
        # Overall total
        self.total_vehicles = 0
        
        # Class names mapping
        from ..core.config import VEHICLE_CLASS_IDS
        self.vehicle_class_ids = VEHICLE_CLASS_IDS
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
            if class_id in self.vehicle_class_ids:
                self.frame_counts[class_id] += 1
                self.total_counts[class_id] += 1
                self.total_vehicles += 1
        
        return {self.class_names.get(k, f"class_{k}"): v for k, v in self.frame_counts.items()}

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