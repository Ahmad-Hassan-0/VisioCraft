"""Object_Extraction/detection_models.py: Data models for detection results"""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DetectionResult:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]

    def get_area(self) -> int:
        return self.bbox[2] * self.bbox[3]

    def contains_point(self, x: int, y: int) -> bool:
        bx, by, bw, bh = self.bbox
        return bx <= x <= bx + bw and by <= y <= by + bh

    def to_dict(self) -> dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'bbox': {'x': self.bbox[0], 'y': self.bbox[1],
                     'width': self.bbox[2], 'height': self.bbox[3]},
            'center': {'x': self.center[0], 'y': self.center[1]},
            'area': self.get_area()
        }