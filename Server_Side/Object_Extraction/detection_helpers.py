"""Object_Extraction/detection_helpers.py: Detection-to-segmentation helpers and workflows"""
import numpy as np
from typing import List
from .detection_models import DetectionResult
from .object_detector import YOLODetector

class DetectionToSegmentation:
    """Converts YOLO detections to SAM-compatible point prompts."""

    @staticmethod
    def to_center_point(d: DetectionResult) -> dict:
        return {'x': d.center[0], 'y': d.center[1], 'label': 1}

    @staticmethod
    def to_points(d: DetectionResult, num_points: int = 5) -> List[dict]:
        x, y, w, h = d.bbox
        points = [{'x': x + w//2, 'y': y + h//2, 'label': 1}]
        grid = int(np.sqrt(num_points - 1)) + 1
        for i in range(grid):
            for j in range(grid):
                if len(points) >= num_points:
                    break
                points.append({'x': x + w*(i+1)//(grid+1),
                                'y': y + h*(j+1)//(grid+1), 'label': 1})
        return points[:num_points]

    @staticmethod
    def to_sam_box(d: DetectionResult) -> List[int]:
        x, y, w, h = d.bbox
        return [x, y, x+w, y+h]


def detect_and_segment(image_path: str, masking_service, session_id: str,
                       detect_class: str = 'person') -> dict:
    """Full YOLO → SAM workflow."""
    detector = YOLODetector()
    if not detector.load():
        return {'success': False, 'error': 'YOLO not available'}

    detections = detector.detect_from_file(image_path, filter_classes=[detect_class])
    if not detections:
        return {'success': False, 'error': f'No {detect_class} detected'}

    main = detector.get_largest(detections)
    points = DetectionToSegmentation.to_points(main, num_points=5)
    mask_path = masking_service.segment_from_points(image_path, points, session_id)

    return {
        'success': True,
        'detection': main.to_dict(),
        'detections_count': len(detections),
        'mask_path': mask_path,
        'points_used': points
    }