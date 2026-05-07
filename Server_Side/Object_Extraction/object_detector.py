"""Object_Extraction/object_detector.py: YOLO model loading and inference"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import config
from .detection_models import DetectionResult

COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

class YOLODetector:
    """Loads YOLO model and runs object detection."""

    def __init__(self, model_path: str = None, confidence: float = 0.5):
        self.model_path = model_path or str(config.YOLO_MODEL_PATH)
        self.confidence = confidence
        self.model = None
        self.is_loaded = False

    def load(self) -> bool:
        if not Path(self.model_path).exists():
            print(f"⚠️  YOLO model not found: {self.model_path}")
            return False
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            print("✓ YOLO loaded")
            return True
        except Exception as e:
            print(f"✗ YOLO load failed: {e}")
            return False

    def detect(self, image: np.ndarray, filter_classes: Optional[List[str]] = None) -> List[DetectionResult]:
        if not self.is_loaded:
            return []
        try:
            results = self.model.predict(image, conf=self.confidence, verbose=False)
            detections = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes.cpu().numpy():
                    cid = int(box.cls[0])
                    name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
                    if filter_classes and name not in filter_classes:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0]
                    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                    detections.append(DetectionResult(
                        class_id=cid, class_name=name,
                        confidence=float(box.conf[0]),
                        bbox=(x, y, w, h),
                        center=(x + w//2, y + h//2)
                    ))
            return detections
        except Exception as e:
            print(f"✗ Detection error: {e}")
            return []

    def detect_from_file(self, image_path: str, filter_classes: Optional[List[str]] = None) -> List[DetectionResult]:
        img = cv2.imread(image_path)
        if img is None:
            return []
        return self.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), filter_classes)

    def get_largest(self, detections: List[DetectionResult]) -> Optional[DetectionResult]:
        return max(detections, key=lambda d: d.get_area()) if detections else None

    def filter_by_confidence(self, detections: List[DetectionResult], min_conf: float) -> List[DetectionResult]:
        return [d for d in detections if d.confidence >= min_conf]

# Backwards compatibility
ObjectDetection = YOLODetector