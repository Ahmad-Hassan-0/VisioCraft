"""
app/domain/models.py
Consolidated domain models for VisioCraft.
Pure data classes with no infrastructure dependencies.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, Optional


# ─── Geometry primitives ─────────────────────────────────────────────────────

@dataclass
class Point:
    """A 2D point in image coordinates."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))

    def distance(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    x: int
    y: int
    width: int
    height: int

    def get_area(self) -> int:
        return self.width * self.height

    def expand(self, pixels: int) -> 'BoundingBox':
        return BoundingBox(
            max(0, self.x - pixels),
            max(0, self.y - pixels),
            self.width + 2 * pixels,
            self.height + 2 * pixels,
        )


# ─── Image domain objects ────────────────────────────────────────────────────

class ImageData:
    """Raw pixel data with metadata."""
    def __init__(self, pixels: np.ndarray, channels: int, color_space: str = "RGB"):
        self.pixels = pixels
        self.channels = channels
        self.color_space = color_space
        self.width = pixels.shape[1]
        self.height = pixels.shape[0]

    def clone(self) -> 'ImageData':
        return ImageData(self.pixels.copy(), self.channels, self.color_space)


class Image:
    """Domain image wrapper."""
    def __init__(self, data: ImageData, image_id: str = None):
        self.image_id = image_id or f"img_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.data = data
        self.width = data.width
        self.height = data.height

    @classmethod
    def from_file(cls, path: str) -> 'Image':
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return cls(ImageData(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 3, "RGB"))

    def crop(self, box: BoundingBox) -> 'Image':
        cropped = self.data.pixels[box.y:box.y + box.height, box.x:box.x + box.width]
        return Image(ImageData(cropped.copy(), self.data.channels, self.data.color_space))


# ─── Mask ─────────────────────────────────────────────────────────────────────

class Mask:
    """Binary segmentation mask."""
    def __init__(self, data: np.ndarray, mask_id: str = None):
        self.mask_id = mask_id or f"mask_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.mask_data = (data > 0.5).astype(np.uint8)
        self.height, self.width = data.shape[:2]

    def get_bounding_box(self) -> BoundingBox:
        coords = cv2.findNonZero(self.mask_data)
        if coords is None:
            return BoundingBox(0, 0, 0, 0)
        x, y, w, h = cv2.boundingRect(coords)
        return BoundingBox(x, y, w, h)

    def expand(self, pixels: int) -> 'Mask':
        k = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
        return Mask(cv2.dilate(self.mask_data, k, iterations=1))

    def blur(self, radius: int) -> 'Mask':
        blurred = cv2.GaussianBlur(
            self.mask_data.astype(float),
            (radius * 2 + 1, radius * 2 + 1), 0,
        )
        return Mask(blurred)

    def save(self, path: str) -> bool:
        try:
            cv2.imwrite(path, (self.mask_data * 255).astype(np.uint8))
            return True
        except Exception as e:
            print(f"✗ Mask save error: {e}")
            return False


# ─── Extracted object ─────────────────────────────────────────────────────────

class ExtractedObject:
    """An object extracted from an image with its mask and bounding box."""
    def __init__(self, image: Image, mask: Mask, bbox: BoundingBox,
                 source_path: str = None):
        self.object_id = f"extracted_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.image = image
        self.mask = mask
        self.bbox = bbox
        self.source_path = source_path


# ─── Detection result ────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Result of object detection (e.g. from YOLO)."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x, y, w, h
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
            'area': self.get_area(),
        }
