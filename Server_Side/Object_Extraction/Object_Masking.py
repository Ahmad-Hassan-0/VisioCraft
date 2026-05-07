"""Object_Extraction/object_masking.py: ObjectMasking service: orchestrates SAM + fallback"""
import numpy as np
from pathlib import Path
from typing import List, Optional
import config
from .masking_models import Image, Mask, BoundingBox, ExtractedObject, ImageData
from .segmentation import SAMModel, FallbackSegmentation

class ObjectMasking:
    """Orchestrates segmentation: SAM first, GrabCut fallback"""

    def __init__(self, model_path: str = None):
        self.sam = None
        self.sam_available = False
        self._load_sam(model_path or str(config.SAM_MODEL_PATH))

    def _load_sam(self, path: str):
        if Path(path).exists():
            sam = SAMModel(path)
            if sam.load():
                self.sam = sam
                self.sam_available = True
                print("✅ Using SAM segmentation")
                return
        print("⚠️  SAM not found, using GrabCut fallback")

    def segment_from_points(self, image_path: str, points: List[dict], session_id: str) -> str:
        """Web API entry point — returns path to saved mask."""
        from .masking_models import Point
        image = Image.from_file(image_path)
        pts = [Point(p['x'], p['y']) for p in points]
        labels = [p.get('label', 1) for p in points]

        mask = None
        if self.sam_available:
            mask = self.sam.segment_with_points(image, pts, labels)

        if mask is None:
            mask = FallbackSegmentation.segment(image, pts)

        if mask is None:
            raise RuntimeError("Segmentation failed")

        # Refine edges
        mask = mask.expand(1).blur(2)

        out = config.get_temp_path(f'mask_{session_id}.png')
        mask.save(str(out))
        return str(out)

    def extract_object(self, image: Image, mask: Mask,
                       padding: int = 10, source_path: str = None) -> Optional[ExtractedObject]:
        """Extract object as RGBA with transparent background."""
        try:
            bbox = mask.get_bounding_box()
            if bbox.get_area() == 0:
                return None

            eb = bbox.expand(padding)
            eb.x = max(0, eb.x)
            eb.y = max(0, eb.y)
            eb.width = min(image.width - eb.x, eb.width)
            eb.height = min(image.height - eb.y, eb.height)

            cropped = image.crop(eb)
            h, w = cropped.height, cropped.width
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = cropped.data.pixels
            rgba[:, :, 3] = mask.mask_data[eb.y:eb.y+h, eb.x:eb.x+w] * 255

            return ExtractedObject(Image(ImageData(rgba, 4, "RGBA")), mask, eb, source_path)
        except Exception as e:
            print(f"✗ Extraction error: {e}")
            return None