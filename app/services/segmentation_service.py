"""
app/services/segmentation_service.py
Orchestrates segmentation backends with fallback chain: SAM → GrabCut.
"""
import logging
import time
import cv2
from typing import List

import config
from app.domain.models import Image, Mask, Point, ImageData
from app.domain.interfaces import ISegmentationBackend
from app.infrastructure.segmentation.sam_backend import SAMBackend
from app.infrastructure.segmentation.grabcut_backend import GrabCutBackend

log = logging.getLogger('visiocraft.segmentation')


class SegmentationService:
    """Orchestrates segmentation: SAM first, GrabCut fallback."""

    def __init__(self, model_path: str = None):
        self.backends: List[ISegmentationBackend] = []
        self.sam_backend = SAMBackend(model_path or str(config.SAM_MODEL_PATH))
        self.sam_available = False

        log.info("Initializing SegmentationService...")

        # 1. Register GrabCut immediately as a zero-wait fallback
        self.backends.append(GrabCutBackend())
        log.info("  GrabCut fallback registered (total backends: %d)", len(self.backends))

        # 2. Load SAM in background thread to avoid blocking server boot
        import threading
        log.info("  Starting background SAM loading thread...")
        threading.Thread(target=self._background_load_sam, daemon=True).start()

    def _background_load_sam(self):
        """Background thread for loading the heavy SAM model."""
        log.info("BACKGROUND SAM LOAD: Started")
        t0 = time.time()
        if self.sam_backend.load():
            # Insert at position 0 to make it the primary backend
            self.backends.insert(0, self.sam_backend)
            self.sam_available = True
            elapsed = time.time() - t0
            log.info("BACKGROUND SAM LOAD: Success (%.1fs) — MobileSAM is now the primary engine", elapsed)
        else:
            log.warning("BACKGROUND SAM LOAD: Failed — project will continue using GrabCut")

    def segment_from_points(self, image_path: str, points: List[dict],
                            session_id: str) -> str:
        """Segment an image using point prompts. Returns path to saved mask."""
        log.info("SEGMENT START — session=%s, points=%d, image=%s",
                 session_id, len(points), image_path)

        t0 = time.time()

        # Load image
        log.debug("Loading image from: %s", image_path)
        image = Image.from_file(image_path)
        log.debug("Image loaded: %dx%d, color_space=%s",
                  image.width, image.height, image.data.color_space)

        pts = [Point(p['x'], p['y']) for p in points]
        labels = [p.get('label', 1) for p in points]
        log.debug("Points converted: %d points, labels=%s",
                  len(pts), labels[:10])

        # Downscale for performance if image is too large
        scale = 1.0
        max_dim = 1024 # Target dimension for SAM speed
        if max(image.width, image.height) > max_dim:
            scale = max_dim / max(image.width, image.height)
            new_w, new_h = int(image.width * scale), int(image.height * scale)
            log.info("  Downscaling image for speed: %dx%d → %dx%d (scale=%.2f)", 
                     image.width, image.height, new_w, new_h, scale)
            
            # Create a scaled copy of the pixel data
            scaled_pixels = cv2.resize(image.data.pixels, (new_w, new_h), interpolation=cv2.INTER_AREA)
            seg_image = Image(ImageData(scaled_pixels, image.data.channels, image.data.color_space))
            
            # Rescale points
            seg_pts = [Point(p.x * scale, p.y * scale) for p in pts]
        else:
            seg_image = image
            seg_pts = pts

        mask = None
        for i, backend in enumerate(self.backends):
            backend_name = backend.__class__.__name__
            log.info("  Trying backend %d/%d: %s",
                     i + 1, len(self.backends), backend_name)
            bt0 = time.time()
            try:
                mask = backend.segment(seg_image, seg_pts, labels)
                elapsed_b = (time.time() - bt0) * 1000
                if mask is not None:
                    log.info("  ✅ %s succeeded in %.0fms (mask: %dx%d)",
                             backend_name, elapsed_b, mask.width, mask.height)
                    
                    # Upscale mask back to original resolution if needed
                    if scale != 1.0:
                        log.debug("  Upscaling mask back to %dx%d", image.width, image.height)
                        upscaled = cv2.resize(mask.mask_data, (image.width, image.height), 
                                            interpolation=cv2.INTER_NEAREST)
                        mask = Mask(upscaled)
                    break
                else:
                    log.warning("  ⚠️ %s returned None after %.0fms",
                                backend_name, elapsed_b)
            except Exception as e:
                elapsed_b = (time.time() - bt0) * 1000
                log.error("  ❌ %s failed after %.0fms: %s",
                          backend_name, elapsed_b, e, exc_info=True)

        if mask is None:
            log.error("SEGMENT FAILED — all %d backends exhausted for session=%s",
                      len(self.backends), session_id)
            raise RuntimeError("Segmentation failed with all backends")

        # Refine edges
        log.debug("Refining mask edges (expand=1, blur=2)...")
        mask = mask.expand(1).blur(2)

        out = config.get_temp_path(f'mask_{session_id}.png')
        mask.save(str(out))

        elapsed = (time.time() - t0) * 1000
        log.info("SEGMENT COMPLETE — session=%s, output=%s, total=%.0fms",
                 session_id, out, elapsed)
        return str(out)
