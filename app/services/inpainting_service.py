"""
app/services/inpainting_service.py
Orchestrates inpainting backends with fallback chain.
"""
import logging
import time
import traceback
import cv2
import numpy as np
from PIL import Image

import config
from app.domain.interfaces import IInpaintingBackend
from app.infrastructure.inpainting.opencv_inpainter import OpenCVInpainter
from app.infrastructure.inpainting.diffusers_inpainter import DiffusersInpainter

log = logging.getLogger('visiocraft.inpainting')


class InpaintingService:
    """Orchestrates inpainting with primary backend + OpenCV emergency fallback."""

    def __init__(self):
        self.primary: IInpaintingBackend = None
        self.fallback = OpenCVInpainter()
        self.available = False

        log.info("Initializing InpaintingService...")

        # Try Diffusers SD first
        diffusers_inpainter = DiffusersInpainter()
        if diffusers_inpainter.available:
            self.primary = diffusers_inpainter
            self.available = True
            log.info("Primary inpainter: Diffusers (Local)")
        else:
            self.primary = self.fallback
            self.available = True
            log.info("Primary inpainter: OpenCV (Diffusers failed to load)")

        # ── BACKGROUND WARMUP ────────────────────────────────────────────────
        # Start a separate thread to warm up models so they don't delay the 1st request.
        import threading
        log.info("Starting background warmup thread for inpainting models...")
        threading.Thread(target=self._background_warmup, daemon=True).start()

    def _background_warmup(self):
        """Sequential warmup: Fast OpenCV first, then the heavy Primary model."""
        t_start = time.time()
        try:
            # 1. OpenCV Warmup (Fast)
            if hasattr(self.fallback, 'warmup'):
                self.fallback.warmup()
            
            # 2. Primary Model Warmup (Heavy - this triggers model load if lazy)
            if self.primary and hasattr(self.primary, 'warmup'):
                self.primary.warmup()
            
            elapsed = time.time() - t_start
            log.info("BACKGROUND WARMUP COMPLETE in %.1fs", elapsed)
            from app.services.progress_manager import progress_manager
            progress_manager.set_system_ready(True)
        except Exception as e:
            log.error("Background warmup failed: %s", e)

    def inpaint_background(self, image_path: str, mask_path: str,
                           session_id: str = "") -> str:
        """Inpaint the masked region, save result, return output path."""
        log.info("INPAINT START — session=%s", session_id)
        log.debug("  image_path=%s", image_path)
        log.debug("  mask_path=%s", mask_path)
        t0 = time.time()

        # ── 1. MASK EXPANSION ──────────────────────────────────────────────────
        # We expand the mask slightly so the model sees more "into" the edges
        # of the original object, resulting in much better seamless blending.
        from app.services.progress_manager import progress_manager
        progress_manager.set_progress(session_id, 5, "Expanding mask for context...")
        
        log.info("  Expanding mask for better blending context...")
        expanded_mask_path = self._expand_mask(mask_path, session_id)

        try:
            log.info("  Using primary backend: %s",
                     self.primary.__class__.__name__)
            progress_manager.set_progress(session_id, 10, f"Initializing {self.primary.__class__.__name__}...")
            
            # Pass the expanded mask to the backend
            result = self.primary.inpaint(image_path, expanded_mask_path, session_id=session_id)
            elapsed_b = (time.time() - t0) * 1000
            log.info("  ✅ Primary inpainting succeeded in %.0fms", elapsed_b)
            progress_manager.set_progress(session_id, 100, "Finishing up...")
        except Exception as e:
            elapsed_b = (time.time() - t0) * 1000
            log.error("  ❌ Primary inpainting failed after %.0fms: %s",
                      elapsed_b, e, exc_info=True)
            log.info("  Falling back to emergency OpenCV...")
            # Use original mask for emergency as it might be safer if expansion failed
            result = self._emergency(image_path, mask_path)
            log.info("  Emergency OpenCV completed")

        out_path = config.get_output_path(f'inpainted_{session_id}.jpg')
        Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save(
            out_path, quality=config.IMAGE_QUALITY)

        elapsed = (time.time() - t0) * 1000
        log.info("INPAINT COMPLETE — session=%s, output=%s, %.0fms",
                 session_id, out_path, elapsed)
        
        # Cleanup progress tracking
        from app.services.progress_manager import progress_manager
        progress_manager.clear_progress(session_id)
        
        return str(out_path)

    def _expand_mask(self, mask_path: str, session_id: str) -> str:
        """Dilates the mask by 15px to ensure the model removes the entire object boundary."""
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return mask_path
            
            # Expansion radius: 15px is a good balance for SD
            radius = 15
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
            dilated = cv2.dilate(mask, kernel, iterations=1)
            
            expanded_path = config.get_temp_path(f'mask_expanded_{session_id}.png')
            cv2.imwrite(str(expanded_path), dilated)
            log.debug("  Expanded mask saved to %s", expanded_path)
            return str(expanded_path)
        except Exception as e:
            log.warning("  Mask expansion failed: %s", e)
            return mask_path

    def _emergency(self, image_path: str, mask_path: str) -> np.ndarray:
        """Last-resort OpenCV inpainting."""
        log.debug("  Emergency: loading image and mask...")
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        result = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS)
        log.debug("  Emergency: OpenCV NS inpaint done")
        return result
