"""
app/services/extraction_service.py
Object extraction and preview generation.
"""
import logging
import time
import cv2
import numpy as np
from PIL import Image

import config

log = logging.getLogger('visiocraft.extraction')


class ExtractionService:
    """Handles extracting objects and creating segmentation previews."""

    def extract_object(self, image_path: str, mask_path: str,
                       session_id: str) -> str:
        """Extract object with transparent background using mask."""
        log.info("EXTRACT START — session=%s", session_id)
        log.debug("  image_path=%s", image_path)
        log.debug("  mask_path=%s", mask_path)
        t0 = time.time()

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            log.error("Failed to read image: %s", image_path)
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            log.error("Failed to read mask: %s", mask_path)
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        log.debug("  Image shape: %s, Mask shape: %s", image.shape, mask.shape)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            log.debug("  Converted BGR → BGRA")
        image[:, :, 3] = mask

        non_zero = cv2.findNonZero(mask)
        if non_zero is None:
            log.error("Mask is empty — no non-zero pixels found")
            raise ValueError("Mask is empty")

        x, y, w, h = cv2.boundingRect(non_zero)
        p = config.MASK_PADDING
        x, y = max(0, x - p), max(0, y - p)
        w = min(image.shape[1] - x, w + 2 * p)
        h = min(image.shape[0] - y, h + 2 * p)
        log.debug("  Bounding rect: x=%d, y=%d, w=%d, h=%d (padding=%d)",
                  x, y, w, h, p)

        out = config.get_output_path(f'extracted_{session_id}.png')
        cv2.imwrite(str(out), image[y:y + h, x:x + w])

        elapsed = (time.time() - t0) * 1000
        log.info("EXTRACT COMPLETE — session=%s, output=%s, size=%dx%d, %.0fms",
                 session_id, out, w, h, elapsed)
        return str(out)

    def create_preview(self, image_path: str, mask_path: str,
                       session_id: str) -> str:
        """Create green overlay preview of segmentation mask."""
        log.info("PREVIEW START — session=%s", session_id)
        t0 = time.time()

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            log.warning("Preview: could not load image/mask, returning original")
            return image_path

        if mask.shape != image.shape[:2]:
            log.debug("  Resizing mask from %s to %s",
                      mask.shape, image.shape[:2])
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        result = image.copy()
        green = np.zeros_like(image)
        green[:] = [0, 255, 0]
        mb = mask_bin > 127
        result[mb] = cv2.addWeighted(image[mb], 0.4, green[mb], 0.6, 0)

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 4)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
        log.debug("  Drew %d contours on preview", len(contours))

        out = config.get_temp_path(f'preview_{session_id}.png')
        cv2.imwrite(str(out), result)

        elapsed = (time.time() - t0) * 1000
        log.info("PREVIEW COMPLETE — session=%s, output=%s, %.0fms",
                 session_id, out, elapsed)
        return str(out)

    def resize_image(self, image_path: str, max_dim: int = None) -> str:
        """Resize image if it exceeds max dimension."""
        max_dim = max_dim or config.MAX_IMAGE_DIMENSION
        img = Image.open(image_path)
        w, h = img.size
        if w <= max_dim and h <= max_dim:
            log.debug("Resize: image %dx%d already within %d limit", w, h, max_dim)
            return image_path
        from pathlib import Path
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        log.info("Resize: %dx%d → %dx%d (scale=%.2f)", w, h, new_w, new_h, scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        out = Path(image_path).parent / f"resized_{Path(image_path).name}"
        img.save(out, quality=config.IMAGE_QUALITY)
        return str(out)
