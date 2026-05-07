"""
app/services/composition_service.py
Server-side image composition — fixes the missing /api/compose endpoint.
"""
import logging
import time
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

import config

log = logging.getLogger('visiocraft.composition')


class CompositionService:
    """Composes extracted object onto inpainted background with transforms."""

    def compose(self, session_data: dict, transform: dict) -> str:
        session_id = session_data.get('id', 'unknown')
        log.info("COMPOSE START — session=%s", session_id)
        log.debug("  Transform: %s", transform)
        t0 = time.time()

        inpainted_path = session_data.get('inpainted_path')
        extracted_path = session_data.get('extracted_path')

        log.debug("  inpainted_path=%s", inpainted_path)
        log.debug("  extracted_path=%s", extracted_path)

        if not inpainted_path or not extracted_path:
            log.error("Missing paths: inpainted=%s, extracted=%s",
                      inpainted_path, extracted_path)
            raise ValueError("Missing inpainted or extracted image paths")
        if not Path(inpainted_path).exists() or not Path(extracted_path).exists():
            log.error("Files not found on disk")
            raise FileNotFoundError("Inpainted or extracted image file not found")

        # Load images
        bg = Image.open(inpainted_path).convert("RGBA")
        obj = Image.open(extracted_path).convert("RGBA")
        log.debug("  BG size: %dx%d, Object size: %dx%d",
                  bg.width, bg.height, obj.width, obj.height)

        # Apply transform
        x = transform.get('x', bg.width // 2)
        y = transform.get('y', bg.height // 2)
        rotation = transform.get('rotation', 0)
        scale = transform.get('scale', 1.0)
        opacity = transform.get('opacity', 1.0)
        log.debug("  Applying: x=%d, y=%d, rot=%d, scale=%.2f, opacity=%.2f",
                  x, y, rotation, scale, opacity)

        if scale != 1.0:
            new_w = max(1, int(obj.width * scale))
            new_h = max(1, int(obj.height * scale))
            obj = obj.resize((new_w, new_h), Image.LANCZOS)
            log.debug("  Scaled object to %dx%d", new_w, new_h)

        if rotation != 0:
            obj = obj.rotate(-rotation, expand=True, resample=Image.BICUBIC)
            log.debug("  Rotated object by %d degrees", rotation)

        if opacity < 1.0:
            alpha = obj.split()[3]
            alpha = alpha.point(lambda a: int(a * opacity))
            obj.putalpha(alpha)
            log.debug("  Applied opacity %.2f", opacity)

        paste_x = int(x - obj.width / 2)
        paste_y = int(y - obj.height / 2)
        log.debug("  Paste position: (%d, %d)", paste_x, paste_y)

        composite = bg.copy()
        composite.paste(obj, (paste_x, paste_y), obj)

        result = composite.convert("RGB")
        out_path = config.get_output_path(f'composite_{session_id}.png')
        result.save(str(out_path), quality=config.IMAGE_QUALITY)

        elapsed = (time.time() - t0) * 1000
        log.info("COMPOSE COMPLETE — session=%s, output=%s, %.0fms",
                 session_id, out_path, elapsed)
        return str(out_path)
