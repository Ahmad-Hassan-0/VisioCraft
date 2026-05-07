"""
app/infrastructure/segmentation/sam_backend.py
MobileSAM segmentation backend.

Implements ISegmentationBackend with disk-backed LRU caching.
"""
import logging
import cv2
import hashlib
import pickle
import threading
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Optional

import config
from app.domain.interfaces import ISegmentationBackend
from app.domain.models import Image, Mask, Point

log = logging.getLogger('visiocraft.infra.sam')

# ── Disk cache ────────────────────────────────────────────────────────────────
_CACHE_DIR = Path(tempfile.gettempdir()) / "seg_cache"
_CACHE_DIR.mkdir(exist_ok=True)
_CACHE_LOCK = threading.Lock()
_MAX_CACHE_FILES = 200


def _cache_key(pixels: np.ndarray, points: List[Point], labels=None) -> str:
    h, w = pixels.shape[:2]
    step_y = max(1, h // 8)
    step_x = max(1, w // 8)
    sample = pixels[::step_y, ::step_x].tobytes()
    pt_str = str([(p.x, p.y) for p in points] + (labels or [])).encode()
    raw = f"{h}:{w}:{pixels.dtype}:".encode() + sample + pt_str
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def _cache_get(key: str) -> Optional[np.ndarray]:
    try:
        p = _CACHE_DIR / key
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


def _cache_set(key: str, arr: np.ndarray) -> None:
    with _CACHE_LOCK:
        try:
            with open(_CACHE_DIR / key, "wb") as f:
                pickle.dump(arr, f, protocol=4)
        except Exception:
            return
        entries = sorted(_CACHE_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
        while len(entries) > _MAX_CACHE_FILES:
            try:
                entries.pop(0).unlink()
            except Exception:
                break


def _best_mask(masks, points, labels, h, w) -> int:
    """Return the index of the mask that covers the last foreground click."""
    resized: dict = {}

    def get(j):
        if j not in resized:
            m = masks[j]
            resized[j] = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) \
                if m.shape != (h, w) else m
        return resized[j]

    for i in range(len(points) - 1, -1, -1):
        if labels[i] == 1:
            x, y = int(points[i][0]), int(points[i][1])
            if 0 <= y < h and 0 <= x < w:
                for j in range(len(masks)):
                    if get(j)[y, x] > 0.5:
                        return j
    return 0


class SAMBackend(ISegmentationBackend):
    """MobileSAM segmentation with disk-backed caching."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(config.SAM_MODEL_PATH)
        self.model = None
        self.is_loaded = False

    def load(self) -> bool:
        if not Path(self.model_path).exists():
            log.warning("SAM model not found: %s", self.model_path)
            return False
        try:
            from ultralytics import SAM
            log.info("Loading MobileSAM from %s...", self.model_path)
            self.model = SAM(self.model_path)
            self.is_loaded = True
            log.info("MobileSAM loaded successfully")
            return True
        except Exception as e:
            log.error("Failed to load SAM: %s", e, exc_info=True)
            return False

    def segment(self, image: Image, points: List[Point],
                labels: List[int] = None) -> Optional[Mask]:
        if not self.is_loaded:
            log.warning("SAM segment called but model not loaded")
            return None

        if labels is None:
            labels = [1] * len(points)

        log.debug("SAM segment: %d points, image=%dx%d",
                  len(points), image.width, image.height)

        pixels = image.data.pixels
        key = _cache_key(pixels, points, labels)
        cached = _cache_get(key)
        if cached is not None:
            log.debug("SAM cache HIT (key=%s)", key[:12])
            return Mask(cached)
        log.debug("SAM cache MISS (key=%s)", key[:12])

        try:
            coords = [[p.x, p.y] for p in points]
            results = self.model.predict(
                pixels,
                points=coords,
                labels=labels,
                verbose=False,
                conf=0.7,
                retina_masks=False,
            )
            if results and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                log.debug("SAM returned %d mask(s)", len(masks))
                if len(masks):
                    idx = _best_mask(masks, coords, labels, image.height, image.width)
                    log.debug("Selected mask index %d/%d", idx, len(masks))
                    mask_data = masks[idx]
                    if mask_data.shape != (image.height, image.width):
                        mask_data = cv2.resize(
                            mask_data, (image.width, image.height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    out = np.ascontiguousarray(mask_data > 0.5, dtype=np.uint8)
                    _cache_set(key, out)
                    log.debug("SAM mask created and cached")
                    return Mask(out)
            else:
                log.warning("SAM returned no masks")
        except Exception as e:
            log.error("SAM segmentation error: %s", e, exc_info=True)
        return None
