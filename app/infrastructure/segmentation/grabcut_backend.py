"""
app/infrastructure/segmentation/grabcut_backend.py
GrabCut fallback segmentation backend.

ROI-based, RAM-efficient implementation with disk caching.
"""
import cv2
import numpy as np
import hashlib
import pickle
import threading
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from app.domain.interfaces import ISegmentationBackend
from app.domain.models import Image, Mask, Point

# ── Global OpenCV setup ──────────────────────────────────────────────────────
cv2.setNumThreads(0)
cv2.setUseOptimized(True)

# ── Disk cache (shared with SAM) ─────────────────────────────────────────────
_CACHE_DIR = Path(tempfile.gettempdir()) / "seg_cache"
_CACHE_DIR.mkdir(exist_ok=True)
_CACHE_LOCK = threading.Lock()
_MAX_CACHE_FILES = 200

# ── GrabCut constants ────────────────────────────────────────────────────────
_MAX_GC_DIM = 512
_GC_BGD = np.zeros((1, 65), np.float64)
_GC_FGD = np.zeros((1, 65), np.float64)


def _cache_key(pixels: np.ndarray, points: List[Point]) -> str:
    h, w = pixels.shape[:2]
    step_y = max(1, h // 8)
    step_x = max(1, w // 8)
    sample = pixels[::step_y, ::step_x].tobytes()
    pt_str = str([(p.x, p.y) for p in points]).encode()
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


# ── Filter helpers ───────────────────────────────────────────────────────────

def _clean(mask: np.ndarray, click: Tuple) -> np.ndarray:
    """Keep only the connected component that contains the click pixel."""
    n, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    lbl = labels[click[1], click[0]]
    return np.uint8(labels == lbl) if lbl > 0 else mask


def _distance_filter(mask: np.ndarray, click: Tuple, max_dist: int) -> np.ndarray:
    h, w = mask.shape
    y_c, x_c = np.ogrid[:h, :w]
    dist_sq = (x_c - click[0]) ** 2 + (y_c - click[1]) ** 2
    return np.uint8(mask & (dist_sq <= max_dist * max_dist))


def _color_filter(mask: np.ndarray, img: np.ndarray, click: Tuple) -> np.ndarray:
    cx, cy = click
    if not (0 <= cy < img.shape[0] and 0 <= cx < img.shape[1]):
        return mask
    color = img[cy, cx].astype(np.float16)
    diff = img.astype(np.float16) - color
    diff_sq = np.einsum("ijk,ijk->ij", diff, diff)
    refined = np.uint8(mask & (diff_sq < 1600))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    if n <= 1:
        return refined
    areas = stats[:, cv2.CC_STAT_AREA]
    lut = np.uint8(areas >= 20)
    lut[0] = 0
    return lut[labels]


class GrabCutBackend(ISegmentationBackend):
    """ROI-based, RAM-efficient GrabCut fallback."""

    def segment(self, image: Image, points: List[Point],
                labels: List[int] = None) -> Optional[Mask]:
        if not points:
            return None

        pixels = image.data.pixels
        key = _cache_key(pixels, points)
        cached = _cache_get(key)
        if cached is not None:
            return Mask(cached)

        try:
            img_bgr = (
                cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                if image.data.color_space == "RGB"
                else pixels
            )
            h, w = image.height, image.width

            # downscale
            scale = 1.0
            if max(h, w) > _MAX_GC_DIM:
                scale = _MAX_GC_DIM / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_s = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                s_pts = [
                    Point(
                        max(0, min(int(p.x * scale), new_w - 1)),
                        max(0, min(int(p.y * scale), new_h - 1)),
                    )
                    for p in points
                ]
            else:
                img_s = img_bgr
                s_pts = points
                new_h, new_w = h, w

            run = (
                self._single_point
                if len(s_pts) == 1
                else self._multi_point
            )
            result = run(img_s, s_pts[0] if len(s_pts) == 1 else s_pts, new_w, new_h)

            if result is None:
                return None

            out = (
                cv2.resize(result, (w, h), interpolation=cv2.INTER_NEAREST)
                if scale < 1.0
                else result
            )
            out = np.ascontiguousarray(out, dtype=np.uint8)
            _cache_set(key, out)
            return Mask(out)

        except Exception as e:
            print(f"  Fallback error: {e}")
            return None

    @staticmethod
    def _single_point(img: np.ndarray, point: Point,
                      w: int, h: int) -> Optional[np.ndarray]:
        cx = max(0, min(int(point.x), w - 1))
        cy = max(0, min(int(point.y), h - 1))
        center = (cx, cy)
        radius = max(8, min(w, h) // 30)

        pad = radius * 6
        rx0 = max(0, cx - pad)
        ry0 = max(0, cy - pad)
        rx1 = min(w, cx + pad)
        ry1 = min(h, cy + pad)

        roi = img[ry0:ry1, rx0:rx1]
        rh, rw = roi.shape[:2]
        if rh < 4 or rw < 4:
            roi = img
            rx0 = ry0 = 0
            rh, rw = h, w

        lc = (cx - rx0, cy - ry0)
        gc_mask = np.full((rh, rw), cv2.GC_BGD, dtype=np.uint8)
        cv2.circle(gc_mask, lc, radius, cv2.GC_PR_FGD, -1)
        cv2.circle(gc_mask, lc, max(2, radius // 6), cv2.GC_FGD, -1)

        bgd, fgd = _GC_BGD.copy(), _GC_FGD.copy()
        cv2.grabCut(roi, gc_mask, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)

        b_roi = np.uint8((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))
        b_roi = _clean(b_roi, lc)
        b_roi = _distance_filter(b_roi, lc, radius * 4)
        b_roi = _color_filter(b_roi, roi, lc)

        full = np.zeros((h, w), dtype=np.uint8)
        full[ry0:ry0 + rh, rx0:rx0 + rw] = b_roi
        return full

    @staticmethod
    def _multi_point(img: np.ndarray, points: List[Point],
                     w: int, h: int) -> Optional[np.ndarray]:
        xs = np.array([int(p.x) for p in points], dtype=np.int32)
        ys = np.array([int(p.y) for p in points], dtype=np.int32)
        margin = max(20, min(w, h) // 15)

        rx0 = max(0, int(xs.min()) - margin)
        ry0 = max(0, int(ys.min()) - margin)
        rx1 = min(w, int(xs.max()) + margin)
        ry1 = min(h, int(ys.max()) + margin)

        roi = img[ry0:ry1, rx0:rx1]
        rh, rw = roi.shape[:2]
        if rh < 4 or rw < 4:
            roi = img
            rx0 = ry0 = 0
            rh, rw = h, w

        rect = (max(1, margin), max(1, margin),
                max(2, rw - 2 * margin), max(2, rh - 2 * margin))

        gc_mask = np.zeros((rh, rw), np.uint8)
        bgd, fgd = _GC_BGD.copy(), _GC_FGD.copy()
        cv2.grabCut(roi, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)

        b_roi = np.uint8((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))

        full = np.zeros((h, w), dtype=np.uint8)
        full[ry0:ry0 + rh, rx0:rx0 + rw] = b_roi
        return full
