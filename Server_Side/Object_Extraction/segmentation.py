"""Object_Extraction/segmentation.py — SAM model and GrabCut fallback segmentation
Performance strategy (8 GB RAM + virtual memory):
  • Disk-backed LRU cache  → repeated images/points cost ~0ms
  • ROI-only GrabCut       → process only the bounding box, not the full image
  • float16 color math     → half the memory of float32, still plenty of precision
  • einsum dot-product     → colour diff with zero intermediate array allocations
  • Vectorised LUT mask    → replaces per-label Python loop entirely
  • cv2.setNumThreads(0)   → OpenCV auto-uses all logical cores
  • Lazy heavy imports     → nothing loaded until actually needed
"""

import cv2
import numpy as np
import hashlib
import pickle
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import config
from .masking_models import Image, Mask, Point

# ── Global one-time OpenCV setup ─────────────────────────────────────────────
cv2.setNumThreads(0)       # use all logical cores automatically
cv2.setUseOptimized(True)  # enable SIMD / IPP paths

# ── Disk cache ────────────────────────────────────────────────────────────────
_CACHE_DIR = Path(tempfile.gettempdir()) / "seg_cache"
_CACHE_DIR.mkdir(exist_ok=True)
_CACHE_LOCK = threading.Lock()
_MAX_CACHE_FILES = 200   # ~few KB each → <10 MB total on disk

# ── GrabCut constants ─────────────────────────────────────────────────────────
_MAX_GC_DIM = 512          # downscale ceiling before GrabCut
_GC_BGD = np.zeros((1, 65), np.float64)
_GC_FGD = np.zeros((1, 65), np.float64)


# ─────────────────────────── Disk cache helpers ───────────────────────────────

def _cache_key(pixels: np.ndarray, points: List[Point], labels=None) -> str:
    """Cheap hash: image shape + sparse pixel sample + point coords."""
    h, w = pixels.shape[:2]
    step_y = max(1, h // 8)
    step_x = max(1, w // 8)
    sample  = pixels[::step_y, ::step_x].tobytes()
    pt_str  = str([(p.x, p.y) for p in points] + (labels or [])).encode()
    raw     = f"{h}:{w}:{pixels.dtype}:".encode() + sample + pt_str
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


# ─────────────────────────── SAM Model ───────────────────────────────────────

class SAMModel:
    """Loads and runs MobileSAM segmentation."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(config.SAM_MODEL_PATH)
        self.model      = None
        self.is_loaded  = False
        self.sam_available = False

    def load(self) -> bool:
        if not Path(self.model_path).exists():
            print(f"  SAM model not found: {self.model_path}")
            return False
        try:
            from ultralytics import SAM
            self.model         = SAM(self.model_path)
            self.is_loaded     = True
            self.sam_available = True
            print("MobileSAM loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load SAM: {e}")
            return False

    def segment_with_points(
        self,
        image: Image,
        points: List[Point],
        labels: List[int],
    ) -> Optional[Mask]:

        if not self.is_loaded:
            return None

        pixels = image.data.pixels

        # ── cache hit → free ──
        key    = _cache_key(pixels, points, labels)
        cached = _cache_get(key)
        if cached is not None:
            return Mask(cached)

        try:
            coords  = [[p.x, p.y] for p in points]
            results = self.model.predict(
                pixels,
                points=coords,
                labels=labels,
                verbose=False,
                conf=0.7,
                retina_masks=False,   # smaller output → less GPU→CPU transfer
            )
            if results and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                if len(masks):
                    idx       = _best_mask(masks, coords, labels, image.height, image.width)
                    mask_data = masks[idx]
                    if mask_data.shape != (image.height, image.width):
                        mask_data = cv2.resize(
                            mask_data, (image.width, image.height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    out = np.ascontiguousarray(mask_data > 0.5, dtype=np.uint8)
                    _cache_set(key, out)
                    return Mask(out)
        except Exception as e:
            print(f"SAM segmentation error: {e}")
        return None


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


# ─────────────────────────── Fallback Segmentation ───────────────────────────

class FallbackSegmentation:
    """ROI-based, RAM-efficient GrabCut fallback."""

    @staticmethod
    def segment(image: Image, points: List[Point]) -> Optional[Mask]:
        if not points:
            return None

        pixels = image.data.pixels

        # ── cache hit → free ──
        key    = _cache_key(pixels, points)
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

            # ── downscale to _MAX_GC_DIM on longest side ──
            scale = 1.0
            if max(h, w) > _MAX_GC_DIM:
                scale   = _MAX_GC_DIM / max(h, w)
                new_w   = int(w * scale)
                new_h   = int(h * scale)
                img_s   = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                s_pts   = [
                    Point(
                        max(0, min(int(p.x * scale), new_w - 1)),
                        max(0, min(int(p.y * scale), new_h - 1)),
                    )
                    for p in points
                ]
            else:
                img_s   = img_bgr
                s_pts   = points
                new_h, new_w = h, w

            run = (
                FallbackSegmentation._single_point
                if len(s_pts) == 1
                else FallbackSegmentation._multi_point
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
            print(f"Fallback error: {e}")
            return None

    # ── single-point ─────────────────────────────────────────────────────────

    @staticmethod
    def _single_point(
        img: np.ndarray, point: Point, w: int, h: int
    ) -> Optional[np.ndarray]:

        cx     = max(0, min(int(point.x), w - 1))
        cy     = max(0, min(int(point.y), h - 1))
        center = (cx, cy)
        radius = max(8, min(w, h) // 30)

        # ── ROI: tight crop around click (huge speedup on large images) ──
        pad = radius * 6
        rx0 = max(0, cx - pad);  ry0 = max(0, cy - pad)
        rx1 = min(w, cx + pad);  ry1 = min(h, cy + pad)

        roi = img[ry0:ry1, rx0:rx1]
        rh, rw = roi.shape[:2]
        if rh < 4 or rw < 4:                     # degenerate → use full image
            roi = img
            rx0 = ry0 = 0
            rh, rw = h, w

        lc = (cx - rx0, cy - ry0)               # click in ROI coords

        gc_mask = np.full((rh, rw), cv2.GC_BGD, dtype=np.uint8)
        cv2.circle(gc_mask, lc, radius,               cv2.GC_PR_FGD, -1)
        cv2.circle(gc_mask, lc, max(2, radius // 6),  cv2.GC_FGD,    -1)

        bgd, fgd = _GC_BGD.copy(), _GC_FGD.copy()
        cv2.grabCut(roi, gc_mask, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)

        b_roi  = np.uint8((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))
        b_roi  = _clean(b_roi, lc)
        b_roi  = _distance_filter(b_roi, lc, radius * 4)
        b_roi  = _color_filter(b_roi, roi, lc)

        full = np.zeros((h, w), dtype=np.uint8)
        full[ry0:ry0 + rh, rx0:rx0 + rw] = b_roi
        return full

    # ── multi-point ──────────────────────────────────────────────────────────

    @staticmethod
    def _multi_point(
        img: np.ndarray, points: List[Point], w: int, h: int
    ) -> Optional[np.ndarray]:

        xs     = np.array([int(p.x) for p in points], dtype=np.int32)
        ys     = np.array([int(p.y) for p in points], dtype=np.int32)
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


# ─────────────────────────── Shared filter helpers ───────────────────────────

def _clean(mask: np.ndarray, click: Tuple) -> np.ndarray:
    """Keep only the connected component that contains the click pixel."""
    n, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    lbl = labels[click[1], click[0]]
    return np.uint8(labels == lbl) if lbl > 0 else mask


def _distance_filter(mask: np.ndarray, click: Tuple, max_dist: int) -> np.ndarray:
    """Zero pixels farther than max_dist from click — integer only, no sqrt."""
    h, w  = mask.shape
    y_c, x_c = np.ogrid[:h, :w]
    dist_sq   = (x_c - click[0]) ** 2 + (y_c - click[1]) ** 2
    return np.uint8(mask & (dist_sq <= max_dist * max_dist))


def _color_filter(mask: np.ndarray, img: np.ndarray, click: Tuple) -> np.ndarray:
    """Keep pixels whose colour is within 40 units of the click pixel's colour."""
    cx, cy = click
    if not (0 <= cy < img.shape[0] and 0 <= cx < img.shape[1]):
        return mask

    # float16: half the RAM of float32, no sqrt, einsum avoids temp arrays
    color  = img[cy, cx].astype(np.float16)
    diff   = img.astype(np.float16) - color           # (H, W, 3)
    diff_sq = np.einsum("ijk,ijk->ij", diff, diff)    # (H, W) — no temp array
    refined = np.uint8(mask & (diff_sq < 1600))       # 40² threshold

    # ── area filter: vectorised LUT, zero Python loops ──
    n, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    if n <= 1:
        return refined

    areas = stats[:, cv2.CC_STAT_AREA]                # shape (n,)
    lut   = np.uint8(areas >= 20)                     # 1 = keep, 0 = drop
    lut[0] = 0                                         # always drop background
    return lut[labels]                                 # O(H*W) index, no loop