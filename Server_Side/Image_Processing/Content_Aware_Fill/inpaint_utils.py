"""Image_Processing/Content_Aware_Fill/inpaint_utils.py: Shared inpainting utilities"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

def load_image_and_mask(image_path: str, mask_path: str):
    """Load and binarize image + mask."""
    image = cv2.imread(image_path)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"Could not load image or mask")
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return image, mask

def blend_edges(inpainted: np.ndarray, original: np.ndarray, mask: np.ndarray,
                feather: int = 10) -> np.ndarray:
    """Feathered blend at mask boundary using distance transform."""
    dist   = distance_transform_edt(mask == 0)
    weight = np.clip(dist / feather, 0, 1)
    weight = np.stack([weight] * 3, axis=-1)
    blended = (inpainted * (1 - weight) + original * weight).astype(np.uint8)
    mask3   = np.stack([mask] * 3, axis=-1) > 0
    return np.where(mask3, blended, original)

def match_colors(inpainted: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Transfer color statistics from border region into inpainted area."""
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    border  = cv2.dilate(mask, kernel, iterations=1) - mask
    b_px    = original[border > 0]
    i_px    = inpainted[mask > 0]
    if not len(b_px) or not len(i_px):
        return inpainted
    b_mean, b_std = np.mean(b_px, 0), np.std(b_px, 0)
    i_mean, i_std = np.mean(i_px, 0), np.std(i_px, 0)
    i_std   = np.where(i_std == 0, 1, i_std)
    result  = inpainted.copy().astype(np.float32)
    mask3   = np.stack([mask] * 3, axis=-1) > 0
    adjusted = np.clip((result - i_mean) * (b_std / i_std) + b_mean, 0, 255)
    return np.where(mask3, adjusted, inpainted).astype(np.uint8)

def remove_artifacts(result: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Median filter only on inpainted region."""
    mask3 = np.stack([mask] * 3, axis=-1) > 0
    return np.where(mask3, cv2.medianBlur(result, 3), result)