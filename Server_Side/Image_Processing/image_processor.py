"""Image_Processing/image_processor.py: Object extraction, preview, and resize operations"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import config

def extract_object(image_path: str, mask_path: str, session_id: str) -> str:
    """Extract object with transparent background using mask."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image[:, :, 3] = mask

    x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
    p = config.MASK_PADDING
    x, y = max(0, x-p), max(0, y-p)
    w = min(image.shape[1]-x, w+2*p)
    h = min(image.shape[0]-y, h+2*p)

    out = config.get_output_path(f'extracted_{session_id}.png')
    cv2.imwrite(str(out), image[y:y+h, x:x+w])
    return str(out)

def create_preview(image_path: str, mask_path: str, session_id: str) -> str:
    """Create green overlay preview of segmentation mask."""
    image = cv2.imread(image_path)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return image_path
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    result   = image.copy()
    green    = np.zeros_like(image)
    green[:] = [0, 255, 0]
    mb       = mask_bin > 127
    result[mb] = cv2.addWeighted(image[mb], 0.4, green[mb], 0.6, 0)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 4)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 2)

    out = config.get_temp_path(f'preview_{session_id}.png')
    cv2.imwrite(str(out), result)
    return str(out)

def resize_image(image_path: str, max_dim: int = None) -> str:
    """Resize image if it exceeds max dimension, preserving aspect ratio."""
    max_dim = max_dim or config.MAX_IMAGE_DIMENSION
    img     = Image.open(image_path)
    w, h    = img.size
    if w <= max_dim and h <= max_dim:
        return image_path
    scale  = max_dim / max(w, h)
    img    = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    out    = Path(image_path).parent / f"resized_{Path(image_path).name}"
    img.save(out, quality=config.IMAGE_QUALITY)
    return str(out)