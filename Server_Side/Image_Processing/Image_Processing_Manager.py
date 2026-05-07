"""
Image_Processing/Image_Processing_Manager.py
Orchestrator — uses local Flux inpainting only.
"""
from PIL import Image
import cv2
import traceback
import numpy as np
import config
from .image_processor import extract_object, create_preview, resize_image
from .Content_Aware_Fill.sd_inpaint import SDInpaint

class ImageProcessingManager:
    def __init__(self):
        print("Initializing ImageProcessingManager...")
        try:
            self.inpainter = SDInpaint()  # Loads Flux once
            print("🎨 Inpainting: Local Flux.1-schnell (CPU)")
        except Exception as e:
            print(f"❌ Flux failed to load: {e}")
            self.inpainter = None
            print("Will use emergency OpenCV only if needed")

    def extract_object(self, image_path: str, mask_path: str, session_id: str) -> str:
        return extract_object(image_path, mask_path, session_id)

    def create_preview(self, image_path: str, mask_path: str, session_id: str) -> str:
        return create_preview(image_path, mask_path, session_id)

    def resize_image(self, image_path: str, max_dim: int = None) -> str:
        return resize_image(image_path, max_dim)

    def inpaint_background(self, image_path: str, mask_path: str,
                           method: str = None, session_id: str = "") -> str:
        print(f"Starting background inpaint for session {session_id}")
        try:
            if not self.inpainter:
                raise RuntimeError("Flux not loaded")
            result = self.inpainter.inpaint(image_path, mask_path)
            print("Flux inpainting successful")
        except Exception as e:
            print(f"Inpainting error: {e}")
            traceback.print_exc()
            result = self._emergency(image_path, mask_path)
            print("Used emergency OpenCV fallback")

        out_path = config.get_output_path(f'inpainted_{session_id}.jpg')
        Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save(
            out_path, quality=config.IMAGE_QUALITY)
        print(f"Saved inpainted result: {out_path}")
        return str(out_path)

    def _emergency(self, image_path: str, mask_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return cv2.inpaint(img, mask, 10, cv2.INPAINT_NS)