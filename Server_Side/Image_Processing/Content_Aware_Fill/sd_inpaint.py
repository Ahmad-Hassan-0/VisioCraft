"""
Image_Processing/Content_Aware_Fill/sd_inpaint.py
Lightweight context-aware inpainting — fast OpenCV, no heavy models.
Replicate SD API used if token is set, otherwise falls back to OpenCV.
"""
import cv2
import numpy as np
import os
import config
from .inpaint_utils import load_image_and_mask, blend_edges, match_colors, remove_artifacts


class SDInpaint:
    """
    Context-aware inpainting.
    Primary:  Replicate SD API (if REPLICATE_API_TOKEN set in config)
    Fallback: Fast OpenCV two-pass (TELEA + NS)
    """

    MODEL = "stability-ai/stable-diffusion-inpainting"

    def __init__(self):
        self.token     = getattr(config, 'REPLICATE_API_TOKEN', None)
        self.available = self._check_replicate()
        if self.available:
            print("🎨 Inpainting: ✅ Replicate SD")
        else:
            print("🎨 Inpainting: ⚡ Fast OpenCV (set REPLICATE_API_TOKEN for AI quality)")

    def _check_replicate(self) -> bool:
        if not self.token or self.token in ("", "YOUR_REPLICATE_TOKEN_HERE"):
            return False
        try:
            import replicate
            return True
        except ImportError:
            return False

    def inpaint(self, image_path: str, mask_path: str, prompt: str = "") -> np.ndarray:
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            raise FileNotFoundError("Image or mask missing")

        image, mask = load_image_and_mask(image_path, mask_path)

        if self.available:
            try:
                result = self._replicate(image_path, mask_path, image, mask, prompt)
                print("✅ Replicate SD inpainting complete")
                return result
            except Exception as e:
                print(f"⚠️  Replicate failed: {e} — falling back to OpenCV")

        return self._opencv(image, mask)

    def _replicate(self, image_path, mask_path, image, mask, prompt):
        import replicate, requests, io, os
        from PIL import Image as PILImage

        os.environ["REPLICATE_API_TOKEN"] = self.token

        if not prompt:
            prompt = self._auto_prompt(image, mask)
        print(f"   → Prompt: {prompt}")

        with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
            output = replicate.run(
                self.MODEL,
                input={
                    "image":               img_f,
                    "mask":                mask_f,
                    "prompt":              prompt,
                    "negative_prompt":     "blurry, artifacts, distorted, unrealistic",
                    "num_inference_steps": 30,
                    "guidance_scale":      7.5,
                }
            )

        url = output[0] if isinstance(output, list) else output
        r   = requests.get(str(url), timeout=60)
        pil = PILImage.open(io.BytesIO(r.content)).convert("RGB")
        result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        if result.shape[:2] != image.shape[:2]:
            result = cv2.resize(result, (image.shape[1], image.shape[0]))

        result = blend_edges(result, image, mask)
        result = match_colors(result, image, mask)
        result = remove_artifacts(result, mask)
        return result

    def _auto_prompt(self, image: np.ndarray, mask: np.ndarray) -> str:
        kernel    = np.ones((20, 20), np.uint8)
        border    = cv2.dilate(mask, kernel, iterations=2) - mask
        bg_pixels = image[border > 0]
        if not len(bg_pixels):
            return "seamless background, photorealistic, high quality"
        b, g, r = np.mean(bg_pixels, axis=0).astype(int)[:3]
        if g > r and g > b:
            scene = "green grass outdoor scene"
        elif b > r and b > g:
            scene = "blue sky background"
        elif r > 150 and g > 150 and b > 150:
            scene = "light neutral wall interior"
        else:
            scene = "natural realistic background"
        return f"{scene}, seamless, photorealistic, high quality, no artifacts"

    def _opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fast two-pass OpenCV fallback."""
        print("   → Fast inpainting: OpenCV TELEA + NS")
        result = cv2.inpaint(image, mask, 15, cv2.INPAINT_TELEA)
        result = cv2.inpaint(result, mask, 10, cv2.INPAINT_NS)
        result = blend_edges(result, image, mask)
        result = match_colors(result, image, mask)
        result = remove_artifacts(result, mask)
        print("   → Fast inpainting completed")
        return result