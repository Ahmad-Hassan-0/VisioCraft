# sd_inpaint.py - Local Flux Inpainting (CPU-friendly)
import torch
from diffusers import FluxInpaintPipeline
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import cv2
import os
from .inpaint_utils import blend_edges, match_colors, remove_artifacts  # Keep your helpers

class SDInpaint:
    def __init__(self):
        print("Loading Flux.1-schnell for inpainting (CPU mode)...")
        model_id = "black-forest-labs/FLUX.1-schnell"  # Fast distilled version
        self.pipe = FluxInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # CPU safe
        )
        self.pipe.enable_model_cpu_offload()  # Crucial for low RAM/CPU
        print("Flux loaded successfully!")

    def inpaint(self, image_path: str, mask_path: str, prompt: str = "") -> np.ndarray:
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            raise FileNotFoundError("Image or mask file missing")

        init_image = load_image(image_path).convert("RGB")
        mask_image = load_image(mask_path).convert("L")  # Grayscale mask

        # Good default prompt for your context-aware fill
        if not prompt:
            prompt = "seamless natural background continuation, photorealistic, high quality, consistent lighting, no artifacts"

        negative_prompt = "blurry, low quality, artifacts, seams, distortion, mismatched colors, people, text"

        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image,
                strength=0.80,               # Adjust 0.7-0.9 for how aggressive
                guidance_scale=3.0,
                num_inference_steps=20,      # Low for CPU speed (~1-5 min on average laptop)
            ).images[0]

            output = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

            # Apply your post-processing
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            original = cv2.imread(image_path)
            output = blend_edges(output, original, mask_gray)
            output = match_colors(output, original, mask_gray)
            output = remove_artifacts(output, mask_gray)

            return output

        except Exception as e:
            print(f"Flux inpaint failed: {e}")
            raise  # Or fallback to OpenCV here if you want