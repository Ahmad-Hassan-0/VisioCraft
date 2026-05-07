"""
app/infrastructure/generation/image_generators.py
AI image generation providers implementing IImageGenerator.
"""
import io
import urllib.parse
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

import config
from app.domain.interfaces import IImageGenerator

try:
    import requests
except ImportError:
    requests = None


class PollinationsGenerator(IImageGenerator):
    """Pollinations.ai — free, no API key needed."""

    def generate(self, prompt: str, session_id: str) -> Optional[str]:
        if not requests:
            return None
        encoded = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return None
        img = Image.open(io.BytesIO(r.content))
        out = config.get_output_path(f'generated_{session_id}.png')
        img.save(out)
        return str(out)


class HuggingFaceGenerator(IImageGenerator):
    """Hugging Face Inference API — requires free token."""

    def generate(self, prompt: str, session_id: str) -> Optional[str]:
        if not requests:
            return None
        token = getattr(config, 'HUGGINGFACE_TOKEN', None)
        if not token:
            return None
        url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": prompt, "options": {"wait_for_model": True}},
            timeout=90,
        )
        if r.status_code != 200:
            return None
        img = Image.open(io.BytesIO(r.content))
        out = config.get_output_path(f'generated_{session_id}.png')
        img.save(out)
        return str(out)


class PlaceholderGenerator(IImageGenerator):
    """Gradient placeholder when all APIs fail."""

    def generate(self, prompt: str, session_id: str) -> Optional[str]:
        w, h = 1024, 1024
        img = Image.new('RGB', (w, h))
        pixels = img.load()
        for y_pos in range(h):
            for x_pos in range(w):
                pixels[x_pos, y_pos] = (
                    int(120 + 80 * x_pos / w),
                    int(80 + 150 * y_pos / h),
                    200,
                )

        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        lines = [
            " AI Image Generation", "",
            f'"{prompt[:80]}"', "",
            "APIs unavailable — check config.py",
        ]
        y0 = h // 2 - 60
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            draw.text(((w - bbox[2]) // 2, y0), line,
                      fill=(255, 255, 200), font=font)
            y0 += 28

        out = config.get_output_path(f'generated_{session_id}.png')
        img.save(out)
        return str(out)
