"""Image_Processing/image_generation.py: AI image generation via external APIs"""
import io
import urllib.parse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import config

try:
    import requests
except ImportError:
    requests = None

def generate_image_from_prompt(prompt: str, session_id: str = "default") -> str:
    """Try providers in order, return path to generated image."""
    if not requests:
        print("⚠️  requests not installed")
        return _placeholder(prompt, session_id)

    for provider in [_pollinations, _huggingface, _placeholder]:
        try:
            result = provider(prompt, session_id)
            if result:
                return result
        except Exception as e:
            print(f"✗ {provider.__name__}: {e}")

    return _placeholder(prompt, session_id)

def _pollinations(prompt: str, session_id: str) -> str:
    """Pollinations.ai — free, no API key needed."""
    encoded = urllib.parse.quote(prompt)
    url     = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
    r       = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    img = Image.open(io.BytesIO(r.content))
    out = config.get_output_path(f'generated_{session_id}.png')
    img.save(out)
    return str(out)

def _huggingface(prompt: str, session_id: str) -> str:
    """Hugging Face Inference API — requires free token."""
    token = getattr(config, 'HUGGINGFACE_TOKEN', None)
    if not token:
        return None
    url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    r   = requests.post(url, headers={"Authorization": f"Bearer {token}"},
                        json={"inputs": prompt, "options": {"wait_for_model": True}},
                        timeout=90)
    if r.status_code != 200:
        return None
    img = Image.open(io.BytesIO(r.content))
    out = config.get_output_path(f'generated_{session_id}.png')
    img.save(out)
    return str(out)

def _placeholder(prompt: str, session_id: str) -> str:
    """Gradient placeholder when all APIs fail."""
    w, h   = 1024, 1024
    img    = Image.new('RGB', (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            pixels[x, y] = (int(120 + 80*x/w), int(80 + 150*y/h), 200)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    lines = [" AI Image Generation", "", f'"{prompt[:80]}"', "",
             "APIs unavailable — check config.py"]
    y0 = h // 2 - 60
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text(((w - bbox[2]) // 2, y0), line, fill=(255, 255, 200), font=font)
        y0 += 28

    out = config.get_output_path(f'generated_{session_id}.png')
    img.save(out)
    return str(out)