"""
VisioCraft AI - Configuration Settings
Central configuration for all application settings.

NOTE: This module contains ONLY constants and pure helper functions.
Directory creation is handled by the app factory (app/__init__.py).
"""

from pathlib import Path
import os

# ── Project Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TEMP_DIR = PROJECT_ROOT / "temp"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Model Paths ──────────────────────────────────────────────────────────────
SAM_MODEL_PATH = MODELS_DIR / "mobile_sam.pt"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

SAM_MODEL_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

# ── Server Configuration ────────────────────────────────────────────────────
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
DEBUG_MODE = True
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# ── Image Processing Settings ───────────────────────────────────────────────
IMAGE_QUALITY = 95
MAX_IMAGE_DIMENSION = 2048
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

INPAINT_RADIUS = 5
PATCH_SIZE = 7

# ── Object Extraction Settings ──────────────────────────────────────────────
MASK_PADDING = 10
MIN_MASK_SIZE = 100
TRANSPARENCY_THRESHOLD = 0.5

# ── AI Image Generation ─────────────────────────────────────────────────────
ENABLE_AI_GENERATION = True

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_STEPS = 30
DEFAULT_CFG_SCALE = 7.0
DEFAULT_MODEL = "stability"

# ── Session Management ──────────────────────────────────────────────────────
SESSION_TIMEOUT = 3600
MAX_SESSIONS = 100
CLEANUP_INTERVAL = 600

# ── Canvas Settings ─────────────────────────────────────────────────────────
CANVAS_DEFAULT_WIDTH = 1024
CANVAS_DEFAULT_HEIGHT = 768
MAX_LAYERS = 50
UNDO_HISTORY_SIZE = 20

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "visiocraft.log"


# ── Helper Functions ─────────────────────────────────────────────────────────

def get_output_path(filename: str) -> Path:
    """Get full path for output file."""
    return OUTPUT_DIR / filename


def get_temp_path(filename: str) -> Path:
    """Get full path for temporary file."""
    return TEMP_DIR / filename


def validate_image_format(filename: str) -> bool:
    """Check if file format is supported."""
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_FORMATS


def get_available_methods():
    """Get list of available inpainting methods."""
    return [
        {"id": "opencv", "name": "OpenCV Fast", "description": "Fast inpainting with edge blending"},
    ]