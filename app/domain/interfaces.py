"""
app/domain/interfaces.py
Abstract base classes defining contracts between layers.
Infrastructure implementations depend on these — never the reverse.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from .models import Image, Mask, Point


class ISegmentationBackend(ABC):
    """Contract for any segmentation strategy (SAM, GrabCut, etc.)."""

    @abstractmethod
    def segment(self, image: Image, points: List[Point],
                labels: List[int] = None) -> Optional[Mask]:
        """Return a binary mask, or None on failure."""
        ...


class IInpaintingBackend(ABC):
    """Contract for any inpainting strategy."""

    @abstractmethod
    def inpaint(self, image_path: str, mask_path: str,
                prompt: str = "", session_id: str = "") -> np.ndarray:
        """Return inpainted image as BGR ndarray."""
        ...


class IImageGenerator(ABC):
    """Contract for AI image generation providers."""

    @abstractmethod
    def generate(self, prompt: str, session_id: str) -> Optional[str]:
        """Generate image from prompt. Return file path or None on failure."""
        ...
