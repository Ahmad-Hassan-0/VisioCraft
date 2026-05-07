"""
app/services/generation_service.py
AI image generation with provider fallback chain.
"""
import logging
import time
from typing import List

from app.domain.interfaces import IImageGenerator
from app.infrastructure.generation.image_generators import (
    PollinationsGenerator,
    HuggingFaceGenerator,
    PlaceholderGenerator,
)

log = logging.getLogger('visiocraft.generation')


class GenerationService:
    """Tries multiple generation providers in order."""

    def __init__(self):
        self.providers: List[IImageGenerator] = [
            PollinationsGenerator(),
            HuggingFaceGenerator(),
            PlaceholderGenerator(),
        ]
        log.info("GenerationService initialized with %d providers: %s",
                 len(self.providers),
                 [p.__class__.__name__ for p in self.providers])

    def generate(self, prompt: str, session_id: str = "default") -> str:
        """Generate image from prompt, trying providers in fallback order."""
        log.info("GENERATE START — session=%s, prompt='%s'",
                 session_id, prompt[:80])
        t0 = time.time()

        for i, provider in enumerate(self.providers):
            name = provider.__class__.__name__
            try:
                log.info("  Trying provider %d/%d: %s",
                         i + 1, len(self.providers), name)
                pt0 = time.time()
                result = provider.generate(prompt, session_id)
                elapsed_p = (time.time() - pt0) * 1000
                if result:
                    elapsed = (time.time() - t0) * 1000
                    log.info("GENERATE COMPLETE — provider=%s, output=%s, "
                             "provider=%.0fms, total=%.0fms",
                             name, result, elapsed_p, elapsed)
                    return result
                else:
                    log.warning("  %s returned None after %.0fms", name, elapsed_p)
            except Exception as e:
                log.error("  %s failed: %s", name, e, exc_info=True)

        # Last resort
        log.warning("All providers failed, using PlaceholderGenerator")
        return PlaceholderGenerator().generate(prompt, session_id)
