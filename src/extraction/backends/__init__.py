"""Optional VLM backends used by the shared extraction core."""

from .base import VLMBackend
from .gemini import GeminiBackend
from .gemini_workers import GeminiGenerationOutcome, GeminiWorkerPool
from .qwen import QwenBackend

__all__ = [
    "GeminiBackend",
    "GeminiGenerationOutcome",
    "GeminiWorkerPool",
    "QwenBackend",
    "VLMBackend",
]
