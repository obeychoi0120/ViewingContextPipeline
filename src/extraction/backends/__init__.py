"""Optional VLM backends used by the shared extraction core."""

from .base import VLMBackend
from .gemini import GeminiBackend
from .qwen import QwenBackend

__all__ = ["GeminiBackend", "QwenBackend", "VLMBackend"]
