"""Compatibility interface for the shared local vLLM backend."""

from extraction.backends.vllm_backend import VLLMBackend, VLLMOutput

QwenOutput = VLLMOutput


class QwenBackend(VLLMBackend):
    source = "qwen"
