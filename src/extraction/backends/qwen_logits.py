"""Loaded inside vLLM GPU workers only; CPU/Windows entrypoints stay importable."""

from vllm.v1.sample.logits_processor import LogitsProcessor

from extraction.backends.qwen_penalty import GeneratedTokenPenalty


class GeneratedTokenLogitsProcessor(GeneratedTokenPenalty, LogitsProcessor):
    pass
