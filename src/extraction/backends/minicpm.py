"""Native MiniCPM-V 4.7 adapter for the shared offline vLLM engine."""

import re
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

from extraction.backends.vllm_backend import VLLMBackend
from extraction.minicpm_config import processor_kwargs


def normalize_newlines(text):
    # Restore Graph line separators, including fenced JSON whitespace, while
    # preserving quoted JSON values and their escaped newlines/backslashes.
    parts = re.split(r'("(?:\\.|[^"\\])*")', text)
    return "".join(part if i % 2 else part.replace(r"\n", "\n") for i, part in enumerate(parts))


class MiniCPMBackend(VLLMBackend):
    source = "minicpm"

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        try:
            installed = version("transformers")
        except PackageNotFoundError as exc:
            raise RuntimeError("MiniCPM requires the minicpm optional dependencies") from exc
        if Version(installed) < Version("5.18.0"):
            raise RuntimeError("MiniCPM-V 4.7 requires transformers>=5.18.0")
        try:
            from vllm.model_executor.models.registry import ModelRegistry
        except ImportError as exc:
            raise RuntimeError("MiniCPM requires a native vLLM build including PR #58674") from exc

        if "MiniCPMV4_7ForConditionalGeneration" not in ModelRegistry.get_supported_archs():
            raise RuntimeError("MiniCPM-V 4.7 requires a native vLLM build including PR #58674")
        return super().from_pretrained(model_path, **kwargs)

    @classmethod
    def engine_options(cls, settings):
        return {
            **{k: v for k, v in settings.items() if k not in {"vision_downsample", "max_slice_nums"}},
            "mm_processor_kwargs": processor_kwargs(settings),
        }

    @staticmethod
    def chat_options():
        return {"enable_thinking": False}

    def normalize_output(self, text):
        return normalize_newlines(text)
