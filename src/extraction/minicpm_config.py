"""MiniCPM's native processor and independent vLLM settings."""

import math
from pathlib import Path

from artifact_io import file_fingerprint

MINICPM_DEFAULTS = {
    "gpu_memory_utilization": 0.9,
    "max_model_len": 8192,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 32,
    "renderer_num_workers": 4,
    "vision_downsample": 16,
    "max_slice_nums": 9,
}
OPTIONAL_ENGINE_SETTINGS = {
    "enable_chunked_prefill", "enable_prefix_caching", "enforce_eager", "async_scheduling",
}
NORMALIZATION_VERSION = "minicpm-graph-newlines/v1"


def minicpm_settings(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - (MINICPM_DEFAULTS.keys() | OPTIONAL_ENGINE_SETTINGS):
        raise ValueError("extraction.minicpm contains unknown settings or is not a mapping")
    result = {**MINICPM_DEFAULTS, **value}
    if type(result["vision_downsample"]) is not int or result["vision_downsample"] not in (4, 16):
        raise ValueError("extraction.minicpm.vision_downsample must be integer 4 or 16")
    memory = result["gpu_memory_utilization"]
    if type(memory) not in (int, float) or not math.isfinite(memory) or not 0 < memory <= 1:
        raise ValueError("extraction.minicpm.gpu_memory_utilization must be in (0, 1]")
    for key in ("max_num_seqs", "max_num_batched_tokens", "renderer_num_workers", "max_slice_nums"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"extraction.minicpm.{key} must be a positive integer")
    length = result["max_model_len"]
    if length != "auto" and (type(length) is not int or length <= 0):
        raise ValueError("extraction.minicpm.max_model_len must be auto or a positive integer")
    for key in OPTIONAL_ENGINE_SETTINGS & result.keys():
        if key == "async_scheduling" and result[key] is None:
            continue
        if type(result[key]) is not bool:
            raise ValueError(f"extraction.minicpm.{key} must be a boolean")
    if result.get("enable_chunked_prefill") is False and length != "auto":
        if result["max_num_batched_tokens"] < length:
            raise ValueError("without chunked prefill max_num_batched_tokens must cover max_model_len")
    return result


def processor_kwargs(settings):
    return {
        "downsample_mode": f"{settings['vision_downsample']}x",
        "max_slice_nums": settings["max_slice_nums"],
    }


def local_vlm_settings(model, value=None):
    if model == "minicpm":
        return minicpm_settings(value)
    if model == "qwen":
        from extraction.qwen_config import qwen_settings

        return qwen_settings(value)
    raise ValueError(f"unsupported local VLM: {model}")


def processor_identity(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): file_fingerprint(p)
            for pattern in ("*processor*.json", "*chat_template*.json", "*.jinja", "chat_templates/*.jinja")
            for p in sorted(directory.glob(pattern)) if p.is_file()}
