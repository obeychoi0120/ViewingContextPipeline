from __future__ import annotations

import math
from typing import Any


QWEN_DEFAULTS = {
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 16384,
    "max_model_len": "auto",
    "renderer_num_workers": 4,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
    "enforce_eager": False,
    "async_scheduling": None,
}


def qwen_settings(value: dict[str, Any] | None = None) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - QWEN_DEFAULTS.keys():
        raise ValueError("extraction.qwen contains unknown settings or is not a mapping")
    result = {**QWEN_DEFAULTS, **value}
    memory = result["gpu_memory_utilization"]
    if type(memory) not in (int, float) or not math.isfinite(memory) or not 0 < memory <= 1:
        raise ValueError("extraction.qwen.gpu_memory_utilization must be in (0, 1]")
    for key in ("max_num_seqs", "max_num_batched_tokens", "renderer_num_workers"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"extraction.qwen.{key} must be a positive integer")
    length = result["max_model_len"]
    if length != "auto" and (type(length) is not int or length <= 0):
        raise ValueError("extraction.qwen.max_model_len must be auto or a positive integer")
    for key in ("enable_chunked_prefill", "enable_prefix_caching", "enforce_eager"):
        if type(result[key]) is not bool:
            raise ValueError(f"extraction.qwen.{key} must be true or false")
    if result["async_scheduling"] is not None and type(result["async_scheduling"]) is not bool:
        raise ValueError("extraction.qwen.async_scheduling must be null, true or false")
    if not result["enable_chunked_prefill"] and length != "auto":
        if result["max_num_batched_tokens"] < length:
            raise ValueError("without chunked prefill max_num_batched_tokens must cover max_model_len")
    return result
