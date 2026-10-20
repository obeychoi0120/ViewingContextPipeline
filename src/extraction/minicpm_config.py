"""MiniCPM Graph-only policy, using the existing local vLLM execution settings."""

from extraction.qwen_config import generation_settings, validate_checkpoint

MODEL = "minicpm"
NORMALIZATION_VERSION = "minicpm-newlines/v1"
PROCESSOR_SETTINGS = {"downsample_mode": "16x"}


def validate_minicpm_checkpoint(model, config):
    if model != MODEL:
        raise ValueError(f"unsupported MiniCPM model: {model}")
    if (
        config.get("model_type") != "minicpmv4_7"
        or "MiniCPMV4_7ForConditionalGeneration" not in config.get("architectures", [])
    ):
        raise ValueError("checkpoint does not match minicpm: expected minicpmv4_7/MiniCPMV4_7ForConditionalGeneration")
    if config.get("quantization_config"):
        raise ValueError("MiniCPM requires a non-quantized BF16 checkpoint")
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype not in (None, "bfloat16"):
        raise ValueError("MiniCPM requires a BF16 checkpoint")


def validate_local_checkpoint(model, config):
    if model == MODEL:
        validate_minicpm_checkpoint(model, config)
    else:
        validate_checkpoint(model, config)


def minicpm_generation_settings(*, repetition_penalty=None, presence_penalty=None):
    settings = generation_settings(
        "qwen3.5", representation="graph",
        repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
    )
    return {**settings, "model": MODEL}
