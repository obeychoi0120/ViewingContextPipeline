"""MiniCPM-specific adaptation of the shared asynchronous vLLM engine."""

from extraction.backends.qwen import QwenBackend
from extraction.minicpm_config import PROCESSOR_SETTINGS, validate_minicpm_checkpoint


class MiniCPMBackend(QwenBackend):
    label = "MiniCPM"
    dependency_extra = "minicpm"

    @classmethod
    def checkpoint_model(cls, config):
        validate_minicpm_checkpoint("minicpm", config)
        return "minicpm"

    @classmethod
    def engine_options(cls):
        return {"mm_processor_kwargs": dict(PROCESSOR_SETTINGS)}

    def validate_task(self, task, config):
        validate_minicpm_checkpoint(task.model, config)

    def template_options(self, task):
        # tokenize=False only renders placeholders; the engine processes images at 16x.
        return {"enable_thinking": False}
