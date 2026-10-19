"""Loaded inside vLLM GPU workers only; CPU/Windows entrypoints stay importable."""

from vllm.v1.sample.logits_processor import LogitsProcessor

from extraction.backends.qwen_penalty import GeneratedTokenPenalty, GeneratedTokenPenaltyV2

try:
    from vllm.v1.worker.gpu.sample.logits_processor.interface import (
        LogitsProcessor as V2LogitsProcessor,
    )
except ModuleNotFoundError as exc:
    # vLLM 0.28 and earlier builds have no V2 processor package. Do not hide
    # dependency failures inside an existing V2 module.
    target = "vllm.v1.worker.gpu.sample.logits_processor.interface"
    if not exc.name or not (target == exc.name or target.startswith(exc.name + ".")):
        raise
    V2LogitsProcessor = None


if V2LogitsProcessor is None:
    class GeneratedTokenLogitsProcessor(GeneratedTokenPenalty, LogitsProcessor):
        pass
else:
    class GeneratedTokenLogitsProcessor(GeneratedTokenPenalty, LogitsProcessor, V2LogitsProcessor):
        """The same FQCN supports both loaders; constructor selects their state contract."""

        def __init__(self, vllm_config, device, is_pin_memory=None):
            self.v2 = None
            if is_pin_memory is None:
                self.v2 = GeneratedTokenPenaltyV2(vllm_config, device)
            else:
                super().__init__(vllm_config, device, is_pin_memory)

        def add_request(self, req_idx, sampling_params):
            return self.v2.add_request(req_idx, sampling_params)

        def apply_staged_writes(self):
            self.v2.apply_staged_writes()

        def apply(self, logits, ctx=None):
            if self.v2 is not None:
                return self.v2.apply(logits, ctx)
            return super().apply(logits)
