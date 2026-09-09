from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Sequence


class GeneratedTokenRepetitionPenalty:
    """Apply a repetition penalty only to tokens generated after the prompt."""

    def __init__(self, penalty: float, prompt_length: int) -> None:
        if penalty < 1.0:
            raise ValueError("repetition penalty must be at least 1.0")
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        generated_ids = input_ids[:, self.prompt_length :]
        if generated_ids.numel() == 0:
            return scores
        repeated_scores = torch.gather(scores, 1, generated_ids)
        repeated_scores = torch.where(
            repeated_scores < 0,
            repeated_scores * self.penalty,
            repeated_scores / self.penalty,
        )
        return scores.scatter(1, generated_ids, repeated_scores)


@dataclass
class QwenBackend:
    model: Any
    processor: Any
    model_id: str

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        use_fc_patch: bool = True,
    ) -> QwenBackend:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen extraction requires the 'qwen' optional dependencies"
            ) from exc

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(model_path, device_map="cuda")
        if use_fc_patch:
            _convert_to_fc_patch(model)
        return cls(model=model, processor=processor, model_id=model_path)

    def generate(
        self,
        images: Sequence[Any],
        prompt: str,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> str:
        content: list[dict[str, Any]] = []
        for image in images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        generation = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if repetition_penalty < 1.0:
            raise ValueError("repetition penalty must be at least 1.0")
        if repetition_penalty > 1.0:
            input_ids = inputs.input_ids
            shape = getattr(input_ids, "shape", None)
            prompt_length = (
                int(shape[-1]) if shape is not None else len(input_ids[0])
            )
            generation["logits_processor"] = [
                GeneratedTokenRepetitionPenalty(
                    repetition_penalty,
                    prompt_length,
                )
            ]
        if do_sample:
            if temperature is None or top_p is None or top_k is None:
                raise ValueError(
                    "sampled Qwen generation requires temperature, top_p, and top_k"
                )
            generation.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )
        seed_context = (
            _seeded_rng(seed, self.model.device)
            if do_sample and seed is not None
            else nullcontext()
        )
        with seed_context:
            generated = self.model.generate(**inputs, **generation)
        trimmed = [
            output[len(source):]
            for source, output in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


@contextmanager
def _seeded_rng(seed: int, device: Any) -> Iterator[None]:
    import torch

    selected = torch.device(device)
    devices = []
    if selected.type == "cuda":
        devices = [
            selected.index if selected.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


def _convert_to_fc_patch(model: Any) -> None:
    try:
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchEmbed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Qwen FC patch requires the 'qwen' optional dependencies"
        ) from exc

    class Qwen3VLVisionPatchEmbedFC(torch.nn.Module):
        def __init__(self, original: Any) -> None:
            super().__init__()
            if not isinstance(original, Qwen3VLVisionPatchEmbed):
                raise TypeError("unexpected Qwen vision patch embedding")
            in_features = (
                original.in_channels
                * original.temporal_patch_size
                * original.patch_size
                * original.patch_size
            )
            self.proj = torch.nn.Linear(in_features, original.embed_dim, bias=True)
            self.proj.weight.data = original.proj.weight.data.view(
                original.embed_dim, in_features
            )
            self.proj.bias.data = original.proj.bias.data

        def forward(self, hidden_states: Any) -> Any:
            return self.proj(hidden_states.to(dtype=self.proj.weight.dtype))

    if not isinstance(model, Qwen3VLForConditionalGeneration):
        raise TypeError("FC patch requires Qwen3VLForConditionalGeneration")
    original = model.model.visual.patch_embed
    if not isinstance(original, Qwen3VLVisionPatchEmbed):
        raise TypeError("unexpected Qwen vision patch embedding")
    model.model.visual.patch_embed = Qwen3VLVisionPatchEmbedFC(original)
