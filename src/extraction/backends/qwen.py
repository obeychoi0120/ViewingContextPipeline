from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from extraction.multimodal import (
    shot_reference_text,
    validate_image_reference_alignment,
)


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
        references: Sequence[dict[str, Any]] = (),
        *,
        do_sample: bool = False,
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> str:
        if references:
            validate_image_reference_alignment(len(images), list(references))
        content: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            content.append({"type": "image", "image": image})
            if references:
                content.append(
                    {"type": "text", "text": shot_reference_text(references[index])}
                )
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
        if do_sample:
            if seed is None or temperature is None or top_p is None or top_k is None:
                raise ValueError(
                    "sampled Qwen generation requires seed, temperature, top_p, and top_k"
                )
            generation.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )
        seed_context = (
            _seeded_rng(seed, self.model.device) if do_sample else nullcontext()
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
