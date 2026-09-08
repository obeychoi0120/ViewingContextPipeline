from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from PIL import Image
import pytest

from extraction.backends import GeminiBackend, QwenBackend
import extraction.backends.gemini as gemini_module
import extraction.backends.qwen as qwen_module


class FakeInputs(dict):
    input_ids = [[1, 2]]

    def to(self, device):
        self["moved_to"] = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return FakeInputs()

    def batch_decode(self, values, **kwargs):
        return ["generated text"]


class FakeModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[1, 2, 3]]


def test_qwen_backend_uses_native_images_and_deterministic_generation() -> None:
    model = FakeModel()
    processor = FakeProcessor()
    backend = QwenBackend(model=model, processor=processor, model_id="qwen")
    image = Image.new("RGB", (8, 8))

    assert backend.generate([image], "prompt", 32) == "generated text"
    assert processor.messages[0]["content"] == [
        {"type": "image", "image": image},
        {"type": "text", "text": "prompt"},
    ]
    assert model.kwargs["max_new_tokens"] == 32
    assert model.kwargs["do_sample"] is False
    assert "temperature" not in model.kwargs
    assert "logits_processor" not in model.kwargs


def test_qwen_backend_penalizes_only_generated_tokens() -> None:
    model = FakeModel()
    backend = QwenBackend(
        model=model,
        processor=FakeProcessor(),
        model_id="qwen",
    )

    assert backend.generate(
        [],
        "summary prompt",
        64,
        repetition_penalty=1.05,
    ) == "generated text"

    processors = model.kwargs["logits_processor"]
    assert len(processors) == 1
    assert isinstance(processors[0], qwen_module.GeneratedTokenRepetitionPenalty)
    assert processors[0].penalty == 1.05
    assert processors[0].prompt_length == 2


def test_qwen_backend_uses_optional_seed_for_sampling(monkeypatch) -> None:
    seeds = []

    @contextmanager
    def fake_seeded_rng(seed, device):
        seeds.append((seed, device))
        yield

    monkeypatch.setattr(qwen_module, "_seeded_rng", fake_seeded_rng)
    model = FakeModel()
    processor = FakeProcessor()
    backend = QwenBackend(model=model, processor=processor, model_id="qwen")

    assert backend.generate(
        [],
        "retry prompt",
        64,
        do_sample=True,
        seed=43,
        temperature=0.1,
        top_p=0.8,
        top_k=20,
    ) == "generated text"

    assert seeds == [(43, "cuda:0")]
    assert model.kwargs["do_sample"] is True
    assert model.kwargs["temperature"] == 0.1
    assert model.kwargs["top_p"] == 0.8
    assert model.kwargs["top_k"] == 20


def test_qwen_backend_allows_unseeded_sampling(monkeypatch) -> None:
    def fail_if_seeded(*_args, **_kwargs):
        raise AssertionError("unseeded sampling must use the ambient RNG state")

    monkeypatch.setattr(qwen_module, "_seeded_rng", fail_if_seeded)
    model = FakeModel()
    backend = QwenBackend(
        model=model,
        processor=FakeProcessor(),
        model_id="qwen",
    )

    assert backend.generate(
        [],
        "summary prompt",
        64,
        do_sample=True,
        temperature=0.2,
        top_p=0.8,
        top_k=20,
    ) == "generated text"

    assert model.kwargs["do_sample"] is True
    assert model.kwargs["temperature"] == 0.2


class FakePart:
    @staticmethod
    def from_bytes(*, data, mime_type):
        return {"kind": "image", "data": data, "mime_type": mime_type}

    @staticmethod
    def from_text(*, text):
        return {"kind": "text", "text": text}


class FakeConfig(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)


class FakeModels:
    def __init__(self) -> None:
        self.call = None

    def generate_content(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(text="gemini text")


def test_gemini_backend_uses_images_prompt_and_operational_controls(monkeypatch) -> None:
    types = SimpleNamespace(
        Part=FakePart,
        GenerateContentConfig=FakeConfig,
        ThinkingConfig=lambda **kwargs: kwargs,
        MediaResolution=lambda value: value,
    )
    monkeypatch.setattr(gemini_module, "_google_genai", lambda: (None, types))
    models = FakeModels()
    backend = GeminiBackend(
        client=SimpleNamespace(models=models),
        model_id="gemini",
        temperature=0.25,
        thinking_level="low",
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
    )
    assert backend.generate([Image.new("RGB", (8, 8))], "prompt", 64) == "gemini text"
    contents = models.call["contents"]
    assert [part["kind"] for part in contents] == ["image", "text"]
    assert contents[1]["text"] == "prompt"
    assert models.call["config"] == {
        "temperature": 0.25,
        "max_output_tokens": 64,
        "thinking_config": {"thinking_level": "low"},
        "media_resolution": "MEDIA_RESOLUTION_MEDIUM",
    }


@pytest.mark.parametrize("reason", ["SAFETY", "MAX_TOKENS", None])
def test_gemini_empty_response_preserves_sdk_diagnostics(monkeypatch, reason) -> None:
    types = pytest.importorskip("google.genai.types")
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(finish_reason=reason, finish_message="details")]
        if reason else None,
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY")
        if reason is None else None,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1500, candidates_token_count=0, thoughts_token_count=1024,
        ),
    )
    monkeypatch.setattr(gemini_module, "_google_genai", lambda: (None, types))
    backend = GeminiBackend(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: response)),
        model_id="gemini",
    )
    with pytest.raises(gemini_module.GeminiEmptyResponseError) as raised:
        backend.generate([], "prompt", 1024)
    diagnostics = raised.value.diagnostics
    assert diagnostics["usage_metadata"]["thoughts_token_count"] == 1024
    if reason:
        assert diagnostics["candidates"] == [{"finish_reason": reason, "finish_message": "details"}]
        assert diagnostics["prompt_feedback"] is None
        assert reason in str(raised.value)
    else:
        assert diagnostics["candidates"] == []
        assert diagnostics["prompt_feedback"]["block_reason"] == "SAFETY"
    assert '"usage_metadata"' in str(raised.value)


def test_gemini_empty_response_without_metadata(monkeypatch) -> None:
    types = SimpleNamespace(Part=FakePart, GenerateContentConfig=FakeConfig)
    monkeypatch.setattr(gemini_module, "_google_genai", lambda: (None, types))
    response = SimpleNamespace(text="  ")
    backend = GeminiBackend(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: response)),
        model_id="gemini",
    )
    with pytest.raises(gemini_module.GeminiEmptyResponseError) as raised:
        backend.generate([], "prompt", 32)
    assert raised.value.diagnostics == {
        "candidates": [], "prompt_feedback": None, "usage_metadata": None,
    }
