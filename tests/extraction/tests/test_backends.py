from __future__ import annotations

from types import SimpleNamespace

from PIL import Image
import pytest

from extraction.backends import GeminiBackend
import extraction.backends.gemini as gemini_module


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
