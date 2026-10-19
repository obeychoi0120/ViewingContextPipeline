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


def test_gemini_client_has_one_http_attempt_and_empty_failure_excludes_diagnostics(monkeypatch):
    types = SimpleNamespace(HttpRetryOptions=FakeConfig, HttpOptions=FakeConfig,
                            Part=FakePart, GenerateContentConfig=FakeConfig)
    clients = []

    def client(**kwargs):
        clients.append(kwargs)
        return SimpleNamespace(models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(text="", candidates=[])))

    monkeypatch.setattr(gemini_module, "_google_genai", lambda: (SimpleNamespace(Client=client), types))
    backend = GeminiBackend.vertex(project_id="project", model_id="gemini")
    assert clients[0]["http_options"]["retry_options"] == {"attempts": 1}
    with pytest.raises(gemini_module.GeminiEmptyResponseError) as raised:
        backend.generate([], "prompt", 64)
    assert str(raised.value) == "Gemini returned an empty response"


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


def test_gemini_backend_retries_resource_exhausted_once_after_30_seconds(monkeypatch) -> None:
    types = SimpleNamespace(Part=FakePart, GenerateContentConfig=FakeConfig)
    monkeypatch.setattr(gemini_module, "_google_genai", lambda: (None, types))
    delays = []
    monkeypatch.setattr(gemini_module.time, "sleep", delays.append)

    class ResourceExhausted(RuntimeError):
        code = 429
        status = "RESOURCE_EXHAUSTED"

    calls = 0

    def generate_content(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ResourceExhausted("capacity unavailable")
        return SimpleNamespace(text="retried response")

    backend = GeminiBackend(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
        model_id="gemini",
    )
    assert backend.generate([], "prompt", 64) == "retried response"
    assert calls == 2
    assert delays == [30]


@pytest.mark.parametrize(
    "code,status",
    [(429, "RESOURCE_EXHAUSTED"), (500, "INTERNAL")],
)
def test_gemini_backend_does_not_retry_beyond_policy(monkeypatch, code, status) -> None:
    types = SimpleNamespace(Part=FakePart, GenerateContentConfig=FakeConfig)
    monkeypatch.setattr(gemini_module, "_google_genai", lambda: (None, types))
    delays = []
    monkeypatch.setattr(gemini_module.time, "sleep", delays.append)

    class ApiError(RuntimeError):
        pass

    error = ApiError("request failed")
    error.code = code
    error.status = status
    calls = 0

    def generate_content(**kwargs):
        nonlocal calls
        calls += 1
        raise error

    backend = GeminiBackend(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
        model_id="gemini",
    )
    with pytest.raises(ApiError):
        backend.generate([], "prompt", 64)
    assert calls == (2 if code == 429 else 1)
    assert delays == ([30] if code == 429 else [])
