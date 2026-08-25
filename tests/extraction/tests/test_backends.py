from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from extraction.backends import GeminiBackend, QwenBackend
import extraction.backends.gemini as gemini_module


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


def test_gemini_backend_interleaves_references(monkeypatch) -> None:
    types = SimpleNamespace(
        Part=FakePart,
        GenerateContentConfig=FakeConfig,
        ThinkingConfig=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(gemini_module, "_google_genai", lambda: (None, types))
    models = FakeModels()
    backend = GeminiBackend(
        client=SimpleNamespace(models=models),
        model_id="gemini",
    )
    references = [
        {
            "kind": "shot_reference",
            "timestamp_seconds": 5,
            "asr_text": "speech",
            "ocr_text": "caption",
        }
    ]

    assert backend.generate([Image.new("RGB", (8, 8))], "prompt", 64, references) == "gemini text"
    contents = models.call["contents"]
    assert [part["kind"] for part in contents] == ["image", "text", "text"]
    assert '"asr_text":"speech"' in contents[1]["text"]
    assert models.call["config"] == {"temperature": 0.0, "max_output_tokens": 64}
