from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from extraction.descriptions import (
    DescriptionError,
    description_summary_prompt,
    extract_scene_descriptions,
    validate_summary,
)
from extraction.summary_validation import summary_soft_warnings


class FakeBackend:
    model_id = "fake"

    def __init__(self, response: str = "visible scene") -> None:
        self.response = response
        self.calls = []

    def generate(self, images, prompt, max_new_tokens, references=()):
        self.calls.append((images, prompt, max_new_tokens, references))
        return self.response


def test_description_scene_uses_chronological_images(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for timestamp in (5, 15, 25):
        Image.new("RGB", (8, 6), "white").save(frames / f"{timestamp:04d}.png")
    timestamp_path = tmp_path / "timestamps.json"
    timestamp_path.write_text(json.dumps([]), encoding="utf-8")
    backend = FakeBackend()

    records = extract_scene_descriptions(
        content_id="demo",
        scenes=[{"scene_idx": 0, "keyframes": [5, 15, 25]}],
        frames_dir=frames,
        timestamp_json_path=timestamp_path,
        backend=backend,
        prompt="describe",
        max_new_tokens=128,
    )

    assert records[0]["keyframes"] == [5, 15, 25]
    assert set(records[0]) == {
        "schema_version",
        "content_id",
        "scene_idx",
        "keyframes",
        "description",
    }
    assert len(backend.calls[0][0]) == 3
    assert backend.calls[0][1:] == ("describe", 128, ())


def test_description_scene_prompt_uses_strict_visible_only_grounding() -> None:
    root = Path(__file__).resolve().parents[3]
    prompt = (root / "config/prompts/description_scene_v1.md").read_text(
        encoding="utf-8"
    )

    assert "using only facts directly" in prompt
    assert "Do not infer story, intent, identity, demographics" in prompt
    assert "purpose, audience" in prompt
    assert "cultural context, or actual" in prompt
    assert "Do not read or transcribe visible text" in prompt


def test_description_summary_preserves_scene_order_and_uses_soft_word_guidance() -> None:
    records = [
        {"schema_version": "scene-description/v1", "scene_idx": 0, "description": "first"},
        {"schema_version": "scene-description/v1", "scene_idx": 1, "description": "second"},
    ]
    prompt = description_summary_prompt("{scenes}", records)
    assert prompt.index("Scene 0") < prompt.index("Scene 1")
    assert len(validate_summary("word " * 150).split()) == 150
    assert len(validate_summary("word " * 162).split()) == 162
    assert validate_summary("short") == "short"
    assert summary_soft_warnings("word " * 162) == [
        "summary_word_guidance_exceeded: observed=162 guidance_max=150"
    ]
    with pytest.raises(DescriptionError, match="must not be empty"):
        validate_summary("")
