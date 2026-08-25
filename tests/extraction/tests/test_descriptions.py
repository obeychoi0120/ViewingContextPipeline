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
    assert len(backend.calls[0][0]) == 3
    assert backend.calls[0][1:] == ("describe", 128, ())


def test_description_summary_preserves_scene_order_and_word_limit() -> None:
    records = [
        {"schema_version": "scene-description/v1", "scene_idx": 0, "description": "first"},
        {"schema_version": "scene-description/v1", "scene_idx": 1, "description": "second"},
    ]
    prompt = description_summary_prompt("{scenes}", records)
    assert prompt.index("Scene 0") < prompt.index("Scene 1")
    assert len(validate_summary("word " * 150).split()) == 150
    with pytest.raises(DescriptionError, match="150-300"):
        validate_summary("short")
