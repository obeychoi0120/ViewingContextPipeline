from __future__ import annotations

import json
from pathlib import Path

from src.scene_context_extraction.graph_core.multimodal import MULTIMODAL_USER_MESSAGE
from src.scene_context_extraction.graph_core.prompt import USER_MESSAGE
from src.scene_context_extraction.ref.extractor import build_gemini_contents


def _scene() -> dict[str, object]:
    return {
        "scene_idx": 0,
        "timeline": [
            {"timestamp": 5, "raw_asr": "first", "raw_ocr": "one"},
            {"timestamp": 15, "raw_asr": "second", "raw_ocr": "two"},
            {"timestamp": 25, "raw_asr": "third", "raw_ocr": "three"},
        ],
    }


def _image_paths(tmp_path: Path) -> list[str]:
    paths = [tmp_path / "0005.png", tmp_path / "0015.png", tmp_path / "0025.png"]
    for path in paths:
        path.write_bytes(b"image")
    return [str(path) for path in paths]


def test_img_only_payload_has_images_and_no_asr_ocr(tmp_path: Path) -> None:
    contents = build_gemini_contents(
        _scene(),
        _image_paths(tmp_path),
        multimodal=False,
    )
    assert len(contents) == 4
    assert contents[-1] == USER_MESSAGE
    assert "first" not in str(contents)
    assert "one" not in str(contents)


def test_multimodal_payload_interleaves_image_then_shot_reference(tmp_path: Path) -> None:
    contents = build_gemini_contents(
        _scene(),
        _image_paths(tmp_path),
        multimodal=True,
    )
    assert len(contents) == 7
    first_reference = json.loads(contents[1].text)
    second_reference = json.loads(contents[3].text)
    third_reference = json.loads(contents[5].text)
    assert first_reference == {
        "kind": "shot_reference",
        "timestamp_seconds": 5,
        "asr_text": "first",
        "ocr_text": "one",
    }
    assert second_reference["timestamp_seconds"] == 15
    assert third_reference["timestamp_seconds"] == 25
    assert third_reference["asr_text"] == "third"
    assert contents[-1] == MULTIMODAL_USER_MESSAGE
