from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import extraction.evidence as evidence


def test_scene_evidence_indexes_frame_directory_once(
    tmp_path: Path, monkeypatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for timestamp in (5, 15, 25, 35):
        Image.new("RGB", (8, 6), "white").save(frames / f"{timestamp:04d}.png")
    scenes = [
        {"scene_idx": 0, "scene_start": 0, "scene_end": 30, "keyframes": [5, 15, 25]},
        {"scene_idx": 1, "scene_start": 30, "scene_end": 40, "keyframes": [35]},
    ]
    timestamps = tmp_path / "timestamps.json"
    timestamps.write_text(json.dumps(scenes), encoding="utf-8")
    original = evidence.list_frame_images
    calls = 0

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(evidence, "list_frame_images", counted)
    rows = evidence.build_scene_evidence(scenes, frames, timestamps)

    assert calls == 1
    assert [row["keyframes"] for row in rows] == [[5, 15, 25], [35]]
    assert [[Path(path).name for path in row["image_paths"]] for row in rows] == [
        ["0005.png", "0015.png", "0025.png"],
        ["0035.png"],
    ]
