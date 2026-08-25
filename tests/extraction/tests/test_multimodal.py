from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from extraction.multimodal import (
    merge_scene_evidence,
    prepare_multimodal_evidence,
    shot_references,
)


def _scenes():
    return [
        {
            "scene_start": 0,
            "scene_end": 30,
            "shot_change_timestamps": [0, 10, 20],
            "keyframe_timestamps": [5, 15, 25],
        }
    ]


def test_merge_scene_evidence_uses_half_open_ocr_intervals() -> None:
    records = merge_scene_evidence(
        _scenes(),
        [{"start": 2.0, "end": 3.0, "word": "hello"}],
        [
            {"start_time": 9, "end_time": 9, "texts": ["At nine"]},
            {"start_time": 10, "end_time": 10, "texts": ["At ten"]},
        ],
        ocr_max_chars=1000,
    )
    timeline = records[0]["timeline"]
    assert timeline[0]["timestamp"] == 5
    assert "At nine" in timeline[0]["raw_ocr"]
    assert "At ten" not in timeline[0]["raw_ocr"]
    assert "At ten" in timeline[1]["raw_ocr"]
    assert shot_references(records[0])[0]["asr_text"] == "hello"


def test_prepare_multimodal_evidence_records_fingerprint(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    timestamps = tmp_path / "timestamps.json"
    timestamps.write_text(json.dumps(_scenes()), encoding="utf-8")
    model_root = tmp_path / "ocr_models"

    def fake_audio(source, destination):
        Path(destination).write_bytes(b"audio")

    def fake_asr(source, destination, **kwargs):
        Path(destination).write_text(
            json.dumps([{"start": 1, "end": 2, "word": "speech"}]),
            encoding="utf-8",
        )

    def fake_frames(source, destination):
        Path(destination).mkdir(parents=True)

    def fake_ocr(source, destination, **kwargs):
        Path(destination).write_text(
            json.dumps([{"frame_time": 1, "texts": ["caption"]}]),
            encoding="utf-8",
        )

    with (
        mock.patch("extraction.data_preparation.audio_extractor.extract_audio", side_effect=fake_audio),
        mock.patch("extraction.data_preparation.asr_processor.process_asr", side_effect=fake_asr),
        mock.patch("extraction.data_preparation.video_processor.extract_frames", side_effect=fake_frames),
        mock.patch("extraction.data_preparation.ocr_processor.process_ocr", side_effect=fake_ocr),
    ):
        manifest = prepare_multimodal_evidence(
            video,
            timestamps,
            tmp_path / "output",
            asr_model="small",
            ocr_model_root=model_root,
        )

    assert manifest["schema_version"] == "multimodal-evidence/v1"
    assert manifest["scene_count"] == 1
    record = json.loads(
        (tmp_path / "output/multimodal_timeline.jsonl").read_text(encoding="utf-8")
    )
    assert record["timeline"][0]["raw_asr"] == "speech"
    assert record["timeline"][0]["raw_ocr"] == "caption"
