from __future__ import annotations

import json
from pathlib import Path

import pytest

from viewing_context_pipeline.stages import StageError, _validate_multimodal_refs


ROOT = Path(__file__).resolve().parents[2]


def test_all_public_contract_ids_are_current_v1() -> None:
    expected = {
        "pipeline.schema.json": "pipeline/v1",
        "prepared_data.schema.json": "prepared-data/v1",
        "visual_manifest.schema.json": "visual-manifest/v1",
        "multimodal_reference.schema.json": "multimodal-reference/v1",
        "video_context.schema.json": "video-context/v1",
        "representations.schema.json": "representations/v1",
        "recommendations.schema.json": "recommendations/v1",
        "diagnosis.schema.json": "diagnosis/v1",
    }
    assert {
        name: json.loads((ROOT / "contracts" / name).read_text(encoding="utf-8"))["$id"]
        for name in expected
    } == expected


def _multimodal_fixture(tmp_path: Path) -> tuple[dict, list[dict]]:
    timestamps = tmp_path / "timestamp.json"
    timestamps.write_text(json.dumps([{"scene_idx": 0, "keyframe_timestamps": [5, 15]}]), encoding="utf-8")
    refs = tmp_path / "multimodal_ref"
    refs.mkdir()
    runtime = {"paths": {"multimodal_ref_dir": str(refs)}}
    visual = [{"content_id": "demo", "timestamp_json": str(timestamps)}]
    return runtime, visual


def test_multimodal_reference_requires_nonempty_file(tmp_path: Path) -> None:
    runtime, visual = _multimodal_fixture(tmp_path)
    with pytest.raises(StageError, match="missing or empty"):
        _validate_multimodal_refs(runtime, visual)


@pytest.mark.parametrize(
    "timeline,error",
    [
        ([{"shot_idx": 0, "timestamp": 5, "raw_asr": "", "raw_ocr": ""}], "image/reference mismatch"),
        ([{"shot_idx": 0, "timestamp": 5, "raw_asr": None, "raw_ocr": ""}, {"shot_idx": 1, "timestamp": 15, "raw_asr": "", "raw_ocr": ""}], "invalid multimodal_ref type"),
    ],
)
def test_multimodal_reference_rejects_alignment_and_type_errors(tmp_path: Path, timeline: list[dict], error: str) -> None:
    runtime, visual = _multimodal_fixture(tmp_path)
    path = Path(runtime["paths"]["multimodal_ref_dir"]) / "demo_multimodal_ref.jsonl"
    path.write_text(json.dumps({"schema_version": "multimodal-reference/v1", "content_id": "demo", "scene_idx": 0, "timeline": timeline}) + "\n", encoding="utf-8")
    with pytest.raises(StageError, match=error):
        _validate_multimodal_refs(runtime, visual)


def test_multimodal_reference_accepts_one_reference_per_image(tmp_path: Path) -> None:
    runtime, visual = _multimodal_fixture(tmp_path)
    path = Path(runtime["paths"]["multimodal_ref_dir"]) / "demo_multimodal_ref.jsonl"
    path.write_text(json.dumps({
        "schema_version": "multimodal-reference/v1", "content_id": "demo", "scene_idx": 0,
        "timeline": [
            {"shot_idx": 0, "timestamp": 5, "raw_asr": "speech", "raw_ocr": "title"},
            {"shot_idx": 1, "timestamp": 15, "raw_asr": "", "raw_ocr": ""},
        ],
    }) + "\n", encoding="utf-8")
    _validate_multimodal_refs(runtime, visual)
