import json

from PIL import Image
import pytest

from src.scene_context_extraction.graph_core.fingerprint import build_visual_evidence_fingerprint
from src.scene_context_extraction.graph_core.profile_text import build_graph_profile_document, serialize_graph_context
from src.scene_description_generation.pipeline import DescriptionError, build_description_profile


def _scenes():
    return [{"scene_idx": 0, "start_seconds": 0, "end_seconds": 30, "keyframes": [5, 15, 25]}]


def _frames(path):
    path.mkdir()
    for timestamp in (5, 15, 25):
        Image.new("RGB", (8, 6), "white").save(path / f"{timestamp:04d}.png")


def _context():
    return {
        "content_axes_4d": {"subject_sociality": 0.1, "media_syntheticity": 0.2, "setting_context": 0.3, "utility_orientation": 0.4},
        "content_axis_distribution": {axis: {"neutral": 1.0} for axis in ("subject_sociality", "media_syntheticity", "setting_context", "utility_orientation")},
        "top_styles": [{"id": "style:mixed", "count": 1}], "top_moods": [{"id": "mood:neutral", "count": 1}],
        "top_scene_functions": [{"id": "scene_function:unknown", "count": 1}], "top_entities": [], "top_motifs": [],
    }


def test_visual_evidence_ignores_asr_ocr_and_records_resize(tmp_path) -> None:
    frames = tmp_path / "frames"
    _frames(frames)
    first = _scenes()
    first[0]["timeline"] = [{"timestamp": value, "raw_asr": "a", "raw_ocr": "x"} for value in (5, 15, 25)]
    second = _scenes()
    second[0]["timeline"] = [{"timestamp": value, "raw_asr": "changed", "raw_ocr": "changed"} for value in (5, 15, 25)]
    a = build_visual_evidence_fingerprint(content_id="demo", scenes=first, frames_dir=frames, shot_interval="fixed_30s")
    b = build_visual_evidence_fingerprint(content_id="demo", scenes=second, frames_dir=frames, shot_interval="fixed_30s")
    assert a["fingerprint"] == b["fingerprint"]
    assert a["components"]["frames"][0]["width"] == 8


def test_graph_serialization_is_canonical(tmp_path) -> None:
    source = tmp_path / "context.json"
    source.write_text("{}", encoding="utf-8")
    context = _context()
    first = serialize_graph_context(context)
    context["top_styles"] = list(reversed(context["top_styles"]))
    assert serialize_graph_context(context) == first
    document = build_graph_profile_document({"content_id": "demo", "context": context, "aggregation_warnings": []}, {"content_id": "demo", "fingerprint": "same"}, source_path=source)
    assert document["status"] == "complete" and document["text"] == first


def test_description_uses_images_only_and_rejects_bad_summary(tmp_path) -> None:
    frames = tmp_path / "frames"
    _frames(frames)
    timestamp = tmp_path / "timestamp.json"
    timestamp.write_text(json.dumps([]), encoding="utf-8")
    calls = []

    def infer(images, prompt, _):
        calls.append((len(images), prompt))
        return "visible scene" if images else "word " * 149

    with pytest.raises(DescriptionError, match="150-300"):
        build_description_profile(content_id="demo", scenes=_scenes(), frames_dir=frames, timestamp_json_path=timestamp, evidence_fingerprint={"content_id": "demo", "fingerprint": "same"}, infer=infer, model_path="qwen")
    assert calls[0][0] == 3
    assert "OCR" in calls[0][1] and "audio" in calls[0][1]


def test_description_complete_profile_has_same_evidence(tmp_path) -> None:
    frames = tmp_path / "frames"
    _frames(frames)
    timestamp = tmp_path / "timestamp.json"
    timestamp.write_text("[]", encoding="utf-8")
    evidence = {"content_id": "demo", "fingerprint": "same"}
    infer = lambda images, prompt, limit: "visible scene" if images else "word " * 150
    document, records = build_description_profile(content_id="demo", scenes=_scenes(), frames_dir=frames, timestamp_json_path=timestamp, evidence_fingerprint=evidence, infer=infer, model_path="qwen")
    assert document["status"] == "complete"
    assert document["evidence_fingerprint"] == evidence
    assert len(records) == 1
