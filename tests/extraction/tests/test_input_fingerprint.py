from __future__ import annotations

from PIL import Image
import pytest

from viewing_context_pipeline.extraction.scene_context_extraction.graph_core.fingerprint import build_input_fingerprint


def _scene(asr: str = "hello", ocr: str = "title") -> list[dict[str, object]]:
    return [
        {
            "scene_idx": 0,
            "keyframes": [0],
            "timeline": [
                {
                    "timestamp": 0,
                    "raw_asr": asr,
                    "raw_ocr": ocr,
                }
            ],
        }
    ]


def _fingerprint(tmp_path, *, multimodal: bool, scenes):
    return build_input_fingerprint(
        content_id="demo",
        scenes=scenes,
        frames_dir=tmp_path,
        multimodal=multimodal,
        backend="ref",
        model_config={"model": "test"},
    )["fingerprint"]


def test_asr_ocr_change_invalidates_only_multimodal(tmp_path) -> None:
    Image.new("RGB", (8, 8), "white").save(tmp_path / "0000.png")
    original = _scene()
    changed = _scene(asr="changed", ocr="changed")

    assert _fingerprint(tmp_path, multimodal=False, scenes=original) == _fingerprint(
        tmp_path, multimodal=False, scenes=changed
    )
    assert _fingerprint(tmp_path, multimodal=True, scenes=original) != _fingerprint(
        tmp_path, multimodal=True, scenes=changed
    )


def test_keyframe_change_invalidates_both_tracks(tmp_path) -> None:
    frame = tmp_path / "0000.png"
    Image.new("RGB", (8, 8), "white").save(frame)
    before = {
        modality: _fingerprint(tmp_path, multimodal=modality, scenes=_scene())
        for modality in (False, True)
    }
    Image.new("RGB", (8, 8), "black").save(frame)

    for modality in (False, True):
        assert before[modality] != _fingerprint(
            tmp_path, multimodal=modality, scenes=_scene()
        )


def test_multimodal_rejects_non_string_asr_ocr(tmp_path) -> None:
    Image.new("RGB", (8, 8), "white").save(tmp_path / "0000.png")
    invalid = _scene()
    invalid[0]["timeline"][0]["raw_asr"] = None
    with pytest.raises(ValueError, match="raw_asr must be a string"):
        _fingerprint(tmp_path, multimodal=True, scenes=invalid)


def test_fixed_30s_fingerprint_records_sampling_and_ocr_contract(tmp_path) -> None:
    Image.new("RGB", (8, 8), "white").save(tmp_path / "0005.png")
    document = build_input_fingerprint(
        content_id="demo",
        scenes=_scene(),
        frames_dir=tmp_path,
        multimodal=True,
        backend="ref",
        shot_interval="fixed_30s",
    )

    contract = document["components"]["sampling_contract"]
    assert contract["scene_seconds"] == 30
    assert contract["reference_seconds"] == 10
    assert contract["keyframe_offsets_seconds"] == [5, 15, 25]
    assert contract["ocr_sampling_fps"] == 1
    assert contract["ocr_max_chars"] == 1000

    img_only = build_input_fingerprint(
        content_id="demo",
        scenes=_scene(),
        frames_dir=tmp_path,
        multimodal=False,
        backend="ref",
        shot_interval="fixed_30s",
    )
    assert "ocr_sampling_fps" not in img_only["components"]["sampling_contract"]
