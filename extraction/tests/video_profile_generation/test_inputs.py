from __future__ import annotations

import json

import pytest

from src.video_profile_generation.inputs import (
    MAX_KEYFRAMES_PER_VIDEO,
    InputError,
    load_manifest_content_ids,
    load_content_bundle,
)

from .conftest import (
    CONTENT_ID,
    write_local_inputs,
    write_manifest,
)


REF_NAME = f"ViewingContextPipeline/asset/fixed_15s/ref_jsonl/{CONTENT_ID}_ref.jsonl"


def load_test_bundle(tmp_path, input_objects: dict[str, str]):
    return load_content_bundle(
        write_local_inputs(tmp_path, input_objects),
        CONTENT_ID,
    )


def timeline_item(shot_idx: int, timestamp: object) -> dict[str, object]:
    return {
        "shot_idx": shot_idx,
        "timestamp": timestamp,
        "raw_asr": "",
        "raw_ocr": "",
    }


def test_manifest_controls_content_ids_and_loads_bundle(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    input_objects[
        "ViewingContextPipeline/asset/fixed_15s/ref_jsonl/stale_not_in_manifest_ref.jsonl"
    ] = input_objects[REF_NAME]
    assert load_manifest_content_ids(write_manifest(tmp_path)) == [CONTENT_ID]
    bundle = load_content_bundle(
        write_local_inputs(tmp_path, input_objects),
        CONTENT_ID,
    )

    assert bundle.content_id == CONTENT_ID
    assert bundle.frame_count == 2
    assert [shot.raw_asr for shot in bundle.scenes[0].timeline] == [
        "영수증 앱테크를",
        "소개합니다.",
    ]
    assert [shot.raw_ocr for shot in bundle.scenes[0].timeline] == [
        "영수증 적립",
        "영수증 적립",
    ]
    assert not hasattr(bundle.scenes[0], "vlm_visual_graph")


def test_manifest_rejects_duplicate_content_id(
    tmp_path,
) -> None:
    manifest_path = write_manifest(
        tmp_path,
        (
            f"content_id,url\n{CONTENT_ID},https://example.com\n"
            f"{CONTENT_ID},https://example.com/duplicate\n"
        ),
    )

    with pytest.raises(InputError, match="duplicates"):
        load_manifest_content_ids(manifest_path)


def test_optional_metadata_header_is_ignored(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    header = {"_type": "video_metadata", "title": "비신뢰 중복 metadata"}
    input_objects[REF_NAME] = "\n".join([json.dumps(header), input_objects[REF_NAME]])

    bundle = load_test_bundle(tmp_path, input_objects)

    assert bundle.frame_count == 2
    assert bundle.metadata["title"] == "매일 사용하는 고효율 앱테크"


def test_ref_scenes_are_sorted_by_scene_idx(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    first = {"scene_idx": 2, "timeline": [timeline_item(0, 34)]}
    second = {"scene_idx": 1, "timeline": [timeline_item(0, 0)]}
    input_objects[REF_NAME] = "\n".join([json.dumps(first), json.dumps(second)])

    bundle = load_test_bundle(tmp_path, input_objects)

    assert [scene.scene_idx for scene in bundle.scenes] == [1, 2]


def test_unsorted_timestamps_within_scene_are_rejected(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    ref_scene = json.loads(input_objects[REF_NAME])
    ref_scene["timeline"][0]["timestamp"] = 34
    ref_scene["timeline"][1]["timestamp"] = 0
    input_objects[REF_NAME] = json.dumps(ref_scene)

    with pytest.raises(InputError, match="strictly increasing timestamps"):
        load_test_bundle(tmp_path, input_objects)


def test_duplicate_shot_idx_within_scene_is_rejected(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    ref_scene = json.loads(input_objects[REF_NAME])
    ref_scene["timeline"][1]["shot_idx"] = ref_scene["timeline"][0]["shot_idx"]
    input_objects[REF_NAME] = json.dumps(ref_scene)

    with pytest.raises(InputError, match="shot_idx values must not contain duplicates"):
        load_test_bundle(tmp_path, input_objects)


def test_same_shot_idx_is_allowed_in_different_scenes(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    first = {"scene_idx": 0, "timeline": [timeline_item(0, 0)]}
    second = {"scene_idx": 1, "timeline": [timeline_item(0, 34)]}
    input_objects[REF_NAME] = "\n".join([json.dumps(first), json.dumps(second)])

    bundle = load_test_bundle(tmp_path, input_objects)

    assert [scene.timeline[0].shot_idx for scene in bundle.scenes] == [0, 0]


def test_timestamps_must_increase_across_scenes(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    first = {"scene_idx": 0, "timeline": [timeline_item(0, 34)]}
    second = {"scene_idx": 1, "timeline": [timeline_item(0, 0)]}
    input_objects[REF_NAME] = "\n".join([json.dumps(first), json.dumps(second)])

    with pytest.raises(InputError, match="strictly increasing across scenes"):
        load_test_bundle(tmp_path, input_objects)


def test_timestamps_must_not_repeat_across_scenes(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    first = {"scene_idx": 0, "timeline": [timeline_item(0, 0)]}
    second = {"scene_idx": 1, "timeline": [timeline_item(0, 0)]}
    input_objects[REF_NAME] = "\n".join([json.dumps(first), json.dumps(second)])

    with pytest.raises(InputError, match="strictly increasing across scenes"):
        load_test_bundle(tmp_path, input_objects)


def test_maximum_keyframe_count_is_allowed(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    ref_scene = json.loads(input_objects[REF_NAME])
    ref_scene["timeline"] = [
        timeline_item(shot_idx, shot_idx) for shot_idx in range(MAX_KEYFRAMES_PER_VIDEO)
    ]
    input_objects[REF_NAME] = json.dumps(ref_scene)
    for timestamp_seconds in range(MAX_KEYFRAMES_PER_VIDEO):
        input_objects[
            f"ViewingContextPipeline/asset/fixed_15s/resized_keyframes/{CONTENT_ID}/{timestamp_seconds:04d}.png"
        ] = "image"

    bundle = load_test_bundle(tmp_path, input_objects)

    assert bundle.frame_count == MAX_KEYFRAMES_PER_VIDEO


def test_keyframe_count_above_limit_is_rejected(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    ref_scene = json.loads(input_objects[REF_NAME])
    ref_scene["timeline"] = [
        timeline_item(shot_idx, shot_idx)
        for shot_idx in range(MAX_KEYFRAMES_PER_VIDEO + 1)
    ]
    input_objects[REF_NAME] = json.dumps(ref_scene)

    with pytest.raises(InputError, match="1441 keyframes exceed.*1440"):
        load_test_bundle(tmp_path, input_objects)


def test_missing_keyframe_fails(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    del input_objects[f"ViewingContextPipeline/asset/fixed_15s/resized_keyframes/{CONTENT_ID}/0034.png"]

    with pytest.raises(InputError, match="missing keyframes"):
        load_test_bundle(tmp_path, input_objects)


def test_missing_metadata_fails(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    del input_objects[f"ViewingContextPipeline/asset/metadata/{CONTENT_ID}.json"]

    with pytest.raises(InputError, match="failed to read metadata"):
        load_test_bundle(tmp_path, input_objects)


def test_missing_metadata_title_fails(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    metadata_name = f"ViewingContextPipeline/asset/metadata/{CONTENT_ID}.json"
    metadata = json.loads(input_objects[metadata_name])
    del metadata["title"]
    input_objects[metadata_name] = json.dumps(metadata, ensure_ascii=False)

    with pytest.raises(InputError, match="metadata.title must be a string"):
        load_test_bundle(tmp_path, input_objects)


def test_missing_ref_jsonl_fails(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    del input_objects[REF_NAME]

    with pytest.raises(InputError, match="failed to read Ref JSONL"):
        load_test_bundle(tmp_path, input_objects)


def test_numeric_string_timestamp_is_rejected(
    input_objects: dict[str, str],
    tmp_path,
) -> None:
    ref_scene = json.loads(input_objects[REF_NAME])
    ref_scene["timeline"][0]["timestamp"] = "0"
    input_objects[REF_NAME] = json.dumps(ref_scene)

    with pytest.raises(InputError, match="invalid Ref JSONL line"):
        load_test_bundle(tmp_path, input_objects)
