from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from PIL import Image

from preparation.video.microlens import prepare_catalog


def test_verified_keyframe_cache_rejects_corrupt_or_wrong_size_images(tmp_path: Path) -> None:
    from preparation.video.fixed30 import resized_keyframes_match_timestamps

    timestamp_path = tmp_path / "timestamps.json"
    timestamp_path.write_text('[{"keyframe_timestamps": [5]}]', encoding="utf-8")
    frames = tmp_path / "frames"
    frames.mkdir()
    image_path = frames / "0005.png"

    Image.new("RGB", (64, 48), "white").save(image_path)
    assert resized_keyframes_match_timestamps(timestamp_path, frames, (64, 48))

    Image.new("RGB", (32, 24), "white").save(image_path)
    assert not resized_keyframes_match_timestamps(timestamp_path, frames, (64, 48))

    image_path.write_bytes(b"not-a-png")
    assert not resized_keyframes_match_timestamps(timestamp_path, frames, (64, 48))

    image_path.write_bytes(b"")
    assert not resized_keyframes_match_timestamps(timestamp_path, frames, (64, 48))

    Image.new("RGB", (64, 48), "white").save(image_path)
    payload = image_path.read_bytes()
    image_path.write_bytes(payload[: len(payload) // 2])
    assert not resized_keyframes_match_timestamps(timestamp_path, frames, (64, 48))


def test_prepare_catalog_processes_exact_cohort(tmp_path: Path) -> None:
    catalog = [
        {"item_id": "1", "content_id": "microlens_100k_00001", "source_video_path": str(tmp_path / "1.mp4"), "duration_seconds": 30.0},
        {"item_id": "2", "content_id": "microlens_100k_00002", "source_video_path": str(tmp_path / "2.mp4"), "duration_seconds": 31.0},
    ]
    failure_path = tmp_path / "run/cohort/preparation_failures.jsonl"
    failure_path.parent.mkdir(parents=True)
    failure_path.write_text('{"error": "stale"}\n', encoding="utf-8")

    with (
        mock.patch(
            "preparation.video.microlens.prepare_visual_item",
            side_effect=lambda **kwargs: {"content_id": kwargs["content_id"], "reused_frames": 6, "extracted_frames": 0},
        ) as process,
        mock.patch(
            "preparation.video.microlens.ThreadPoolExecutor",
            side_effect=lambda **kwargs: ThreadPoolExecutor(**kwargs),
        ) as executor,
        mock.patch("preparation.video.microlens.tqdm") as progress_factory,
    ):
        progress = progress_factory.return_value.__enter__.return_value
        result = prepare_catalog(
            catalog,
            assets_root=tmp_path / "run/cohort/source_assets",
            output_root=tmp_path / "shared",
            failure_path=failure_path,
            image_size=(640, 352),
        )

    assert result["succeeded"] == 2 and result["failed"] == 0
    assert result["workers"] == 8
    progress_factory.assert_called_once_with(
        total=2,
        desc="Prepare visual evidence",
        unit="video",
        dynamic_ncols=True,
    )
    assert progress.update.call_count == 2
    progress.update.assert_has_calls([mock.call(1), mock.call(1)])
    executor.assert_called_once_with(max_workers=8)
    assert sorted(call.kwargs["source_video_path"].name for call in process.call_args_list) == [
        "1.mp4",
        "2.mp4",
    ]
    assert sorted(call.kwargs["duration_seconds"] for call in process.call_args_list) == [
        30.0,
        31.0,
    ]
    assert all("metadata" not in call.kwargs for call in process.call_args_list)
    assert not failure_path.exists()
    assert not (tmp_path / "run/cohort/extraction_manifest.csv").exists()
