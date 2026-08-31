from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from unittest import mock

from PIL import Image

from extraction.data_preparation.fixed30 import (
    build_fixed_30s_windows,
    prepare_visual_item,
)
from extraction.data_preparation.microlens import prepare_catalog
from extraction.data_preparation.video_processor import (
    _last_decodable_frame_timestamp_seconds,
    extract_resized_keyframes,
)


def test_fixed_30s_sampling_uses_5_15_25_second_keyframes() -> None:
    assert build_fixed_30s_windows(31) == [
        {
            "scene_start": 0,
            "scene_end": 30,
            "duration": 30,
            "shot_change_timestamps": [0, 10, 20],
            "keyframe_timestamps": [5, 15, 25],
        },
        {
            "scene_start": 30,
            "scene_end": 31,
            "duration": 1,
            "shot_change_timestamps": [30],
            "keyframe_timestamps": [30],
        },
    ]


def test_direct_keyframe_clamps_trailing_timestamp_to_last_decodable_frame(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    destination = tmp_path / ".keyframes.direct_tmp" / "0301.png"

    def extract_frame(command: list[str], **kwargs: object) -> mock.Mock:
        destination.touch()
        return mock.Mock(returncode=0, stderr="")

    with (
        mock.patch(
            "extraction.data_preparation.video_processor.subprocess.run",
            side_effect=extract_frame,
        ) as run,
        mock.patch(
            "extraction.data_preparation.video_processor.verified_image_size",
            side_effect=[None, (64, 48)],
        ),
        mock.patch(
            "extraction.data_preparation.video_processor._last_decodable_frame_timestamp_seconds",
            return_value=300.96,
        ),
    ):
        extract_resized_keyframes(video, [301], tmp_path / "keyframes", (64, 48))

    assert run.call_count == 2
    assert run.call_args_list[0].args[0][6] == "301"
    assert run.call_args_list[1].args[0][6] == "300.96"


def test_verified_keyframe_cache_rejects_corrupt_or_wrong_size_images(tmp_path: Path) -> None:
    from extraction.data_preparation.fixed30 import resized_keyframes_match_timestamps

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


def test_last_decodable_frame_uses_latest_ffprobe_frame_timestamp(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    with mock.patch(
        "extraction.data_preparation.video_processor.subprocess.run",
        return_value=mock.Mock(stdout="0.000000\n300.960000\n", stderr="", returncode=0),
    ) as run:
        assert _last_decodable_frame_timestamp_seconds(video) == 300.96

    command = run.call_args.args[0]
    assert command[0] == "ffprobe"
    assert "-show_frames" in command


def test_prepare_visual_item_reuses_catalog_duration_and_removes_legacy_metadata(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    output_root = tmp_path / "run"
    metadata_path = output_root / "data/cohort/metadata/content-1.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text('{"title": "stale"}\n', encoding="utf-8")

    with mock.patch(
        "extraction.data_preparation.fixed30.extract_resized_keyframes"
    ) as extract:
        result = prepare_visual_item(
            content_id="content-1",
            source_video_path=video,
            assets_root=tmp_path / "assets",
            output_root=output_root,
            duration_seconds=30.1,
            image_size=(640, 352),
        )

    assert result == {"content_id": "content-1"}
    timestamp_path = tmp_path / "assets/content-1/assets/timestamp_fixed_30s.json"
    scenes = json.loads(timestamp_path.read_text(encoding="utf-8"))
    assert scenes[-1]["scene_end"] == 31
    assert extract.call_args.args[1] == [5, 15, 25, 30]
    assert not metadata_path.exists()
    assert not metadata_path.parent.exists()


def test_prepare_catalog_processes_exact_cohort(tmp_path: Path) -> None:
    catalog = [
        {"item_id": "1", "content_id": "microlens_100k_00001", "source_video_path": str(tmp_path / "1.mp4"), "duration_seconds": 30.0},
        {"item_id": "2", "content_id": "microlens_100k_00002", "source_video_path": str(tmp_path / "2.mp4"), "duration_seconds": 31.0},
    ]
    failure_path = tmp_path / "run/data/cohort/preparation_failures.jsonl"
    failure_path.parent.mkdir(parents=True)
    failure_path.write_text('{"error": "stale"}\n', encoding="utf-8")

    with (
        mock.patch(
            "extraction.data_preparation.microlens.prepare_visual_item",
            side_effect=lambda **kwargs: {"content_id": kwargs["content_id"]},
        ) as process,
        mock.patch(
            "extraction.data_preparation.microlens.ThreadPoolExecutor",
            side_effect=lambda **kwargs: ThreadPoolExecutor(**kwargs),
        ) as executor,
        mock.patch("extraction.data_preparation.microlens.tqdm") as progress_factory,
    ):
        progress = progress_factory.return_value.__enter__.return_value
        result = prepare_catalog(
            catalog,
            assets_root=tmp_path / "assets",
            output_root=tmp_path / "run",
            image_size=(640, 352),
        )

    assert result["succeeded"] == 2 and result["failed"] == 0
    assert result["workers"] == 4
    progress_factory.assert_called_once_with(
        total=2,
        desc="Prepare input data",
        unit="content",
    )
    assert progress.update.call_count == 2
    progress.update.assert_has_calls([mock.call(1), mock.call(1)])
    executor.assert_called_once_with(max_workers=4)
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
    assert not (tmp_path / "run/data/cohort/extraction_manifest.csv").exists()
