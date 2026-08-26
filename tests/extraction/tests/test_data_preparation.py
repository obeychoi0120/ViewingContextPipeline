from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

from extraction.data_preparation.fixed30 import build_fixed_30s_windows
from extraction.data_preparation.microlens import prepare_catalog
from extraction.data_preparation.video_processor import (
    _last_decodable_frame_timestamp_seconds,
    extract_resized_keyframes,
)


def _write_values(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "value"])
        writer.writerows(rows)


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
    image = np.zeros((48, 64, 3), dtype=np.uint8)
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
            "extraction.data_preparation.video_processor.cv2.imread",
            side_effect=[None, image],
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


def test_prepare_catalog_processes_exact_cohort(tmp_path: Path) -> None:
    titles = tmp_path / "titles.csv"
    tags = tmp_path / "tags.csv"
    _write_values(titles, [("1", "one"), ("2", "two")])
    _write_values(tags, [("1", "tag-a"), ("2", "tag-b")])
    catalog = [
        {"item_id": "1", "content_id": "microlens_100k_00001", "source_video_path": str(tmp_path / "1.mp4"), "duration_seconds": 30.0},
        {"item_id": "2", "content_id": "microlens_100k_00002", "source_video_path": str(tmp_path / "2.mp4"), "duration_seconds": 31.0},
    ]

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
            titles_csv=titles,
            tags_csv=tags,
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
    assert not (tmp_path / "run/data/cohort/extraction_manifest.csv").exists()
