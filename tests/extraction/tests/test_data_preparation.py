from __future__ import annotations

import csv
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from viewing_context_pipeline.extraction.common.manifest import read_manifest_rows
from viewing_context_pipeline.extraction.data_preparation.microlens import prepare_catalog
from viewing_context_pipeline.extraction.data_preparation.raw_pipeline import normalize_shot_interval
from viewing_context_pipeline.extraction.data_preparation.video_processor import (
    _last_decodable_frame_timestamp_seconds,
    extract_resized_keyframes,
)


def _write_values(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "value"])
        writer.writerows(rows)


def test_sampling_accepts_fixed_30s_only() -> None:
    assert normalize_shot_interval("fixed_30s") == "fixed_30s"
    with pytest.raises(ValueError, match="fixed_30s"):
        normalize_shot_interval("fixed_15s")


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
            "viewing_context_pipeline.extraction.data_preparation.video_processor.subprocess.run",
            side_effect=extract_frame,
        ) as run,
        mock.patch(
            "viewing_context_pipeline.extraction.data_preparation.video_processor.cv2.imread",
            side_effect=[None, image],
        ),
        mock.patch(
            "viewing_context_pipeline.extraction.data_preparation.video_processor._last_decodable_frame_timestamp_seconds",
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
        "viewing_context_pipeline.extraction.data_preparation.video_processor.subprocess.run",
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

    with mock.patch(
        "viewing_context_pipeline.extraction.data_preparation.microlens.process_local_source",
        side_effect=lambda **kwargs: {"content_id": kwargs["name"]},
    ) as process:
        result = prepare_catalog(
            catalog,
            titles_csv=titles,
            tags_csv=tags,
            assets_root=tmp_path / "assets",
            output_root=tmp_path / "run",
            processing_config=tmp_path / "processing.json",
        )

    assert result["succeeded"] == 2 and result["failed"] == 0
    assert [call.kwargs["source_video_path"].name for call in process.call_args_list] == ["1.mp4", "2.mp4"]
    assert read_manifest_rows(result["manifest"]) == [
        {"content_id": "microlens_100k_00001"},
        {"content_id": "microlens_100k_00002"},
    ]
