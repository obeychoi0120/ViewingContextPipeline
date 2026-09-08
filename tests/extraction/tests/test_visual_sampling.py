from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from unittest import mock

from PIL import Image
import pytest

from extraction.data_preparation.fixed30 import (
    resized_keyframes_match_timestamps,
    selected_keyframe_timestamps,
)
from extraction.data_preparation.video_processor import extract_resized_keyframes
from extraction.evidence import build_scene_evidence, image_path_for_timestamp
from validation.diagnosis_scenes import _nonempty_timestamp_list
from visual_sampling import build_fixed_windows, timestamp_stem, truncate_timestamp


@pytest.mark.parametrize(
    "duration,scene_duration,count,expected",
    [
        (30, 30, 6, [[2.5, 7.5, 12.5, 17.5, 22.5, 27.5]]),
        (42, 30, 6, [[2.5, 7.5, 12.5, 17.5, 22.5, 27.5], [32.5, 37.5, 41]]),
        (30.1, 30, 6, [[2.5, 7.5, 12.5, 17.5, 22.5, 27.5], [30]]),
        (31, 30, 6, [[2.5, 7.5, 12.5, 17.5, 22.5, 27.5], [30.5]]),
        (31.1, 30, 6, [[2.5, 7.5, 12.5, 17.5, 22.5, 27.5], [30.5]]),
        (10, 10, 3, [[1.6, 5, 8.3]]),
        (21, 20, 4, [[2.5, 7.5, 12.5, 17.5], [20.5]]),
        (0.05, 30, 6, [[0]]),
        (0.38, 1, 8, [[0, 0.1, 0.3]]),
    ],
)
def test_sampling_keeps_full_bins_and_clips_only_the_tail(
    duration, scene_duration, count, expected,
):
    scenes = build_fixed_windows(
        duration, scene_duration=scene_duration, num_keyframes=count,
    )
    assert [scene["keyframe_timestamps"] for scene in scenes] == expected
    assert scenes[-1]["scene_end"] == duration
    for scene in scenes:
        timestamps = scene["keyframe_timestamps"]
        assert timestamps == sorted(set(timestamps))
        assert all(scene["scene_start"] <= value < scene["scene_end"] for value in timestamps)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True, None])
def test_sampling_rejects_invalid_duration(duration):
    with pytest.raises(ValueError):
        build_fixed_windows(duration, scene_duration=30, num_keyframes=6)


@pytest.mark.parametrize(
    "value,expected,name",
    [
        (2.5, 2.5, "0002_5"),
        (7.5, 7.5, "0007_5"),
        (Fraction(5, 3), 1.6, "0001_6"),
        (1.29, 1.2, "0001_2"),
        (0.3, 0.3, "0000_3"),
        (59.99, 59.9, "0059_9"),
        (5.0, 5, "0005"),
        (10002.59, 10002.5, "10002_5"),
    ],
)
def test_timestamp_truncation_and_filename_agree(value, expected, name):
    assert truncate_timestamp(value) == expected
    assert timestamp_stem(value) == name


def test_fractional_seek_json_cache_and_image_selection_agree(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    frames = tmp_path / "frames"
    stamps = tmp_path / "timestamps.json"
    scenes = build_fixed_windows(10, scene_duration=10, num_keyframes=3)
    stamps.write_text(json.dumps(scenes), encoding="utf-8")
    requested = selected_keyframe_timestamps(stamps)
    seeks = []

    def extract(command, **_kwargs):
        seeks.append(command[command.index("-ss") + 1])
        Image.new("RGB", (16, 8)).save(command[-1])
        return mock.Mock(returncode=0, stderr="")

    with mock.patch(
        "extraction.data_preparation.video_processor.subprocess.run", side_effect=extract,
    ):
        extract_resized_keyframes(video, requested, frames, (16, 8))

    assert seeks == ["1.6", "5", "8.3"]
    assert {path.name for path in frames.iterdir()} == {"0001_6.png", "0005.png", "0008_3.png"}
    assert resized_keyframes_match_timestamps(stamps, frames, (16, 8))
    row, = build_scene_evidence(scenes, frames, stamps)
    assert row["keyframes"] == [1.6, 5, 8.3]
    assert [Path(path).name for path in row["image_paths"]] == [
        "0001_6.png", "0005.png", "0008_3.png",
    ]
    assert image_path_for_timestamp(list(frames.iterdir()), 1.6) == frames / "0001_6.png"
    assert _nonempty_timestamp_list(row["keyframes"])


@pytest.mark.parametrize(
    "timestamps",
    [[2.55], [float("nan")], [float("inf")], [-0.1], [True], [2.5, 2.5], [7.5, 2.5]],
)
def test_invalid_keyframes_are_rejected_before_extracting(tmp_path, timestamps):
    video = tmp_path / "video.mp4"
    video.touch()
    with pytest.raises(ValueError):
        extract_resized_keyframes(video, timestamps, tmp_path / "frames", (16, 8))
    assert not _nonempty_timestamp_list(timestamps)
    assert not (tmp_path / "frames").exists()
