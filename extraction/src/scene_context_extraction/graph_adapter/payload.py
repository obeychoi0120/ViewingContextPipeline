from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_scene_timestamps(timestamp_json_path: str | Path | None) -> list[dict[str, Any]]:
    if not timestamp_json_path:
        return []
    path = Path(timestamp_json_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def select_scene_image_paths(
    frames_dir: str | Path,
    item: dict[str, Any],
    scene_timestamps: list[dict[str, Any]],
    fallback_idx: int,
) -> list[str]:
    frame_paths = list_frame_images(frames_dir)
    if not frame_paths:
        return []

    timestamps = normalize_keyframe_timestamps(get_keyframe_timestamps(item, scene_timestamps, fallback_idx))
    timestamp_matches = []
    seen: set[Path] = set()
    for timestamp in timestamps:
        match = image_path_for_timestamp(frame_paths, timestamp)
        if match is not None and match not in seen:
            timestamp_matches.append(match)
            seen.add(match)
    return [str(path) for path in timestamp_matches]


def list_frame_images(frames_dir: str | Path) -> list[Path]:
    path = Path(frames_dir)
    if not path.is_dir():
        return []
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_keyframe_timestamps(
    item: dict[str, Any],
    scene_timestamps: list[dict[str, Any]],
    idx: int,
) -> list[Any]:
    timeline = item.get("timeline")
    if isinstance(timeline, list) and timeline:
        timeline_timestamps = [
            shot.get("timestamp")
            for shot in timeline
            if isinstance(shot, dict) and shot.get("timestamp") is not None
        ]
        if len(timeline_timestamps) == len(timeline):
            return timeline_timestamps

    item_timestamps = item.get("keyframe_timestamps")
    if isinstance(item_timestamps, list) and item_timestamps:
        return item_timestamps
    timestamp_index = item.get("scene_idx", idx)
    if type(timestamp_index) is not int:
        timestamp_index = idx
    if scene_timestamps and 0 <= timestamp_index < len(scene_timestamps):
        timestamp_record = scene_timestamps[timestamp_index]
        if isinstance(timestamp_record, dict):
            timestamps = timestamp_record.get("keyframe_timestamps", [])
            if isinstance(timestamps, list):
                return timestamps
    return []


def normalize_keyframe_timestamps(timestamps: list[Any]) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for timestamp in timestamps:
        try:
            timestamp_sec = int(round(float(timestamp)))
        except (TypeError, ValueError):
            continue
        if timestamp_sec not in seen:
            normalized.append(timestamp_sec)
            seen.add(timestamp_sec)
    return normalized


def image_path_for_timestamp(frame_paths: list[Path], timestamp: Any) -> Path | None:
    try:
        timestamp_sec = int(round(float(timestamp)))
    except (TypeError, ValueError):
        return None

    timestamp_name = f"{timestamp_sec:04d}"
    for image_path in frame_paths:
        if image_path.stem == timestamp_name:
            return image_path
    return None
