from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_images(image_paths: list[str]) -> list[Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("image loading requires Pillow") from exc
    images: list[Any] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def load_scene_timestamps(path: str | Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).is_file():
        return []
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def build_scene_evidence(
    scenes: list[dict[str, Any]],
    frames_dir: str | Path,
    timestamp_json_path: str | Path,
) -> list[dict[str, Any]]:
    timeline = load_scene_timestamps(timestamp_json_path)
    frame_index = _frame_image_index(frames_dir)
    rows: list[dict[str, Any]] = []
    for fallback_idx, scene in enumerate(scenes):
        keyframes = normalize_keyframe_timestamps(
            get_keyframe_timestamps(scene, timeline, fallback_idx)
        )
        rows.append({
            "fallback_idx": fallback_idx,
            "scene_idx": int(scene.get("scene_idx", fallback_idx)),
            "scene_start_seconds": _scene_boundary(
                scene, timeline, fallback_idx, "scene_start", keyframes[0] if keyframes else 0
            ),
            "scene_end_seconds": _scene_boundary(
                scene,
                timeline,
                fallback_idx,
                "scene_end",
                keyframes[-1] if keyframes else 0,
            ),
            "keyframes": keyframes,
            "image_paths": select_scene_image_paths(
                frames_dir,
                scene,
                timeline,
                fallback_idx,
                frame_index=frame_index,
            ),
        })
    return rows


def select_scene_image_paths(
    frames_dir: str | Path,
    scene: dict[str, Any],
    timestamps: list[dict[str, Any]],
    fallback_idx: int,
    *,
    frame_index: dict[str, Path] | None = None,
) -> list[str]:
    indexed = frame_index if frame_index is not None else _frame_image_index(frames_dir)
    selected: list[Path] = []
    seen: set[Path] = set()
    for timestamp in normalize_keyframe_timestamps(
        get_keyframe_timestamps(scene, timestamps, fallback_idx)
    ):
        match = indexed.get(f"{timestamp:04d}")
        if match is not None and match not in seen:
            selected.append(match)
            seen.add(match)
    return [str(path) for path in selected]


def list_frame_images(frames_dir: str | Path) -> list[Path]:
    path = Path(frames_dir)
    if not path.is_dir():
        return []
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )


def _frame_image_index(frames_dir: str | Path) -> dict[str, Path]:
    return {path.stem: path for path in list_frame_images(frames_dir)}


def get_keyframe_timestamps(
    scene: dict[str, Any],
    timestamps: list[dict[str, Any]],
    fallback_idx: int,
) -> list[Any]:
    timeline = scene.get("timeline")
    if isinstance(timeline, list) and timeline:
        values = [
            shot.get("timestamp")
            for shot in timeline
            if isinstance(shot, dict) and shot.get("timestamp") is not None
        ]
        if len(values) == len(timeline):
            return values
    direct = scene.get("keyframe_timestamps") or scene.get("keyframes")
    if isinstance(direct, list) and direct:
        return direct
    index = scene.get("scene_idx", fallback_idx)
    if type(index) is not int:
        index = fallback_idx
    if timestamps and 0 <= index < len(timestamps):
        values = timestamps[index].get("keyframe_timestamps", [])
        if isinstance(values, list):
            return values
    return []


def normalize_keyframe_timestamps(values: list[Any]) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            timestamp = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        if timestamp >= 0 and timestamp not in seen:
            normalized.append(timestamp)
            seen.add(timestamp)
    return sorted(normalized)


def _scene_boundary(
    scene: dict[str, Any],
    timeline: list[dict[str, Any]],
    fallback_idx: int,
    key: str,
    fallback: int,
) -> int | float:
    value = scene.get(key)
    if value is None and 0 <= fallback_idx < len(timeline):
        value = timeline[fallback_idx].get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return int(number) if number.is_integer() else number


def image_path_for_timestamp(paths: list[Path], timestamp: Any) -> Path | None:
    try:
        name = f"{int(round(float(timestamp))):04d}"
    except (TypeError, ValueError):
        return None
    return next((path for path in paths if path.stem == name), None)
