from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from .video_processor import extract_resized_keyframes
from extraction.image_validation import verified_image_size


SCENE_SECONDS = 30
REFERENCE_SECONDS = 10
KEYFRAME_OFFSETS = (5, 15, 25)


def prepare_visual_item(
    *,
    content_id: str,
    source_video_path: str | Path,
    assets_root: str | Path,
    output_root: str | Path,
    duration_seconds: object,
    image_size: tuple[int, int],
    force: bool = False,
) -> dict[str, str]:
    source = Path(source_video_path)
    if not source.is_file():
        raise FileNotFoundError(f"local video source not found: {source}")
    content_id = _safe_content_id(content_id)
    item_root = Path(assets_root) / content_id
    item_assets = item_root / "assets"
    timestamp_path = item_assets / "timestamp_fixed_30s.json"
    frames_dir = (
        Path(output_root)
        / "data"
        / "fixed_30s"
        / "resized_keyframes"
        / content_id
    )
    legacy_metadata_path = (
        Path(output_root) / "data" / "cohort" / "metadata" / f"{content_id}.json"
    )
    legacy_metadata_path.unlink(missing_ok=True)
    try:
        legacy_metadata_path.parent.rmdir()
    except OSError:
        pass
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")
    complete = (
        timestamp_path.is_file()
        and resized_keyframes_match_timestamps(timestamp_path, frames_dir, image_size)
    )
    if force or not complete:
        shutil.rmtree(frames_dir, ignore_errors=True)
        duration = _ceil_duration_seconds(duration_seconds)
        scenes = build_fixed_30s_windows(duration)
        _write_json(timestamp_path, scenes)
        extract_resized_keyframes(
            source,
            selected_keyframe_timestamps(timestamp_path),
            frames_dir,
            image_size,
        )
    return {"content_id": content_id}


def build_fixed_30s_windows(video_duration: int) -> list[dict[str, Any]]:
    duration = int(video_duration)
    if duration <= 0:
        raise ValueError("video_duration must be positive")
    windows: list[dict[str, Any]] = []
    for scene_start in range(0, duration, SCENE_SECONDS):
        scene_end = min(scene_start + SCENE_SECONDS, duration)
        boundaries = list(range(scene_start, scene_end, REFERENCE_SECONDS))
        keyframes = [
            (start + min(start + REFERENCE_SECONDS, scene_end)) // 2
            for start in boundaries
        ]
        windows.append(
            {
                "scene_start": scene_start,
                "scene_end": scene_end,
                "duration": scene_end - scene_start,
                "shot_change_timestamps": boundaries,
                "keyframe_timestamps": keyframes,
            }
        )
    return windows


def selected_keyframe_timestamps(timestamp_file: str | Path) -> list[int]:
    scenes = json.loads(Path(timestamp_file).read_text(encoding="utf-8"))
    if not isinstance(scenes, list):
        raise ValueError("fixed-30s timestamp file must contain a list")
    values: list[int] = []
    seen: set[int] = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ValueError("fixed-30s timestamp scene must be an object")
        for raw in scene.get("keyframe_timestamps", []):
            timestamp = int(round(float(raw)))
            if timestamp not in seen:
                values.append(timestamp)
                seen.add(timestamp)
    if not values:
        raise ValueError("fixed-30s timestamp file contains no keyframes")
    return values


def resized_keyframes_match_timestamps(
    timestamp_file: str | Path,
    output_dir: str | Path,
    image_size: tuple[int, int],
) -> bool:
    try:
        expected = {
            f"{timestamp:04d}.png"
            for timestamp in selected_keyframe_timestamps(timestamp_file)
        }
        output = Path(output_dir)
        actual = {
            path.name
            for path in output.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        }
        if actual != expected:
            return False
        for name in expected:
            if verified_image_size(output / name) != image_size:
                return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _safe_content_id(value: object) -> str:
    text = str(value or "").strip()
    if not text or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for char in text):
        raise ValueError(f"content_id is not filesystem-safe: {value!r}")
    return text


def _ceil_duration_seconds(value: object) -> int:
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"duration_seconds must be a positive finite number: {value!r}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"duration_seconds must be a positive finite number: {value!r}")
    return max(1, math.ceil(duration))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
