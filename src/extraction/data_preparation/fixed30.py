from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .video_processor import extract_resized_keyframes
from extraction.image_validation import verified_image_size
from visual_sampling import build_fixed_windows, timestamp_stem, truncate_timestamp


def prepare_visual_item(
    *,
    content_id: str,
    source_video_path: str | Path,
    assets_root: str | Path,
    output_root: str | Path,
    duration_seconds: object,
    image_size: tuple[int, int],
    scene_duration: int = 30,
    num_keyframes: int = 6,
    force: bool = False,
) -> dict[str, str]:
    source = Path(source_video_path)
    if not source.is_file():
        raise FileNotFoundError(f"local video source not found: {source}")
    content_id = _safe_content_id(content_id)
    item_root = Path(assets_root) / content_id
    item_assets = item_root / "assets"
    timestamp_path = item_assets / f"timestamp_fixed_{scene_duration}s.json"
    frames_dir = Path(output_root) / "data" / "resized_keyframes" / content_id
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
    complete = visual_evidence_matches(
        timestamp_path, frames_dir, image_size, duration_seconds,
        scene_duration=scene_duration, num_keyframes=num_keyframes,
    )
    if force or not complete:
        scenes = build_fixed_windows(
            duration_seconds, scene_duration=scene_duration, num_keyframes=num_keyframes,
        )
        extract_resized_keyframes(
            source,
            [timestamp for scene in scenes for timestamp in scene["keyframe_timestamps"]],
            frames_dir,
            image_size,
        )
        # Commit the timestamp only after the complete frame directory is installed.
        _write_json(timestamp_path, scenes)
    return {"content_id": content_id}


def build_fixed_30s_windows(video_duration: int) -> list[dict[str, Any]]:
    """Legacy three-keyframe entrypoint, using the shared timestamp precision."""
    return build_fixed_windows(video_duration, scene_duration=30, num_keyframes=3)


def selected_keyframe_timestamps(timestamp_file: str | Path) -> list[int | float]:
    scenes = json.loads(Path(timestamp_file).read_text(encoding="utf-8"))
    if not isinstance(scenes, list):
        raise ValueError("fixed-window timestamp file must contain a list")
    values: list[int | float] = []
    seen: set[int | float] = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ValueError("fixed-window timestamp scene must be an object")
        for raw in scene.get("keyframe_timestamps", []):
            timestamp = truncate_timestamp(raw)
            if timestamp not in seen:
                values.append(timestamp)
                seen.add(timestamp)
    if not values:
        raise ValueError("fixed-window timestamp file contains no keyframes")
    return values


def resized_keyframes_match_timestamps(
    timestamp_file: str | Path,
    output_dir: str | Path,
    image_size: tuple[int, int],
) -> bool:
    try:
        expected = {
            f"{timestamp_stem(timestamp)}.png"
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


def visual_evidence_matches(
    timestamp_file: Path,
    output_dir: Path,
    image_size: tuple[int, int],
    duration_seconds: object,
    *,
    scene_duration: int = 30,
    num_keyframes: int = 6,
) -> bool:
    try:
        expected = build_fixed_windows(
            duration_seconds, scene_duration=scene_duration, num_keyframes=num_keyframes,
        )
        observed = json.loads(timestamp_file.read_text(encoding="utf-8"))
        return observed == expected and resized_keyframes_match_timestamps(
            timestamp_file, output_dir, image_size
        )
    except (OSError, TypeError, ValueError):
        return False


def _safe_content_id(value: object) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for char in text):
        raise ValueError(f"content_id is not filesystem-safe: {value!r}")
    return text


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
