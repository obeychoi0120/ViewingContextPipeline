from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from .video_processor import extract_resized_keyframes, get_video_duration_seconds


SCENE_SECONDS = 30
REFERENCE_SECONDS = 10
KEYFRAME_OFFSETS = (5, 15, 25)


def prepare_visual_item(
    *,
    content_id: str,
    source_video_path: str | Path,
    assets_root: str | Path,
    output_root: str | Path,
    metadata: dict[str, Any],
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
    metadata_path = Path(output_root) / "data" / "cohort" / "metadata" / f"{content_id}.json"
    contract_path = item_assets / "processing_contract_fixed_30s.json"
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")
    contract = {
        "schema_version": "local-video-processing-contract/v4",
        "source_sha256": file_sha256(source),
        "sampling": {
            "scene_seconds": SCENE_SECONDS,
            "reference_seconds": REFERENCE_SECONDS,
            "frames_per_scene": len(KEYFRAME_OFFSETS),
            "keyframe_offsets_seconds": list(KEYFRAME_OFFSETS),
        },
        "image_resolution": [width, height],
    }
    current = _read_json(contract_path)
    complete = (
        current == contract
        and timestamp_path.is_file()
        and resized_keyframes_match_timestamps(timestamp_path, frames_dir, image_size)
    )
    if force or not complete:
        shutil.rmtree(frames_dir, ignore_errors=True)
        duration = max(1, math.ceil(get_video_duration_seconds(source)))
        scenes = build_fixed_30s_windows(duration)
        _write_json(timestamp_path, scenes)
        extract_resized_keyframes(
            source,
            selected_keyframe_timestamps(timestamp_path),
            frames_dir,
            image_size,
        )
        _write_json(contract_path, contract)
    _write_json(
        metadata_path,
        {
            **metadata,
            "content_id": content_id,
            "title": str(metadata.get("title") or ""),
            "tags": str(metadata.get("tags") or ""),
        },
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
        from PIL import Image

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
            with Image.open(output / name) as image:
                if image.size != image_size:
                    return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_content_id(value: object) -> str:
    text = str(value or "").strip()
    if not text or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for char in text):
        raise ValueError(f"content_id is not filesystem-safe: {value!r}")
    return text


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
