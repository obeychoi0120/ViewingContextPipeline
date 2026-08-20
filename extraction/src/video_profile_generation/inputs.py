from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.common.manifest import ManifestContractError, read_manifest_rows
from src.video_data_collection.raw_pipeline import (
    normalize_shot_interval,
    ref_jsonl_relative_path,
)


REF_JSONL_SUFFIX = "_ref.jsonl"
MAX_KEYFRAMES_PER_VIDEO = 1_440


class InputError(ValueError):
    """Raised when a video's inputs violate the expected contract."""


class TimelineReference(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    shot_idx: int = Field(ge=0)
    timestamp: int = Field(ge=0)
    raw_asr: str
    raw_ocr: str


class RefScene(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    scene_idx: int = Field(ge=0)
    timeline: list[TimelineReference] = Field(min_length=1)

    @field_validator("timeline")
    @classmethod
    def validate_timeline(cls, values: list[TimelineReference]) -> list[TimelineReference]:
        timestamps = [item.timestamp for item in values]
        if any(current >= following for current, following in zip(timestamps, timestamps[1:])):
            raise ValueError("timeline timestamps must be strictly increasing timestamps in seconds")
        shot_ids = [item.shot_idx for item in values]
        if len(shot_ids) != len(set(shot_ids)):
            raise ValueError("timeline shot_idx values must not contain duplicates")
        return values


@dataclass(frozen=True)
class ContentBundle:
    content_id: str
    metadata: dict[str, Any]
    scenes: list[RefScene]
    keyframe_dir: Path

    @property
    def frame_count(self) -> int:
        return sum(len(scene.timeline) for scene in self.scenes)


def load_manifest_content_ids(manifest_path: str | Path) -> list[str]:
    try:
        rows = read_manifest_rows(manifest_path)
    except ManifestContractError as exc:
        raise InputError(str(exc)) from exc

    content_ids: list[str] = []
    for line_number, row in enumerate(rows, start=2):
        content_id = row["content_id"]
        try:
            _validate_content_id(content_id)
        except InputError as exc:
            raise InputError(f"manifest line {line_number}: {exc}") from exc
        content_ids.append(content_id)
    return content_ids


def load_content_bundle(
    local_input_dir: str | Path,
    content_id: str,
    shot_interval: str = "fixed_15s",
) -> ContentBundle:
    mode = normalize_shot_interval(shot_interval)
    _validate_content_id(content_id)
    input_root = Path(local_input_dir)
    metadata = _load_metadata(input_root, content_id)
    scenes = _load_scenes(input_root, content_id, mode)

    frame_count = sum(len(scene.timeline) for scene in scenes)
    if frame_count > MAX_KEYFRAMES_PER_VIDEO:
        raise InputError(
            f"{content_id}: {frame_count} keyframes exceed the per-request limit "
            f"of {MAX_KEYFRAMES_PER_VIDEO}"
        )

    keyframe_dir = (
        input_root / "asset" / mode / "resized_keyframes" / content_id
    )
    expected_paths = {
        keyframe_dir / f"{timestamp_seconds:04d}.png"
        for scene in scenes
        for timestamp_seconds in (shot.timestamp for shot in scene.timeline)
    }
    missing = sorted(str(path) for path in expected_paths if not path.is_file())
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise InputError(f"{content_id}: missing keyframes: {preview}{suffix}")

    return ContentBundle(
        content_id=content_id,
        metadata=metadata,
        scenes=scenes,
        keyframe_dir=keyframe_dir,
    )


def _load_metadata(input_root: Path, content_id: str) -> dict[str, Any]:
    path = input_root / "asset" / "metadata" / f"{content_id}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{content_id}: invalid metadata JSON at {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise InputError(
            f"{content_id}: failed to read metadata at {path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise InputError(f"{content_id}: metadata must be a JSON object")
    for field in ("title", "channel", "upload_date", "description"):
        if field not in raw or not isinstance(raw[field], str):
            raise InputError(f"{content_id}: metadata.{field} must be a string")
    for field in ("channel", "upload_date"):
        if not raw[field].strip():
            raise InputError(f"{content_id}: metadata.{field} must not be empty")
    return raw


def _load_scenes(
    input_root: Path,
    content_id: str,
    shot_interval: str,
) -> list[RefScene]:
    path = (
        input_root
        / ref_jsonl_relative_path(shot_interval)
        / f"{content_id}{REF_JSONL_SUFFIX}"
    )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(
            f"{content_id}: failed to read Ref JSONL at {path}: {exc}"
        ) from exc

    ref_scenes: list[RefScene] = []
    metadata_header_seen = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record must be a JSON object")
            if record.get("_type") == "video_metadata":
                if metadata_header_seen or ref_scenes:
                    raise ValueError("video_metadata header must appear at most once before all scenes")
                metadata_header_seen = True
                continue
            ref_scenes.append(RefScene.model_validate(record))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise InputError(f"{content_id}: invalid Ref JSONL line {line_number}: {exc}") from exc

    if not ref_scenes:
        raise InputError(f"{content_id}: Ref JSONL contains no scenes")
    scene_ids = [scene.scene_idx for scene in ref_scenes]
    if len(scene_ids) != len(set(scene_ids)):
        raise InputError(f"{content_id}: duplicate scene_idx values")
    ordered_ref_scenes = sorted(ref_scenes, key=lambda scene: scene.scene_idx)

    timestamps = [
        shot.timestamp
        for scene in ordered_ref_scenes
        for shot in scene.timeline
    ]
    if any(current >= following for current, following in zip(timestamps, timestamps[1:])):
        raise InputError(f"{content_id}: keyframe timestamps must be strictly increasing across scenes")
    return ordered_ref_scenes


def _validate_content_id(content_id: str) -> None:
    if not content_id or content_id.strip() != content_id or "/" in content_id or "\\" in content_id:
        raise InputError(f"invalid contents_id: {content_id!r}")
