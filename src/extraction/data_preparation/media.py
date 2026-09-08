"""Source-bound duration checkpoints for resumable visual preparation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from extraction.evidence_reuse import SOURCE_IDENTITY_KEYS, source_matches_inventory
from pipeline_runtime import read_json, write_json
from validation.cohort import probe_duration


def _valid_duration(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _duration_path(assets_root: Path, row: dict[str, Any]) -> Path:
    path = assets_root / row["content_id"] / "assets" / "video_duration.json"
    if not path.resolve().is_relative_to(assets_root.resolve()):
        raise ValueError("duration checkpoint must remain inside source_assets")
    return path


def cached_duration(assets_root: Path, row: dict[str, Any]) -> float | None:
    # Existing runs already carry a probed duration in their source inventory.
    if _valid_duration(row.get("duration_seconds")):
        return row["duration_seconds"]
    try:
        cached = read_json(_duration_path(assets_root, row))
        if (
            isinstance(cached, dict)
            and cached.get("schema_version") == "video-duration/v1"
            and all(key in row and cached.get(key) == row[key] for key in SOURCE_IDENTITY_KEYS)
            and _valid_duration(cached.get("duration_seconds"))
        ):
            return cached["duration_seconds"]
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return None


def save_duration(assets_root: Path, row: dict[str, Any], duration: float) -> None:
    if not _valid_duration(duration):
        raise ValueError("duration must be positive and finite")
    path = _duration_path(assets_root, row)
    # Without a matching source checkpoint, existing PNGs may belong to a replaced
    # video of the same duration. Drop completion markers before saving the probe;
    # a failed extraction must not make those old images reusable on the next run.
    for timestamp in path.parent.glob("timestamp_fixed_*s.json"):
        timestamp.unlink()
    write_json(path, {
        "schema_version": "video-duration/v1",
        **{key: row[key] for key in SOURCE_IDENTITY_KEYS},
        "duration_seconds": duration,
    })


def resolve_duration(assets_root: Path, row: dict[str, Any]) -> float:
    duration = cached_duration(assets_root, row)
    if duration is not None:
        return duration
    if not source_matches_inventory(row):
        raise ValueError("source video changed or is missing before duration probing")
    duration = probe_duration(Path(row["source_video_path"]))
    if not source_matches_inventory(row):
        raise ValueError("source video changed during duration probing")
    # Save before extracting: even a later extraction failure can reuse this probe.
    save_duration(assets_root, row, duration)
    return duration
