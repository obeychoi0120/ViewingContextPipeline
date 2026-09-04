from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from extraction.data_preparation.fixed30 import (
    selected_keyframe_timestamps,
    visual_evidence_matches,
)
from extraction.errors import ExtractionStepError
from pipeline_runtime import read_jsonl


SOURCE_KEYS = (
    "item_id",
    "content_id",
    "source_video_path",
    "source_file_size",
    "source_mtime_ns",
    "duration_seconds",
)


def evidence_paths(run_root: Path, content_id: str) -> tuple[Path, Path]:
    return (
        run_root / "data/cohort/source_assets" / content_id / "assets/timestamp_fixed_30s.json",
        run_root / "data/fixed_30s/resized_keyframes" / content_id,
    )


def source_matches_inventory(row: dict[str, Any]) -> bool:
    try:
        source = Path(row["source_video_path"])
        stat = source.stat()
        return (
            source.is_file()
            and stat.st_size == row["source_file_size"]
            and stat.st_mtime_ns == row["source_mtime_ns"]
        )
    except (OSError, TypeError, KeyError):
        return False


def donor_inventory(run_root: Path) -> dict[str, dict[str, Any]]:
    try:
        rows = read_jsonl(run_root / "data/cohort/item_inventory.jsonl")
        by_item = {str(row["item_id"]): row for row in rows}
        return by_item if len(by_item) == len(rows) else {}
    except (OSError, TypeError, KeyError, ValueError):
        return {}


def copy_matching_evidence(
    *,
    target_root: Path,
    donor_root: Path,
    current: dict[str, Any],
    donor: dict[str, Any] | None,
    image_size: tuple[int, int],
) -> bool:
    if (
        donor is None
        or donor.get("eligible") is not True
        or any(key not in donor or donor[key] != current[key] for key in SOURCE_KEYS)
        or not source_matches_inventory(current)
    ):
        return False
    source_timestamp, source_frames = evidence_paths(donor_root, current["content_id"])
    if not visual_evidence_matches(
        source_timestamp, source_frames, image_size, current["duration_seconds"]
    ):
        return False
    timestamp, frames = evidence_paths(target_root, current["content_id"])
    root = target_root.resolve()
    if any(not path.resolve().is_relative_to(root) for path in (timestamp, frames)):
        raise ExtractionStepError("evidence destination must remain inside the target run")
    if timestamp.is_symlink() or frames.is_symlink():
        raise ExtractionStepError("cannot replace a symlinked evidence destination")
    frames.parent.mkdir(parents=True, exist_ok=True)
    timestamp.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{frames.name}.reuse-", dir=frames.parent) as temp:
        staging = Path(temp)
        staged_frames = staging / "frames"
        staged_timestamp = staging / "timestamp.json"
        backup = staging / "previous_frames"
        installed = False
        try:
            staged_frames.mkdir()
            shutil.copy2(source_timestamp, staged_timestamp)
            for value in selected_keyframe_timestamps(staged_timestamp):
                name = f"{value:04d}.png"
                shutil.copy2(source_frames / name, staged_frames / name)
            if not visual_evidence_matches(
                staged_timestamp, staged_frames, image_size, current["duration_seconds"]
            ) or not source_matches_inventory(current):
                return False
            if frames.exists():
                frames.replace(backup)
            staged_frames.replace(frames)
            installed = True
            # Timestamp is the final completion marker; no donor paths are embedded.
            staged_timestamp.replace(timestamp)
        except BaseException as exc:
            if installed:
                frames.replace(staged_frames)
            if backup.exists():
                backup.replace(frames)
            if isinstance(exc, (OSError, ValueError)):
                return False
            raise
    return True
