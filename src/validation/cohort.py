from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

from .cohort_selection import CohortError, content_id_for_item, normalize_item_id



def load_pairs(path: Path) -> list[tuple[str, list[str]]]:
    users: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2 or not parts[0].strip():
                raise CohortError(f"invalid pairs row {line_number}")
            user_id = parts[0].strip()
            if user_id in seen:
                raise CohortError(f"duplicate user {user_id}")
            seen.add(user_id)
            users.append((user_id, [normalize_item_id(item) for item in parts[1].split()]))
    return users


def load_metadata_titles(path: Path, *, keep_blank: bool = False) -> dict[str, str]:
    """Blank titles outside the required catalog do not block preparation."""
    titles: dict[str, str] = {}
    seen: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise CohortError(f"failed to read metadata titles {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            if "," not in raw:
                raise CohortError(f"invalid metadata title row {line_number}: missing comma")
            raw_item_id, raw_title = raw.split(",", 1)
            try:
                item_id = normalize_item_id(raw_item_id)
            except CohortError as exc:
                raise CohortError(f"invalid metadata title row {line_number}: {exc}") from exc
            if item_id in seen:
                raise CohortError(
                    f"duplicate metadata title for item {item_id} at row {line_number}"
                )
            seen.add(item_id)
            title = raw_title.strip()
            if title or keep_blank:
                titles[item_id] = title
    return titles


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def probe_duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    duration = float(json.loads(completed.stdout)["format"]["duration"])
    if not _positive_finite(duration):
        raise ValueError("duration must be positive and finite")
    return duration


def build_item_inventory(
    referenced_items: set[str],
    videos_dir: Path,
    probe: Callable[[Path], float] | None = probe_duration,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    videos: dict[str, list[Path]] = {}
    for path in sorted(videos_dir.glob("*.mp4")):
        try:
            item_id = normalize_item_id(path.stem)
        except CohortError:
            continue
        if item_id in referenced_items:
            videos.setdefault(item_id, []).append(path)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item_id in sorted(referenced_items, key=int):
        matches = videos.get(item_id, [])
        path = matches[0] if matches else None
        reasons: list[str] = []
        duration = size = mtime = None
        if path is None:
            reasons.append("missing_video")
            failures.append({"item_id": item_id, "reason": "missing_video"})
        else:
            try:
                if len(matches) != 1:
                    raise ValueError(f"multiple video files normalize to item {item_id}")
                stat = path.stat()
                size, mtime = stat.st_size, stat.st_mtime_ns
                if not path.is_file() or size <= 0:
                    raise ValueError("video must be a non-empty file")
                if probe is not None:
                    duration = probe(path)
                    if not _positive_finite(duration):
                        raise ValueError("duration must be positive and finite")
            except Exception as exc:
                duration = None
                reasons.append("invalid_video")
                failures.append({"item_id": item_id, "reason": "invalid_video", "error": str(exc)})
        rows.append(
            {
                "item_id": item_id,
                "content_id": content_id_for_item(item_id),
                "source_video_path": str(path.resolve()) if path else None,
                "duration_seconds": duration,
                "source_file_size": size,
                "source_mtime_ns": mtime,
                "eligible": not reasons,
                "exclusion_reasons": reasons,
            }
        )
    return rows, failures
