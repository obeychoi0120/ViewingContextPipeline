from __future__ import annotations

from pathlib import Path
from typing import Any


def video_names(catalog: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in catalog:
        content_id = str(row["content_id"])
        source = str(row.get("source_video_path") or "").strip()
        names[content_id] = Path(source.replace("\\", "/")).name if source else f"{content_id}.mp4"
    return names
