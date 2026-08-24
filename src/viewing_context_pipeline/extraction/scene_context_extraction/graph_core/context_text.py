from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


AXES = ("subject_sociality", "media_syntheticity", "setting_context", "utility_orientation")
BUCKETS = ("negative", "neutral", "positive")
LIST_FIELDS = ("top_styles", "top_moods", "top_scene_functions", "top_entities", "top_motifs")


def _ranked(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("graph aggregate list field must be a list")
    normalized = [{"id": str(row["id"]), "count": int(row["count"])} for row in rows]
    return sorted(normalized, key=lambda row: (-row["count"], row["id"]))


def serialize_graph_context(context: dict[str, Any]) -> str:
    axes = context.get("content_axes_4d")
    distributions = context.get("content_axis_distribution")
    if not isinstance(axes, dict) or not isinstance(distributions, dict):
        raise ValueError("graph context is missing axis fields")
    lines = ["Visual graph video context."]
    lines.append("Content axes: " + "; ".join(f"{axis.replace('_', ' ')}={float(axes[axis]):.3f}" for axis in AXES) + ".")
    distribution_parts = []
    for axis in AXES:
        row = distributions.get(axis, {})
        distribution_parts.append(axis.replace("_", " ") + " [" + ", ".join(f"{bucket}={float(row.get(bucket, 0.0)):.3f}" for bucket in BUCKETS) + "]")
    lines.append("Axis distributions: " + "; ".join(distribution_parts) + ".")
    for field in LIST_FIELDS:
        rows = _ranked(context.get(field, []))
        label = field.removeprefix("top_").replace("_", " ").capitalize()
        values = ", ".join(f"{row['id'].split(':', 1)[-1].replace('_', ' ')} ({row['count']})" for row in rows) or "none"
        lines.append(f"{label}: {values}.")
    return "\n".join(lines)


def write_json_atomic(path: str | Path, document: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
