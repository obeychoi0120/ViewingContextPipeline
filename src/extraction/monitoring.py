from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def video_names(catalog: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in catalog:
        content_id = str(row["content_id"])
        source = str(row.get("source_video_path") or "").strip()
        names[content_id] = (
            Path(source.replace("\\", "/")).name
            if source
            else f"{content_id}.mp4"
        )
    return names


def scene_messages(
    video_name: str,
    records: list[dict[str, Any]],
    *,
    arm: str,
) -> list[str]:
    if arm not in {"graph", "description"}:
        raise ValueError(f"unsupported scene monitor arm: {arm}")
    messages: list[str] = []
    for record in sorted(records, key=lambda row: int(row["scene_idx"])):
        scene_idx = int(record["scene_idx"])
        if arm == "graph":
            label = "Graph"
            content = json.dumps(
                {"triples": record["triples"]},
                ensure_ascii=False,
                indent=2,
            )
        else:
            label = "Desc"
            content = str(record["description"]).strip()
        messages.append(f"[{label}] {video_name} | scene #{scene_idx:03d}\n{content}")
    return messages


def summary_message(
    video_name: str,
    *,
    arm: str,
    scene_count: int,
    text: str,
) -> str:
    if arm not in {"graph", "description"}:
        raise ValueError(f"unsupported summary monitor arm: {arm}")
    label = "Summary_graph" if arm == "graph" else "Summary_desc"
    return f"[{label}] {video_name} | {scene_count} scenes\n{text.strip()}"
