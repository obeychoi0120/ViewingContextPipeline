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
    source: str | None = None,
) -> list[str]:
    if arm not in {"graph", "description"}:
        raise ValueError(f"unsupported scene monitor arm: {arm}")
    messages: list[str] = []
    for record in sorted(records, key=lambda row: int(row["scene_idx"])):
        scene_idx = int(record["scene_idx"])
        if arm == "graph":
            label = f"Graph_{source}" if source else "Graph"
            content = json.dumps(
                record["graph"],
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
    source: str | None = None,
) -> str:
    if arm not in {"graph", "description"}:
        raise ValueError(f"unsupported summary monitor arm: {arm}")
    label = (
        f"Summary_graph_{source}"
        if arm == "graph" and source
        else "Summary_graph"
        if arm == "graph"
        else "Summary_desc"
    )
    return f"[{label}] {video_name} | {scene_count} scenes\n{text.strip()}"


def graph_skip_message(
    video_name: str,
    record: dict[str, Any],
    *,
    source: str | None = None,
) -> str:
    scene_idx = int(record["scene_idx"])
    error = " ".join(str(record.get("error") or "JSON repair failed").splitlines())
    label = f"Graph_skip_{source}" if source else "Graph_skip"
    return f"[{label}] {video_name} | scene #{scene_idx:03d}\n{error}"
