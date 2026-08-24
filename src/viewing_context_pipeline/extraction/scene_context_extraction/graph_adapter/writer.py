from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_graph_record(
    item: dict[str, Any],
    idx: int,
    keyframes: list[int],
    observation: dict[str, Any],
) -> dict[str, Any]:
    if not observation:
        raise ValueError("successful graph records require a non-empty observation")
    return {
        "scene_idx": item.get("scene_idx", item.get("scene_id", idx)),
        "keyframes": keyframes,
        "vlm_visual_graph": observation,
    }


def open_jsonl_writer(output_path: str | Path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def write_graph_record(output_file, record: dict[str, Any]) -> None:
    output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    output_file.flush()
