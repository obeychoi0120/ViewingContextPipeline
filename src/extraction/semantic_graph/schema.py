from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from extraction.backends import VLMBackend
from extraction.evidence import build_scene_evidence, load_images
from extraction.json_repair import extract_json
from extraction.semantic_graph.taxonomy import SCENE_SCHEMA_VERSION


GRAPH_SUMMARY_SCHEMA = "graph-video-summary/v1"


class SemanticGraphError(RuntimeError):
    pass


def parse_graph_output(text: str) -> dict[str, Any]:
    """Parse a JSON object without enforcing semantic graph fields or references."""
    payload = extract_json(text)
    if payload is None:
        raise SemanticGraphError("VLM output does not contain a JSON object")
    return payload


def extract_scene_graphs(
    *,
    content_id: str,
    scenes: list[dict[str, Any]],
    frames_dir: str | Path,
    timestamp_json_path: str | Path,
    backend: VLMBackend,
    prompt: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene in build_scene_evidence(scenes, frames_dir, timestamp_json_path):
        image_paths = scene["image_paths"]
        if not scene["keyframes"] or len(image_paths) != len(scene["keyframes"]):
            raise SemanticGraphError(
                f"scene {scene['scene_idx']} has {len(image_paths)} of "
                f"{len(scene['keyframes'])} keyframes"
            )
        graph = parse_graph_output(
            backend.generate(load_images(image_paths), prompt, max_new_tokens)
        )
        records.append(
            {
                "schema_version": SCENE_SCHEMA_VERSION,
                "content_id": content_id,
                "scene_idx": scene["scene_idx"],
                "scene_start_seconds": scene["scene_start_seconds"],
                "scene_end_seconds": scene["scene_end_seconds"],
                "keyframes": scene["keyframes"],
                "image_paths": image_paths,
                "graph": graph,
            }
        )
    if not records:
        raise SemanticGraphError("video has no scenes")
    return records


def graph_summary_prompt(template: str, records: list[dict[str, Any]]) -> str:
    if not records:
        raise SemanticGraphError("graph summary requires scene records")
    ordered = sorted(records, key=lambda row: int(row["scene_idx"]))
    scenes = [
        "\n".join(
            [
                f"Scene {int(record['scene_idx'])} "
                f"({record['scene_start_seconds']}-{record['scene_end_seconds']}s):",
                json.dumps(record["graph"], ensure_ascii=False, sort_keys=True),
            ]
        )
        for record in ordered
    ]
    return template.format(scenes="\n\n".join(scenes))


def validate_summary(text: str) -> str:
    summary = str(text or "").strip()
    words = len(summary.split())
    if not 1 <= words <= 150:
        raise SemanticGraphError(
            f"video summary must contain 1-150 words; got {words}"
        )
    return summary
