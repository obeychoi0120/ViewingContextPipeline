from __future__ import annotations

from pathlib import Path
from typing import Any

from extraction.backends import VLMBackend
from extraction.evidence import (
    get_keyframe_timestamps,
    load_images,
    load_scene_timestamps,
    normalize_keyframe_timestamps,
    select_scene_image_paths,
)


SCENE_SCHEMA_VERSION = "scene-description/v1"
SUMMARY_SCHEMA_VERSION = "description-video-summary/v1"


class DescriptionError(RuntimeError):
    pass


def extract_scene_descriptions(
    *,
    content_id: str,
    scenes: list[dict[str, Any]],
    frames_dir: str | Path,
    timestamp_json_path: str | Path,
    backend: VLMBackend,
    prompt: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    timeline = load_scene_timestamps(timestamp_json_path)
    records: list[dict[str, Any]] = []
    for fallback_idx, scene in enumerate(scenes):
        scene_idx = int(scene.get("scene_idx", fallback_idx))
        keyframes = normalize_keyframe_timestamps(
            get_keyframe_timestamps(scene, timeline, fallback_idx)
        )
        image_paths = select_scene_image_paths(
            frames_dir, scene, timeline, fallback_idx
        )
        if not keyframes or len(image_paths) != len(keyframes):
            raise DescriptionError(
                f"scene {scene_idx} has {len(image_paths)} of {len(keyframes)} keyframes"
            )
        description = backend.generate(
            load_images(image_paths), prompt, max_new_tokens
        ).strip()
        if not description:
            raise DescriptionError(f"scene {scene_idx} produced an empty description")
        records.append(
            {
                "schema_version": SCENE_SCHEMA_VERSION,
                "content_id": content_id,
                "scene_idx": scene_idx,
                "keyframes": keyframes,
                "image_paths": image_paths,
                "description": description,
            }
        )
    if not records:
        raise DescriptionError("video has no scenes")
    return records


def description_summary_prompt(template: str, records: list[dict[str, Any]]) -> str:
    if not records:
        raise DescriptionError("description summary requires scene records")
    lines: list[str] = []
    for record in records:
        if (
            record.get("schema_version") != SCENE_SCHEMA_VERSION
            or not str(record.get("description", "")).strip()
        ):
            raise DescriptionError("description summary received an invalid scene record")
        lines.append(f"Scene {record['scene_idx']}: {record['description']}")
    return template.format(scenes="\n".join(lines))


def validate_summary(text: str) -> str:
    summary = str(text or "").strip()
    words = len(summary.split())
    if not 150 <= words <= 300:
        raise DescriptionError(f"video summary must contain 150-300 words; got {words}")
    return summary
