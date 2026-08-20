from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .multimodal import shot_references
from .ontology import ONTOLOGY_CONTRACT_PATH
from .prompt import SCENE_EXTRACTION_PROMPT, USER_MESSAGE


FINGERPRINT_SCHEMA = "scene-context-input-fingerprint/v1"
MANIFEST_NAME = "input_fingerprint_manifest.json"
VISUAL_EVIDENCE_SCHEMA = "visual-evidence-fingerprint/v1"


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def build_input_fingerprint(
    *,
    content_id: str,
    scenes: Iterable[dict[str, Any]],
    frames_dir: str | Path,
    multimodal: bool,
    backend: str,
    model_config: dict[str, Any] | None = None,
    shot_interval: str | None = None,
) -> dict[str, Any]:
    ordered_scenes = []
    for fallback_idx, scene in enumerate(scenes):
        item: dict[str, Any] = {
            "scene_idx": scene.get("scene_idx", fallback_idx),
            "timestamps": [
                entry.get("timestamp")
                for entry in scene.get("timeline", [])
                if isinstance(entry, dict)
            ],
            "keyframes": list(scene.get("keyframes") or scene.get("keyframe_timestamps") or []),
        }
        if multimodal:
            item["shot_references"] = shot_references(scene)
        ordered_scenes.append(item)

    frame_root = Path(frames_dir)
    frame_hashes = (
        [
            {"name": path.name, "sha256": _sha256_file(path)}
            for path in sorted(frame_root.iterdir(), key=lambda value: value.name)
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if frame_root.is_dir()
        else [{"keyframe_contract": scene["keyframes"]} for scene in ordered_scenes]
    )
    prompt_contract = {
        "system": SCENE_EXTRACTION_PROMPT,
        "user": USER_MESSAGE,
        "multimodal": multimodal,
    }
    components = {
        "backend": backend,
        "model_config": model_config or {},
        "prompt_sha256": _canonical_hash(prompt_contract),
        "ontology_sha256": _sha256_file(Path(ONTOLOGY_CONTRACT_PATH)),
        "ordered_scenes_sha256": _canonical_hash(ordered_scenes),
        "keyframes_sha256": _canonical_hash(frame_hashes),
    }
    if shot_interval == "fixed_30s":
        sampling_contract = {
            "scene_seconds": 30,
            "reference_seconds": 10,
            "keyframe_offsets_seconds": [5, 15, 25],
        }
        if multimodal:
            sampling_contract.update(
                {
                    "ocr_sampling_fps": 1,
                    "ocr_interval_max_gap_seconds": 2.0,
                    "ocr_interval_similarity_threshold": 0.75,
                    "ocr_dedup_similarity_threshold": 0.8,
                    "ocr_max_chars": 1000,
                }
            )
        components["sampling_contract"] = sampling_contract
    return {
        "schema_version": FINGERPRINT_SCHEMA,
        "content_id": content_id,
        "modality": "multimodal" if multimodal else "img_only",
        "fingerprint": _canonical_hash(components),
        "components": components,
    }


def build_visual_evidence_fingerprint(
    *,
    content_id: str,
    scenes: Iterable[dict[str, Any]],
    frames_dir: str | Path,
    shot_interval: str,
) -> dict[str, Any]:
    """Fingerprint only shared visual evidence, never model or text modalities."""
    ordered_scenes: list[dict[str, Any]] = []
    for fallback_idx, scene in enumerate(scenes):
        timestamps = list(scene.get("keyframes") or scene.get("keyframe_timestamps") or [])
        timeline = scene.get("timeline")
        if isinstance(timeline, list) and timeline:
            timestamps = [row.get("timestamp") for row in timeline if isinstance(row, dict)]
        ordered_scenes.append({
            "scene_idx": scene.get("scene_idx", fallback_idx),
            "start_seconds": scene.get("start_seconds", scene.get("start_time")),
            "end_seconds": scene.get("end_seconds", scene.get("end_time")),
            "keyframe_timestamps": timestamps,
        })
    frame_rows: list[dict[str, Any]] = []
    root = Path(frames_dir)
    if root.is_dir():
        for path in sorted(root.iterdir(), key=lambda value: value.name):
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            with Image.open(path) as image:
                width, height = image.size
            frame_rows.append({"name": path.name, "sha256": _sha256_file(path), "width": width, "height": height})
    components = {
        "shot_interval": shot_interval,
        "sampling_contract": {"scene_seconds": 30, "reference_seconds": 10, "keyframe_offsets_seconds": [5, 15, 25]} if shot_interval == "fixed_30s" else {"mode": shot_interval},
        "ordered_scenes": ordered_scenes,
        "frames": frame_rows,
    }
    return {
        "schema_version": VISUAL_EVIDENCE_SCHEMA,
        "content_id": content_id,
        "fingerprint": _canonical_hash(components),
        "components": components,
    }


def manifest_path(scene_context_path: str | Path) -> Path:
    return Path(scene_context_path).parent / MANIFEST_NAME


def fingerprint_matches(scene_context_path: str | Path, fingerprint: dict[str, Any]) -> bool:
    path = manifest_path(scene_context_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    record = manifest.get("contents", {}).get(fingerprint["content_id"])
    return isinstance(record, dict) and record.get("fingerprint") == fingerprint.get("fingerprint")


def write_fingerprint(scene_context_path: str | Path, fingerprint: dict[str, Any]) -> None:
    path = manifest_path(scene_context_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {"schema_version": FINGERPRINT_SCHEMA, "contents": {}}
    contents = manifest.setdefault("contents", {})
    contents[fingerprint["content_id"]] = fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
