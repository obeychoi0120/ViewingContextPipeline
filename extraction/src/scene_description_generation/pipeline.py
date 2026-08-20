from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from src.scene_context_extraction.graph_adapter.payload import (
    get_keyframe_timestamps,
    load_scene_timestamps,
    normalize_keyframe_timestamps,
    select_scene_image_paths,
)
from src.scene_context_extraction.graph_core.profile_text import write_json_atomic
from src.scene_context_extraction.ondevice.extractor import image_content_for_model, load_images, move_inputs_to_model


SCENE_PROMPT = """Describe only what is visibly present across these chronological keyframes, trying to not miss any important detail that characterize it. Write one factual English paragraph.

Use only the supplied images. Do not infer from audio, speech, subtitles, OCR, titles, genres, categories, or metadata.

Try to capture, among others, the following aspects in the description:
- All the characters, places, scenes etc names (if known) or description of them. If necessary, explain who they are in a few words so that the reader knows more about them.
- Core Content: Clearly describe what is happening in the video, including the key actions, events, and subjects involved. Mention the setting, objects, and notable visual details.
- Mood and Emotion: Identify the overall mood or emotion that is probably conveyed in the video (e.g., exciting, calming, dramatic, humorous, etc.).
- Context and Audience: Highlight the possible intended audience or purpose of the video (e.g., for children, niche hobbyists, general entertainment, etc.) in one sentence.
- Cultural context: add a few words for cultural context or localization if something might not be clear to the reader.

Create a self-containing text, without subsections. Avoid repetitions."""

SUMMARY_PROMPT = """Using chronological scene descriptions below, write a coherent English visual summary of the entire video. \n\n{descriptions}"""
SCHEMA_VERSION = "description-video-profile/v1"

class DescriptionError(RuntimeError):
    pass


Infer = Callable[[list[Any], str, int], str]


def prompt_fingerprint() -> str:
    return hashlib.sha256((SCENE_PROMPT + "\n" + SUMMARY_PROMPT).encode("utf-8")).hexdigest()


def qwen_infer(model: Any, processor: Any, images: list[Any], prompt: str, max_new_tokens: int) -> str:
    content = image_content_for_model(images, "qwen3_vl") if images else []
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = move_inputs_to_model(inputs, model, "qwen3_vl")
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def build_description_profile(
    *,
    content_id: str,
    scenes: list[dict[str, Any]],
    frames_dir: str | Path,
    timestamp_json_path: str | Path,
    evidence_fingerprint: dict[str, Any],
    infer: Infer,
    scene_max_new_tokens: int = 384,
    summary_max_new_tokens: int = 512,
    model_path: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scene_timestamps = load_scene_timestamps(timestamp_json_path)
    records: list[dict[str, Any]] = []
    for fallback_idx, scene in enumerate(scenes):
        scene_idx = scene.get("scene_idx", fallback_idx)
        normalized_scene = scene
        if not scene.get("timeline") and not scene.get("keyframe_timestamps") and scene.get("keyframes"):
            normalized_scene = {**scene, "keyframe_timestamps": scene["keyframes"]}
        keyframes = normalize_keyframe_timestamps(get_keyframe_timestamps(normalized_scene, scene_timestamps, fallback_idx))
        image_paths = select_scene_image_paths(frames_dir, normalized_scene, scene_timestamps, fallback_idx)
        if not keyframes or len(image_paths) != len(keyframes):
            raise DescriptionError(f"scene {scene_idx} has {len(image_paths)} of {len(keyframes)} keyframes")
        description = infer(load_images(image_paths), SCENE_PROMPT, scene_max_new_tokens).strip()
        if not description:
            raise DescriptionError(f"scene {scene_idx} produced an empty description")
        records.append({"scene_idx": scene_idx, "keyframes": keyframes, "image_paths": image_paths, "description": description})
    if not records:
        raise DescriptionError("video has no scenes")
    chronological = "\n".join(f"Scene {row['scene_idx']}: {row['description']}" for row in records)
    summary = infer([], SUMMARY_PROMPT.format(descriptions=chronological), summary_max_new_tokens).strip()
    words = len(summary.split())
    if not 150 <= words <= 300:
        raise DescriptionError(f"video summary must contain 150-300 words; got {words}")
    document = {
        "schema_version": SCHEMA_VERSION, "content_id": content_id,
        "profile_type": "description", "status": "complete", "text": summary,
        "scene_descriptions": records, "evidence_fingerprint": evidence_fingerprint,
        "model": {"family": "qwen3_vl", "path": model_path, "do_sample": False},
        "prompt_fingerprint": prompt_fingerprint(), "warnings": [],
    }
    return document, records


def write_description_outputs(profile_path: str | Path, scene_path: str | Path, document: dict[str, Any], records: list[dict[str, Any]]) -> None:
    write_json_atomic(profile_path, document)
    rows = [{"schema_version": "scene-description/v1", "content_id": document["content_id"], "evidence_fingerprint": document["evidence_fingerprint"], **record} for record in records]
    target = Path(scene_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
