from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from viewing_context_pipeline.extraction.common.gemini import create_client
from viewing_context_pipeline.extraction.common.local_images import local_image_part
from viewing_context_pipeline.extraction.scene_context_extraction.graph_adapter.payload import get_keyframe_timestamps, load_scene_timestamps, normalize_keyframe_timestamps, select_scene_image_paths
from viewing_context_pipeline.extraction.scene_context_extraction.graph_core.fingerprint import build_input_fingerprint, build_visual_evidence_fingerprint
from viewing_context_pipeline.extraction.scene_context_extraction.graph_core.multimodal import shot_reference_text, shot_references, validate_image_reference_alignment
from viewing_context_pipeline.extraction.scene_context_extraction.graph_core.context_text import write_json_atomic
from viewing_context_pipeline.extraction.scene_context_extraction.ondevice.pipeline import load_scene_context_source, read_manifest

from .pipeline import SCENE_PROMPT, SUMMARY_PROMPT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate descriptive video context with Gemini.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    output_root = Path(os.environ["OUTPUT_SAVE_PATH"])
    project = os.getenv("GCP_PROJECT_ID", "").strip()
    if not project:
        raise ValueError("GCP_PROJECT_ID is required")
    multimodal = bool(settings["multimodal"])
    model_name = str(settings["gemini_model"])
    client = create_client(project, str(settings.get("gemini_location", "global")), bounded_retry=True)
    jobs = read_manifest(args.manifest, shot_interval="fixed_30s", multimodal=multimodal)
    mode = "multimodal" if multimodal else "img_only"
    output_dir = output_root / "viewing_context" / mode / "fixed_30s" / "video_context_desc_gemini"
    failures = 0
    for index, job in enumerate(jobs, start=1):
        destination = output_dir / f"{job.content_id}_context_desc_gemini.json"
        try:
            scenes = load_scene_context_source(job)
            evidence = (
                build_input_fingerprint(content_id=job.content_id, scenes=scenes, frames_dir=job.frames_dir, multimodal=True, backend="gemini", shot_interval="fixed_30s", model_config=settings)
                if multimodal
                else build_visual_evidence_fingerprint(content_id=job.content_id, scenes=scenes, frames_dir=job.frames_dir, shot_interval="fixed_30s")
            )
            if destination.is_file() and not args.force:
                current = json.loads(destination.read_text(encoding="utf-8"))
                if current.get("status") == "complete" and current.get("evidence_fingerprint", {}).get("fingerprint") == evidence["fingerprint"]:
                    print(f"[SKIP] {job.content_id}", flush=True)
                    continue
            timestamps = load_scene_timestamps(job.timestamp_json)
            descriptions: list[str] = []
            for fallback, scene in enumerate(scenes):
                keyframes = normalize_keyframe_timestamps(get_keyframe_timestamps(scene, timestamps, fallback))
                image_paths = select_scene_image_paths(job.frames_dir, scene, timestamps, fallback)
                if len(image_paths) != len(keyframes) or not image_paths:
                    raise ValueError(f"scene {fallback} image/keyframe mismatch")
                parts = []
                references = shot_references(scene) if multimodal else []
                if multimodal:
                    validate_image_reference_alignment(len(image_paths), references)
                for image_index, image_path in enumerate(image_paths):
                    parts.append(local_image_part(image_path))
                    if multimodal:
                        from google.genai import types
                        parts.append(types.Part.from_text(text=shot_reference_text(references[image_index])))
                parts.append(SCENE_PROMPT)
                response = client.models.generate_content(model=model_name, contents=parts)
                text = str(response.text or "").strip()
                if not text:
                    raise ValueError(f"scene {fallback} produced empty description")
                descriptions.append(text)
            summary_prompt = SUMMARY_PROMPT.format(descriptions="\n".join(f"Scene {i}: {text}" for i, text in enumerate(descriptions)))
            response = client.models.generate_content(model=model_name, contents=[summary_prompt])
            summary = str(response.text or "").strip()
            if not summary:
                raise ValueError("empty Gemini video summary")
            write_json_atomic(destination, {
                "schema_version": "description-video-context/v1", "content_id": job.content_id,
                "context_type": "description", "status": "complete", "text": summary,
                "evidence_fingerprint": evidence, "model": {"family": "gemini", "name": model_name},
            })
            print(f"[PROGRESS] gemini_desc {index}/{len(jobs)} {job.content_id}", flush=True)
        except Exception as exc:
            failures += 1
            destination.unlink(missing_ok=True)
            print(f"[FAILURE] {job.content_id}: {exc}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
