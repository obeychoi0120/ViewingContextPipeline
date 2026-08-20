from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.scene_context_extraction.graph_core.fingerprint import build_visual_evidence_fingerprint
from src.scene_context_extraction.graph_core.profile_text import write_json_atomic
from src.scene_context_extraction.ondevice.pipeline import init_vlm_model, load_config, load_scene_context_source, manifest_row_to_job, model_family_from_config, output_save_path

from .pipeline import DescriptionError, build_description_profile, prompt_fingerprint, qwen_infer, write_description_outputs


DEFAULT_SETTINGS = "config/scene_description_generation.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual-only scene descriptions and video summaries.")
    parser.add_argument("--manifest", required=True, help="Selection JSONL from ViewingContextValidation.")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def read_selection(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_config(args.settings)
    family = model_family_from_config(settings)
    if family != "qwen3_vl" or settings["shot_interval"] != "fixed_30s" or settings["multimodal"]:
        raise ValueError("Description track requires qwen3_vl, fixed_30s, and multimodal=false")
    rows = read_selection(args.manifest)
    if args.limit is not None:
        rows = rows[:args.limit]
    model_path = str(settings.get("MODEL_PATH", "")).strip()
    if not model_path:
        raise ValueError("MODEL_PATH is required")
    model, processor = init_vlm_model(model_path, family)
    failures = 0
    for row in rows:
        job = manifest_row_to_job(row, family, "fixed_30s", False)
        root = Path(job.scene_context_jsonl).parent.parent
        profile_path = root / "video_profile_desc_qwen" / f"{job.content_id}_vp_desc.json"
        scene_path = root / "scene_description_qwen" / f"{job.content_id}_scene_descriptions.jsonl"
        failure_path = output_save_path() / "failures" / "viewing_context" / "img_only" / "fixed_30s" / "video_profile_desc_qwen" / f"{job.content_id}_failures.json"
        try:
            scenes = load_scene_context_source(job)
            evidence = build_visual_evidence_fingerprint(content_id=job.content_id, scenes=scenes, frames_dir=job.frames_dir, shot_interval="fixed_30s")
            if profile_path.is_file() and not args.force:
                existing = json.loads(profile_path.read_text(encoding="utf-8"))
                if existing.get("status") == "complete" and existing.get("evidence_fingerprint", {}).get("fingerprint") == evidence["fingerprint"] and existing.get("prompt_fingerprint") == prompt_fingerprint():
                    continue
            infer = lambda images, prompt, limit: qwen_infer(model, processor, images, prompt, limit)
            document, records = build_description_profile(content_id=job.content_id, scenes=scenes, frames_dir=job.frames_dir, timestamp_json_path=job.timestamp_json, evidence_fingerprint=evidence, infer=infer, scene_max_new_tokens=int(settings.get("scene_max_new_tokens", 384)), summary_max_new_tokens=int(settings.get("summary_max_new_tokens", 512)), model_path=model_path)
            write_description_outputs(profile_path, scene_path, document, records)
            failure_path.unlink(missing_ok=True)
        except (OSError, ValueError, DescriptionError, RuntimeError) as exc:
            failures += 1
            profile_path.unlink(missing_ok=True)
            write_json_atomic(failure_path, {"schema_version": "description-failure/v1", "content_id": job.content_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {job.content_id}: {exc}", file=sys.stderr)
    print(f"Processed {len(rows)} description profiles; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
