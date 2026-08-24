from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from viewing_context_pipeline.extraction.scene_context_extraction.graph_core.fingerprint import build_input_fingerprint, build_visual_evidence_fingerprint
from viewing_context_pipeline.extraction.scene_context_extraction.graph_core.context_text import write_json_atomic
from viewing_context_pipeline.extraction.scene_context_extraction.ondevice.pipeline import init_qwen3vl_model, load_config, load_scene_context_source, manifest_row_to_job, output_save_path

from .pipeline import DescriptionError, build_description_context, prompt_fingerprint, qwen_infer, write_description_outputs


DEFAULT_SETTINGS = "config/extraction/scene_description_generation.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual-only scene descriptions and video summaries.")
    parser.add_argument("--manifest", required=True, help="MicroLens cohort catalog JSONL.")
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
    if settings["shot_interval"] != "fixed_30s":
        raise ValueError("Description track requires fixed_30s")
    multimodal = bool(settings["multimodal"])
    rows = read_selection(args.manifest)
    if args.limit is not None:
        rows = rows[:args.limit]
    model_path = str(settings.get("MODEL_PATH", "")).strip()
    if not model_path:
        raise ValueError("MODEL_PATH is required")
    model, processor = init_qwen3vl_model(model_path, use_fc_patch=True)
    failures = 0
    for index, row in enumerate(rows, start=1):
        job = manifest_row_to_job(row, "fixed_30s", multimodal)
        root = Path(job.scene_context_jsonl).parent.parent
        context_path = root / "video_context_desc_qwen" / f"{job.content_id}_context_desc_qwen.json"
        scene_path = root / "scene_description_qwen" / f"{job.content_id}_scene_descriptions.jsonl"
        failure_path = output_save_path() / "failures" / "viewing_context" / ("multimodal" if multimodal else "img_only") / "fixed_30s" / "video_context_desc_qwen" / f"{job.content_id}_failures.json"
        try:
            scenes = load_scene_context_source(job)
            evidence = (
                build_input_fingerprint(content_id=job.content_id, scenes=scenes, frames_dir=job.frames_dir, multimodal=True, backend="qwen", shot_interval="fixed_30s", model_config=settings)
                if multimodal
                else build_visual_evidence_fingerprint(content_id=job.content_id, scenes=scenes, frames_dir=job.frames_dir, shot_interval="fixed_30s")
            )
            if context_path.is_file() and not args.force:
                existing = json.loads(context_path.read_text(encoding="utf-8"))
                if existing.get("status") == "complete" and existing.get("evidence_fingerprint", {}).get("fingerprint") == evidence["fingerprint"] and existing.get("prompt_fingerprint") == prompt_fingerprint():
                    print(f"[SKIP] ondevice_desc {index}/{len(rows)} {job.content_id}", flush=True)
                    continue
            def infer(images, prompt, limit, references=None):
                return qwen_infer(model, processor, images, prompt, limit, references)

            document, records = build_description_context(content_id=job.content_id, scenes=scenes, frames_dir=job.frames_dir, timestamp_json_path=job.timestamp_json, evidence_fingerprint=evidence, infer=infer, scene_max_new_tokens=int(settings.get("scene_max_new_tokens", 384)), summary_max_new_tokens=int(settings.get("summary_max_new_tokens", 512)), model_path=model_path, multimodal=multimodal)
            write_description_outputs(context_path, scene_path, document, records)
            failure_path.unlink(missing_ok=True)
            print(f"[PROGRESS] ondevice_desc {index}/{len(rows)} {job.content_id}", flush=True)
        except (OSError, ValueError, DescriptionError, RuntimeError) as exc:
            failures += 1
            context_path.unlink(missing_ok=True)
            write_json_atomic(failure_path, {"schema_version": "description-failure/v1", "content_id": job.content_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {job.content_id}: {exc}", file=sys.stderr)
    print(f"Processed {len(rows)} description profiles; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
