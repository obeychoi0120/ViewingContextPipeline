from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.scene_context_extraction.graph_core.fingerprint import build_input_fingerprint, build_visual_evidence_fingerprint, fingerprint_matches
from src.scene_context_extraction.graph_core.profile_text import build_graph_profile_document, write_json_atomic
from src.scene_context_extraction.graph_core.scene_failures import read_scene_failures
from src.scene_context_extraction.ondevice.pipeline import load_config, load_scene_context_source, manifest_row_to_job, model_family_from_config, ondevice_context_path, ondevice_failure_path, output_save_path

from .cli import read_selection


DEFAULT_GRAPH_SETTINGS = "config/scene_context_extraction_ondevice.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serialize deterministic Graph video profiles.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--settings", default=DEFAULT_GRAPH_SETTINGS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_config(args.settings)
    family = model_family_from_config(settings)
    if family != "qwen3_vl" or settings["shot_interval"] != "fixed_30s" or settings["multimodal"]:
        raise ValueError("Graph profile track requires qwen3_vl, fixed_30s, and multimodal=false")
    rows = read_selection(args.manifest)
    if args.limit is not None:
        rows = rows[:args.limit]
    failures = 0
    for row in rows:
        job = manifest_row_to_job(row, family, "fixed_30s", False)
        source_path = ondevice_context_path(job)
        output_path = Path(job.scene_context_jsonl).parent.parent / "video_profile_graph_qwen" / f"{job.content_id}_vp_graph.json"
        failure_path = output_save_path() / "failures" / "viewing_context" / "img_only" / "fixed_30s" / "video_profile_graph_qwen" / f"{job.content_id}_failures.json"
        try:
            if not source_path.is_file():
                raise FileNotFoundError(f"missing GraphVideoContext: {source_path}")
            scene_failures = read_scene_failures(ondevice_failure_path(job))
            if scene_failures:
                raise ValueError(f"Graph profile has {len(scene_failures)} failed scenes")
            scenes = load_scene_context_source(job)
            source_fingerprint = build_input_fingerprint(content_id=job.content_id, scenes=scenes, frames_dir=job.frames_dir, multimodal=False, backend="qwen", shot_interval="fixed_30s", model_config=settings)
            if not fingerprint_matches(job.scene_context_jsonl, source_fingerprint):
                raise ValueError("Graph Scene Context fingerprint does not match current visual inputs and settings")
            evidence = build_visual_evidence_fingerprint(content_id=job.content_id, scenes=scenes, frames_dir=job.frames_dir, shot_interval="fixed_30s")
            if output_path.is_file() and not args.force:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                if existing.get("status") == "complete" and existing.get("evidence_fingerprint", {}).get("fingerprint") == evidence["fingerprint"]:
                    continue
            context = json.loads(source_path.read_text(encoding="utf-8"))
            write_json_atomic(output_path, build_graph_profile_document(context, evidence, source_path=source_path, complete=True))
            failure_path.unlink(missing_ok=True)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures += 1
            output_path.unlink(missing_ok=True)
            write_json_atomic(failure_path, {"schema_version": "graph-profile-failure/v1", "content_id": job.content_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {job.content_id}: {exc}")
    print(f"Processed {len(rows)} graph profiles; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
