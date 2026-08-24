from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from viewing_context_pipeline.extraction.common.manifest import read_manifest_rows
from viewing_context_pipeline.extraction.data_preparation.raw_pipeline import (
    multimodal_from_config,
    multimodal_ref_relative_path,
    shot_interval_from_config,
)
from .extractor import (
    SceneContextGeminiConfig,
    build_extraction_config,
    extract_scene_contexts_gemini,
    load_scene_context_rows,
    scene_context_gemini_output_dir,
    video_context_graph_gemini_output_dir,
)
from viewing_context_pipeline.extraction.common.gemini import create_client


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG = SceneContextGeminiConfig(gcp_project_id="")
DEFAULT_SETTINGS_PATH = (
    PROJECT_ROOT / "config" / "extraction" / "gemini.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract scene-level visual graph context with Gemini.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="MicroLens cohort extraction manifest CSV.",
    )
    parser.add_argument("--training-output", default=None)
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--gcp-project-id", default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--thinking-level", default=None)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract every scene even when complete Gemini outputs already exist.",
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_settings_config(args)
    if not config.gcp_project_id:
        raise RuntimeError("GCP project id is required; configure local.yaml or pass --gcp-project-id.")

    scene_context_output_dir = scene_context_gemini_output_dir(
        config.output_dir,
        config.shot_interval,
        config.multimodal,
    )
    video_context_output_dir = video_context_graph_gemini_output_dir(
        config.output_dir,
        config.shot_interval,
        config.multimodal,
    )
    print_startup_config(
        args=args,
        config=config,
        scene_context_output_dir=scene_context_output_dir,
        video_context_output_dir=video_context_output_dir,
    )

    print("[step] Loading canonical visual or multimodal evidence...", flush=True)
    scene_context_rows, multimodal_ref_file_count = load_context_rows(
        config,
        args.manifest,
    )
    print_input_summary(
        config=config,
        multimodal_ref_file_count=multimodal_ref_file_count,
        content_count=len({row["content_id"] for row in scene_context_rows}),
        scene_count=len(scene_context_rows),
    )

    print("[step] Creating Gemini client...", flush=True)
    client = create_client(
        config.gcp_project_id,
        config.location,
        bounded_retry=True,
    )
    extraction_config = build_extraction_config(
        config.thinking_level,
        config.multimodal,
    )
    print("[step] Starting scene extraction...", flush=True)
    summary = extract_scene_contexts_gemini(
        scene_context_rows=scene_context_rows,
        output_path=scene_context_output_dir / "scene_context_gemini.jsonl",
        client=client,
        config=config,
        extraction_config=extraction_config,
        sleep_sec=args.sleep_sec,
        resume=not args.force,
        max_scenes=args.max_scenes,
        output_dir=scene_context_output_dir,
        training_output_path=args.training_output,
        video_context_output_dir=video_context_output_dir,
    )
    print(f"Wrote Gemini scene context records to {scene_context_output_dir}")
    print(f"Wrote Gemini video contexts to {video_context_output_dir}")
    print(
        "Summary: "
        f"processed={summary['processed']} "
        f"succeeded={summary['succeeded']} "
        f"failed={summary['failed']} "
        f"skipped={summary['skipped']} "
        f"video_contexts_written={summary['video_contexts_written']}"
    )
    return 1 if summary["failed"] else 0


def print_startup_config(
    args: argparse.Namespace,
    config: SceneContextGeminiConfig,
    scene_context_output_dir: Path,
    video_context_output_dir: Path,
    multimodal_ref_file_count: int | None = None,
    content_count: int | None = None,
    scene_count: int | None = None,
) -> None:
    print("[config]")
    print(f"  path: {Path(args.settings).resolve()}")
    print(f"  manifest: {Path(args.manifest).resolve()}")
    print(f"  gcp_project_id: {config.gcp_project_id}")
    print(f"  gemini_location: {config.location}")
    print(f"  gemini_model: {config.model}")
    print(f"  gemini_thinking_level: {config.thinking_level}")
    print(f"  shot_interval: {config.shot_interval}")
    print(f"  multimodal: {config.multimodal}")
    print(f"  local_output_dir: {Path(config.output_dir).resolve()}")
    print(f"  scene_context_graph_gemini_dir: {scene_context_output_dir}")
    print(f"  video_context_graph_gemini_dir: {video_context_output_dir}")
    print(f"  training_output: {args.training_output or ''}")
    print(f"  resume: {not args.force}")
    print(f"  force: {args.force}")
    print(f"  max_scenes: {args.max_scenes if args.max_scenes is not None else ''}")
    print(f"  sleep_sec: {args.sleep_sec}")
    if multimodal_ref_file_count is not None and content_count is not None and scene_count is not None:
        print_input_summary(config, multimodal_ref_file_count, content_count, scene_count)


def print_input_summary(
    config: SceneContextGeminiConfig,
    multimodal_ref_file_count: int,
    content_count: int,
    scene_count: int,
) -> None:
    print("[input]")
    print(
        "  multimodal_ref_source: "
        f"{(Path(config.output_dir) / multimodal_ref_relative_path(config.shot_interval)).resolve()}"
    )
    print(f"  multimodal_ref_files: {multimodal_ref_file_count}")
    print(f"  contents: {content_count}")
    print(f"  scenes: {scene_count}", flush=True)


def load_context_rows(
    config: SceneContextGeminiConfig,
    manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], int]:
    content_ids = [
        row["content_id"]
        for row in read_manifest_rows(manifest_path)
    ]
    if config.multimodal:
        return load_scene_context_rows(
            Path(config.output_dir) / multimodal_ref_relative_path(config.shot_interval),
            content_ids=content_ids,
            multimodal=True,
        )
    visual_manifest = Path(config.output_dir) / "data" / config.shot_interval / "visual_manifest.jsonl"
    if not visual_manifest.is_file():
        raise FileNotFoundError(f"visual manifest is required for visual_only Gemini extraction: {visual_manifest}")
    selected = set(content_ids)
    rows: list[dict[str, Any]] = []
    documents = [json.loads(line) for line in visual_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for document in documents:
        content_id = str(document.get("content_id") or "")
        if content_id not in selected:
            continue
        for index, scene in enumerate(document.get("scenes") or []):
            keyframes = scene.get("keyframe_timestamps") if isinstance(scene, dict) else None
            if not isinstance(keyframes, list) or not keyframes:
                raise ValueError(f"{content_id}: visual manifest scene {index} has no keyframes")
            rows.append({"content_id": content_id, "scene_idx": scene.get("scene_idx", index), "keyframes": keyframes})
    missing = sorted(selected - {row["content_id"] for row in rows})
    if missing:
        raise ValueError(f"visual manifest is missing {len(missing)} selected contents")
    return rows, len({row["content_id"] for row in rows})


def load_settings_config(args: argparse.Namespace) -> SceneContextGeminiConfig:
    raw = read_config_json(args.settings)
    output_dir = args.output_dir
    if output_dir is None:
        configured_root = os.getenv("OUTPUT_SAVE_PATH", "").strip()
        output_dir = configured_root
    if not output_dir:
        raise RuntimeError("pipeline output directory is required")
    return SceneContextGeminiConfig(
        gcp_project_id=args.gcp_project_id or os.getenv("GCP_PROJECT_ID", "").strip(),
        output_dir=output_dir,
        location=args.location or str(raw.get("gemini_location") or DEFAULT_CONFIG.location),
        model=args.model or str(raw.get("gemini_model") or DEFAULT_CONFIG.model),
        thinking_level=args.thinking_level
        or str(raw.get("gemini_thinking_level") or DEFAULT_CONFIG.thinking_level),
        shot_interval=shot_interval_from_config(raw),
        multimodal=multimodal_from_config(raw),
    )


def read_config_json(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
