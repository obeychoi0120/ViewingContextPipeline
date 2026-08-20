from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from src.common.manifest import CANONICAL_MANIFEST_PATH, read_manifest_rows
from src.common.output_paths import custom_output_root
from src.video_data_collection.raw_pipeline import (
    multimodal_from_config,
    ref_jsonl_relative_path,
    shot_interval_from_config,
)
from .extractor import (
    SceneContextRefConfig,
    build_extraction_config,
    extract_scene_contexts_ref,
    load_scene_context_rows,
    scene_context_ref_output_dir,
    video_context_graph_ref_output_dir,
)
from src.common.gemini import create_client
from ..graph_core.report import write_viewing_context_report


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = SceneContextRefConfig(gcp_project_id="")
DEFAULT_SETTINGS_PATH = (
    PROJECT_ROOT / "config" / "scene_context_extraction_ref.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract scene-level visual graph Reference context with Gemini.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CANONICAL_MANIFEST_PATH,
        help="Input video manifest CSV. Defaults to contracts/manifest.csv.",
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
        help="Re-extract every scene even when complete Reference outputs already exist.",
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_settings_config(args)
    if not config.gcp_project_id:
        raise RuntimeError("GCP_PROJECT_ID is missing. Set it in config/.env or pass --gcp-project-id.")

    scene_context_output_dir = scene_context_ref_output_dir(
        config.output_dir,
        config.shot_interval,
        config.multimodal,
    )
    video_context_output_dir = video_context_graph_ref_output_dir(
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

    print("[step] Loading local Ref JSONL scene inputs...", flush=True)
    scene_context_rows, ref_jsonl_file_count = load_context_rows(
        config,
        args.manifest,
    )
    print_input_summary(
        config=config,
        ref_jsonl_file_count=ref_jsonl_file_count,
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
    summary = extract_scene_contexts_ref(
        scene_context_rows=scene_context_rows,
        output_path=scene_context_output_dir / "scene_context_ref.jsonl",
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
    print(f"Wrote scene context Reference records to {scene_context_output_dir}")
    print(f"Wrote Reference video profiles to {video_context_output_dir}")
    print(
        "Summary: "
        f"processed={summary['processed']} "
        f"succeeded={summary['succeeded']} "
        f"failed={summary['failed']} "
        f"skipped={summary['skipped']} "
        f"video_contexts_written={summary['video_contexts_written']}"
    )
    write_viewing_context_report(
        config.output_dir,
        multimodal=config.multimodal,
        mode=config.shot_interval,
        source="ref",
        payload=summary,
    )
    return 1 if summary["failed"] else 0


def print_startup_config(
    args: argparse.Namespace,
    config: SceneContextRefConfig,
    scene_context_output_dir: Path,
    video_context_output_dir: Path,
    ref_jsonl_file_count: int | None = None,
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
    print(f"  scene_context_graph_ref_dir: {scene_context_output_dir}")
    print(f"  video_context_graph_ref_dir: {video_context_output_dir}")
    print(f"  training_output: {args.training_output or ''}")
    print(f"  resume: {not args.force}")
    print(f"  force: {args.force}")
    print(f"  max_scenes: {args.max_scenes if args.max_scenes is not None else ''}")
    print(f"  sleep_sec: {args.sleep_sec}")
    if ref_jsonl_file_count is not None and content_count is not None and scene_count is not None:
        print_input_summary(config, ref_jsonl_file_count, content_count, scene_count)


def print_input_summary(
    config: SceneContextRefConfig,
    ref_jsonl_file_count: int,
    content_count: int,
    scene_count: int,
) -> None:
    print("[input]")
    print(
        "  ref_jsonl_source: "
        f"{(Path(config.output_dir) / ref_jsonl_relative_path(config.shot_interval)).resolve()}"
    )
    print(f"  ref_jsonl_files: {ref_jsonl_file_count}")
    print(f"  contents: {content_count}")
    print(f"  scenes: {scene_count}", flush=True)


def load_context_rows(
    config: SceneContextRefConfig,
    manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], int]:
    content_ids = [
        row["content_id"]
        for row in read_manifest_rows(manifest_path)
    ]
    return load_scene_context_rows(
        Path(config.output_dir) / ref_jsonl_relative_path(config.shot_interval),
        content_ids=content_ids,
        multimodal=config.multimodal,
    )


def load_settings_config(args: argparse.Namespace) -> SceneContextRefConfig:
    load_dotenv(PROJECT_ROOT / "config" / ".env")
    raw = read_config_json(args.settings)
    legacy_keys = sorted({"gcp_project_id", "output_dir"} & raw.keys())
    if legacy_keys:
        raise RuntimeError(
            f"Move {', '.join(legacy_keys)} from {args.settings} to "
            "GCP_PROJECT_ID/OUTPUT_SAVE_PATH in config/.env"
        )
    output_dir = args.output_dir
    if output_dir is None:
        configured_root = os.getenv("OUTPUT_SAVE_PATH", "").strip()
        output_dir = str(custom_output_root(configured_root)) if configured_root else ""
    if not output_dir:
        raise RuntimeError("OUTPUT_SAVE_PATH is required in config/.env or the environment")
    return SceneContextRefConfig(
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
