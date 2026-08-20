from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.video_profile_generation import run_pipeline
from src.video_profile_generation.config import ConfigError, load_video_profile_config
from src.video_profile_generation.pipeline import PipelineError
from src.common.gemini import create_client
from src.common.manifest import CANONICAL_MANIFEST_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "video_profile_generation.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured video profiles with Gemini on Vertex AI.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CANONICAL_MANIFEST_PATH,
        help="Input video manifest CSV. Defaults to contracts/manifest.csv.",
    )
    parser.add_argument("--contents-id", dest="content_ids", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_video_profile_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    print("[config]")
    print(f"  path: {Path(args.config).resolve()}")
    print(f"  manifest: {Path(args.manifest).resolve()}")
    print(f"  gcp_project_id: {config.gcp_project_id}")
    print(f"  gemini_location: {config.gemini_location}")
    print(f"  gemini_model: {config.gemini_model}")
    print(f"  gemini_thinking_level: {config.gemini_thinking_level}")
    print(f"  shot_interval: {config.shot_interval}")
    print(
        "  ontology_contract_path: "
        f"{config.resolve_ontology_contract_path(PROJECT_ROOT)}"
    )
    print(f"  local_output_dir: {config.resolve_local_output_dir(PROJECT_ROOT)}")

    try:
        client = create_client(
            config.gcp_project_id,
            config.gemini_location,
            bounded_retry=True,
        )
        summary = run_pipeline(
            client=client,
            config=config,
            project_root=PROJECT_ROOT,
            manifest_path=args.manifest,
            content_ids=args.content_ids,
            overwrite=args.overwrite,
        )
    except PipelineError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1

    print(
        "Summary: "
        f"succeeded={summary.succeeded} "
        f"skipped={summary.skipped} "
        f"failed={summary.failed}"
    )
    if summary.errors:
        print("Failed contents:")
        for content_id, error in summary.errors.items():
            print(f"  {content_id}: {error}")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
