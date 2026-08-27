from __future__ import annotations

import argparse
import sys

from viewing_context_pipeline.pipeline import run_pipeline
from viewing_context_pipeline.runtime import RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed MicroLens Graph-vs-Description pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--gpus",
        type=_positive_int,
        help="Number of CUDA devices for Qwen extraction and summarization stages.",
    )
    args = parser.parse_args(argv)
    try:
        return run_pipeline(
            RunContext.load(args.run_id),
            dry_run=args.dry_run,
            gpus=args.gpus,
        )
    except KeyboardInterrupt:
        print("[INTERRUPTED] pipeline", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed
