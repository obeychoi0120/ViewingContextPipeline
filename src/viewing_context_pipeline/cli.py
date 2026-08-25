from __future__ import annotations

import argparse
import sys

from viewing_context_pipeline.pipeline import STAGES, run_pipeline
from viewing_context_pipeline.runtime import RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed MicroLens Graph-vs-Description pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force-stage", action="append", choices=STAGES, default=[])
    args = parser.parse_args(argv)
    try:
        return run_pipeline(
            RunContext.load(args.run_id),
            resume=args.resume,
            dry_run=args.dry_run,
            force_stages=set(args.force_stage),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 1
