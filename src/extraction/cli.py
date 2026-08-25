from __future__ import annotations

import argparse
import sys

from extraction.steps import STEP_HANDLERS
from viewing_context_pipeline.pipeline import invalidate_descendants
from viewing_context_pipeline.runtime import RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Viewing Context extraction step.")
    parser.add_argument("step", choices=tuple(STEP_HANDLERS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        context = RunContext.load(args.run_id)
        if args.force:
            context.initialize()
            invalidate_descendants(context, {args.step})
        STEP_HANDLERS[args.step](context, force=args.force)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
