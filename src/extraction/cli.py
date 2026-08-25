from __future__ import annotations

import argparse
import sys

from extraction.steps import STEP_HANDLERS
from viewing_context_pipeline.pipeline import GPU_STAGES, invalidate_descendants
from viewing_context_pipeline.runtime import RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Viewing Context extraction step.")
    parser.add_argument("step", choices=tuple(STEP_HANDLERS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--gpus",
        type=_positive_int,
        help="Number of visible CUDA devices to use.",
    )
    args = parser.parse_args(argv)
    try:
        context = RunContext.load(args.run_id)
        if args.force:
            context.initialize()
            invalidate_descendants(context, {args.step})
        kwargs = {"force": args.force}
        if args.step in GPU_STAGES:
            kwargs["gpus"] = args.gpus
        elif args.gpus is not None:
            raise ValueError(f"--gpus is only supported for: {', '.join(sorted(GPU_STAGES))}")
        STEP_HANDLERS[args.step](context, **kwargs)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed
