from __future__ import annotations

import argparse
import sys

from validation.steps import STEP_HANDLERS
from pipeline_runtime import RunContext


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Viewing Context validation step.")
    parser.add_argument("step", choices=tuple(STEP_HANDLERS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--target", nargs="+",
        help="Sources to include (embedding/recommendation/diagnosis; default: configured active arms).",
    )
    parser.add_argument(
        "--workers-per-gpu", type=_positive_int,
        help="Independent combination processes per GPU (run-recommendation only; default: one).",
    )
    parser.add_argument("--compare-run-id", help="Reference run for paired Graph comparison (run-diagnosis only).")
    args = parser.parse_args(argv)
    try:
        if args.compare_run_id is not None and args.step != "run-diagnosis":
            raise ValueError("--compare-run-id is only supported by run-diagnosis")
        if args.target is not None and args.step not in {"embed-representations", "run-recommendation", "run-diagnosis"}:
            raise ValueError("--target is only supported by run-recommendation/run-diagnosis")
        if args.workers_per_gpu is not None and args.step != "run-recommendation":
            raise ValueError("--workers-per-gpu is only supported by run-recommendation")
        context = RunContext.load(args.run_id)
        kwargs = {"force": args.force}
        if args.compare_run_id is not None:
            kwargs["compare_run_id"] = args.compare_run_id
        if args.target is not None:
            from arm_registry import select_arms
            select_arms(context.config, args.target)
            kwargs["target"] = args.target
        if args.workers_per_gpu is not None:
            kwargs["workers_per_gpu"] = args.workers_per_gpu
        STEP_HANDLERS[args.step](context, **kwargs)
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
