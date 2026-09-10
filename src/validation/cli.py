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
        "--gpus", type=_positive_int,
        help="Number of visible CUDA devices (v4 run-recommendation only; default: one).",
    )
    parser.add_argument(
        "--workers-per-gpu", type=_positive_int,
        help="Independent combination processes per GPU (v4 run-recommendation only; default: one).",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Freeze users and list required items without media/title validation (prepare-cohort only).",
    )
    args = parser.parse_args(argv)
    try:
        if args.plan_only and args.step != "prepare-cohort":
            raise ValueError("--plan-only is only supported by prepare-cohort")
        if (args.gpus is not None or args.workers_per_gpu is not None) and args.step != "run-recommendation":
            raise ValueError("--gpus/--workers-per-gpu are only supported by run-recommendation")
        context = RunContext.load(args.run_id)
        kwargs = {"force": args.force}
        if args.step == "prepare-cohort":
            kwargs["plan_only"] = args.plan_only
        if args.gpus is not None:
            kwargs["gpus"] = args.gpus
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
