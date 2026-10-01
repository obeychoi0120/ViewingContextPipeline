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
        "--target",
        nargs="+",
        required=True,
        help="Explicit arms to include (required for embedding/recommendation/diagnosis; no default).",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=_positive_int,
        help="Independent combination processes per GPU (run-recommendation only; default: one).",
    )
    parser.add_argument(
        "--compare-run-id", help="Reference run for paired Graph comparison (run-diagnosis only)."
    )
    parser.add_argument("--representation-mode", choices=("text", "graph"), required=True)
    parser.add_argument("--scene-aggregation", choices=("mean", "attention"))
    parser.add_argument(
        "--profile-every",
        type=_positive_int,
        help="Profile first 3 batches and every Nth batch per epoch with synchronized CUDA timings "
        "(run-recommendation only; disabled by default).",
    )
    parser.add_argument(
        "--profile-operators", action="store_true",
        help="Capture one warmed training batch with PyTorch operator/ backward tracing "
        "(requires --profile-every; run-recommendation only).",
    )
    args = parser.parse_args(argv)
    try:
        from validation.graph_context import validate_mode

        validate_mode(args.representation_mode, args.scene_aggregation, args.step, args.target)
        if args.compare_run_id is not None and args.step != "run-diagnosis":
            raise ValueError("--compare-run-id is only supported by run-diagnosis")
        if args.target is not None and args.step not in {
            "embed-representations",
            "run-recommendation",
            "run-diagnosis",
        }:
            raise ValueError(
                "--target is only supported by embed-representations/run-recommendation/run-diagnosis"
            )
        if args.workers_per_gpu is not None and args.step != "run-recommendation":
            raise ValueError("--workers-per-gpu is only supported by run-recommendation")
        if args.profile_every is not None and args.step != "run-recommendation":
            raise ValueError("--profile-every is only supported by run-recommendation")
        if args.profile_operators and (args.step != "run-recommendation" or args.profile_every is None):
            raise ValueError("--profile-operators requires run-recommendation and --profile-every")
        context = RunContext.load(args.run_id)
        kwargs = {"force": args.force, "representation_mode": args.representation_mode}
        if args.scene_aggregation is not None:
            kwargs["scene_aggregation"] = args.scene_aggregation
        if args.compare_run_id is not None:
            kwargs["compare_run_id"] = args.compare_run_id
        if args.target is not None:
            from arm_registry import select_arms

            select_arms(context.config, args.target)
            kwargs["target"] = args.target
        if args.workers_per_gpu is not None:
            kwargs["workers_per_gpu"] = args.workers_per_gpu
        if args.profile_every is not None:
            kwargs["profile_every"] = args.profile_every
        if args.profile_operators:
            kwargs["profile_operators"] = True
        STEP_HANDLERS[args.step](context, **kwargs)
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
