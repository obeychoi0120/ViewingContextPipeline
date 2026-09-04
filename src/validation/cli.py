from __future__ import annotations

import argparse
import sys

from validation.steps import STEP_HANDLERS
from pipeline_runtime import RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Viewing Context validation step.")
    parser.add_argument("step", choices=tuple(STEP_HANDLERS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Freeze users and list required items without media/title validation (prepare-cohort only).",
    )
    args = parser.parse_args(argv)
    try:
        if args.plan_only and args.step != "prepare-cohort":
            raise ValueError("--plan-only is only supported by prepare-cohort")
        context = RunContext.load(args.run_id)
        kwargs = {"force": args.force}
        if args.step == "prepare-cohort":
            kwargs["plan_only"] = args.plan_only
        STEP_HANDLERS[args.step](context, **kwargs)
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
