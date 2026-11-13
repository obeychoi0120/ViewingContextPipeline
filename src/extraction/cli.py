from __future__ import annotations

import argparse
import sys

from extraction.steps import GRAPH_SOURCES, LOCAL_MODELS, STEP_HANDLERS
from extraction.qwen_config import penalty_schedule, validate_presence_penalty
from pipeline_runtime import DATASETS, RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Viewing Context extraction step.")
    parser.add_argument("step", choices=tuple(STEP_HANDLERS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all scenes for this step, including completed scenes.",
    )
    parser.add_argument("--schema", help="Path to one Markdown prompt (required for generation).")
    parser.add_argument(
        "--model",
        choices=GRAPH_SOURCES,
        help="Required generation model for scene extraction.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        help="Local vLLM generated-token repetition penalty in [1, 2]. Defaults: Graph scenes 1.05; Description scenes 1.0.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        help="Local vLLM presence penalty in [-2, 2]. Defaults: Graph scenes 0; Description scenes 1.5. Use 0 to disable.",
    )
    parser.add_argument(
        "--stop-repeated-lines",
        action="store_true",
        help="Local vLLM Graph scenes only: end the output instead of writing a line identical "
        "to an earlier line (the repeated line is dropped). Off by default.",
    )
    parser.add_argument(
        "--arm",
        help="Optional generation source; defaults to the one set by --model and the extraction step.",
    )
    args = parser.parse_args(argv)
    try:
        if args.schema is None:
            raise ValueError(f"{args.step} requires --schema PATH.md")
        if args.model is None:
            raise ValueError("generation requires --model")
        options = {}
        if args.repetition_penalty is not None:
            if args.model not in LOCAL_MODELS:
                raise ValueError("--repetition-penalty is supported only for local vLLM models")
            options["repetition_penalty"] = penalty_schedule([args.repetition_penalty])[0]
        if args.presence_penalty is not None:
            if args.model not in LOCAL_MODELS:
                raise ValueError("--presence-penalty is supported only for local vLLM models")
            options["presence_penalty"] = validate_presence_penalty(args.presence_penalty)
        if args.stop_repeated_lines:
            if args.model not in LOCAL_MODELS or args.step != "extract-graph-scenes":
                raise ValueError("--stop-repeated-lines is supported only for local vLLM Graph scenes")
            options["stop_repeated_lines"] = True
        context = RunContext.load(args.run_id, dataset=args.dataset)
        result = STEP_HANDLERS[args.step](
            context,
            force=args.force,
            schema=context.prompt_path(args.schema),
            model=args.model,
            arm=args.arm,
            **options,
        )
        if result.get("failure_count", 0):
            raise RuntimeError(
                f"{result['failure_count']} output(s) failed; completed outputs are preserved. "
                "Rerun this step to retry incomplete outputs."
            )
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
