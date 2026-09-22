from __future__ import annotations

import argparse
import sys

from extraction.steps import GRAPH_SOURCES, STEP_HANDLERS
from pipeline_runtime import RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Viewing Context extraction step.")
    parser.add_argument("step", choices=tuple(STEP_HANDLERS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate all outputs for this step, including completed summaries.",
    )
    parser.add_argument("--schema", help="Path to one Markdown prompt (required for generation).")
    parser.add_argument("--model", choices=GRAPH_SOURCES,
                        help="Required generation model for extraction and summarization.")
    parser.add_argument("--arm", help="Scene or Summary arm.")
    parser.add_argument("--summary-model", choices=GRAPH_SOURCES,
                        help="Summary model to copy with migrate-arm-layout.")
    args = parser.parse_args(argv)
    try:
        extract = args.step.startswith("extract-")
        summary = args.step.startswith("summarize-")
        if extract or summary:
            if args.schema is None:
                raise ValueError(f"{args.step} requires --schema PATH.md")
            if args.model is None or args.arm is None:
                raise ValueError("generation requires --model qwen|gemini and --arm")
            if args.summary_model is not None:
                raise ValueError("--summary-model is only for migrate-arm-layout")
        elif args.step == "migrate-arm-layout":
            if args.summary_model is None:
                raise ValueError("migrate-arm-layout requires --summary-model")
            if any((args.schema, args.model, args.arm, args.force)):
                raise ValueError("migration accepts only --run-id and --summary-model")
        elif any((args.schema, args.model, args.arm, args.summary_model)):
            raise ValueError("generation arguments are not supported for this step")
        context = RunContext.load(args.run_id)
        kwargs = {"force": args.force}
        if extract or summary:
            kwargs["schema"] = context.prompt_path(args.schema)
            kwargs["model"] = args.model
            kwargs["arm"] = args.arm
        if args.step == "migrate-arm-layout":
            kwargs = {"summary_model": args.summary_model}
        STEP_HANDLERS[args.step](context, **kwargs)
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
