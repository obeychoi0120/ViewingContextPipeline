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
        "--force",
        action="store_true",
        help="Regenerate all outputs for this step, including completed summaries.",
    )
    parser.add_argument("--schema", help="Path to one Markdown prompt (required for generation).")
    parser.add_argument(
        "--model",
        choices=GRAPH_SOURCES,
        help="Required generation model for extraction and summarization.",
    )
    parser.add_argument(
        "--arm",
        help="Generation source (v7: graph_qwen, desc_qwen, graph_gemini, desc_gemini); not a metadata-concat arm.",
    )
    args = parser.parse_args(argv)
    try:
        if args.schema is None:
            raise ValueError(f"{args.step} requires --schema PATH.md")
        if args.model is None or args.arm is None:
            raise ValueError("generation requires --model qwen|gemini and --arm")
        context = RunContext.load(args.run_id)
        STEP_HANDLERS[args.step](
            context,
            force=args.force,
            schema=context.prompt_path(args.schema),
            model=args.model,
            arm=args.arm,
        )
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
