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
    parser.add_argument("--model", choices=GRAPH_SOURCES)
    parser.add_argument("--source", choices=GRAPH_SOURCES)
    parser.add_argument(
        "--gpus",
        type=_positive_int,
        help="Number of visible CUDA devices to use.",
    )
    args = parser.parse_args(argv)
    try:
        extract = args.step.startswith("extract-")
        summary = args.step.startswith("summarize-")
        if extract or summary:
            if args.schema is None:
                raise ValueError(f"{args.step} requires --schema PATH.md")
            if extract and (args.model is None or args.source is not None):
                raise ValueError("extraction requires --model qwen|gemini and does not accept --source")
            if summary and (args.source is None or args.model is not None):
                raise ValueError("summary requires --source qwen|gemini and does not accept --model")
        elif args.schema is not None or args.model is not None or args.source is not None:
            raise ValueError("--schema/--model/--source are only supported for generation")
        gpu_enabled = summary or extract and args.model == "qwen"
        if args.gpus is not None and not gpu_enabled:
            raise ValueError("--gpus is not supported for this step/model")
        context = RunContext.load(args.run_id)
        kwargs = {"force": args.force}
        if extract or summary:
            kwargs["schema"] = context.prompt_path(args.schema)
            kwargs["model" if extract else "source"] = args.model if extract else args.source
        if gpu_enabled:
            kwargs["gpus"] = args.gpus
        STEP_HANDLERS[args.step](context, **kwargs)
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed
