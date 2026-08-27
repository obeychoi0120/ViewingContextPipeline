from __future__ import annotations

import argparse
import sys

from extraction.steps import GRAPH_SOURCES, STEP_HANDLERS
from viewing_context_pipeline.runtime import RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Viewing Context extraction step.")
    parser.add_argument("step", choices=tuple(STEP_HANDLERS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", choices=GRAPH_SOURCES)
    parser.add_argument("--source", choices=GRAPH_SOURCES)
    parser.add_argument(
        "--gpus",
        type=_positive_int,
        help="Number of visible CUDA devices to use.",
    )
    args = parser.parse_args(argv)
    try:
        if args.step == "extract-graph-scenes":
            if args.model is None:
                raise ValueError("extract-graph-scenes requires --model qwen|gemini")
            if args.source is not None:
                raise ValueError("--source is only supported by summarize-graph")
        elif args.step == "summarize-graph":
            if args.source is None:
                raise ValueError("summarize-graph requires --source qwen|gemini")
            if args.model is not None:
                raise ValueError("--model is only supported by extract-graph-scenes")
        else:
            if args.model is not None or args.source is not None:
                raise ValueError("--model/--source are only supported by Graph steps")
        if args.step == "extract-graph-scenes" and args.model == "gemini" and args.gpus:
            raise ValueError("--gpus cannot be used with --model gemini")
        gpu_enabled = (
            args.step in {"summarize-graph", "extract-description-scenes", "summarize-description"}
            or args.step == "extract-graph-scenes" and args.model == "qwen"
        )
        if args.gpus is not None and not gpu_enabled:
            raise ValueError("--gpus is not supported for this step")
        context = RunContext.load(args.run_id)
        kwargs = {"force": args.force}
        if args.step == "extract-graph-scenes":
            kwargs["model"] = args.model
        elif args.step == "summarize-graph":
            kwargs["source"] = args.source
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
