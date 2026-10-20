from __future__ import annotations

import argparse
import sys

from extraction.steps import GRAPH_SOURCES, LOCAL_MODELS, STEP_HANDLERS
from extraction.qwen_config import penalty_schedule, validate_presence_penalty
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
        "--repetition-penalty",
        type=float,
        help="Local vLLM generated-token repetition penalty in [1, 2]. Defaults: Graph scenes 1.05; Description scenes and summaries 1.0.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        help="Local vLLM presence penalty in [-2, 2]. Defaults: scenes 1.5; summaries 2.0. Use 0 to disable.",
    )
    parser.add_argument(
        "--arm",
        help="Generation source; minicpm Graph scenes default to graph_minicpm. Other models require --arm.",
    )
    args = parser.parse_args(argv)
    try:
        if args.schema is None:
            raise ValueError(f"{args.step} requires --schema PATH.md")
        if args.model == "minicpm":
            if args.step != "extract-graph-scenes":
                raise ValueError("MiniCPM supports only extract-graph-scenes")
            if args.arm is None:
                args.arm = "graph_minicpm"
            elif args.arm != "graph_minicpm":
                raise ValueError("minicpm Graph scenes require --arm graph_minicpm")
        if args.step == "summarize" and args.arm == "graph_minicpm":
            raise ValueError("MiniCPM source supports only Graph scene extraction")
        if args.model is None or args.arm is None:
            raise ValueError("generation requires --model and --arm")
        options = {}
        if args.repetition_penalty is not None:
            if args.model not in LOCAL_MODELS:
                raise ValueError("--repetition-penalty is supported only for local vLLM models")
            options["repetition_penalty"] = penalty_schedule([args.repetition_penalty])[0]
        if args.presence_penalty is not None:
            if args.model not in LOCAL_MODELS:
                raise ValueError("--presence-penalty is supported only for local vLLM models")
            options["presence_penalty"] = validate_presence_penalty(args.presence_penalty)
        context = RunContext.load(args.run_id)
        STEP_HANDLERS[args.step](
            context,
            force=args.force,
            schema=context.prompt_path(args.schema),
            model=args.model,
            arm=args.arm,
            **options,
        )
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
