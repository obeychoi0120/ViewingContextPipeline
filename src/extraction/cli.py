from __future__ import annotations

import argparse
import sys

from extraction.steps import GRAPH_SOURCES, STEP_HANDLERS
from extraction.qwen_config import QWEN_MODELS, penalty_schedule, validate_presence_penalty
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
        help="Qwen generated-token repetition penalty in [1, 2]. Defaults: Graph scenes 1.05; Description scenes and summaries 1.0.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        help="Qwen presence penalty in [-2, 2]. Defaults: scenes 1.5; summaries 2.0. Use 0 to disable.",
    )
    parser.add_argument(
        "--arm",
        help="Generation source (v7: graph_qwen3vl, desc_qwen3vl, graph_qwen3.5, desc_qwen3.5, graph_gemini, desc_gemini); not a metadata-concat arm.",
    )
    args = parser.parse_args(argv)
    try:
        if args.schema is None:
            raise ValueError(f"{args.step} requires --schema PATH.md")
        if args.model is None or args.arm is None:
            raise ValueError("generation requires --model qwen3vl|qwen3.5|gemini and --arm")
        options = {}
        if args.repetition_penalty is not None:
            if args.model not in QWEN_MODELS:
                raise ValueError("--repetition-penalty is supported only for Qwen models")
            options["repetition_penalty"] = penalty_schedule([args.repetition_penalty])[0]
        if args.presence_penalty is not None:
            if args.model not in QWEN_MODELS:
                raise ValueError("--presence-penalty is supported only for Qwen models")
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
