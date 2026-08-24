from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .pipeline import STAGES, runner_main, stage_cli_main
from .stages import STAGE_HANDLERS


def _execute_stage(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    args = parser.parse_args(argv)
    try:
        STAGE_HANDLERS[args.stage](Path(args.runtime))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.stage} error={exc}", file=sys.stderr, flush=True)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"-h", "--help"}:
        print("usage: viewing-context-pipeline {run," + ",".join(STAGES) + "} [options]")
        return 0
    command = arguments.pop(0) if arguments else "run"
    if command == "run":
        return runner_main(arguments)
    if command in STAGES:
        return stage_cli_main(command, arguments)
    if command == "_execute-stage":
        return _execute_stage(arguments)
    choices = ", ".join(("run", *STAGES))
    raise SystemExit(f"unknown command {command!r}; choose one of: {choices}")
