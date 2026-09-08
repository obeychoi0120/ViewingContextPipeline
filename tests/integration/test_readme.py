from __future__ import annotations
from collections import Counter
from pathlib import Path
import re
import shlex
import pytest
import extraction.cli as extraction_cli
import validation.cli as validation_cli

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


def test_readme_relative_links_exist() -> None:
    text = README.read_text(encoding="utf-8")
    for first, second in re.findall(r"\[[^\]]+\]\((?:<([^>]+)>|([^)]+))\)", text):
        target = first or second
        if "://" not in target and not target.startswith("#"):
            assert (ROOT / target.split("#", 1)[0]).exists(), target


def _stage_commands():
    return [shlex.split(line) for line in README.read_text(encoding="utf-8").splitlines()
            if re.match(r"^python -m (extraction|validation) ", line)]


def test_readme_documents_complete_stage_matrix() -> None:
    stages = Counter((args[2], args[3]) for args in _stage_commands()
                     if "--plan-only" not in args)
    assert stages == Counter({
        ("validation", "prepare-cohort"): 1,
        ("extraction", "prepare-input-data"): 1,
        ("extraction", "extract-graph-scenes"): 2,
        ("extraction", "summarize-graph"): 2,
        ("extraction", "extract-description-scenes"): 1,
        ("extraction", "summarize-description"): 1,
        ("validation", "embed-representations"): 1,
        ("validation", "run-recommendation"): 1,
        ("validation", "run-diagnosis"): 1,
    })
    assert sum("--plan-only" in args for args in _stage_commands()) == 1


@pytest.mark.parametrize("command", _stage_commands())
def test_readme_commands_are_accepted_and_dispatch_documented_options(command, monkeypatch):
    cli = extraction_cli if command[2] == "extraction" else validation_cli
    context = object()
    calls = []
    monkeypatch.setattr(cli.RunContext, "load", lambda run_id: context)
    monkeypatch.setitem(cli.STEP_HANDLERS, command[3],
                        lambda ctx, **kwargs: calls.append((ctx, kwargs)))
    assert "--run-id" in command
    assert cli.main(command[3:]) == 0
    assert len(calls) == 1 and calls[0][0] is context
    kwargs = calls[0][1]
    for flag in ("--model", "--source", "--gpus"):
        if flag in command:
            expected = command[command.index(flag) + 1]
            assert kwargs[flag[2:]] == (int(expected) if flag == "--gpus" else expected)
    if "--plan-only" in command:
        assert kwargs["plan_only"] is True
