from pathlib import Path
import re
import shlex

import pytest
from extraction.cli import main as extraction
from validation.cli import main as validation

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text()
COMMANDS = [line for line in README.splitlines() if line.startswith("python -m extraction ") or line.startswith("python -m validation ")]


def test_readme_relative_links_exist():
    for target in re.findall(r"\]\(([^)]+)\)", README):
        if not target.startswith(("http", "#")):
            assert (ROOT / target.split("#")[0]).exists(), target


def test_readme_documents_complete_stage_matrix():
    assert len(COMMANDS) == 17
    for step in ("extract-description-scenes", "extract-graph-scenes", "summarize-description", "summarize-graph"):
        assert sum(f" {step} " in command for command in COMMANDS) == 2


@pytest.mark.parametrize("command", COMMANDS)
def test_documented_commands_dispatch(v5_context, monkeypatch, command):
    module = "extraction.cli" if "-m extraction" in command else "validation.cli"
    cli = __import__(module, fromlist=["STEP_HANDLERS"])
    args = shlex.split(command.replace("$RUN_ID", v5_context.run_id))[3:]
    monkeypatch.setattr(cli.RunContext, "load", lambda _: v5_context)
    received = []
    monkeypatch.setitem(cli.STEP_HANDLERS, args[0], lambda context, **options: received.append(options))
    assert (extraction if module.startswith("extraction") else validation)(args) == 0
    assert len(received) == 1
