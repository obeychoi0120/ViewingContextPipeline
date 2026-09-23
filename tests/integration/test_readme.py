from pathlib import Path
import shlex

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text()
COMMANDS = [
    line
    for line in README.splitlines()
    if "migrate-arm-layout" not in line
    and line.startswith(
        ("python -m preparation ", "python -m extraction ", "python -m validation ")
    )
]


@pytest.mark.parametrize("command", COMMANDS)
def test_documented_commands_dispatch(current_context, monkeypatch, command):
    module = f"{shlex.split(command)[2]}.cli"
    cli = __import__(module, fromlist=["STEP_HANDLERS"])
    args = shlex.split(command.replace("$RUN_ID", current_context.run_id))[3:]
    monkeypatch.setattr(cli.RunContext, "load", lambda _: current_context)
    received = []
    monkeypatch.setitem(
        cli.STEP_HANDLERS, args[0], lambda context, **options: received.append(options)
    )
    assert cli.main(args) == 0
    assert len(received) == 1
