from pathlib import Path
import shlex

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text()
COMMANDS = [line for line in README.splitlines() if line.startswith(("python -m preparation ", "python -m extraction ", "python -m validation "))]


@pytest.mark.parametrize("command", COMMANDS)
def test_documented_commands_dispatch(v5_context, monkeypatch, command):
    from arm_registry import registry
    historical = "migrate-arm-layout" in command
    if historical:
        v5_context.config["schema_version"] = "viewing-context-config/v6"
    else:
        v5_context.config.pop("schema_version")
        v5_context.config["experiment_config_version"] = "v4"
    v5_context.config["protocol"]["description_extractors"] = ["qwen", "gemini"] if historical else ["qwen"]
    v5_context.config["protocol"]["arms"] = ["meta"]
    v5_context.config["protocol"]["arms"] = list(registry(v5_context.config))
    module = f"{shlex.split(command)[2]}.cli"
    cli = __import__(module, fromlist=["STEP_HANDLERS"])
    args = shlex.split(command.replace("$RUN_ID", v5_context.run_id))[3:]
    monkeypatch.setattr(cli.RunContext, "load", lambda _: v5_context)
    received = []
    monkeypatch.setitem(cli.STEP_HANDLERS, args[0], lambda context, **options: received.append(options))
    assert cli.main(args) == 0
    assert len(received) == 1
