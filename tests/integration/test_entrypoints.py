"""Validate the shipped execution instructions without model or API calls."""
import importlib
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("document", ["README.md", "docs/runtime.md", "docs/graph_training.md", "docs/TODO.md"])
def test_document_links_exist(document):
    source = ROOT / document
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", source.read_text()):
        if "://" in target or target.startswith("#"):
            continue
        assert (source.parent / target.split("#", 1)[0]).exists(), (document, target)


@pytest.mark.parametrize("script,count", [("run_pipeline_v7.sh", 2), ("script_graph_v7.sh", 5)])
def test_scripts_dispatch_supported_commands(tmp_path, monkeypatch, current_context, script, count):
    executable = tmp_path / "python"
    executable.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "sys.exit(int(os.environ.get('FAKE_EXIT', '0')))\n"
    )
    executable.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
           "RUN_ID": "script-test", "CALL_LOG": str(log)}
    subprocess.run(["bash", "-n", str(ROOT / script)], check=True)
    subprocess.run(["bash", str(ROOT / script)], cwd=tmp_path, env=env, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == count
    for argv in calls:
        assert argv[0] == "-m"
        cli = importlib.import_module(f"{argv[1]}.cli")
        args = argv[2:]
        assert args[0] in cli.STEP_HANDLERS
        assert args[args.index("--run-id")+1] == "script-test"
        received = []
        with monkeypatch.context() as patch:
            patch.setattr(cli.RunContext, "load", lambda _: current_context)
            patch.setitem(cli.STEP_HANDLERS, args[0], lambda ctx, **options: received.append(options))
            assert cli.main(args) == 0
            assert len(received) == 1
    log.unlink()
    failed = subprocess.run(["bash", str(ROOT / script)], cwd=tmp_path,
                            env={**env, "FAKE_EXIT": "7"})
    assert failed.returncode == 7
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.parametrize("command", ["summarize-graph", "summarize-description", "migrate-arm-layout", "migrate-scene-schema"])
def test_retired_extraction_commands_are_rejected(command):
    from extraction.cli import main
    with pytest.raises(SystemExit) as exc:
        main([command, "--run-id", "unused"])
    assert exc.value.code == 2
