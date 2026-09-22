from copy import deepcopy

import pytest

import extraction.cli as cli
from pipeline_logging import step_settings
from pipeline_fixtures import context as context


def test_lora_override_is_scoped_to_graph_extraction(context, monkeypatch, tmp_path):
    original = deepcopy(context.config)
    merged = tmp_path / "merged-lora-2.1"
    merged.mkdir()
    monkeypatch.setattr(cli.RunContext, "load", lambda _: context)
    observed = []

    def capture(actual, **kwargs):
        step = "extract-graph-scenes" if "model" in kwargs else "summarize-graph"
        observed.append((actual.path("models", "qwen"), step_settings(actual, step, **kwargs)))

    monkeypatch.setitem(cli.STEP_HANDLERS, "extract-graph-scenes", capture)
    monkeypatch.setitem(cli.STEP_HANDLERS, "summarize-graph", capture)
    assert cli.main(["extract-graph-scenes", "--run-id", "lora", "--model", "qwen",
                     "--qwen-model-path", str(merged)]) == 0
    assert cli.main(["summarize-graph", "--run-id", "lora", "--source", "qwen"]) == 0
    assert observed[0][0] == merged.resolve()
    assert observed[0][1]["models.qwen"] == str(merged.resolve())
    assert observed[1][0] == context.path("models", "qwen")
    assert context.config == original


@pytest.mark.parametrize("args", [
    ["extract-graph-scenes", "--model", "gemini"],
    ["summarize-graph", "--source", "qwen"],
    ["extract-description-scenes"],
    ["prepare-input-data"],
])
def test_lora_override_rejects_other_stages_before_loading(args, monkeypatch, capsys):
    def unexpected_load(_):
        raise AssertionError("invalid options must be rejected before loading the run")

    monkeypatch.setattr(cli.RunContext, "load", unexpected_load)
    assert cli.main([*args, "--run-id", "test", "--qwen-model-path", "merged"]) == 1
    assert "requires extract-graph-scenes --model qwen" in capsys.readouterr().err
