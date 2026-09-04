from __future__ import annotations

from types import SimpleNamespace

import pytest

import extraction.cli as cli_module


def test_documented_extraction_step_matrix_is_stable() -> None:
    assert tuple(cli_module.STEP_HANDLERS) == (
        "prepare-input-data",
        "extract-graph-scenes",
        "summarize-graph",
        "extract-description-scenes",
        "summarize-description",
    )


def test_gpu_count_is_forwarded_to_cuda_step(monkeypatch: pytest.MonkeyPatch) -> None:
    received = {}

    def handler(context, *, model, force=False, gpus=None):
        received.update(context=context, model=model, force=force, gpus=gpus)

    context = SimpleNamespace()
    monkeypatch.setattr(cli_module.RunContext, "load", lambda _: context)
    monkeypatch.setitem(cli_module.STEP_HANDLERS, "extract-graph-scenes", handler)

    assert cli_module.main([
        "extract-graph-scenes",
        "--run-id",
        "demo",
        "--model",
        "qwen",
        "--gpus",
        "2",
    ]) == 0
    assert received == {
        "context": context,
        "model": "qwen",
        "force": False,
        "gpus": 2,
    }


def test_graph_source_arguments_are_required() -> None:
    assert cli_module.main(["extract-graph-scenes", "--run-id", "demo"]) == 1
    assert cli_module.main(["summarize-graph", "--run-id", "demo"]) == 1


def test_graph_summary_source_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    received = {}

    def handler(context, *, source, force=False, gpus=None):
        received.update(context=context, source=source, force=force, gpus=gpus)

    context = SimpleNamespace()
    monkeypatch.setattr(cli_module.RunContext, "load", lambda _: context)
    monkeypatch.setitem(cli_module.STEP_HANDLERS, "summarize-graph", handler)

    assert cli_module.main([
        "summarize-graph",
        "--run-id",
        "demo",
        "--source",
        "gemini",
        "--gpus",
        "2",
    ]) == 0
    assert received == {
        "context": context,
        "source": "gemini",
        "force": False,
        "gpus": 2,
    }


def test_gpus_are_rejected_for_gemini_extraction() -> None:
    assert cli_module.main([
        "extract-graph-scenes",
        "--run-id",
        "demo",
        "--model",
        "gemini",
        "--gpus",
        "2",
    ]) == 1


def test_gpu_count_is_rejected_for_cpu_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module.RunContext, "load", lambda _: SimpleNamespace())
    assert cli_module.main([
        "prepare-input-data",
        "--run-id",
        "demo",
        "--gpus",
        "2",
    ]) == 1


def test_gpu_count_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        cli_module.main([
            "extract-graph-scenes",
            "--run-id",
            "demo",
            "--model",
            "qwen",
            "--gpus",
            "0",
        ])


def test_keyboard_interrupt_returns_shell_interrupt_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module.RunContext, "load", lambda _: SimpleNamespace())

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setitem(cli_module.STEP_HANDLERS, "prepare-input-data", interrupt)
    assert cli_module.main(["prepare-input-data", "--run-id", "demo"]) == 130


def test_reuse_run_is_forwarded_to_input_preparation(monkeypatch):
    received = {}
    monkeypatch.setattr(cli_module.RunContext, "load", lambda _: SimpleNamespace())

    def prepare(_context, **kwargs):
        received.update(kwargs)

    monkeypatch.setitem(cli_module.STEP_HANDLERS, "prepare-input-data", prepare)
    assert cli_module.main([
        "prepare-input-data", "--run-id", "new", "--reuse-run-id", "old",
    ]) == 0
    assert received == {"force": False, "reuse_run_id": "old"}


@pytest.mark.parametrize("args", [
    ["extract-graph-scenes", "--model", "qwen"],
    ["extract-graph-scenes", "--model", "gemini"],
    ["summarize-graph", "--source", "qwen"],
    ["summarize-graph", "--source", "gemini"],
    ["extract-description-scenes"],
    ["summarize-description"],
    ["prepare-input-data", "--force"],
])
def test_reuse_run_is_rejected_outside_its_stage_or_with_force(args, capsys):
    assert cli_module.main([*args, "--run-id", "new", "--reuse-run-id", "old"]) == 1
    assert "--reuse-run-id" in capsys.readouterr().err


def test_plan_only_is_not_an_extraction_flag():
    with pytest.raises(SystemExit) as raised:
        cli_module.main(["prepare-input-data", "--run-id", "new", "--plan-only"])
    assert raised.value.code == 2
