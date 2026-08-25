from __future__ import annotations

from types import SimpleNamespace

import pytest

import extraction.cli as cli_module


def test_gpu_count_is_forwarded_to_cuda_step(monkeypatch: pytest.MonkeyPatch) -> None:
    received = {}

    def handler(context, *, force=False, gpus=None):
        received.update(context=context, force=force, gpus=gpus)

    context = SimpleNamespace()
    monkeypatch.setattr(cli_module.RunContext, "load", lambda _: context)
    monkeypatch.setitem(cli_module.STEP_HANDLERS, "extract-graph-scenes", handler)

    assert cli_module.main([
        "extract-graph-scenes",
        "--run-id",
        "demo",
        "--gpus",
        "2",
    ]) == 0
    assert received == {"context": context, "force": False, "gpus": 2}


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
            "--gpus",
            "0",
        ])
