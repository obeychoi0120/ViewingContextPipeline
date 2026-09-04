from __future__ import annotations

import validation.cli as cli_module
import pytest


def test_documented_validation_step_matrix_is_stable() -> None:
    assert tuple(cli_module.STEP_HANDLERS) == (
        "prepare-cohort",
        "embed-representations",
        "run-recommendation",
        "run-diagnosis",
    )


def test_plan_only_is_forwarded_only_to_prepare_cohort(monkeypatch):
    received = {}
    context = object()
    monkeypatch.setattr(cli_module.RunContext, "load", lambda _: context)

    def prepare(actual, **kwargs):
        assert actual is context
        received.update(kwargs)

    monkeypatch.setitem(cli_module.STEP_HANDLERS, "prepare-cohort", prepare)
    assert cli_module.main(["prepare-cohort", "--run-id", "test", "--plan-only"]) == 0
    assert received == {"force": False, "plan_only": True}


@pytest.mark.parametrize("step", ["embed-representations", "run-recommendation", "run-diagnosis"])
def test_plan_only_is_rejected_on_other_validation_steps(step, capsys):
    assert cli_module.main([step, "--run-id", "test", "--plan-only"]) == 1
    assert "--plan-only is only supported by prepare-cohort" in capsys.readouterr().err


@pytest.mark.parametrize("args", [
    ["unknown", "--run-id", "test"],
    ["prepare-cohort"],
    ["prepare-cohort", "--run-id", "test", "--reuse-run-id", "old"],
])
def test_validation_cli_syntax_errors_keep_exit_code_two(args):
    with pytest.raises(SystemExit) as raised:
        cli_module.main(args)
    assert raised.value.code == 2


def test_cohort_interrupt_returns_130(monkeypatch):
    monkeypatch.setattr(cli_module.RunContext, "load", lambda _: object())

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setitem(cli_module.STEP_HANDLERS, "prepare-cohort", interrupt)
    assert cli_module.main(["prepare-cohort", "--run-id", "test", "--plan-only"]) == 130
