"""Sampled profiling is an explicit recommendation-only execution option."""

import pytest

from validation import cli


@pytest.mark.parametrize("step", ["embed-representations", "run-diagnosis"])
def test_profile_rejected_before_loading_other_steps(step, monkeypatch, capsys):
    monkeypatch.setattr(cli.RunContext, "load", lambda *_: pytest.fail("must reject before load"))
    assert (
        cli.main(
            [
                step,
                "--run-id",
                "unused",
                "--target",
                "meta",
                "--representation-mode",
                "text",
                "--profile-every",
                "100",
            ]
        )
        == 1
    )
    assert "--profile-every is only supported" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_profile_interval_must_be_positive(value, monkeypatch):
    monkeypatch.setattr(cli.RunContext, "load", lambda *_: pytest.fail("must reject before load"))
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "run-recommendation",
                "--run-id",
                "unused",
                "--target",
                "meta",
                "--representation-mode",
                "text",
                "--profile-every",
                value,
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["text", "graph"])
@pytest.mark.parametrize("operators", [False, True])
def test_profile_cli_passes_through_step_to_rolling(ready_context, monkeypatch, mode, operators):
    calls = []
    monkeypatch.setattr(cli.RunContext, "load", lambda *_: ready_context)
    monkeypatch.setattr(
        "validation.rolling_recommendation.run_rolling",
        lambda context, **kwargs: calls.append((context, kwargs)),
    )
    extra = ["--scene-aggregation", "mean"] if mode == "graph" else []
    if operators:
        extra.append("--profile-operators")
    assert (
        cli.main(
            [
                "run-recommendation",
                "--run-id",
                "test",
                "--target",
                "meta",
                "--representation-mode",
                mode,
                "--profile-every",
                "17",
                *extra,
            ]
        )
        == 0
    )
    assert calls[0][1]["profile_every"] == 17
    assert calls[0][1].get("profile_operators", False) == operators
    if mode == "graph":
        assert calls[0][0].scene_aggregation == "mean"


@pytest.mark.parametrize("step", ["run-recommendation", "embed-representations", "run-diagnosis"])
def test_operator_profile_requires_recommendation_and_interval(step, monkeypatch, capsys):
    monkeypatch.setattr(cli.RunContext, "load", lambda *_: pytest.fail("must reject before load"))
    assert cli.main([
        step, "--run-id", "unused", "--target", "meta", "--representation-mode", "text",
        "--profile-operators",
    ]) == 1
    assert "--profile-operators requires" in capsys.readouterr().err
