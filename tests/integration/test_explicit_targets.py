"""Explicit execution targets and the four-source, six-arm contract."""
from types import SimpleNamespace

import pytest

from arm_registry import registry, generation_registry
from pipeline_runtime import _validate_config
from validation import cli
from validation.cli import main
from validation.recommendation_contracts import resolve_target_arms


@pytest.mark.parametrize('step', ['embed-representations', 'run-recommendation', 'run-diagnosis'])
def test_cli_requires_target_before_loading_run(step, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('missing target must be rejected before loading a run')
    monkeypatch.setattr('validation.cli.RunContext.load', forbidden)
    with pytest.raises(SystemExit) as exc:
        main([step, '--run-id', 'unused'])
    assert exc.value.code == 2
    assert '--target' in capsys.readouterr().err


@pytest.mark.parametrize('step', ['embed-representations', 'run-recommendation', 'run-diagnosis'])
def test_cli_passes_only_explicit_selection(step, ready_context, monkeypatch):
    calls = []
    monkeypatch.setattr('validation.cli.RunContext.load', lambda _: ready_context)
    monkeypatch.setitem(cli.STEP_HANDLERS,
                        step, lambda ctx, **kw: calls.append(kw))
    assert main([step, '--run-id', 'test', '--target', 'desc_gemini_meta', '--representation-mode', 'text']) == 0
    assert calls == [{'force': False, 'target': ['desc_gemini_meta'], 'representation_mode': 'text'}]


def test_config_without_arms_and_gemini_description_source(ready_context):
    assert 'arms' not in ready_context.config['protocol']
    _validate_config(ready_context.config)
    arm = registry(ready_context.config)['desc_gemini_meta']
    assert (arm.scene_arm, arm.model, arm.representation, arm.uses_title) == (
        'desc_gemini', 'gemini', 'description', True)
    assert generation_registry(ready_context.config)['desc_gemini'].uses_title is False
    for target in [None, [], ['desc_qwen'], ['desc_gemini'], ['graph_gemini'], ['meta', 'meta']]:
        with pytest.raises(ValueError):
            resolve_target_arms(target, config=ready_context.config)


@pytest.mark.parametrize('name', ['embed_representations', 'run_recommendation', 'run_diagnosis'])
def test_public_steps_have_no_implicit_target(name):
    from validation import steps
    # Missing scope must fail even without a usable context or model runtime.
    with pytest.raises(ValueError, match='--target is required'):
        getattr(steps, name)(SimpleNamespace())


