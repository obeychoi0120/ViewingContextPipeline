"""Explicit execution targets and the four-source, six-arm contract."""
from types import SimpleNamespace

import pytest

from arm_registry import registry, generation_registry, LEGACY_CONCAT_ARM_CONTRACT
from pipeline_runtime import _validate_config, write_json
from validation import cli
from validation.cli import main
from validation.recommendation_contracts import resolve_target_arms
from validation.selection import diagnosis_context, cohort_directory


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
    assert main([step, '--run-id', 'test', '--target', 'desc_gemini_meta']) == 0
    assert calls == [{'force': False, 'target': ['desc_gemini_meta']}]


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


def test_old_manifest_retains_previous_arm_layout(ready_context):
    write_json(cohort_directory(ready_context) / 'manifest.json', {
        'arm_contract': LEGACY_CONCAT_ARM_CONTRACT,
        'arms': ['meta', 'desc_qwen', 'desc_qwen_meta'],
    })
    historical = diagnosis_context(ready_context)
    assert 'desc_qwen' in registry(historical.config)
    assert 'desc_gemini_meta' not in registry(historical.config)
    assert 'arms' not in ready_context.config['protocol']
    from validation.diagnosis_statistics import comparison_families
    assert len(comparison_families(historical.config)['title_input']) == 2
    assert list(resolve_target_arms(['desc_qwen'], config=historical.config)) == ['desc_qwen']
