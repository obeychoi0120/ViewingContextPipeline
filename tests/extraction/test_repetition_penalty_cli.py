"""CLI overrides reach requests, identities, reuse checks and forced regeneration."""

import pytest

from artifact_io import read_json
from extraction.cli import main


@pytest.mark.parametrize('model', ['qwen3vl', 'qwen3.5'])
@pytest.mark.parametrize('representation', ['graph', 'description'])
def test_cli_penalty_reaches_scene_and_summary_requests_and_reuse_checks(
    ready_context, fake_models, monkeypatch, model, representation,
):
    context = ready_context
    monkeypatch.setattr('extraction.cli.RunContext.load', lambda _: context)
    arm = f"{'graph' if representation == 'graph' else 'desc'}_{model}"
    for summary in (False, True):
        step = 'summarize' if summary else f'extract-{representation}-scenes'
        version = 8 if representation == 'graph' else (5 if summary else 3)
        schema = f"prompts/{'summary' if summary else 'scene'}_{representation}_v{version}.md"
        args = [step, '--run-id', context.run_id, '--model', model, '--arm', arm,
                '--schema', schema, '--repetition-penalty', '1.12']
        assert main(args) == 0
        assert all(task.repetition_penalty == 1.12 for task in fake_models[-1])
        if summary:
            directory = context.summary_arm_dir(arm)
            identity_path = next(directory.glob('*.json'))
            settings = read_json(identity_path)['provenance']['settings']
            assert settings['actual_repetition_penalty'] == 1.12
        else:
            directory = context.scene_arm_dir(arm)
            identity_path = directory / 'generation_identity.json'
            settings = read_json(identity_path)['identity']['settings']
        assert settings['generation']['repetition_penalty'] == 1.12
        assert settings['repetition_penalty'] == [1.12]
        before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        calls = len(fake_models)
        assert main(args) == 0
        assert len(fake_models) == calls
        changed = [*args[:-1], '1.15']
        assert main(changed) == 1
        assert len(fake_models) == calls
        assert before == {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        assert main([*changed, '--force']) == 0
        assert all(task.repetition_penalty == 1.15 for task in fake_models[-1])
        updated = read_json(identity_path)
        settings = updated['provenance' if summary else 'identity']['settings']
        assert settings['generation']['repetition_penalty'] == 1.15
        assert settings['repetition_penalty'] == [1.15]


@pytest.mark.parametrize('value', ['0.9', '2.1', 'nan'])
def test_invalid_penalty_is_rejected_before_loading_run(value, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid penalty must be rejected before loading a Run')

    monkeypatch.setattr('extraction.cli.RunContext.load', forbidden)
    assert main(['extract-graph-scenes', '--run-id', 'unused', '--model', 'qwen3.5',
                 '--arm', 'graph_qwen3.5', '--schema', 'unused.md',
                 f'--repetition-penalty={value}']) == 1
    assert 'repetition penalty' in capsys.readouterr().err


@pytest.mark.parametrize('step', ['extract-graph-scenes', 'extract-description-scenes', 'summarize'])
def test_gemini_penalty_is_rejected_before_loading_run(step, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('Gemini penalty must be rejected before loading a Run')

    monkeypatch.setattr('extraction.cli.RunContext.load', forbidden)
    assert main([step, '--run-id', 'unused', '--model', 'gemini', '--arm', 'graph_gemini',
                 '--schema', 'unused.md', '--repetition-penalty', '1.05']) == 1
    assert 'only for Qwen' in capsys.readouterr().err
