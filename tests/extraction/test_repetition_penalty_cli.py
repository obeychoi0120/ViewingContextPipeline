"""CLI overrides reach pending requests while completed outputs remain reusable."""

import pytest

from artifact_io import read_json
from extraction.cli import main


@pytest.mark.parametrize('penalty,initial,updated_value', [
    ('repetition_penalty', 1.12, 1.15), ('presence_penalty', -2.0, 0.0),
])
@pytest.mark.parametrize('model', ['qwen3vl', 'qwen3.5'])
@pytest.mark.parametrize('representation', ['graph', 'description'])
def test_cli_penalty_reaches_scene_and_summary_requests_and_reuse_checks(
    ready_context, fake_models, monkeypatch, model, representation, penalty, initial, updated_value,
):
    context = ready_context
    monkeypatch.setattr('extraction.cli.RunContext.load', lambda _: context)
    arm = f"{'graph' if representation == 'graph' else 'desc'}_{model}"
    for summary in (False, True):
        step = 'summarize' if summary else f'extract-{representation}-scenes'
        version = 8 if representation == 'graph' else (5 if summary else 3)
        schema = f"prompts/{'summary' if summary else 'scene'}_{representation}_v{version}.md"
        args = [step, '--run-id', context.run_id, '--model', model, '--arm', arm,
                '--schema', schema, '--' + penalty.replace('_', '-'), str(initial)]
        assert main(args) == 0
        assert all(getattr(task, penalty) == initial and not task.do_sample for task in fake_models[-1])
        if summary:
            directory = context.summary_arm_dir(arm)
            identity_path = next(directory.glob('*.json'))
            settings = read_json(identity_path)['provenance']['settings']
            if penalty == 'repetition_penalty':
                assert settings['actual_repetition_penalty'] == initial
        else:
            directory = context.scene_arm_dir(arm)
            identity_path = directory / 'generation_identity.json'
            settings = read_json(identity_path)['identity']['settings']
        assert settings['generation'][penalty] == initial
        assert settings['generation']['temperature'] == 0.0
        if penalty == 'repetition_penalty':
            assert settings['repetition_penalty'] == [initial]
        before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        calls = len(fake_models)
        assert main(args) == 0
        assert len(fake_models) == calls
        changed = [*args[:-1], str(updated_value)]
        assert main(changed) == 0
        assert len(fake_models) == calls
        assert before == {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        # Missing outputs use the new settings while other completed outputs stay intact.
        output = next(directory.glob('*.json' if summary else '*.jsonl'))
        output.unlink()
        assert main(changed) == 0
        assert len(fake_models) == calls + 1
        assert all(getattr(task, penalty) == updated_value for task in fake_models[-1])
        assert main([*changed, '--force']) == 0
        assert all(getattr(task, penalty) == updated_value for task in fake_models[-1])
        updated = read_json(identity_path)
        settings = updated['provenance' if summary else 'identity']['settings']
        assert settings['generation'][penalty] == updated_value
        if penalty == 'repetition_penalty':
            assert settings['repetition_penalty'] == [updated_value]


@pytest.mark.parametrize('option,value', [
    ('repetition', '0.9'), ('repetition', '2.1'), ('repetition', 'nan'),
    ('presence', '-2.1'), ('presence', '2.1'), ('presence', 'nan'), ('presence', 'inf'),
])
def test_invalid_penalty_is_rejected_before_loading_run(option, value, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid penalty must be rejected before loading a Run')

    monkeypatch.setattr('extraction.cli.RunContext.load', forbidden)
    assert main(['extract-graph-scenes', '--run-id', 'unused', '--model', 'qwen3.5',
                 '--arm', 'graph_qwen3.5', '--schema', 'unused.md',
                 f'--{option}-penalty={value}']) == 1
    assert f'{option} penalty' in capsys.readouterr().err


@pytest.mark.parametrize('option', ['repetition', 'presence'])
@pytest.mark.parametrize('step', ['extract-graph-scenes', 'extract-description-scenes', 'summarize'])
def test_gemini_penalty_is_rejected_before_loading_run(option, step, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('Gemini penalty must be rejected before loading a Run')

    monkeypatch.setattr('extraction.cli.RunContext.load', forbidden)
    assert main([step, '--run-id', 'unused', '--model', 'gemini', '--arm', 'graph_gemini',
                 '--schema', 'unused.md', f'--{option}-penalty', '1.05']) == 1
    assert 'only for Qwen' in capsys.readouterr().err
