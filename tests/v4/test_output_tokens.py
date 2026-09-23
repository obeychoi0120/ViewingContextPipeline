"""Backend usage is persisted without counting serialized Graph or Summary text."""
from contextlib import contextmanager

import pytest

from extraction import steps
from extraction.backends import GeminiGenerationOutcome
from extraction.failures import FailureLog
from extraction.scene_storage import read_scene_records, write_scene_records
from extraction.token_usage import gemini_output_tokens, qwen_output_tokens
from pipeline_runtime import read_json, read_jsonl

GRAPH = '[Context]\nmedium: animation\nformat: unknown\ntopics: none\n[Entities]\nnone\n[Actions]\nnone'


def install_generators(monkeypatch, representation, *, raw=False, cutoff=None):
    def response(task):
        scene = bool(task.image_paths)
        text = ((GRAPH if representation == 'graph' else 'A person walks.') if scene
                else 'A short animation.')
        if raw and scene:
            text = 'Not a formatted graph.'
        return text, 123 if scene else 321, cutoff == ('scene' if scene else 'summary')

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                text, tokens, truncated = response(task)
                kwargs['runtime'].current_result = {
                    'output_tokens': tokens, 'prompt_tokens': 999,
                    'finish_reason': 'length' if truncated else 'stop',
                }
                callback(task.task_id, text)
            return {}
        yield generate

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            for task in tasks:
                text, tokens, truncated = response(task)
                callback(GeminiGenerationOutcome(task.task_id, text, response_diagnostics={
                    'usage_metadata': {'candidates_token_count': tokens,
                                       'prompt_token_count': 999, 'thoughts_token_count': 77},
                    'candidates': [{'finish_reason': 'MAX_TOKENS' if truncated else 'STOP'}],
                }))

    monkeypatch.setattr(steps, 'qwen_generator', generator)
    monkeypatch.setattr(steps, 'GeminiWorkerPool', Pool)


def extract(context, representation, model='qwen'):
    context.config['extraction'][f'{representation}_repetition_penalty'] = [1.0]
    name = f"{'desc' if representation == 'description' else 'graph'}_{model}"
    result = getattr(steps, f'extract_{representation}_scenes')(
        context, model=model, arm=name,
        schema=f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md")
    return name, result


@pytest.mark.parametrize('representation,model,raw', [
    ('graph', 'qwen', False), ('graph', 'gemini', False), ('description', 'qwen', False),
    ('graph', 'qwen', True), ('graph', 'gemini', True),
])
def test_scene_output_tokens_and_exact_field_order(ready_context, monkeypatch, representation, model, raw):
    context = ready_context
    install_generators(monkeypatch, representation, raw=raw)
    arm, result = extract(context, representation, model)
    assert result['failure_count'] == 0
    expected = (['content_id', 'scene_idx', 'tokens', 'warning', 'scene_graph']
                if representation == 'graph' else ['content_id', 'scene_idx', 'tokens', 'description'])
    for path in context.scene_arm_dir(arm).glob('*.jsonl'):
        row = read_jsonl(path)[0]
        assert list(row) == expected
        assert row['tokens'] == 123
        if raw:
            assert isinstance(row['scene_graph'], str) and row['warning']
        restored = read_scene_records(path)
        assert restored[0]['tokens'] == 123
        write_scene_records(path, restored)
        assert read_jsonl(path)[0] == row


@pytest.mark.parametrize('model', ['qwen', 'gemini'])
def test_cutoff_failure_retains_tokens_after_failure_log_reload(ready_context, monkeypatch, model):
    install_generators(monkeypatch, 'graph', cutoff='scene')
    arm, result = extract(ready_context, 'graph', model)
    assert result['failure_count'] == 4
    directory = ready_context.scene_arm_dir(arm)
    assert not list(directory.glob('*.jsonl'))
    failures = FailureLog(directory, scenes=True)
    assert all(row['tokens'] == 123 and row['raw_output'] == GRAPH for row in failures.rows.values())


@pytest.mark.parametrize('representation', ['graph', 'description'])
@pytest.mark.parametrize('model', ['qwen', 'gemini'])
@pytest.mark.parametrize('cutoff', [None, 'summary'])
def test_summary_tokens_cover_sources_backends_and_failure_resume(
    ready_context, monkeypatch, representation, model, cutoff
):
    from extraction.errors import ExtractionStepError

    context = ready_context
    install_generators(monkeypatch, representation, cutoff=cutoff)
    arm, _ = extract(context, representation)
    context.config['extraction']['summary_repetition_penalty'] = [1.0]

    def summarize():
        return steps.summarize(context, model=model, arm=arm,
                               schema=f'prompts/summary_{representation}_v5.md')

    if cutoff and model == 'gemini':
        with pytest.raises(ExtractionStepError, match='summaries failed'):
            summarize()
    else:
        summarize()
    directory = context.summary_arm_dir(arm)
    files = list(directory.glob('*.json'))
    assert len(files) == 4
    for path in files:
        doc = read_json(path)
        fields = list(doc)
        assert fields[fields.index('content_id') + 1] == 'tokens'
        assert doc['tokens'] == 321
        assert doc['status'] == ('failed' if cutoff else 'complete')
    if cutoff and model == 'qwen':
        # Rebuild terminal artifacts from persisted failure records, without generation.
        for path in files:
            path.unlink()
        summarize()
        assert all(read_json(path)['tokens'] == 321 for path in files)


def test_unknown_usage_is_not_estimated_or_inherited():
    assert qwen_output_tokens({'output_tokens': 0}) == 0
    assert qwen_output_tokens({}) is None
    assert qwen_output_tokens({'output_tokens': True}) is None
    assert gemini_output_tokens(None) is None
    assert gemini_output_tokens({'usage_metadata': {'total_token_count': 888}}) is None
    assert gemini_output_tokens({'usage_metadata': {'candidatesTokenCount': 17}}) == 17


@pytest.mark.parametrize('value', [-1, True, '100', 1.5])
def test_invalid_persisted_tokens_are_rejected(tmp_path, value):
    with pytest.raises(ValueError, match='tokens'):
        write_scene_records(tmp_path / 'video.jsonl', [{'scene_idx': 0, 'graph': GRAPH, 'tokens': value}])
