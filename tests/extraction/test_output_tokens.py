"""Backend usage is persisted without counting serialized Graph or Summary text."""
from contextlib import contextmanager

import pytest

from extraction import steps
from extraction.backends import GeminiGenerationOutcome
from extraction.failures import FailureLog

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


def extract(context, representation, model='qwen3vl'):
    context.config['extraction'][f'{representation}_repetition_penalty'] = [1.0]
    name = f"{'desc' if representation == 'description' else 'graph'}_{model}"
    result = getattr(steps, f'extract_{representation}_scenes')(
        context, model=model, arm=name,
        schema=f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md")
    return name, result


@pytest.mark.parametrize('model', ['qwen3vl', 'gemini'])
def test_cutoff_failure_retains_tokens_after_failure_log_reload(ready_context, monkeypatch, model):
    install_generators(monkeypatch, 'graph', cutoff='scene')
    arm, result = extract(ready_context, 'graph', model)
    assert result['failure_count'] == 4
    directory = ready_context.scene_arm_dir(arm)
    assert not list(directory.glob('*.jsonl'))
    failures = FailureLog(directory, scenes=True)
    assert all(row['tokens'] == 123 and row['raw_output'] == GRAPH for row in failures.rows.values())
