"""Official generation policy stays fixed through failure and explicit reruns."""
from contextlib import contextmanager

import pytest

from artifact_io import read_json, read_jsonl
from extraction import steps

GRAPH_TEXT = "[Context]\nmedium: live_action\nformat: other\ntopics: none\n[Entities]\nnone\n[Actions]\nnone"


@pytest.mark.parametrize('model', ['qwen3vl', 'qwen3.5'])
@pytest.mark.parametrize('representation', ['graph', 'description'])
def test_fixed_scene_policy_retries_only_failed_scenes(ready_context, monkeypatch, model, representation):
    context = ready_context
    source = f"{'graph' if representation == 'graph' else 'desc'}_{model}"
    ids = [v['content_id'] for v in steps.visual_rows(context)]
    calls = []
    succeed = False

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                cid = task.task_id.rsplit(':', 1)[0]
                calls.append((cid, task.repetition_penalty))
                failed = cid != ids[0] and not succeed
                kwargs['runtime'].current_result = {'finish_reason': 'length' if failed else 'stop'}
                callback(task.task_id, GRAPH_TEXT if representation == 'graph' else 'A person walks.')
            return {}
        yield generate

    monkeypatch.setattr(steps, 'qwen_generator', generator)
    extract = getattr(steps, f'extract_{representation}_scenes')
    options = dict(arm=source, model=model, schema=f"prompts/scene_{representation}_v{8 if representation == 'graph' else 2}.md")
    assert extract(context, **options)['failure_count'] == 3
    assert calls == [(cid, 1.0) for cid in ids]
    directory = context.scene_arm_dir(source)
    saved = (directory / f'{ids[0]}.jsonl').read_bytes()
    succeed = True
    assert extract(context, **options)['failure_count'] == 0
    assert calls[4:] == [(cid, 1.0) for cid in ids[1:]]
    assert (directory / f'{ids[0]}.jsonl').read_bytes() == saved
    assert all(not (directory / 'failures' / f'{cid}.jsonl').exists() for cid in ids)
    assert extract(context, **options)['failure_count'] == 0
    assert len(calls) == 7


@pytest.mark.parametrize('model', ['qwen3vl', 'qwen3.5'])
@pytest.mark.parametrize('representation', ['graph', 'description'])
def test_fixed_summary_policy_retries_failures_on_rerun(ready_context, fake_models, monkeypatch, model, representation):
    context = ready_context
    source = f"{'graph' if representation == 'graph' else 'desc'}_{model}"
    getattr(steps, f'extract_{representation}_scenes')(context, arm=source, model=model,
        schema=f"prompts/scene_{representation}_v{8 if representation == 'graph' else 2}.md")
    calls = []
    succeed = False

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                calls.append((task.task_id, task.repetition_penalty))
                kwargs['runtime'].current_result = {'finish_reason': 'stop' if succeed else 'length'}
                callback(task.task_id, 'A person walks.')
            return {}
        yield generate

    monkeypatch.setattr(steps, 'qwen_generator', generator)
    options = dict(arm=source, model=model, schema=f"prompts/summary_{representation}_v{8 if representation == 'graph' else 5}.md")
    assert steps.summarize(context, **options)['failure_count'] == 4
    directory = context.summary_arm_dir(source)
    assert all(row['repetition_penalty'] == 1.0 for row in read_jsonl(directory / 'failures.jsonl'))
    assert all(read_json(path)['text'] == '' for path in directory.glob('*.json'))
    succeed = True
    assert steps.summarize(context, **options)['failure_count'] == 0
    assert len(calls) == 8 and all(penalty == 1.0 for _, penalty in calls)
    assert steps.summarize(context, **options)['failure_count'] == 0
    assert len(calls) == 8
