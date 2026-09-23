from contextlib import contextmanager

import pytest


@pytest.mark.parametrize("representation", ["graph", "description"])
def test_summary_publish_error_requires_regeneration_without_journal(
    ready_context,
    fake_models,
    monkeypatch,
    representation,
):
    import extraction.steps as steps
    import extraction.summary_executor as executor

    context = ready_context
    extract = getattr(steps, f"extract_{representation}_scenes")
    extract(
        context,
        arm=f"{'graph' if representation == 'graph' else 'desc'}_qwen",
        model="qwen",
        schema=f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md",
    )
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                calls.append(task.task_id)
                callback(task.task_id, "A person walks outdoors.")
            return {}

        yield generate

    monkeypatch.setattr(steps, "qwen_generator", generator)
    original = executor.atomic_write_json

    def fault(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(executor, "atomic_write_json", fault)
    summarize = getattr(steps, f"summarize_{representation}")
    options = {
        "arm": f"{'graph' if representation == 'graph' else 'desc'}_qwen",
        "schema": f"prompts/summary_{representation}_v5.md",
    }
    with pytest.raises(OSError, match="disk full"):
        summarize(context, model="qwen", **options)
    directory = context.summary_arm_dir(f"{'graph' if representation == 'graph' else 'desc'}_qwen")
    assert not list(directory.rglob("*.json"))
    monkeypatch.setattr(executor, "atomic_write_json", original)
    assert summarize(context, model="qwen", **options)["failure_count"] == 0
    assert calls[0] == calls[1] and len(calls) == 5
