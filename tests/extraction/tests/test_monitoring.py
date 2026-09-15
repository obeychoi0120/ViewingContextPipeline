from __future__ import annotations


import pytest

import extraction.summary_executor as summary_executor
from extraction.backends.qwen_workers import QwenGenerationTask


def test_generator_uses_visible_gpu_pool_and_passes_completed_request_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from extraction.qwen_runtime import QwenRuntime

    runtime = QwenRuntime()

    class Pool:
        def __init__(self, model_path, **kwargs):
            kwargs["on_runtime"]({"worker_index": 0, "gpu_id": "0"})

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def generate(self, tasks, callback):
            for task in reversed(tasks):
                assert task.repetition_penalty == {"first": 1.05, "second": 1.2}[task.prompt]
                self.last_result = {"task_id": task.task_id, "worker_index": 0}
                callback(task.task_id, f"result:{task.prompt}")
                assert runtime.current_result is None
            return {}

    monkeypatch.setattr(summary_executor, "QwenWorkerPool", Pool)
    tasks = [
        QwenGenerationTask("a", (), "first", 10, repetition_penalty=1.05),
        QwenGenerationTask("b", (), "second", 10, repetition_penalty=1.2),
    ]
    completed = []

    def complete(task_id, text):
        assert runtime.current_result["task_id"] == task_id
        completed.append((task_id, text))

    with summary_executor.qwen_generator(model_path=tmp_path, runtime=runtime) as generate:
        results = generate(tasks, complete)

    assert completed == [("b", "result:second"), ("a", "result:first")]
    assert results == {}
