from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

import extraction.backends.qwen as qwen_module
import extraction.backends.qwen_workers as workers
import extraction.evidence as evidence_module
from extraction.backends.qwen_workers import (
    QwenGenerationTask,
    QwenWorkerPool,
    _visible_gpu_ids,
    _worker_main,
    assign_worker_indices,
)


@pytest.fixture
def pool_factory(monkeypatch):
    def create(task_queues, result_queue, processes):
        monkeypatch.setattr(
            workers, "_visible_gpu_ids", lambda count: [str(i) for i in range(count)]
        )
        monkeypatch.setattr(
            workers.mp, "get_context", lambda method: SimpleNamespace(Queue=lambda: result_queue)
        )
        monkeypatch.setattr(
            workers,
            "_start_worker",
            lambda context, index, gpu, model, result: (task_queues[index], processes[index]),
        )
        return QwenWorkerPool(len(task_queues), "model")

    return create


def test_tasks_are_assigned_round_robin_by_gpu_count() -> None:
    assert assign_worker_indices(7, 3) == [0, 1, 2, 0, 1, 2, 0]


@pytest.mark.parametrize("gpu_count", [0, -1])
def test_gpu_count_must_be_positive(gpu_count: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        assign_worker_indices(1, gpu_count)


def test_gpu_count_selects_first_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 3))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7,9")
    assert _visible_gpu_ids(2) == ["4", "7"]


@pytest.mark.parametrize("penalty", [1.0, 1.05, 1.25])
def test_worker_selects_cuda_device_and_reuses_one_model(
    monkeypatch: pytest.MonkeyPatch,
    penalty: float,
) -> None:
    initialized = []

    class Backend:
        @classmethod
        def from_pretrained(cls, model_path, *, use_fc_patch):
            initialized.append((model_path, use_fc_patch, os.environ["CUDA_VISIBLE_DEVICES"]))
            return cls()

        def generate(self, images, prompt, max_new_tokens, **generation):
            assert generation == {
                "do_sample": False,
                "seed": None,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "repetition_penalty": penalty,
            }
            return f"{images}:{prompt}:{max_new_tokens}"

    class Queue:
        def __init__(self, values=()):
            self.values = list(values)

        def get(self):
            return self.values.pop(0)

        def put(self, value):
            self.values.append(value)

    task = QwenGenerationTask("task", ("a.png",), "prompt", 32, repetition_penalty=penalty)
    result_queue = Queue()
    monkeypatch.setattr(qwen_module, "QwenBackend", Backend)
    monkeypatch.setattr(evidence_module, "load_images", lambda paths: paths)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original")

    _worker_main(2, "7", "model", Queue([task, None]), result_queue)

    assert initialized == [("model", True, "7")]
    assert result_queue.values[0]["text"] == "['a.png']:prompt:32"


def test_pool_callback_uses_worker_completion_order(pool_factory) -> None:
    class TaskQueue:
        def __init__(self) -> None:
            self.values = []

        def put(self, value) -> None:
            self.values.append(value)

    class ResultQueue:
        def __init__(self) -> None:
            self.values = [
                {"ok": True, "task_id": "b", "text": "second"},
                {"ok": True, "task_id": "a", "text": "first"},
            ]

        def get(self, timeout):
            return self.values.pop(0)

    pool = pool_factory([TaskQueue(), TaskQueue()], ResultQueue(), [None, None])
    tasks = [
        QwenGenerationTask("a", (), "a", 1),
        QwenGenerationTask("b", (), "b", 1),
    ]
    completed = []

    results = pool.generate(
        tasks,
        lambda task_id, text: completed.append((task_id, text)),
    )

    assert completed == [("b", "second"), ("a", "first")]
    assert results == {}


def test_pool_interrupt_terminates_then_kills_stubborn_workers(pool_factory) -> None:
    class TaskQueue:
        def __init__(self) -> None:
            self.cancelled = False
            self.closed = False

        def put(self, value) -> None:
            return None

        def cancel_join_thread(self) -> None:
            self.cancelled = True

        def close(self) -> None:
            self.closed = True

    class InterruptingResultQueue(TaskQueue):
        def get(self, timeout):
            raise KeyboardInterrupt

    class StubbornProcess:
        def __init__(self) -> None:
            self.alive = True
            self.terminate_calls = 0
            self.kill_calls = 0
            self.join_timeouts = []

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1
            self.alive = False

        def join(self, timeout) -> None:
            self.join_timeouts.append(timeout)

    task_queue = TaskQueue()
    result_queue = InterruptingResultQueue()
    process = StubbornProcess()
    pool = pool_factory([task_queue], result_queue, [process])

    with pytest.raises(KeyboardInterrupt):
        pool.generate([QwenGenerationTask("a", (), "a", 1)])

    assert pool._closed is True
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.join_timeouts == [0.2, 0.2]
    assert task_queue.cancelled is True
    assert task_queue.closed is True
    assert result_queue.cancelled is True
    assert result_queue.closed is True


def test_pool_starts_workers_on_construction_and_closes_once(monkeypatch):
    events = []
    queues = []

    class Queue:
        def __init__(self):
            self.index = len(queues)
            queues.append(self)

        def put(self, value):
            events.append(("put", self.index, value))

        def close(self):
            events.append(("close", self.index))

    class Process:
        def __init__(self, *, target, args, daemon):
            assert target is workers._worker_main
            index, gpu, model, task_queue, result_queue = args
            assert gpu == str(index + 4)
            assert model == "model"
            assert task_queue is queues[index + 1]
            assert result_queue is queues[0]
            assert daemon is True
            self.index = index

        def start(self):
            events.append(("start", self.index))

        def join(self, timeout):
            events.append(("join", self.index, timeout))

        def is_alive(self):
            return False

    def get_context(method):
        assert method == "spawn"
        return SimpleNamespace(Queue=Queue, Process=Process)

    monkeypatch.setattr(workers, "_visible_gpu_ids", lambda count: ["4", "5"])
    monkeypatch.setattr(workers.mp, "get_context", get_context)
    pool = QwenWorkerPool(2, "model")
    assert events == [("start", 0), ("start", 1)]
    with pool:
        pass
    pool.close()
    assert events == [
        ("start", 0),
        ("start", 1),
        ("put", 1, None),
        ("put", 2, None),
        ("join", 0, 5),
        ("join", 1, 5),
        ("close", 1),
        ("close", 2),
        ("close", 0),
    ]
