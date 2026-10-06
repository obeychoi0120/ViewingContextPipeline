from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

import extraction.backends.qwen_workers as workers
from extraction.backends.qwen_workers import QwenGenerationTask, QwenWorkerPool


class Channel:
    def __init__(self, values=()):
        self.values = list(values)
        self.closed = self.cancelled = False

    def get(self, timeout=None):
        if not self.values:
            raise queue.Empty
        return self.values.pop(0)

    def put(self, value):
        self.values.append(value)

    put_nowait = put

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        self.cancelled = True


class Process:
    def __init__(self, alive=True):
        self.alive = alive
        self.terminated = self.killed = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.alive = False

    def join(self, timeout):
        pass


@pytest.fixture
def pool_factory(monkeypatch):
    def create(count=2, events=(), **kwargs):
        task_queues = [Channel() for _ in range(count)]
        results = Channel(events)
        processes = [Process() for _ in range(count)]
        monkeypatch.setattr(workers, "_visible_gpu_ids", lambda: [str(i) for i in range(count)])
        monkeypatch.setattr(workers.mp, "get_context", lambda method: SimpleNamespace(Queue=lambda: results))
        monkeypatch.setattr(workers, "_start_worker",
                            lambda context, index, *args: (task_queues[index], processes[index]))
        monkeypatch.setattr(QwenWorkerPool, "_signal_groups", lambda *args: None)
        return QwenWorkerPool("model", settings={"max_num_seqs": 1}, **kwargs)
    return create


def task(name):
    return QwenGenerationTask(name, (), name, 32)


def event(name, index=0):
    return {"ok": True, "kind": "result", "worker_index": index, "gpu_id": str(index),
            "task_id": name, "text": name.upper(), "output_tokens": 2}


def test_completion_refills_fast_gpu_before_saving_without_batch_barrier(pool_factory):
    # Two slots per GPU. GPU 1 completes three requests before GPU 0 completes any.
    pool = pool_factory(events=[event("b", 1), event("d", 1), event("e", 1),
                                event("a", 0), event("c", 0), event("f", 1)])
    completed = []
    stats = []
    pool.on_progress = stats.append

    def save(name, text):
        completed.append((name, text))
        if name == "b":
            assert [t.task_id for t in pool._task_queues[1].values] == ["b", "d", "e"]
            assert [t.task_id for t in pool._task_queues[0].values] == ["a", "c"]

    assert pool.generate((task(x) for x in "abcdef"), save) == {}
    assert completed == [(x, x.upper()) for x in "bdeacf"]
    assert stats[-1]["completed"] == 6
    assert stats[-1]["inflight"] == 0
    assert [t.task_id for t in pool._task_queues[1].values] == ["b", "d", "e", "f"]
    pool.abort()


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), OSError("disk full")])
def test_callback_failure_keeps_prior_saves_and_stops_all_workers(pool_factory, failure):
    pool = pool_factory(count=1, events=[event("a"), event("b")])
    saved = []

    def save(name, text):
        if name == "b":
            raise failure
        saved.append(name)

    with pytest.raises(type(failure)):
        pool.generate([task("a"), task("b")], save)
    assert saved == ["a"]
    assert pool._closed
    assert all(p.terminated and p.killed for p in pool._processes)
    assert all(q.closed and q.cancelled for q in [*pool._task_queues, pool._result_queue])


@pytest.mark.parametrize("events,dead,match", [
    ([event("unknown")], False, "Unexpected"),
    ([{"ok": False, "worker_index": 0, "error": "CUDA out of memory"}], False, "out of memory"),
    ([], True, "exited before finishing"),
])
def test_engine_failure_is_fatal(pool_factory, events, dead, match):
    pool = pool_factory(count=1, events=events)
    pool._processes[0].alive = not dead
    with pytest.raises(RuntimeError, match=match):
        pool.generate([task("a")])
    assert pool._closed
