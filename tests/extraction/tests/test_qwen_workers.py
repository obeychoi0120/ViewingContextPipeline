from __future__ import annotations

import asyncio
import os
import queue
import sys
from types import SimpleNamespace

import pytest

import extraction.backends.qwen as qwen_module
import extraction.backends.qwen_workers as workers
from extraction.backends.qwen import QwenOutput
from extraction.backends.qwen_workers import QwenGenerationTask, QwenWorkerPool, _visible_gpu_ids


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
        monkeypatch.setattr(workers, "_visible_gpu_ids", lambda count: [str(i) for i in range(count)])
        monkeypatch.setattr(workers.mp, "get_context", lambda method: SimpleNamespace(Queue=lambda: results))
        monkeypatch.setattr(workers, "_start_worker",
                            lambda context, index, *args: (task_queues[index], processes[index]))
        monkeypatch.setattr(QwenWorkerPool, "_signal_groups", lambda *args: None)
        return QwenWorkerPool(count, "model", settings={"max_num_seqs": 1}, **kwargs)
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


def test_initial_admission_is_bounded_and_runtime_events_are_delivered(pool_factory):
    ready = []
    pool = pool_factory(count=1, events=[
        {"kind": "started", "worker_index": 0, "process_group": None},
        {"kind": "ready", "ok": True, "worker_index": 0},
        event("a"), event("b"), event("c"),
    ], on_runtime=ready.append)
    consumed = []

    def tasks():
        for name in "abc":
            consumed.append(name)
            yield task(name)

    original_get = pool._result_queue.get

    def get(timeout):
        if not ready:
            assert consumed == ["a", "b"]
        return original_get(timeout)

    pool._result_queue.get = get
    assert pool.generate(tasks()) == {"a": "A", "b": "B", "c": "C"}
    assert len(ready) == 1
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


def test_duplicate_ids_rejected(pool_factory):
    pool = pool_factory(count=1)
    with pytest.raises(ValueError, match="unique"):
        pool.generate([task("a"), task("a")])


def test_engine_readiness_can_be_measured_before_request_submission(pool_factory):
    pool = pool_factory(count=1, events=[
        {"kind": "started", "worker_index": 0, "process_group": None},
        {"kind": "ready", "worker_index": 0, "ok": True}, event("a"),
    ])
    pool.wait_ready()
    assert pool._task_queues[0].values == []
    assert pool.generate([task("a")]) == {"a": "A"}
    pool.wait_ready()  # Already-ready engines are reused without another queue wait.
    pool.abort()


@pytest.mark.parametrize("gpu_count", [0, -1, True])
def test_invalid_gpu_count(gpu_count):
    with pytest.raises(ValueError, match="positive"):
        _visible_gpu_ids(gpu_count)


def test_visible_devices_preserved(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 3)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7,9")
    assert _visible_gpu_ids(2) == ["4", "7"]
    with pytest.raises(RuntimeError, match="only 3"):
        _visible_gpu_ids(4)


def test_worker_reuses_one_engine_and_passes_requests_concurrently(monkeypatch):
    initialized, closed, active = [], [], []
    peak = 0

    class Backend:
        runtime_info = {"gpu_name": "fake", "settings": {}}

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            initialized.append((path, os.environ["CUDA_VISIBLE_DEVICES"], kwargs))
            return cls()

        async def generate(self, item):
            nonlocal peak
            active.append(item.task_id)
            peak = max(peak, len(active))
            if item.task_id == "a":
                # A cannot finish until B has been admitted to the same GPU.
                while "b" not in active:
                    await asyncio.sleep(0.001)
            await asyncio.sleep(0.02)
            active.remove(item.task_id)
            return QwenOutput(item.prompt, 8, 2)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(qwen_module, "QwenBackend", Backend)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original")
    incoming, outgoing = Channel([task("a"), task("b"), None]), Channel()
    workers._worker_main(0, "7", "model", incoming, outgoing, {"max_num_seqs": 1}, 6)
    assert initialized == [("model", "7", {"settings": {"max_num_seqs": 1}, "image_limit": 6})]
    assert peak == 2
    assert closed == [True]
    assert {v["task_id"] for v in outgoing.values if v.get("kind") == "result"} == {"a", "b"}
    assert outgoing.values[1]["kind"] == "ready"


def test_spawn_uses_non_daemon_and_bounded_queue():
    calls = []

    class Spawned:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def start(self):
            calls.append("start")

    context = SimpleNamespace(Queue=lambda **kwargs: kwargs, Process=Spawned)
    tasks, _ = workers._start_worker(context, 0, "2", "model", Channel(), {"max_num_seqs": 3}, 6)
    assert tasks == {"maxsize": 7}
    assert calls[0]["daemon"] is False
    assert calls[0]["args"][-1] == 6
    assert calls[1] == "start"


def test_partial_startup_failure_disposes_first_worker(pool_factory, monkeypatch):
    pool = pool_factory(count=1)
    first_queue, first_process = pool._task_queues[0], pool._processes[0]
    pool.abort()
    first_process.alive = True

    def start(context, index, *args):
        if index:
            raise RuntimeError("spawn failed")
        return first_queue, first_process

    monkeypatch.setattr(workers, "_start_worker", start)
    with pytest.raises(RuntimeError, match="spawn failed"):
        QwenWorkerPool(2, "model")
    assert first_process.killed
    assert first_queue.closed
