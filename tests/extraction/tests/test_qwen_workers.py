from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

import extraction.backends.qwen as qwen_module
import extraction.evidence as evidence_module
from extraction.backends.qwen_workers import (
    QwenGenerationTask,
    _visible_gpu_ids,
    _worker_main,
    assign_worker_indices,
)


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


def test_worker_selects_cuda_device_and_reuses_one_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = []

    class Backend:
        @classmethod
        def from_pretrained(cls, model_path, *, use_fc_patch):
            initialized.append((model_path, use_fc_patch, os.environ["CUDA_VISIBLE_DEVICES"]))
            return cls()

        def generate(self, images, prompt, max_new_tokens):
            return f"{images}:{prompt}:{max_new_tokens}"

    class Queue:
        def __init__(self, values=()):
            self.values = list(values)

        def get(self):
            return self.values.pop(0)

        def put(self, value):
            self.values.append(value)

    task = QwenGenerationTask("task", ("a.png",), "prompt", 32)
    result_queue = Queue()
    monkeypatch.setattr(qwen_module, "QwenBackend", Backend)
    monkeypatch.setattr(evidence_module, "load_images", lambda paths: paths)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original")

    _worker_main(2, "7", "model", Queue([task, None]), result_queue)

    assert initialized == [("model", True, "7")]
    assert result_queue.values[0]["text"] == "['a.png']:prompt:32"
