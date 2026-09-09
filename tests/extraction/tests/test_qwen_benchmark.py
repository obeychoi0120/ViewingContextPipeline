from __future__ import annotations

from dataclasses import asdict

import pytest

from benchmarks import qwen_benchmark as benchmark
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.qwen_config import qwen_settings
from pipeline_runtime import read_json


@pytest.mark.parametrize("fail", [False, True])
def test_benchmark_excludes_warmup_and_records_partial_failures(tmp_path, monkeypatch, fail):
    submissions = []
    shutdown = []
    manifest = {"requests": [asdict(QwenGenerationTask(str(i), (), "prompt", 32)) for i in range(4)],
                "image_hashes": {}, "model_path": "model", "image_limit": 6,
                "stage": "description-scenes"}

    class Pool:
        def __init__(self, count, model, **kwargs):
            kwargs["on_runtime"]({"gpu_id": "0"})

        def wait_ready(self):
            pass

        def generate(self, tasks, callback):
            submissions.append([task.task_id for task in tasks])
            for task in tasks:
                if fail and task.task_id == "3":
                    raise RuntimeError("CUDA out of memory")
                self.last_result = {"output_tokens": 2, "prompt_tokens": 8}
                callback(task.task_id, "a visible person")

        def close(self):
            shutdown.append("close")

        def abort(self):
            shutdown.append("abort")

    monkeypatch.setattr(benchmark, "QwenWorkerPool", Pool)
    monkeypatch.setattr(benchmark, "_visible_gpu_ids", lambda _: ["0"])
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda *args, **kwargs: "GPU-uuid, 1000\n")
    output = tmp_path / "report.json"
    if fail:
        with pytest.raises(RuntimeError, match="out of memory"):
            benchmark.measure(manifest, "vllm", qwen_settings(), 1, output)
    else:
        benchmark.measure(manifest, "vllm", qwen_settings(), 1, output)
    report = read_json(output)
    assert submissions == [["0"], ["1", "2", "3"]]
    assert [r["task_id"] for r in report["results"]] == (["1", "2"] if fail else ["1", "2", "3"])
    assert report["startup_seconds"] is not None
    assert report["warmup_seconds"] is not None
    assert report["failure_rate"] == (1 / 3 if fail else 0)
    assert report["peak_gpu_memory_mib"] == {"GPU-uuid": 1000}
    assert shutdown == (["abort"] if fail else ["close"])


def test_output_validation_uses_stage_contract():
    assert benchmark.check_output("graph-scenes", "not json")
    assert benchmark.check_output("graph-summary-gemini", "not structured")
    assert benchmark.check_output("description-summary", "not structured")
    assert benchmark.check_output("description-scenes", "")
    assert benchmark.check_output("description-scenes", "person running") is None
