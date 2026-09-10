from dataclasses import replace
import io
import multiprocessing as mp

import numpy as np
import pytest
from tqdm import tqdm

from pipeline_runtime import read_json, write_json
from test_rolling import full_context as full_context
from validation.recommendation_contracts import RECOMMENDATION_ARMS
from validation.rolling_data import EventTable, iter_jsonl
from validation.rolling_recommendation import (
    combination_dir, prepare_split, run_combination, run_rolling, worker_devices,
)
from validation.rolling_workers import run_parallel
from validation.steps import validation_config


class Progress:
    def __init__(self):
        self.completed = 0
        self.fp = io.StringIO()

    def update(self, count):
        self.completed += count

    def write(self, value, *, file):
        file.write(value + "\n")

    def refresh(self):
        pass


def prepare_embeddings(context):
    context.representations_dir.mkdir(parents=True)
    write_json(context.representations_dir / "item_index.json", {str(i): i - 1 for i in range(1, 5)})
    write_json(context.representations_dir / "graph_gemini_fallbacks.json", {"fallbacks": []})
    for branch in RECOMMENDATION_ARMS.values():
        values = np.random.default_rng(4).normal(size=(4, 1024)).astype(np.float32)
        np.savez(context.representations_dir / f"{branch}_embeddings.npz", values=values)


def two_jobs(context):
    split = context.require_ready_cohort()["plan"]["splits"][0]
    return [(split, {
        "run_id": context.run_id, "evaluation_date": split["evaluation_date"],
        "seed": seed, "arm": "SASRec_METADATA",
    }, "metadata") for seed in (42, 43)]


@pytest.mark.torch
def test_spawned_training_matches_serial_parameters_and_metrics(full_context):
    import torch

    context = full_context
    prepare_embeddings(context)
    jobs = two_jobs(context)
    stream = io.StringIO()
    before_completion = []
    children_before = {child.pid for child in mp.active_children()}
    with tqdm(total=2, desc="Rolling recommendation", file=stream) as progress:
        original_write = progress.write

        def observe_write(message, **kwargs):
            original_write(message, **kwargs)
            if "epoch=" in message and progress.n == 0:
                before_completion.append(stream.getvalue().rsplit(message, 1)[-1])

        progress.write = observe_write
        assert run_parallel(context, jobs, ["cpu", "cpu"], progress) == 2
        assert progress.n == 2
    assert before_completion
    assert all("Rolling recommendation:" in tail and "0/2" in tail for tail in before_completion)
    assert "refit epochs=" in stream.getvalue() and "test device=" in stream.getvalue()
    assert {child.pid for child in mp.active_children()} == children_before

    table = EventTable(iter_jsonl(context.cohort_dir / "events.jsonl"))
    config = validation_config(context)
    serial = replace(context, run_root=context.run_root / "serial")
    prepare_embeddings(serial)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for split, identity, branch in jobs:
            run_combination(serial, config, table, split, identity, branch,
                            prepare_split(table, split), torch.device("cpu"))
            parallel_dir = combination_dir(context, split["evaluation_date"], identity["seed"], identity["arm"])
            serial_dir = combination_dir(serial, split["evaluation_date"], identity["seed"], identity["arm"])
            for filename in ("complete.json", "per_event_metrics.jsonl"):
                assert (parallel_dir / filename).read_bytes() == (serial_dir / filename).read_bytes()
            a, b = (read_json(path / "training.json") for path in (parallel_dir, serial_dir))
            a.pop("elapsed_seconds")
            b.pop("elapsed_seconds")
            assert a == b
            a, b = (torch.load(path / "sasrec.pt", weights_only=True)["state_dict"]
                    for path in (parallel_dir, serial_dir))
            assert a.keys() == b.keys()
            assert all(torch.equal(a[key], b[key]) for key in a)
    finally:
        torch.set_num_threads(original_threads)


@pytest.mark.torch
def test_worker_failure_is_reported_and_all_children_are_joined(full_context):
    context = full_context
    children_before = {child.pid for child in mp.active_children()}
    # Missing embeddings fail inside a real spawned child, after task dispatch.
    with pytest.raises(RuntimeError, match="rolling worker failed:.*device=cpu"):
        run_parallel(context, two_jobs(context), ["cpu", "cpu"], Progress())
    assert {child.pid for child in mp.active_children()} == children_before
    assert not list(context.recommendations_dir.rglob("complete.json"))


@pytest.mark.torch
def test_parent_interrupt_keeps_completed_artifacts_for_resume(full_context, monkeypatch):
    context = full_context
    prepare_embeddings(context)

    class InterruptProgress(Progress):
        def update(self, count):
            raise KeyboardInterrupt

    children_before = {child.pid for child in mp.active_children()}
    with pytest.raises(KeyboardInterrupt):
        run_parallel(context, two_jobs(context), ["cpu", "cpu"], InterruptProgress())
    assert {child.pid for child in mp.active_children()} == children_before
    protected = {path: path.read_bytes() for path in context.recommendations_dir.rglob("complete.json")}
    assert protected
    monkeypatch.setattr("validation.rolling_recommendation.worker_devices", lambda *_: ["cpu", "cpu"])
    pending = []

    def capture(context, jobs, devices, progress):
        pending.extend(jobs)
        return len(jobs)

    monkeypatch.setattr("validation.rolling_workers.run_parallel", capture)
    result = run_rolling(context)
    assert result["skipped"] == len(protected)
    assert len(pending) == 84 - len(protected)
    assert all(path.read_bytes() == original for path, original in protected.items())
    assert all(combination_dir(context, split["evaluation_date"], identity["seed"], identity["arm"])
               / "complete.json" not in protected for split, identity, _ in pending)


def abrupt_worker(*args):
    import os
    os._exit(9)


def test_abrupt_worker_exit_does_not_hang(full_context, monkeypatch):
    monkeypatch.setattr("validation.rolling_workers.combination_worker", abrupt_worker)
    children_before = {child.pid for child in mp.active_children()}
    with pytest.raises(RuntimeError, match="workers? .*exit"):
        run_parallel(full_context, two_jobs(full_context), ["cpu", "cpu"], Progress())
    assert {child.pid for child in mp.active_children()} == children_before


def test_dispatch_is_unique_and_skips_completed_work(full_context, monkeypatch):
    context = full_context
    prepare_embeddings(context)
    monkeypatch.setattr("validation.rolling_recommendation.worker_devices",
                        lambda *_: ["cuda:0", "cuda:1"] * 2)
    reused = ("2022-09-05", 42, "SASRec_METADATA")
    monkeypatch.setattr("validation.rolling_recommendation.combination_complete",
                        lambda directory, identity, count:
                        tuple(identity[k] for k in ("evaluation_date", "seed", "arm")) == reused)
    dispatched = []

    def run(context, jobs, devices, progress):
        assert devices == ["cuda:0", "cuda:1"] * 2
        keys = [tuple(identity[k] for k in ("evaluation_date", "seed", "arm"))
                for _, identity, _ in jobs]
        assert len(set(keys)) == len(keys)
        dispatched.append(keys)
        return len(jobs)

    monkeypatch.setattr("validation.rolling_workers.run_parallel", run)
    assert run_rolling(context, gpus=2, workers_per_gpu=2) == {
        "stage": "run-recommendation", "completed": 83, "skipped": 1,
    }
    assert reused not in dispatched[-1]
    assert run_rolling(context, force=True, gpus=2, workers_per_gpu=2)["completed"] == 84
    assert reused in dispatched[-1]


@pytest.mark.torch
def test_gpu_assignment_respects_visible_devices(monkeypatch):
    monkeypatch.setattr("validation.model.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("validation.model.torch.cuda.device_count", lambda: 4)
    assert worker_devices(4, 2) == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"] * 2
    assert worker_devices(None, 1) == ["cuda:0"]
    with pytest.raises(ValueError, match="only 4"):
        worker_devices(5, 1)
    with pytest.raises(ValueError, match="positive"):
        worker_devices(4, 0)
    monkeypatch.setattr("validation.model.torch.cuda.is_available", lambda: False)
    assert worker_devices(None, 1) == ["cpu"]
    for gpus, workers in [(1, 1), (None, 2)]:
        with pytest.raises(ValueError, match="only 0"):
            worker_devices(gpus, workers)
