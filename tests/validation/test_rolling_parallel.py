from dataclasses import replace
import io
import multiprocessing as mp
import shutil

import numpy as np
import pytest
from tqdm import tqdm

from pipeline_runtime import read_json, write_json
from arm_registry import registry
from validation.rolling_data import EventTable
from validation.selection import load_validation_cohort, prepare_validation_cohort, training_signature
from validation.rolling_recommendation import (
    combination_dir, prepare_split, run_combination, run_rolling,
)
from validation.rolling_workers import run_parallel
from validation.steps import validation_config


@pytest.fixture
def full_context(ready_context, fake_models):
    import extraction.steps as steps
    for source in ("qwen", "gemini"):
        for kind in ("description", "graph"):
            getattr(steps, f"extract_{kind}_scenes")(
                ready_context, model=source,
                schema=f"prompts/{kind}_scene_v{'3' if kind == 'graph' else '2'}.md")
            getattr(steps, f"summarize_{kind}")(
                ready_context, source=source, model="qwen", schema=f"prompts/{kind}_summary_v4.md")
    prepare_validation_cohort(ready_context, "qwen")
    return ready_context


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
    context.representations_dir.mkdir(parents=True, exist_ok=True)
    write_json(context.representations_dir / "item_index.json", {str(i): i - 1 for i in range(1, 5)})
    write_json(context.representations_dir / "graph_gemini_fallbacks.json", {"fallbacks": []})
    for branch in registry(context.config):
        values = np.random.default_rng(4).normal(size=(4, 1024)).astype(np.float32)
        np.savez(context.representations_dir / f"{branch}_embeddings.npz", values=values)


def two_jobs(context):
    cohort = load_validation_cohort(context)
    split = cohort["plan"]["splits"][0]
    training_hash = training_signature(context, cohort, validation_config(context))
    return [(split, {
        "run_id": context.run_id, "evaluation_date": split["evaluation_date"],
        "seed": seed, "arm": "metadata", "training_input_hash": training_hash,
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

    table = EventTable(load_validation_cohort(context)["events"])
    config = validation_config(context)
    serial = replace(context, run_root=context.run_root.parent / "serial")
    shutil.copytree(context.run_root / "extraction", serial.run_root / "extraction")
    prepare_validation_cohort(serial, "qwen")
    prepare_embeddings(serial)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for split, identity, branch in jobs:
            run_combination(serial, config, table, split, identity, branch,
                            prepare_split(table, split), torch.device("cpu"))
            parallel_dir = combination_dir(context, split["evaluation_date"], identity["seed"], identity["arm"])
            serial_dir = combination_dir(serial, split["evaluation_date"], identity["seed"], identity["arm"])
            for filename in ("per_event_metrics.jsonl",):
                assert (parallel_dir / filename).read_bytes() == (serial_dir / filename).read_bytes()
            complete_a, complete_b = (read_json(path / "complete.json") for path in (parallel_dir, serial_dir))
            complete_a.pop("checksums")
            complete_b.pop("checksums")
            assert complete_a == complete_b
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
    monkeypatch.setattr("validation.rolling_recommendation.verify_representations", lambda *a, **k: None)
    monkeypatch.setattr("validation.representation_provenance.recommendation_identity", lambda *a: {})
    result = run_rolling(context)
    assert result["skipped"] == len(protected)
    assert len(pending) == 105 - len(protected)
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
    monkeypatch.setattr("validation.rolling_recommendation.require_torch", lambda: None)
    context = full_context
    prepare_embeddings(context)
    monkeypatch.setattr("validation.rolling_recommendation.worker_devices",
                        lambda *_: ["cuda:0", "cuda:1"] * 2)
    reused = ("2022-09-05", 42, "metadata")
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
    monkeypatch.setattr("validation.rolling_recommendation.verify_representations", lambda *a, **k: None)
    monkeypatch.setattr("validation.representation_provenance.recommendation_identity", lambda *a: {})
    assert run_rolling(context, workers_per_gpu=2) == {
        "stage": "run-recommendation", "completed": 104, "skipped": 1,
    }
    assert reused not in dispatched[-1]
    assert run_rolling(context, force=True, workers_per_gpu=2)["completed"] == 105
    assert reused in dispatched[-1]
