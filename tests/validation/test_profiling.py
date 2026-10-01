"""Profiler must preserve optimization/evaluation and remain inert when disabled."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from graph_fixture import synthetic_store  # noqa: E402
from validation.config import GraphExecutionConfig  # noqa: E402
from validation.graph_model import GraphSASRec  # noqa: E402
from validation.model import SASRec, seed_everything  # noqa: E402
from validation.profiling import RunProfiler, is_profiling, sampled, span  # noqa: E402
from validation.rolling_data import EventTable, DAY  # noqa: E402
from validation.rolling_recommendation import train_epoch, rank_batches  # noqa: E402


@pytest.fixture(
    params=["cpu"] + ([os.environ["GRAPH_TEST_CUDA"]] if os.environ.get("GRAPH_TEST_CUDA") else [])
)
def device(request):
    torch.set_num_threads(1)
    return torch.device(request.param)


def profiler(path, device="cpu", every=4, *, operators=False):
    path.mkdir(exist_ok=True)
    return RunProfiler(
        path, {"evaluation_date": "test", "seed": 42, "arm": "graph_qwen"}, device, every,
        operators=operators,
    )


def records(path):
    return [json.loads(line) for line in (path / "profile.jsonl").read_text().splitlines()]


def test_disabled_is_inert_and_sampling_resets_after_error(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU/disabled profiling must not access CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", forbidden)
    monkeypatch.setattr(torch.profiler, "profile", forbidden)
    with sampled(None, "selection", 1, 1), span("anything"):
        assert not is_profiling()
    assert not list(tmp_path.iterdir())
    p = profiler(tmp_path)
    for batch in range(1, 10):
        with p.sample("selection", 1, batch):
            assert is_profiling() == (batch <= 3 or batch % 4 == 0)
            with span("parent"):
                with span("child", cuda=False):
                    pass
    with pytest.raises(ValueError, match="original failure"):
        with p.sample("selection", 2, 1):
            with span("failure"):
                raise ValueError("original failure")
    assert not is_profiling()
    rows = records(tmp_path)
    assert [r["batch"] for r in rows if r.get("status") == "ok"] == [1, 2, 3, 4, 8]
    row = rows[1]
    assert row["seconds"]["parent"] >= row["seconds"]["parent/child"]
    assert "exclusive_seconds" not in row
    assert rows[-1]["status"] == "error"
    p2 = profiler(tmp_path)
    assert p2.session != p.session and len(records(tmp_path)) == len(rows) + 1


def test_compact_records_preserve_session_context_and_measurements(tmp_path):
    identity = {
        "run_id": "run",
        "evaluation_date": "2022-09-05",
        "seed": 42,
        "arm": "graph_qwen",
        "embedding_hash": "features",
        "training_input_hash": "training",
        "representation_mode": "graph",
        "scene_aggregation": "mean",
        "graph_model": {"layers": 1},
        "item_model": {"normalization": "final_layernorm"},
    }
    p = RunProfiler(tmp_path, identity, "cpu", 100)
    with p.sample("selection", 1, 1, examples=512):
        with span("parent"):
            with span("child"):
                pass
    with p.sample("preparation", kind="stage"):
        pass
    p.phase("selection", 1, 10.0, 1024)
    start, batch, stage, phase = records(tmp_path)
    assert start["schema_version"] == "recommendation-profile/v2"
    assert all(start[key] == value for key, value in identity.items())
    assert start["device"] == "cpu" and "pid" in start
    for row in (batch, stage, phase):
        assert row["session_id"] == start["session_id"]
        assert row["timestamp"]
        assert not set(identity).intersection(row)
        assert not {"schema_version", "pid", "device", "exclusive_seconds",
                    "examples_per_second"}.intersection(row)
    assert batch["examples"] == 512 and batch["status"] == "ok"
    assert batch["calls"] == {"parent/child": 1, "parent": 1}
    assert batch["seconds"]["parent"] >= batch["seconds"]["parent/child"]
    assert not {"batch", "seconds", "calls", "workload"}.intersection(stage)
    assert phase["wall_seconds"] == 10.0 and phase["examples"] == 1024
    other = RunProfiler(tmp_path, {**identity, "embedding_hash": "new"}, "cpu", 100)
    assert records(tmp_path)[-1]["embedding_hash"] == "new"
    assert other.session != p.session


def test_operator_backward_attribution_separates_shared_projection_and_other_linear(tmp_path):
    p = profiler(tmp_path, operators=True)
    shared = torch.nn.Linear(4, 2)
    other = torch.nn.Linear(2, 1)
    with p.sample("selection", 1, 3, examples=3):
        with span("node_linear"):
            nodes = shared(torch.ones(3, 4))
        with span("context_projection"):
            contexts = shared(torch.ones(3, 4))
        with span("backward"):
            other(nodes + contexts).sum().backward()
    row = records(tmp_path)[-1]
    summary = json.loads((tmp_path / row["operator_summary"]).read_text())
    groups = summary["projection_backward"]["projection_nodes"]
    assert set(groups) == {"node_linear", "context_projection"}
    for group in groups.values():
        assert group["node_types"] == {"AddmmBackward0": 1, "TBackward0": 1}
        assert group["nodes"] == 2
    with p.sample("selection", 2, 3), span("node_linear"):
        shared(torch.ones(3, 4))
    with p.sample("validation", 2, 3):
        pass
    assert sum(bool(r.get("operator_profiled")) for r in records(tmp_path)) == 1


def test_operator_profile_error_preserves_original_and_resets_context(tmp_path):
    p = profiler(tmp_path, operators=True)
    with pytest.raises(ValueError, match="original failure"):
        with p.sample("selection", 1, 3), span("node_linear"):
            raise ValueError("original failure")
    assert not is_profiling()
    row = records(tmp_path)[-1]
    assert row["status"] == "error" and row["operator_profiled"]
    assert "operator_trace" not in row


@pytest.mark.parametrize("mode", ["text", "mean", "attention"])
@pytest.mark.parametrize("checkpoint", ["never", "always"])
def test_profile_preserves_training_weights_ranks_and_records_work(
    tmp_path, device, mode, checkpoint
):
    table = EventTable(
        [
            {
                "event_id": user * 10 + i,
                "user_id": str(user),
                "item_id": str((i + user) % 8 + 1),
                "timestamp": i * DAY,
            }
            for user in range(2)
            for i in range(10)
        ]
    )
    config = SimpleNamespace(model=SimpleNamespace(batch_size=4))
    ids = table.select()
    counts = np.bincount(table.targets[ids], minlength=len(table.items) + 1)
    probabilities = counts / counts.sum()
    results, states, ranked = [], [], []
    for enabled in (False, True, "operators"):
        # Isolate profiling from the optional nondeterministic training policy.
        seed_everything(42, deterministic=True)
        kwargs = dict(
            item_count=len(table.items),
            max_length=10,
            embedding_dim=16,
            num_blocks=1,
            num_heads=2,
            dropout=0.1,
            arm="graph",
        )
        model = (
            SASRec(**kwargs, item_features=np.ones((len(table.items), 12), dtype=np.float32))
            if mode == "text"
            else GraphSASRec(
                **kwargs,
                store=synthetic_store(len(table.items), feature_dim=32),
                aggregation=mode,
                execution=GraphExecutionConfig(chunk_items=3, checkpoint=checkpoint),
            )
        ).to(device)
        p = profiler(tmp_path, device, operators=enabled == "operators") if enabled else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        result = train_epoch(
            model,
            optimizer,
            table,
            ids,
            probabilities,
            np.random.default_rng(7),
            config,
            device,
            profiler=p,
            phase="selection",
            epoch=1,
        )
        results.append(result)
        states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        ranked.append(
            [
                ranks
                for _, ranks in rank_batches(
                    model, table, ids, config, device, profiler=p, phase="validation", epoch=1
                )
            ]
        )
    assert results[0] == results[1]
    assert ranked[0] == ranked[1]
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
        torch.testing.assert_close(states[0][key], states[2][key], rtol=0, atol=0)
    assert results[0] == results[2] and ranked[0] == ranked[2]
    rows = [r for r in records(tmp_path) if r["session_id"] == p.session]
    training = [r for r in rows if r.get("phase") == "selection"]
    assert [r["batch"] for r in training] == [1, 2, 3, 4]
    for row in training:
        assert all(
            name in row["seconds"]
            for name in (
                "input_prepare",
                "sasrec_forward",
                "logits_loss",
                "backward",
                "clip_grad",
                "optimizer",
            )
        )
        assert all(0 <= value <= row["wall_seconds"] for value in row["seconds"].values())
    if mode != "text":
        row = training[0]
        assert row["workload"]["nodes"] > 0
        assert row["workload"]["chunks"] >= 2
        assert row["workload"]["checkpoint_batches"] == int(checkpoint == "always")
        for name in (
            "graph_items/graph_batch/cpu_pack",
            "graph_items/graph_encoder/message_passing",
            "graph_items/graph_encoder/video_pooling",
            "graph_items/graph_encoder/node_projection/node_linear",
            "graph_items/graph_encoder/scene_readout/context_projection",
            "graph_items/item_fusion/title_projection",
        ):
            assert name in row["seconds"]
        if checkpoint == "always":
            assert "backward/graph_encoder/message_passing" in row["seconds"]
            assert not any(
                "graph_batch" in key for key in row["seconds"] if key.startswith("backward/")
            )
    catalog = [r for r in rows if r.get("kind") == "catalog"]
    assert len(catalog) == 1 and "catalog_encode" in catalog[0]["seconds"]
    if device.type == "cuda":
        assert training[0]["cuda_memory_mib"]["peak_allocated"] > 0
    traced = [r for r in rows if r.get("operator_profiled")]
    assert len(traced) == 1 and traced[0]["batch"] == 3
    summary = json.loads((tmp_path / traced[0]["operator_summary"]).read_text())
    trace = json.loads((tmp_path / traced[0]["operator_trace"]).read_text())
    assert trace["traceEvents"]
    backward = summary["projection_backward"]
    assert backward["matched_nodes"] > 0
    assert 0 < backward["projection_cpu_pct_of_backward_nodes"] <= 100
    paths = backward["projection_nodes"]
    expected = {"title_projection"} if mode == "text" else {
        "node_linear", "context_projection", "title_projection"
    }
    assert expected <= {path.rsplit("/", 1)[-1] for path in paths}
    assert all(g["cpu_total_us"] > 0 for g in paths.values())
    if device.type == "cuda":
        assert backward["all_backward_device_total_us"] > 0
        assert backward["projection_device_pct_of_backward_nodes"] > 0
    assert not is_profiling()
