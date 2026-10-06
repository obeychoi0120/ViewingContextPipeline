"""Dense scene GCN math, isolation, joint training and execution contracts."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from graph_fixture import synthetic_store  # noqa: E402
from validation.config import GraphExecutionConfig  # noqa: E402
from validation.graph_batching import PackedGraphStore  # noqa: E402
from validation.graph_context import graph_contract, validate_mode  # noqa: E402
from validation.graph_model import GraphSASRec, SceneRoleEncoder  # noqa: E402
from validation.scene_gcn import SceneGCN, normalized_similarity, scene_groups  # noqa: E402
from validation.profiling import RunProfiler  # noqa: E402


@pytest.fixture(params=["cpu"] + ([os.environ["GRAPH_TEST_CUDA"]]
                                if os.environ.get("GRAPH_TEST_CUDA") else []))
def device(request):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch.device(request.param)


def oracle_similarity(x):
    unit = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    a = np.maximum(unit @ unit.T, 0)
    np.fill_diagonal(a, 1)
    degree = a.sum(-1)
    return a / np.sqrt(degree[:, None] * degree[None, :])


def test_similarity_and_residual_gcn_match_numpy_with_zero_and_negative_cosine(device):
    x = np.array([[1., 0.], [1., 1.], [-1., 0.], [0., 0.]], dtype=np.float32)
    h = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
    w = np.array([[1., -1., 2.], [0., 2., -1.], [-1., 0., 1.]], dtype=np.float32)
    bias = np.array([.1, -.2, .3], dtype=np.float32)
    model = SceneGCN(3).to(device)
    with torch.no_grad():
        model.projection.weight.copy_(torch.as_tensor(w, device=device))
        model.projection.bias.copy_(torch.as_tensor(bias, device=device))
    xt = torch.tensor(x, device=device, requires_grad=True)
    ht = torch.tensor(h, device=device, requires_grad=True)
    expected_a = oracle_similarity(x)
    torch.testing.assert_close(normalized_similarity(xt), torch.tensor(expected_a, device=device))
    assert expected_a[0, 2] == 0 and expected_a[3, 3] == 1
    indices = [torch.arange(4, device=device).reshape(1, 4)]
    out = model(ht, xt, indices)
    expected = h + np.maximum((expected_a @ h) @ w.T + bias, 0)
    torch.testing.assert_close(out, torch.tensor(expected, device=device))
    out.square().sum().backward()
    assert xt.grad is not None and torch.isfinite(xt.grad).all() and xt.grad.abs().sum() > 0
    assert ht.grad is not None and model.projection.weight.grad.abs().sum() > 0


def test_batched_ragged_videos_equal_independent_computation_and_permutation(device):
    torch.manual_seed(41)
    model = SceneGCN(8).to(device)
    h = torch.randn(8, 8, device=device)
    x = torch.randn(8, 6, device=device)
    offsets = [0, 3, 3, 4, 7, 8]
    groups = [torch.tensor(a, device=device) for a in scene_groups(offsets).values()]
    result = model(h, x, groups)
    expected = torch.cat([h[a:b] + torch.relu(model.projection(
        normalized_similarity(x[a:b]) @ h[a:b]))
        for a, b in zip(offsets[:-1], offsets[1:], strict=True) if b > a])
    torch.testing.assert_close(result, expected)
    perm = torch.tensor([2, 0, 1, 3, 5, 6, 4, 7], device=device)
    torch.testing.assert_close(model(h[perm], x[perm], groups), result[perm])
    changed = h.clone()
    changed[:3] += 100
    torch.testing.assert_close(model(changed, x, groups)[3:], result[3:])
    # Different adjacency with identical messages changes the result.
    altered = x.clone()
    altered[:3] = 0
    assert not torch.allclose(model(h, altered, groups)[:3], result[:3])


def test_title_does_not_enter_similarity_and_missing_fallback_bypasses_gcn(device):
    store = synthetic_store(24, feature_dim=32)
    batch = PackedGraphStore(store).batch([1, 2, 11, 0], device, group_scenes=True)
    model = SceneRoleEncoder(32, 8, "gnn").to(device)
    seen = []
    hook = model.gcn.register_forward_pre_hook(lambda _, args: seen.append(args[1].detach().clone()))
    result = model(batch)
    changed = {**batch, "titles": batch["titles"] + 10}
    result_changed = model(changed)
    hook.remove()
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
    assert seen[0].shape[-1] == 24
    assert not torch.allclose(result[:2], result_changed[:2])
    fallback = model.scene_mlp(torch.cat([
        model.title_projection(batch["titles"])[2:3], result.new_zeros((1, 24))], -1))
    torch.testing.assert_close(result[2:3], fallback)
    assert not result[3].any()
    empty = PackedGraphStore(store).batch([11, 0], device, group_scenes=True)
    hook = model.gcn.register_forward_hook(lambda *_: pytest.fail("no scenes: bypass GCN"))
    assert torch.isfinite(model(empty)).all()
    hook.remove()
    original_content = seen[0]
    # Changing Context changes content-based similarity input.
    changed = {**batch, "contexts": batch["contexts"] + torch.randn_like(batch["contexts"])}
    seen = []
    hook = model.gcn.register_forward_pre_hook(lambda _, args: seen.append(args[1].detach().clone()))
    model(changed)
    hook.remove()
    group = next(value for key, value in batch.items()
                 if key.startswith("scene_group_") and value.shape[1] > 1)
    assert not torch.allclose(normalized_similarity(seen[0][group]),
                              normalized_similarity(original_content[group]))


def make_model(store, device, *, pool="gnn", pairs=1048576, checkpoint="never", cache=0):
    return GraphSASRec(
        store=store, aggregation=pool, item_count=len(store), max_length=10,
        embedding_dim=32, num_blocks=2, num_heads=2, dropout=0, arm="graph",
        execution=GraphExecutionConfig(chunk_items=64, chunk_nodes=100000,
            chunk_scene_pairs=pairs, checkpoint=checkpoint, feature_cache_mb=cache),
    ).to(device)


def training_loss(model, device):
    sequences = torch.tensor([[2, 3, 0], [4, 5, 0], [6, 7, 0]], device=device)
    model.prepare_items(torch.arange(1, len(model.store) + 1, device=device))
    logits = model.user_vectors(sequences) @ model.catalog_vectors().T
    return torch.nn.functional.cross_entropy(logits, torch.tensor([3, 5, 7], device=device))


@pytest.mark.parametrize("checkpoint", ["always", "never"])
def test_joint_loss_chunk_checkpoint_fixed_features_and_reload(device, checkpoint, tmp_path):
    torch.manual_seed(91)
    store = synthetic_store(24, feature_dim=32)
    features = store["features"].copy()
    reference = make_model(store, device)
    model = make_model(store, device, pairs=1, checkpoint=checkpoint, cache=2048)
    model.load_state_dict(reference.state_dict())
    losses = [training_loss(m, device) for m in (reference, model)]
    torch.testing.assert_close(*losses, atol=2e-6, rtol=1e-5)
    for loss in losses:
        loss.backward()
    for (name, a), (_, b) in zip(reference.named_parameters(), model.named_parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4, msg=name)
    for layer in (model.graph_encoder.entity_projection, model.graph_encoder.action_projection,
                  model.graph_encoder.context_projection, model.graph_encoder.role_fusion,
                  model.graph_encoder.scene_mlp.fc1, model.graph_encoder.gcn.projection,
                  model.encoder.layers[0].linear1):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    assert model.execution_counts["chunks"] > reference.execution_counts["chunks"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    model.eval()
    vectors = model.catalog_vectors().detach().clone()
    assert not model.item_vectors(torch.tensor([0, 1], device=device)).any()
    np.testing.assert_array_equal(store["features"], features)
    torch.save(model.state_dict(), tmp_path / "gcn.pt")
    restored = make_model(store, device)
    restored.load_state_dict(torch.load(tmp_path / "gcn.pt", weights_only=True))
    restored.eval()
    sequences = torch.tensor([[2, 3, 0]], device=device)
    torch.testing.assert_close(restored.catalog_vectors(), vectors)
    torch.testing.assert_close(restored.score_catalog(sequences), model.score_catalog(sequences))
    with pytest.raises(RuntimeError):
        make_model(store, device, pool="mean").load_state_dict(model.state_dict())


def test_single_and_actionless_scene_zero_similarity_and_empty_nodes(device):
    model = SceneRoleEncoder(4, 2, "gnn").to(device)
    batch = {"offsets": [0, 1], "features": torch.zeros(0, 4, device=device),
             "types": torch.zeros(0, dtype=torch.long, device=device),
             "node_scenes": torch.zeros(0, dtype=torch.long, device=device),
             "titles": torch.zeros(1, 4, device=device),
             "title_available": torch.zeros(1, dtype=torch.bool, device=device),
             "available": torch.ones(1, dtype=torch.bool, device=device),
             "contexts": torch.zeros(1, 4, device=device),
             "scene_videos": torch.zeros(1, dtype=torch.long, device=device)}
    output = model(batch)
    h = model.scene_mlp(torch.cat([torch.zeros(1, 6, device=device),
                                  model.context_projection(batch["contexts"])], -1))
    torch.testing.assert_close(output, h + torch.relu(model.gcn.projection(h)))
    assert torch.isfinite(output).all()
    torch.testing.assert_close(normalized_similarity(torch.zeros(1, 3, 6, device=device)),
                               torch.eye(3, device=device).unsqueeze(0))


def test_profile_preserves_outputs_gradients_and_counts_pairs(device, tmp_path):
    store = synthetic_store(12, feature_dim=32)
    plain = make_model(store, device)
    measured = make_model(store, device)
    measured.load_state_dict(plain.state_dict())
    loss = training_loss(plain, device)
    loss.backward()
    p = RunProfiler(tmp_path, {"scene_aggregation": "gnn", "evaluation_date": "test", "seed": 42,
                              "arm": "graph_qwen"}, device, 1)
    with p.sample("selection", 1, 1):
        observed = training_loss(measured, device)
        observed.backward()
    torch.testing.assert_close(loss, observed)
    for a, b in zip(plain.parameters(), measured.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    import json
    row = json.loads((tmp_path / "profile.jsonl").read_text().splitlines()[-1])
    assert row["workload"]["scene_pairs"] == int(np.square(np.diff(store["video_offsets"])).sum())
    assert any(name.endswith("scene_similarity") for name in row["seconds"])
    assert any(name.endswith("scene_gcn") for name in row["seconds"])


def test_contracts_cli_and_scene_pair_execution_config():
    mean_version, mean = graph_contract("mean", "graph_qwen")
    version, gnn = graph_contract("gnn", "graph_qwen")
    assert mean_version == "sasrec-role-graph/v7" and "gcn" not in mean
    assert version == "sasrec-role-graph/v8" and gnn["gcn_applied"]
    assert graph_contract("gnn", "meta")[1]["gcn_applied"] is False
    assert gnn["gcn"]["similarity_dim"] == 384
    validate_mode("graph", "gnn", "run-recommendation", ["graph_qwen"])
    validate_mode("graph", "gnn", "run-diagnosis", ["meta"])
    for mode, step in (("text", "run-recommendation"), ("graph", "embed-representations")):
        with pytest.raises(ValueError):
            validate_mode(mode, "gnn", step, ["meta"])
    with pytest.raises(ValueError):
        GraphExecutionConfig(chunk_scene_pairs=0)
    assert GraphExecutionConfig().chunk_scene_pairs == 1048576


def test_gnn_auto_checkpoint_accounts_for_quadratic_pairs(monkeypatch):
    model = make_model(synthetic_store(12, feature_dim=32), "cpu", checkpoint="auto")
    class DeviceParameter:
        device = torch.device("cuda:0")
    monkeypatch.setattr(type(model.title_projection), "weight", DeviceParameter(), raising=False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (1024 ** 3, 1024 ** 3))
    # Scene pairs alone exceed the auto threshold, despite few nodes.
    model.packed.scene_counts = np.full(12, 5000)
    assert model._use_checkpoint([2])
