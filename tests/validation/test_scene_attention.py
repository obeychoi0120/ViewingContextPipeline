"""Attention pooling and attention GCN math, start state, isolation, training and execution."""

import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from graph_fixture import synthetic_store  # noqa: E402
from validation.config import GraphExecutionConfig  # noqa: E402
from validation.graph_batching import PackedGraphStore  # noqa: E402
from validation.graph_context import ATTENTION_AGGREGATIONS, SCENE_EXCHANGES  # noqa: E402
from validation.graph_model import GraphSASRec, SceneRoleEncoder, segment_mean  # noqa: E402
from validation.scene_attention import (  # noqa: E402
    SceneAttentionGCN, SceneAttentionPool, recorded_weights, segment_softmax, unit_rms,
)
from validation.scene_gcn import SceneGCN, scene_groups  # noqa: E402

OFFSETS = [0, 3, 3, 4, 7, 8]  # Videos of 3, 0, 1, 3 and 1 scenes.


@pytest.fixture(params=["cpu"] + ([os.environ["GRAPH_TEST_CUDA"]]
                                if os.environ.get("GRAPH_TEST_CUDA") else []))
def device(request):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch.device(request.param)


def videos_of(offsets, device):
    return torch.as_tensor(np.repeat(np.arange(len(offsets) - 1), np.diff(offsets)), device=device)


def groups_of(offsets, device):
    return [torch.tensor(a, device=device) for a in scene_groups(offsets).values()]


def open_pool(pool):
    """The score output starts at zero (exact mean pooling); open it to exercise the weights."""
    with torch.no_grad():
        pool.score.weight.copy_(torch.randn_like(pool.score.weight))
    return pool


def test_segment_softmax_and_pool_match_numpy_including_large_scores(device):
    scores = np.array([0.5, -1.0, 2.0, 7.0, 1000.0, 1001.0, 999.0, -3.0], dtype=np.float32)
    videos = videos_of(OFFSETS, device)
    weights = segment_softmax(torch.tensor(scores, device=device), videos, 5)
    expected = np.zeros_like(scores)
    for a, b in zip(OFFSETS[:-1], OFFSETS[1:], strict=True):
        if b > a:
            shifted = np.exp(scores[a:b] - scores[a:b].max())
            expected[a:b] = shifted / shifted.sum()
    torch.testing.assert_close(weights, torch.tensor(expected, device=device))
    torch.manual_seed(1)
    pool = open_pool(SceneAttentionPool(6, 4).to(device))
    h = torch.randn(8, 6, device=device)
    beta = pool.weights(h, videos, 5)
    unit = h / h.norm(dim=-1, keepdim=True) * math.sqrt(6)
    scores = (torch.tanh(unit @ pool.hidden.weight.T + pool.hidden.bias) @ pool.score.weight.T)[:, 0]
    manual = torch.stack([
        (torch.softmax(scores[a:b], 0)[:, None] * h[a:b]).sum(0) if b > a else h.new_zeros(6)
        for a, b in zip(OFFSETS[:-1], OFFSETS[1:], strict=True)])
    torch.testing.assert_close(pool(h, videos, 5), manual)
    torch.testing.assert_close(beta[[3, 7]], torch.ones(2, device=device))  # Single scenes.
    # Scores see only scene directions; the pooled values keep their scale.
    torch.testing.assert_close(pool.weights(3 * h, videos, 5), beta)
    torch.testing.assert_close(pool(3 * h, videos, 5), 3 * pool(h, videos, 5))


def test_zero_score_pool_is_mean_and_opens_through_the_score_weight(device):
    torch.manual_seed(2)
    pool = SceneAttentionPool(6, 4).to(device)
    assert pool.score.bias is None and not pool.score.weight.any()
    h = torch.randn(8, 6, device=device, requires_grad=True)
    videos = videos_of(OFFSETS, device)
    target = torch.randn(5, 6, device=device)
    output = pool(h, videos, 5)
    torch.testing.assert_close(output, segment_mean(h, videos, 5))
    (output * target).sum().backward()
    # At zero only the score weight receives gradient; the hidden layer follows once opened.
    assert pool.score.weight.grad.abs().sum() > 0
    assert not pool.hidden.weight.grad.any()
    torch.optim.AdamW(pool.parameters(), lr=1e-2).step()
    pool.zero_grad(set_to_none=True)
    (pool(h, videos, 5) * target).sum().backward()
    assert pool.hidden.weight.grad.abs().sum() > 0


def test_pool_is_scene_permutation_invariant_and_isolates_videos(device):
    torch.manual_seed(3)
    pool = open_pool(SceneAttentionPool(6, 4).to(device))
    h = torch.randn(8, 6, device=device)
    videos = videos_of(OFFSETS, device)
    result = pool(h, videos, 5)
    perm = torch.tensor([2, 0, 1, 3, 6, 4, 5, 7], device=device)  # Within each video.
    torch.testing.assert_close(pool(h[perm], videos, 5), result)
    changed = h.clone()
    changed[:3] += 100
    torch.testing.assert_close(pool(changed, videos, 5)[1:], result[1:])
    assert not torch.allclose(pool(changed, videos, 5)[0], result[0])


def test_attention_gcn_matches_manual_softmax_and_the_gnn_residual(device):
    torch.manual_seed(4)
    model = SceneAttentionGCN(5, 3, 4).to(device)
    x = torch.randn(4, 3, device=device, requires_grad=True)
    h = torch.randn(4, 5, device=device, requires_grad=True)
    unit = x / x.norm(dim=-1, keepdim=True) * math.sqrt(3)
    q, k = unit @ model.query.weight.T, unit @ model.key.weight.T
    adjacency = torch.softmax(q @ k.T / math.sqrt(4), dim=-1)
    torch.testing.assert_close(model.adjacency(x), adjacency)
    torch.testing.assert_close(model.adjacency(5 * x), adjacency)  # Scale-free, like cosine.
    expected = h + torch.relu((adjacency @ h) @ model.projection.weight.T + model.projection.bias)
    out = model(h, x, [torch.arange(4, device=device).reshape(1, 4)])
    torch.testing.assert_close(out, expected)
    out.square().sum().backward()
    for grad in (x.grad, model.query.weight.grad, model.key.weight.grad,
                 model.projection.weight.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    with torch.no_grad():  # Zero queries give uniform neighbour weights, self included.
        model.query.weight.zero_()
    torch.testing.assert_close(model.adjacency(x), torch.full((4, 4), 0.25, device=device))


def test_attention_gcn_groups_equal_independent_computation_and_permutation(device):
    torch.manual_seed(5)
    model = SceneAttentionGCN(8, 6, 4).to(device)
    h = torch.randn(8, 8, device=device)
    x = torch.randn(8, 6, device=device)
    groups = groups_of(OFFSETS, device)
    result = model(h, x, groups)
    expected = torch.cat([h[a:b] + torch.relu(model.projection(model.adjacency(x[a:b]) @ h[a:b]))
                          for a, b in zip(OFFSETS[:-1], OFFSETS[1:], strict=True) if b > a])
    torch.testing.assert_close(result, expected)
    perm = torch.tensor([2, 0, 1, 3, 5, 6, 4, 7], device=device)
    torch.testing.assert_close(model(h[perm], x[perm], groups), result[perm])
    changed = h.clone()
    changed[:3] += 100
    torch.testing.assert_close(model(changed, x, groups)[3:], result[3:])
    altered = x.clone()
    altered[:3] = torch.randn_like(altered[:3])
    assert not torch.allclose(model(h, altered, groups)[:3], result[:3])


def test_attention_gcn_differs_from_gnn_only_by_query_and_key():
    gnn, attention = SceneGCN(512), SceneAttentionGCN(512, 384, 128)
    assert attention.adjacency_span != gnn.adjacency_span
    missing, unexpected = attention.load_state_dict(gnn.state_dict(), strict=False)
    assert missing == ["query.weight", "key.weight"] and not unexpected
    added = sum(p.numel() for p in attention.parameters()) - sum(p.numel() for p in gnn.parameters())
    assert added == 2 * 384 * 128 == 98_304
    assert sum(p.numel() for p in SceneAttentionPool(512, 128).parameters()) == 65_792


def test_attention_gcn_starts_at_the_scene_transformer_logit_scale():
    store = synthetic_store(24, feature_dim=1024)
    torch.manual_seed(37)
    model = make_model(store, "cpu", pool="gnn_attn", dimension=512)
    gcn = model.graph_encoder.gcn  # Re-initialized after the shared xavier reset.
    for layer in (gcn.query, gcn.key):
        assert float(layer.weight.detach().std()) == pytest.approx((2 * 384) ** -0.5, rel=0.02)
    u = unit_rms(torch.randn(4000, 384))
    with torch.no_grad():
        logits = (gcn.query(u[:2000]) * gcn.key(u[2000:])).sum(-1) / math.sqrt(128)
    assert float(logits.std()) == pytest.approx(0.5, rel=0.1)


def test_scene_exchange_scope():
    assert {"gnn", "gnn_attn", "transformer", "transformer_pos", "transformer_local"} == SCENE_EXCHANGES
    assert "attn_pool" not in SCENE_EXCHANGES and "mean" not in SCENE_EXCHANGES


def make_model(store, device, *, pool, items=64, pairs=1048576, checkpoint="never", cache=0,
               dimension=32):
    return GraphSASRec(
        store=store, aggregation=pool, item_count=len(store), max_length=10,
        embedding_dim=dimension, num_blocks=2, num_heads=2, dropout=0, arm="graph",
        execution=GraphExecutionConfig(chunk_items=items, chunk_nodes=100000,
            chunk_scene_pairs=pairs, checkpoint=checkpoint, feature_cache_mb=cache),
    ).to(device)


def training_loss(model, device):
    sequences = torch.tensor([[2, 3, 0], [4, 5, 0], [6, 7, 0]], device=device)
    model.prepare_items(torch.arange(1, len(model.store) + 1, device=device))
    logits = model.user_vectors(sequences) @ model.catalog_vectors().T
    return torch.nn.functional.cross_entropy(logits, torch.tensor([3, 5, 7], device=device))


def test_attention_pool_starts_as_mean_pooling_and_reports_its_weights(device):
    store = synthetic_store(24, feature_dim=32)
    torch.manual_seed(23)
    mean = make_model(store, device, pool="mean").eval()
    with torch.no_grad():
        expected = mean.catalog_vectors().clone()
    model = make_model(store, device, pool="attn_pool")
    missing, unexpected = model.load_state_dict(mean.state_dict(), strict=False)
    assert not unexpected and all(".attention_pool." in name for name in missing)
    pool = model.graph_encoder.attention_pool
    assert not pool.score.weight.any()  # Re-zeroed after the shared Linear reset.
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.catalog_vectors(), expected)
    start = model.execution_report()["scene_attention_summary"]
    counts = np.diff(store["video_offsets"])
    assert start["videos"] == int((counts >= 2).sum())
    for name in ("normalized_entropy", "max_weight_ratio", "first_scene_weight_ratio",
                 "last_scene_weight_ratio"):
        assert start[f"{name}_mean"] == pytest.approx(1, abs=1e-5)
    assert start["score_weight_norm"] == 0
    model.train()
    for _ in range(5):
        model.zero_grad(set_to_none=True)
        training_loss(model, device).backward()
        torch.optim.AdamW(model.parameters(), lr=1e-2).step()
    chunks = model.execution_counts["chunks"]
    learned = model.execution_report()["scene_attention_summary"]
    assert model.execution_counts["chunks"] == chunks  # The summary pass is not counted.
    assert learned["score_weight_norm"] > 0 and learned["normalized_entropy_mean"] < 1
    assert learned["max_weight_ratio_mean"] > 1
    assert "scene_attention_summary" not in mean.execution_report()


def test_attention_gcn_reports_row_weights(device):
    store = synthetic_store(24, feature_dim=32)
    torch.manual_seed(29)
    model = make_model(store, device, pool="gnn_attn").eval()
    summary = model.execution_report()["scene_attention_summary"]
    counts = np.diff(store["video_offsets"])
    assert summary["rows"] == int(counts[counts >= 2].sum())
    assert 0 < summary["normalized_entropy_mean"] <= 1 + 1e-6
    assert summary["max_weight_ratio_mean"] >= 1 - 1e-6
    assert summary["self_weight_ratio_mean"] > 0
    assert summary["query_weight_norm"] > 0 and summary["key_weight_norm"] > 0
    with torch.no_grad():
        model.graph_encoder.gcn.query.weight.zero_()
    uniform = model.execution_report()["scene_attention_summary"]
    for name in ("normalized_entropy", "max_weight_ratio", "self_weight_ratio"):
        assert uniform[f"{name}_mean"] == pytest.approx(1, abs=1e-5)


def test_recorded_weights_match_direct_computation(device):
    torch.manual_seed(31)
    pool = open_pool(SceneAttentionPool(6, 4).to(device))
    h = torch.randn(8, 6, device=device)
    videos = videos_of(OFFSETS, device)
    with recorded_weights(pool) as summary:
        pool(h, videos, 5)
    beta = pool.weights(h, videos, 5).detach().cpu().numpy()
    rows = [beta[a:b] for a, b in zip(OFFSETS[:-1], OFFSETS[1:], strict=True) if b - a >= 2]
    assert summary["videos"] == 2
    assert summary["normalized_entropy_mean"] == pytest.approx(
        np.mean([-(w * np.log(w)).sum() / np.log(len(w)) for w in rows]), rel=1e-5)
    assert summary["first_scene_weight_ratio_mean"] == pytest.approx(
        np.mean([w[0] * len(w) for w in rows]), rel=1e-5)
    assert summary["last_scene_weight_ratio_mean"] == pytest.approx(
        np.mean([w[-1] * len(w) for w in rows]), rel=1e-5)
    assert summary["max_weight_ratio_mean"] == pytest.approx(
        np.mean([w.max() * len(w) for w in rows]), rel=1e-5)
    pool(h, videos, 5)  # The hook is removed after the block.
    assert summary["videos"] == 2


@pytest.mark.parametrize("pool", ATTENTION_AGGREGATIONS)
def test_title_only_fallback_and_empty_videos_bypass_attention(device, pool):
    store = synthetic_store(24, feature_dim=32)
    exchange = pool in SCENE_EXCHANGES
    batch = PackedGraphStore(store).batch([1, 2, 11, 0], device, group_scenes=exchange)
    model = SceneRoleEncoder(32, 8, pool).to(device)
    if pool == "attn_pool":
        open_pool(model.attention_pool)
    module = model.attention_pool if pool == "attn_pool" else model.gcn
    result = model(batch)
    fallback = model.scene_mlp(torch.cat([
        model.title_projection(batch["titles"])[2:3], result.new_zeros((1, 24))], -1))
    torch.testing.assert_close(result[2:3], fallback)
    assert not result[3].any()
    empty = PackedGraphStore(store).batch([11, 0], device, group_scenes=exchange)
    hook = module.register_forward_hook(lambda *_: pytest.fail("no scenes: bypass attention"))
    assert torch.isfinite(model(empty)).all()
    hook.remove()


def test_title_does_not_enter_attention_gcn_weights(device):
    store = synthetic_store(24, feature_dim=32)
    batch = PackedGraphStore(store).batch([1, 2, 3], device, group_scenes=True)
    model = SceneRoleEncoder(32, 8, "gnn_attn").to(device)
    seen = []
    hook = model.gcn.register_forward_pre_hook(lambda _, args: seen.append(args[1].detach().clone()))
    result = model(batch)
    changed = model({**batch, "titles": batch["titles"] + 10})
    hook.remove()
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
    assert seen[0].shape[-1] == 24
    assert not torch.allclose(result, changed)


@pytest.mark.parametrize("pool", ATTENTION_AGGREGATIONS)
@pytest.mark.parametrize("checkpoint", ["always", "never"])
def test_joint_loss_chunk_checkpoint_and_reload(device, pool, checkpoint, tmp_path):
    torch.manual_seed(91)
    store = synthetic_store(24, feature_dim=32)
    reference = make_model(store, device, pool=pool)
    if pool == "attn_pool":
        open_pool(reference.graph_encoder.attention_pool)
    model = make_model(store, device, pool=pool, items=5, checkpoint=checkpoint, cache=2048)
    model.load_state_dict(reference.state_dict())
    losses = [training_loss(m, device) for m in (reference, model)]
    torch.testing.assert_close(*losses, atol=2e-6, rtol=1e-5)
    for loss in losses:
        loss.backward()
    for (name, a), (_, b) in zip(reference.named_parameters(), model.named_parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4, msg=name)
    encoder = model.graph_encoder
    added = ((encoder.attention_pool.hidden.weight, encoder.attention_pool.score.weight)
             if pool == "attn_pool" else
             (encoder.gcn.query.weight, encoder.gcn.key.weight, encoder.gcn.projection.weight))
    for weight in (encoder.entity_projection.weight, encoder.action_projection.weight,
                   encoder.context_projection.weight, encoder.role_fusion.weight,
                   encoder.scene_mlp.fc1.weight, *added, model.encoder.layers[0].linear1.weight):
        assert weight.grad is not None and weight.grad.abs().sum() > 0
    assert model.execution_counts["chunks"] > reference.execution_counts["chunks"]
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    model.eval()
    vectors = model.catalog_vectors().detach().clone()
    assert not model.item_vectors(torch.tensor([0, 1], device=device)).any()
    torch.save(model.state_dict(), tmp_path / "attention.pt")
    restored = make_model(store, device, pool=pool)
    restored.load_state_dict(torch.load(tmp_path / "attention.pt", weights_only=True))
    restored.eval()
    sequences = torch.tensor([[2, 3, 0]], device=device)
    torch.testing.assert_close(restored.catalog_vectors(), vectors)
    torch.testing.assert_close(restored.score_catalog(sequences), model.score_catalog(sequences))
    for other in ("mean", "gnn", "transformer", *(set(ATTENTION_AGGREGATIONS) - {pool})):
        with pytest.raises(RuntimeError):
            make_model(store, device, pool=other).load_state_dict(model.state_dict())


def test_attention_pool_has_no_pairwise_chunking_or_scene_groups(device):
    store = synthetic_store(24, feature_dim=32)
    counts = {}
    for pool in ("attn_pool", "gnn_attn"):
        model = make_model(store, device, pool=pool, pairs=1).eval()
        with torch.no_grad():
            model.catalog_vectors()
        counts[pool] = model.execution_counts["chunks"]
    assert counts["attn_pool"] == 1 and counts["gnn_attn"] > 1
    batch = PackedGraphStore(store).batch(np.arange(24), device, group_scenes=False)
    assert not any(key.startswith("scene_group_") for key in batch)
    model = SceneRoleEncoder(32, 8, "attn_pool").to(device)
    assert torch.isfinite(model(batch)).all()
