"""Scene Transformer math, variant contracts, isolation, joint training and execution."""

import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from graph_fixture import synthetic_store  # noqa: E402
from validation.config import GraphExecutionConfig  # noqa: E402
from validation.graph_batching import PackedGraphStore  # noqa: E402
from validation.graph_context import SCENE_TRANSFORMER, TRANSFORMER_VARIANTS  # noqa: E402
from validation.graph_model import GraphSASRec, SceneRoleEncoder  # noqa: E402
from validation.scene_gcn import scene_groups  # noqa: E402
from validation.scene_transformer import SceneTransformer, scene_positions  # noqa: E402

VARIANTS = tuple(TRANSFORMER_VARIANTS)


@pytest.fixture(params=["cpu"] + ([os.environ["GRAPH_TEST_CUDA"]]
                                if os.environ.get("GRAPH_TEST_CUDA") else []))
def device(request):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch.device(request.param)


def activate(transformer):
    """Residual scales start at 0 (exact mean pooling); open them to exercise the block."""
    with torch.no_grad():
        for layer in transformer.layers:
            layer.residual_scale.copy_(torch.tensor([0.7, 0.5]))
    return transformer


def block(variant, dimension, device, dropout=0.0):
    model = SceneTransformer(dimension, dropout=dropout, **SCENE_TRANSFORMER,
                             **TRANSFORMER_VARIANTS[variant]).to(device)
    with torch.no_grad():  # Non-trivial LayerNorm affine terms and biases.
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    return activate(model)


def oracle_positions(count, dimension):
    rank = np.arange(count)[:, None]
    angle = rank / np.power(10000.0, np.arange(0, dimension, 2) / dimension)
    table = np.zeros((count, dimension))
    table[:, 0::2], table[:, 1::2] = np.sin(angle), np.cos(angle)
    return table / np.sqrt(dimension / 2)


def oracle_block(model, x, *, positions=None, self_only=False):
    """Pre-LN block with residual scales, written from the equations using raw weights."""
    layer = model.layers[0]
    attention = layer.attention
    scale = layer.residual_scale
    size, dimension = x.shape
    heads = attention.num_heads
    width = dimension // heads
    norm = torch.nn.functional.layer_norm
    if positions is not None:
        x = x + positions
    hidden = norm(x, (dimension,), layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
    q, k, v = (hidden @ attention.in_proj_weight.T + attention.in_proj_bias).split(dimension, -1)
    q, k, v = (t.reshape(size, heads, width).transpose(0, 1) for t in (q, k, v))
    scores = q @ k.transpose(-1, -2) / math.sqrt(width)
    if self_only:
        scores = scores.masked_fill(~torch.eye(size, dtype=torch.bool, device=x.device), -math.inf)
    mixed = (scores.softmax(-1) @ v).transpose(0, 1).reshape(size, dimension)
    x = x + scale[0] * (mixed @ attention.out_proj.weight.T + attention.out_proj.bias)
    hidden = norm(x, (dimension,), layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
    hidden = torch.relu(hidden @ layer.linear1.weight.T + layer.linear1.bias)
    x = x + scale[1] * (hidden @ layer.linear2.weight.T + layer.linear2.bias)
    return x if positions is None else x - positions


def one_video(size, device):
    return [torch.arange(size, device=device).reshape(1, size)]


def test_positions_are_unit_norm_sinusoids_of_valid_scene_rank(device):
    actual = scene_positions(7, 16, device, torch.float32)
    expected = torch.tensor(oracle_positions(7, 16), dtype=torch.float32, device=device)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(7, device=device))


@pytest.mark.parametrize("variant", VARIANTS)
def test_block_matches_manual_pre_layernorm_attention(device, variant):
    torch.manual_seed(5)
    model = block(variant, 8, device)
    h = torch.randn(4, 8, device=device)
    spec = TRANSFORMER_VARIANTS[variant]
    positions = scene_positions(4, 8, device, h.dtype) if spec["position"] != "none" else None
    expected = oracle_block(model, h, positions=positions,
                            self_only=spec["attention"] == "self_only")
    torch.testing.assert_close(model(h, one_video(4, device)), expected, atol=2e-6, rtol=1e-5)
    model.eval()  # Inference may use PyTorch's fused attention path.
    with torch.no_grad():
        torch.testing.assert_close(model(h, one_video(4, device)), expected, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("variant", VARIANTS)
def test_grouped_videos_equal_independent_computation_and_stay_isolated(device, variant):
    torch.manual_seed(13)
    model = block(variant, 8, device)
    h = torch.randn(8, 8, device=device)
    offsets = [0, 3, 3, 4, 7, 8]
    groups = [torch.tensor(a, device=device) for a in scene_groups(offsets).values()]
    result = model(h, groups)
    expected = torch.cat([model(h[a:b], one_video(b - a, device))
                          for a, b in zip(offsets[:-1], offsets[1:], strict=True) if b > a])
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=1e-5)
    changed = h.clone()
    changed[:3] += 100
    torch.testing.assert_close(model(changed, groups)[3:], result[3:])


def test_variants_separate_scene_interaction_from_scene_order(device):
    torch.manual_seed(17)
    h = torch.randn(5, 8, device=device)
    perm = torch.tensor([3, 0, 4, 1, 2], device=device)
    edited = h.clone()
    edited[1:] += torch.randn_like(edited[1:])
    outputs = {}
    for variant in VARIANTS:
        model = block(variant, 8, device)
        outputs[variant] = (model(h, one_video(5, device)), model(h[perm], one_video(5, device)),
                            model(edited, one_video(5, device)))
    # Without positions, scene outputs follow the permutation, so the video mean is unchanged.
    for variant in ("transformer", "transformer_local"):
        output, permuted, _ = outputs[variant]
        torch.testing.assert_close(permuted, output[perm], atol=2e-6, rtol=1e-5)
    output, permuted, _ = outputs["transformer_pos"]
    assert not torch.allclose(permuted.mean(0), output.mean(0), atol=1e-4)
    # Only all-scene attention lets other scenes change scene 0.
    assert not torch.allclose(outputs["transformer"][2][0], outputs["transformer"][0][0])
    torch.testing.assert_close(outputs["transformer_local"][2][0], outputs["transformer_local"][0][0])


@pytest.mark.parametrize("variant", VARIANTS)
def test_zero_residual_scales_are_identity_and_positions_are_removed(device, variant):
    model = block(variant, 8, device)
    with torch.no_grad():  # Closed sublayers: only the residual stream remains.
        model.layers[0].residual_scale.zero_()
    h = torch.randn(6, 8, device=device)
    torch.testing.assert_close(model(h, one_video(6, device)), h, rtol=0, atol=0)


def test_parameter_count_is_identical_across_variants():
    counts = {variant: sum(p.numel() for p in SceneTransformer(
        512, dropout=0.1, **SCENE_TRANSFORMER, **TRANSFORMER_VARIANTS[variant]).parameters())
        for variant in VARIANTS}
    d, ffn = 512, 512 * SCENE_TRANSFORMER["ffn_ratio"]
    expected = (4 * d * d + 4 * d) + (2 * d * ffn + ffn + d) + 4 * d + 2
    assert set(counts.values()) == {expected} and expected == 3_152_386


def test_every_variant_starts_as_mean_pooling_and_learns_its_scales(device):
    store = synthetic_store(24, feature_dim=32)
    torch.manual_seed(23)
    mean = make_model(store, device, pool="mean").eval()
    with torch.no_grad():
        expected = mean.catalog_vectors().clone()
    for variant in VARIANTS:
        model = make_model(store, device, pool=variant)
        missing, unexpected = model.load_state_dict(mean.state_dict(), strict=False)
        assert not unexpected and all(".scene_transformer." in name for name in missing)
        layer = model.graph_encoder.scene_transformer.layers[0]
        assert not layer.residual_scale.any()
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model.catalog_vectors(), expected, rtol=0, atol=0)
        model.train()
        training_loss(model, device).backward()
        # At zero scale only the scales receive gradient; sublayer weights follow once opened.
        assert layer.residual_scale.grad.abs().min() > 0
        assert not layer.linear1.weight.grad.any()
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        model.zero_grad(set_to_none=True)
        training_loss(model, device).backward()
        assert layer.linear1.weight.grad.abs().sum() > 0
        assert layer.attention.in_proj_weight.grad.abs().sum() > 0
        scales = model.execution_report()["scene_transformer_residual_scales"]
        assert len(scales) == 1 and len(scales[0]) == 2 and all(v != 0 for v in scales[0])
    assert "scene_transformer_residual_scales" not in mean.execution_report()


@pytest.mark.parametrize("variant", VARIANTS)
def test_title_only_fallback_and_empty_videos_bypass_transformer(device, variant):
    store = synthetic_store(24, feature_dim=32)
    batch = PackedGraphStore(store).batch([1, 2, 11, 0], device, group_scenes=True)
    model = SceneRoleEncoder(32, 8, variant).to(device)
    result = model(batch)
    fallback = model.scene_mlp(torch.cat([
        model.title_projection(batch["titles"])[2:3], result.new_zeros((1, 24))], -1))
    torch.testing.assert_close(result[2:3], fallback)
    assert not result[3].any()
    empty = PackedGraphStore(store).batch([11, 0], device, group_scenes=True)
    hook = model.scene_transformer.register_forward_hook(
        lambda *_: pytest.fail("no scenes: bypass the scene Transformer"))
    assert torch.isfinite(model(empty)).all()
    hook.remove()


def make_model(store, device, *, pool, pairs=1048576, checkpoint="never", cache=0, dropout=0):
    return GraphSASRec(
        store=store, aggregation=pool, item_count=len(store), max_length=10,
        embedding_dim=32, num_blocks=2, num_heads=2, dropout=dropout, arm="graph",
        execution=GraphExecutionConfig(chunk_items=64, chunk_nodes=100000,
            chunk_scene_pairs=pairs, checkpoint=checkpoint, feature_cache_mb=cache),
    ).to(device)


def training_loss(model, device):
    sequences = torch.tensor([[2, 3, 0], [4, 5, 0], [6, 7, 0]], device=device)
    model.prepare_items(torch.arange(1, len(model.store) + 1, device=device))
    logits = model.user_vectors(sequences) @ model.catalog_vectors().T
    return torch.nn.functional.cross_entropy(logits, torch.tensor([3, 5, 7], device=device))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("checkpoint", ["always", "never"])
def test_joint_loss_chunk_checkpoint_and_reload(device, variant, checkpoint, tmp_path):
    torch.manual_seed(91)
    store = synthetic_store(24, feature_dim=32)
    reference = make_model(store, device, pool=variant)
    activate(reference.graph_encoder.scene_transformer)
    model = make_model(store, device, pool=variant, pairs=1, checkpoint=checkpoint, cache=2048)
    model.load_state_dict(reference.state_dict())
    losses = [training_loss(m, device) for m in (reference, model)]
    torch.testing.assert_close(*losses, atol=2e-6, rtol=1e-5)
    for loss in losses:
        loss.backward()
    for (name, a), (_, b) in zip(reference.named_parameters(), model.named_parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4, msg=name)
    layer = model.graph_encoder.scene_transformer.layers[0]
    encoder = model.graph_encoder
    for weight in (encoder.entity_projection.weight, encoder.action_projection.weight,
                   encoder.context_projection.weight, encoder.role_fusion.weight,
                   encoder.scene_mlp.fc1.weight, layer.attention.in_proj_weight,
                   layer.linear1.weight, layer.residual_scale,
                   model.encoder.layers[0].linear1.weight):
        assert weight.grad is not None and weight.grad.abs().sum() > 0
    assert model.execution_counts["chunks"] > reference.execution_counts["chunks"]
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    model.eval()
    vectors = model.catalog_vectors().detach().clone()
    assert not model.item_vectors(torch.tensor([0, 1], device=device)).any()
    torch.save(model.state_dict(), tmp_path / "transformer.pt")
    restored = make_model(store, device, pool=variant)
    restored.load_state_dict(torch.load(tmp_path / "transformer.pt", weights_only=True))
    restored.eval()
    sequences = torch.tensor([[2, 3, 0]], device=device)
    torch.testing.assert_close(restored.catalog_vectors(), vectors)
    torch.testing.assert_close(restored.score_catalog(sequences), model.score_catalog(sequences))
    for other in ("mean", "gnn"):
        with pytest.raises(RuntimeError):
            make_model(store, device, pool=other).load_state_dict(model.state_dict())


def test_dropout_trains_stochastically_and_checkpoint_replays_it(device):
    store = synthetic_store(24, feature_dim=32)
    torch.manual_seed(3)
    reference = make_model(store, device, pool="transformer", dropout=0.5)
    activate(reference.graph_encoder.scene_transformer)
    model = make_model(store, device, pool="transformer", checkpoint="always", dropout=0.5)
    model.load_state_dict(reference.state_dict())
    reference.train()
    reference.prepare_items(torch.arange(1, 25, device=device))
    first = reference.catalog_vectors().detach().clone()
    reference.clear_item_cache()
    assert not torch.allclose(reference.catalog_vectors(), first)
    losses = []
    for net in (reference, model):
        net.zero_grad(set_to_none=True)
        net.train()
        torch.manual_seed(11)
        losses.append(training_loss(net, device))
        losses[-1].backward()
    torch.testing.assert_close(*losses, atol=2e-6, rtol=1e-5)
    for (name, a), (_, b) in zip(reference.named_parameters(), model.named_parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4, msg=name)
    model.eval()
    with torch.no_grad():
        once = model.catalog_vectors().clone()
        model.clear_item_cache()
        torch.testing.assert_close(model.catalog_vectors(), once, rtol=0, atol=0)
