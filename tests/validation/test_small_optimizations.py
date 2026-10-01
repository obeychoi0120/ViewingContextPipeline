"""Small-only production contracts and the state lookup optimization."""

import copy
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import config_data  # noqa: E402
from graph_fixture import synthetic_store  # noqa: E402
from validation.config import ValidationConfig  # noqa: E402
from validation.features import FeatureError, _load_bge_runtime  # noqa: E402
from validation.graph_model import GraphSASRec, RoleLayer, new_graph_model  # noqa: E402
from validation.model import seed_everything  # noqa: E402
from validation.profiling import RunProfiler, span  # noqa: E402
from validation.recommendation import _new_model  # noqa: E402
from validation.selection import training_signature  # noqa: E402
from validation.representation_provenance import recommendation_identity  # noqa: E402


@pytest.mark.parametrize("branch", ["meta", "desc_qwen_meta", "graph_qwen"])
def test_small_production_text_dimensions_and_masks(tmp_path, branch):
    torch.set_num_threads(1)
    config = ValidationConfig.model_validate(config_data(tmp_path))
    features = {
        "title_values": np.ones((3, 384), dtype=np.float32),
        "video_values": np.arange(3 * 384, dtype=np.float32).reshape(3, 384) / 384,
        "title_available": np.array([True, False, False]),
        "video_available": np.array([True, True, False]),
    }
    model = _new_model(config, item_count=3, branch=branch, features=features, device="cpu")
    captured = []
    hook = model.item_norm.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    vectors = model.item_vectors(torch.arange(4))
    hook.remove()
    assert vectors.shape == (4, 512)
    assert not vectors[[0, 3]].any()
    if branch == "meta":
        assert (model.item_projection.in_features, model.item_projection.out_features) == (384, 512)
        assert not vectors[2].any()
    else:
        assert (model.title_projection.in_features, model.title_projection.out_features) == (384, 128)
        assert isinstance(model.video_projection, torch.nn.Identity)
        assert not any(k.startswith("video_projection.") for k in model.state_dict())
        torch.testing.assert_close(captured[0][1:3, 128:], torch.from_numpy(features["video_values"][:2]))
        assert not captured[0][2, :128].any()
    old = {**features, "title_values": np.ones((3, 1024)), "video_values": np.ones((3, 1024))}
    with pytest.raises(ValueError, match="dimension mismatch"):
        _new_model(config, item_count=3, branch=branch, features=old, device="cpu")
    legacy = model.state_dict()
    weight = "item_projection.weight" if branch == "meta" else "title_projection.weight"
    legacy[weight] = torch.zeros((legacy[weight].shape[0], 1024))
    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_state_dict(legacy)


@pytest.mark.parametrize("pool", ["mean", "attention"])
def test_small_graph_dimensions_shared_projection_and_empty_items(pool):
    torch.set_num_threads(1)
    store = synthetic_store(12)
    model = GraphSASRec(store=store, aggregation=pool, item_count=len(store), max_length=10,
                       embedding_dim=512, num_blocks=2, num_heads=2, dropout=0, arm="graph")
    encoder = model.graph_encoder
    assert (encoder.projection.in_features, encoder.projection.out_features) == (384, 128)
    assert encoder.output_dim == 384
    calls = []
    hook = encoder.projection.register_forward_hook(lambda _, args, out: calls.append(out.shape))
    vectors = model.catalog_vectors()
    hook.remove()
    assert len(calls) == 2  # The same module projects nodes and contexts.
    assert vectors.shape == (12, 512) and not vectors[0].any()
    assert torch.isfinite(vectors).all()
    assert not any(k.startswith("video_projection.") for k in model.state_dict())


def test_embedding_lookup_duplicate_states_matches_indexing_forward_and_gradients(tmp_path):
    torch.set_num_threads(1)
    seed_everything(42)
    current = RoleLayer(128)
    old = copy.deepcopy(current)
    h = torch.randn(7, 128, requires_grad=True)
    old_h = h.detach().clone().requires_grad_(True)
    roles = [torch.empty((0, 2), dtype=torch.long) for _ in range(10)]
    # Duplicate indices, both states, and empty roles must all retain their meaning.
    states = [torch.tensor([[0, 0], [1, 0], [2, 1], [3, 0], [3, 1]])]
    states += [torch.empty((0, 2), dtype=torch.long) for _ in range(4)]
    messages = torch.zeros_like(old_h)
    sizes = old_h.new_zeros(len(old_h))
    for role, pairs in enumerate(states):
        messages.index_add_(0, pairs[:, 0], old.states[role, pairs[:, 1]])
        sizes.index_add_(0, pairs[:, 0], old_h.new_ones(len(pairs)))
    expected = old.norm(old_h + torch.nn.functional.gelu(
        old.self_projection(old_h) + messages / sizes.clamp_min(1).unsqueeze(-1)))
    profiler = RunProfiler(tmp_path, {"deterministic": False, "evaluation_date": "test",
                                      "seed": 42, "arm": "graph_qwen"}, "cpu", 1, operators=True)
    with profiler.sample("selection", 1, 3):
        with span("message_passing"):
            actual = current(h, roles=roles, states=states)
        with span("backward"):
            (actual * torch.arange(128)).sum().backward()
    (expected * torch.arange(128)).sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(h.grad, old_h.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(current.states.grad, old.states.grad, atol=1e-5, rtol=1e-5)
    assert list(current.state_dict()) == list(old.state_dict())
    rows = [json.loads(line) for line in (tmp_path / "profile.jsonl").read_text().splitlines()]
    assert rows[-1]["calls"]["message_passing/state_lookup"] == 5
    operator_path = tmp_path / rows[-1]["operator_summary"]
    assert "EmbeddingBackward" in operator_path.read_text()


@pytest.mark.parametrize("deterministic", [False, True])
def test_seed_policy_preserves_rng_and_switches_determinism(deterministic):
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        seed_everything(123, deterministic=deterministic)
        first = (random.random(), np.random.random(), torch.rand(5))
        seed_everything(123, deterministic=deterministic)
        second = (random.random(), np.random.random(), torch.rand(5))
        assert first[:2] == second[:2]
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        assert torch.are_deterministic_algorithms_enabled() is deterministic
    finally:
        torch.use_deterministic_algorithms(previous)


def test_determinism_default_and_change_invalidate_training_cache(tmp_path):
    value = config_data(tmp_path)
    value["model"].pop("deterministic")
    default = ValidationConfig.model_validate(value)
    explicit = default.model_copy(update={"model": default.model.model_copy(update={"deterministic": False})})
    enabled = default.model_copy(update={"model": default.model.model_copy(update={"deterministic": True})})
    cohort = {"events": [], "manifest": {"selection_hash": "same-input"}}
    assert default.model.deterministic is False
    assert training_signature(None, cohort, default) == training_signature(None, cohort, explicit)
    assert training_signature(None, cohort, default) != training_signature(None, cohort, enabled)


def test_graph_rejects_large_stored_features_before_training(tmp_path, monkeypatch):
    config = ValidationConfig.model_validate(config_data(tmp_path))
    context = SimpleNamespace(representations_dir=tmp_path, scene_aggregation="mean")
    monkeypatch.setattr("validation.graph_model.GraphStore", lambda _: synthetic_store(16, 1024))
    with pytest.raises(ValueError, match="graph feature dimension mismatch"):
        new_graph_model(context, config, "graph_qwen_meta", "cpu")


def test_encoder_identity_and_video_structure_are_in_result_cache(current_context, monkeypatch):
    from validation.recommendation_cache import cache_for

    monkeypatch.setattr("validation.representation_provenance.read_state",
                        lambda *a: {"recommendation_hash": "fixed-values"})
    current = recommendation_identity(current_context, "desc_qwen_meta")
    assert current["encoder"]["embedding_dim"] == 384
    assert current["item_model"]["video_transform"] == "identity"
    original = cache_for(current_context, current).key
    current_context.config["models"]["bge"] = str(current_context.root / "bge-large-en-v1.5")
    assert cache_for(current_context, recommendation_identity(current_context, "desc_qwen_meta")).key != original
    old = {**current, "text_architecture": "sasrec-content-v4"}
    assert cache_for(current_context, old).key != original


def test_actual_encoder_dimension_checked_before_device_transfer(tmp_path, monkeypatch):
    settings = ValidationConfig.model_validate(config_data(tmp_path)).encoder
    settings.model_path.mkdir()
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        AutoModel=SimpleNamespace(from_pretrained=lambda *a, **k:
                                  SimpleNamespace(config=SimpleNamespace(hidden_size=1024))),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    with pytest.raises(FeatureError, match="required BGE small dimension 384"):
        _load_bge_runtime(settings, "cpu")
