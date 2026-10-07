"""Large-only production contracts and residual item/scene learning."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import config_data  # noqa: E402
from graph_fixture import synthetic_store  # noqa: E402
from validation.config import ValidationConfig  # noqa: E402
from validation.features import FeatureError, _load_bge_runtime  # noqa: E402
from validation.graph_model import GraphSASRec  # noqa: E402
from validation.recommendation import _new_model  # noqa: E402
from validation.representation_provenance import recommendation_identity  # noqa: E402


@pytest.mark.parametrize("branch", ["meta", "desc_qwen_meta", "graph_qwen"])
@pytest.mark.parametrize("device", ["cpu"] + ([os.environ["GRAPH_TEST_CUDA"]] if os.environ.get("GRAPH_TEST_CUDA") else []))
def test_large_production_text_dimensions_and_masks(tmp_path, branch, device):
    torch.set_num_threads(1)
    config = ValidationConfig.model_validate(config_data(tmp_path))
    features = {
        "title_values": np.ones((3, 1024), dtype=np.float32),
        "video_values": np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024) / 1024,
        "title_available": np.array([True, False, False]),
        "video_available": np.array([True, True, False]),
    }
    model = _new_model(config, item_count=3, branch=branch, features=features, device=device)
    captured = []
    hook = model.item_mlp.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    vectors = model.item_vectors(torch.arange(4, device=device))
    hook.remove()
    assert vectors.shape == (4, 512)
    assert not vectors[[0, 3]].any()
    if branch == "meta":
        assert (model.item_projection.in_features, model.item_projection.out_features) == (1024, 512)
        assert not vectors[2].any()
    else:
        assert (model.title_projection.in_features, model.title_projection.out_features) == (1024, 128)
        assert (model.video_projection.in_features, model.video_projection.out_features) == (1024, 384)
        torch.testing.assert_close(
            captured[0][1:3, 128:], model.video_projection(model.video_features[1:3])
        )
        assert not captured[0][2, :128].any()
    old = {**features, "title_values": np.ones((3, 384)), "video_values": np.ones((3, 384))}
    with pytest.raises(ValueError, match="dimension mismatch"):
        _new_model(config, item_count=3, branch=branch, features=old, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    users = model.user_vectors(torch.tensor([[1, 2, 0], [2, 1, 0]], device=device))
    loss = torch.nn.functional.cross_entropy(users @ vectors[1:].T, torch.tensor([0, 1], device=device))
    loss.backward()
    projections = [model.item_projection] if branch == "meta" else [model.title_projection, model.video_projection]
    layers = [*projections, model.item_mlp.fc1, model.encoder.layers[0].linear1]
    before = [layer.weight.detach().clone() for layer in layers]
    for layer in layers:
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    assert not model.title_features.requires_grad and not model.video_features.requires_grad
    optimizer.step()
    assert all(not torch.equal(old, layer.weight) for old, layer in zip(before, layers, strict=True))
    assert not model.item_vectors(torch.tensor([0, 3], device=device)).any()
    legacy = model.state_dict()
    weight = "item_projection.weight" if branch == "meta" else "title_projection.weight"
    legacy[weight] = torch.zeros((legacy[weight].shape[0], 384))
    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_state_dict(legacy)


@pytest.mark.parametrize("pool", ["mean"])
@pytest.mark.parametrize("device", ["cpu"] + ([os.environ["GRAPH_TEST_CUDA"]] if os.environ.get("GRAPH_TEST_CUDA") else []))
def test_large_graph_dimensions_independent_projections_and_empty_items(pool, device):
    torch.set_num_threads(1)
    store = synthetic_store(12)
    model = GraphSASRec(store=store, aggregation=pool, item_count=len(store), max_length=10,
                       embedding_dim=512, num_blocks=2, num_heads=2, dropout=0, arm="graph").to(device)
    encoder = model.graph_encoder
    projections = [encoder.title_projection, encoder.entity_projection,
                   encoder.action_projection, encoder.context_projection]
    assert all((p.in_features, p.out_features) == (1024, 128) for p in projections)
    assert len({p.weight.data_ptr() for p in projections}) == 4
    assert encoder.output_dim == 512
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in encoder.modules())
    vectors = model.catalog_vectors()
    assert vectors.shape == (12, 512) and not vectors[0].any()
    assert torch.isfinite(vectors).all()
    assert not any(k.startswith("video_projection.") for k in model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    users = model.user_vectors(torch.tensor([[2, 3, 0], [3, 4, 0]], device=device))
    loss = torch.nn.functional.cross_entropy(users @ vectors.T, torch.tensor([2, 3], device=device))
    loss.backward()
    layers = [*projections, encoder.role_fusion, encoder.scene_mlp.fc1,
              model.encoder.layers[0].linear1]
    before = [layer.weight.detach().clone() for layer in layers]
    states_before = encoder.role_states.detach().clone()
    for layer in layers:
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    assert encoder.role_states.grad is not None and encoder.role_states.grad.abs().sum() > 0
    optimizer.step()
    assert all(not torch.equal(old, layer.weight) for old, layer in zip(before, layers, strict=True))
    assert not torch.equal(states_before, encoder.role_states)
    model.clear_item_cache()
    assert not model.item_vectors(torch.tensor([0, 1], device=device)).any()


def test_encoder_identity_and_video_structure_are_in_result_cache(current_context, monkeypatch):
    from validation.recommendation_cache import cache_for

    monkeypatch.setattr("validation.representation_provenance.read_state",
                        lambda *a: {"recommendation_hash": "fixed-values"})
    current = recommendation_identity(current_context, "desc_qwen_meta")
    assert current["encoder"]["embedding_dim"] == 1024
    assert current["item_model"]["video_transform"] == "linear"
    original = cache_for(current_context, current).key
    current_context.config["models"]["bge"] = str(current_context.root / "bge-small-en-v1.5")
    assert cache_for(current_context, recommendation_identity(current_context, "desc_qwen_meta")).key != original
    assert current["text_architecture"] == "sasrec-content-v8"
    assert current["item_model"]["item_mlp_linear_layers"] == 1
    old = {**current, "text_architecture": "sasrec-content-v7"}
    assert cache_for(current_context, old).key != original


def test_actual_encoder_dimension_checked_before_device_transfer(tmp_path, monkeypatch):
    settings = ValidationConfig.model_validate(config_data(tmp_path)).encoder
    settings.model_path.mkdir()
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        AutoModel=SimpleNamespace(from_pretrained=lambda *a, **k:
                                  SimpleNamespace(config=SimpleNamespace(hidden_size=384))),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    with pytest.raises(FeatureError, match="required BGE large dimension 1024"):
        _load_bge_runtime(settings, "cpu")
