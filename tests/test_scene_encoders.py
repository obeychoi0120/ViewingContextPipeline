"""Mean/hypergraph Arms, with Role Fusion only in G2."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from arm_registry import registry
from test_scene_extensions import make_model, synthetic_store
from validation.config import GraphExecutionConfig
from validation.graph_batching import PackedGraphStore
from validation.graph_model import DescriptionSceneEncoder, GraphSASRec, SceneRoleEncoder, parameter_report


@pytest.mark.parametrize("name", ["g0", "g1", "g2"])
def test_graph_role_fusion_follows_arm_definition(name):
    arm = registry({})[name]
    store = synthetic_store(videos=8)
    batch = PackedGraphStore(store).batch(np.arange(len(store)), "cpu", scene_segments=bool(arm.extensions))
    encoder = SceneRoleEncoder(16, hidden=4, aggregation=arm.aggregation,
                               role_fusion=arm.role_fusion, extensions=arm.extensions)
    if arm.extensions:
        with torch.no_grad():
            encoder.segment_gates.fill_(0.5)
    actual = encoder(batch)
    modified = deepcopy(batch)
    modified["role_entities"] = modified["role_entities"].flip(1)
    modified["role_states"] = 1 - modified["role_states"]
    if name == "g2":
        assert not torch.allclose(encoder(modified), actual)
        actual.square().sum().backward()
        assert encoder.role_fusion.weight.grad.abs().sum() > 0
    else:
        del modified["role_entities"], modified["role_states"]
        torch.testing.assert_close(encoder(modified), actual, rtol=0, atol=0)


@pytest.mark.parametrize("name,total", [("d", 7624704), ("g0", 7624704),
                                        ("g1", 8675330), ("g2", 8758402)])
def test_registered_parameter_counts_and_absent_fusion_report(name, total):
    arm = registry({})[name]
    store = synthetic_store(videos=4, feature_dim=1024)
    model = GraphSASRec(store=store, aggregation=arm.aggregation, item_count=len(store),
                       max_length=10, embedding_dim=512, num_blocks=2, num_heads=2, dropout=0.1,
                       arm=name, representation=arm.representation,
                       role_fusion=arm.role_fusion, extensions=arm.extensions)
    assert parameter_report(model, name) == {
        "trainable": total, "active_trainable": total,
        "removed_role_fusion": 83072 if name in ("g0", "g1") else 0,
    }
    assert not any("gcn" in key or "query" in key or "key" in key for key in model.state_dict())


@pytest.mark.parametrize("encoder", [SceneRoleEncoder, DescriptionSceneEncoder])
def test_scene_encoders_reject_removed_aggregation(encoder):
    with pytest.raises(ValueError, match="scene aggregation must be mean"):
        encoder(16, aggregation="gnn")


def test_scene_order_and_video_isolation_with_hypergraph():
    store = synthetic_store(videos=12)
    packed = PackedGraphStore(store)
    encoder = SceneRoleEncoder(16, hidden=4, extensions=("segments",))
    with torch.no_grad():
        encoder.segment_gates.fill_(0.5)
    batch = packed.batch(np.arange(len(store)), "cpu", scene_segments=True)
    expected = encoder(batch)
    separate = torch.cat([encoder(packed.batch([video], "cpu", scene_segments=True))
                          for video in range(len(store))])
    torch.testing.assert_close(separate, expected, rtol=1e-5, atol=1e-6)
    permutation = torch.cat([torch.arange(stop - 1, start - 1, -1)
                             for start, stop in zip(batch["offsets"][:-1], batch["offsets"][1:])])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    reordered = deepcopy(batch)
    reordered["contexts"] = batch["contexts"][permutation]
    reordered["node_scenes"] = inverse[batch["node_scenes"]]
    reordered["segment_scenes"] = inverse[batch["segment_scenes"]]
    torch.testing.assert_close(encoder(reordered), expected, rtol=1e-5, atol=1e-6)
    modified = deepcopy(batch)
    modified["titles"][1] += 100
    changed = encoder(modified)
    others = torch.arange(len(store)) != 1
    torch.testing.assert_close(changed[others], expected[others], rtol=0, atol=0)
    assert not torch.allclose(changed[1], expected[1])


def test_legacy_scene_pair_limit_is_ignored_and_checkpoint_preserves_gradients():
    store = synthetic_store(videos=12)
    reference = make_model(store, ("segments",), items=12, role_fusion=False).train()
    with torch.no_grad():
        reference.graph_encoder.segment_gates.fill_(0.5)
    limited = make_model(store, ("segments",), items=12, role_fusion=False).train()
    limited.load_state_dict(reference.state_dict())
    limited.execution = GraphExecutionConfig(chunk_items=3, chunk_nodes=100000,
                                            chunk_scene_pairs=1, checkpoint="always", feature_cache_mb=0)
    expected, actual = reference.catalog_vectors(), limited.catalog_vectors()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for (name, left), (_, right) in zip(reference.named_parameters(), limited.named_parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, rtol=1e-4, atol=1e-4, msg=name)
    assert limited.execution_counts["chunks"] == 4
    assert reference.execution_counts["chunks"] == 1
    assert limited.execution_counts["checkpoint_batches"] == 1
