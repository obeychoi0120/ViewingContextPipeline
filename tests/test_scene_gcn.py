"""Common cosine GCN and the G0 -> G0+A -> G0+A+B experiment."""

from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from arm_registry import registry
from test_aggregation_arms import description_batch
from test_scene_extensions import make_model, synthetic_store
from validation.config import GraphExecutionConfig
from validation.graph_batching import PackedGraphStore
from validation.graph_model import DescriptionSceneEncoder, SceneRoleEncoder, parameter_report
from validation.scene_gcn import SceneGCN, normalized_similarity, scene_groups


def test_cosine_adjacency_handles_negative_and_zero_vectors():
    features = torch.tensor([[1., 0.], [1., 1.], [-1., 0.], [0., 0.]])
    cosine = 1 / math.sqrt(2)
    expected = torch.tensor([
        [1 / (1 + cosine), cosine / (1 + cosine), 0., 0.],
        [cosine / (1 + cosine), 1 / (1 + cosine), 0., 0.],
        [0., 0., 1., 0.], [0., 0., 0., 1.],
    ])
    torch.testing.assert_close(normalized_similarity(features), expected)
    assert torch.isfinite(normalized_similarity(features)).all()


def test_gcn_residual_formula_gradients_permutation_and_video_separation():
    gcn = SceneGCN(4)
    with torch.no_grad():
        gcn.projection.weight.fill_(0.1)
        gcn.projection.bias.fill_(0.2)
    scenes = (torch.rand(6, 4) + 0.5).requires_grad_()
    features = (torch.rand(6, 3) + 0.5).requires_grad_()
    groups = [torch.as_tensor(v) for v in scene_groups([0, 3, 6]).values()]
    actual = gcn(scenes, features, groups)
    adjacent = torch.stack([normalized_similarity(features[:3]), normalized_similarity(features[3:])])
    messages = (adjacent @ scenes.reshape(2, 3, 4)).reshape(6, 4)
    torch.testing.assert_close(actual, scenes + torch.relu(gcn.projection(messages)))
    separate = torch.cat([gcn(scenes[s:s + 3], features[s:s + 3], [torch.arange(3)[None]])
                          for s in (0, 3)])
    torch.testing.assert_close(actual, separate)
    permutation = torch.tensor([2, 0, 1, 5, 3, 4])
    torch.testing.assert_close(gcn(scenes[permutation], features[permutation], groups), actual[permutation])
    actual.square().sum().backward()
    for grad in (features.grad, scenes.grad, gcn.projection.weight.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_description_gcn_uses_title_free_projected_description():
    encoder = DescriptionSceneEncoder(4, hidden=2, aggregation="gnn")
    batch = description_batch()
    captured = []
    hook = encoder.gcn.register_forward_pre_hook(lambda module, args: captured.append(args))
    try:
        result = encoder(batch)
        modified = deepcopy(batch)
        modified["titles"] += 10
        encoder(modified)
    finally:
        hook.remove()
    torch.testing.assert_close(captured[0][1], encoder.description_projection(batch["contexts"]))
    torch.testing.assert_close(captured[0][1], captured[1][1], rtol=0, atol=0)
    assert not torch.allclose(captured[0][0], captured[1][0])
    assert captured[0][0].shape == (3, 8) and captured[0][1].shape == (3, 6)
    torch.testing.assert_close(result[3], torch.zeros(8), rtol=0, atol=0)
    title = encoder.title_projection(batch["titles"])[1]
    torch.testing.assert_close(result[1], encoder.scene_mlp(torch.cat([title, torch.zeros(6)])))
    result.square().sum().backward()
    assert encoder.gcn.projection.weight.grad.abs().sum() > 0
    assert encoder.description_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["g0", "g1", "g2"])
def test_graph_gcn_content_and_role_fusion_follow_arm_definition(name):
    arm = registry({})[name]
    store = synthetic_store(videos=8)
    batch = PackedGraphStore(store).batch(np.arange(len(store)), "cpu", group_scenes=True,
                                         scene_segments=bool(arm.extensions))
    encoder = SceneRoleEncoder(16, hidden=4, aggregation=arm.aggregation,
                               role_fusion=arm.role_fusion, extensions=arm.extensions)
    if arm.extensions:
        with torch.no_grad():
            encoder.segment_gates.fill_(0.5)
    captured, inputs = [], []
    hooks = [encoder.gcn.register_forward_pre_hook(lambda m, args: captured.append(args)),
             encoder.scene_mlp.register_forward_pre_hook(lambda m, args: inputs.append(args[0]))]
    try:
        actual = encoder(batch)
    finally:
        for hook in hooks:
            hook.remove()
    torch.testing.assert_close(captured[0][1], inputs[0][:, 4:], rtol=0, atol=0)
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


def test_gcn_and_module_parameter_counts_and_absent_fusion_report():
    encoders = {}
    for name in ("g0", "g1", "g2"):
        arm = registry({})[name]
        encoders[name] = SceneRoleEncoder(1024, hidden=128, aggregation=arm.aggregation,
                                          role_fusion=arm.role_fusion, extensions=arm.extensions)
        assert sum(p.numel() for p in encoders[name].gcn.parameters()) == 262656
        assert not any("query" in key or "key" in key for key in encoders[name].state_dict())
        model = torch.nn.Module()
        model.graph_encoder = encoders[name]
        assert parameter_report(model, name)["removed_role_fusion"] == (0 if name == "g2" else 83072)
    count = {k: sum(p.numel() for p in v.parameters()) for k, v in encoders.items()}
    assert count["g1"] - count["g0"] == 1050626
    assert count["g2"] - count["g1"] == 83072
    description = DescriptionSceneEncoder(1024, hidden=128, aggregation="gnn")
    assert sum(p.numel() for p in description.gcn.parameters()) == 262656


def test_scene_pair_chunk_limit_and_checkpoint_preserve_outputs_and_gradients():
    store = synthetic_store(videos=12)
    reference = make_model(store, ("segments",), items=12, role_fusion=False).train()
    with torch.no_grad():
        reference.graph_encoder.segment_gates.fill_(0.5)
    limited = make_model(store, ("segments",), items=12, role_fusion=False).train()
    limited.load_state_dict(reference.state_dict())
    limited.execution = GraphExecutionConfig(chunk_items=12, chunk_nodes=100000,
                                            chunk_scene_pairs=1, checkpoint="always", feature_cache_mb=0)
    expected, actual = reference.catalog_vectors(), limited.catalog_vectors()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for (name, left), (_, right) in zip(reference.named_parameters(), limited.named_parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, rtol=1e-4, atol=1e-4, msg=name)
    assert limited.execution_counts["chunks"] > reference.execution_counts["chunks"]
    assert limited.execution_counts["checkpoint_batches"] == 1
