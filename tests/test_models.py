
import numpy as np
import pytest
import torch

from helpers import graph_store, simple_model
from validation.graph_batching import PackedGraphStore
from validation.graph_model import SceneRoleEncoder

pytestmark = pytest.mark.torch




def test_text_model_masks_missing_components():
    features = {
        "title_values": np.array([[1, 2, 3, 4], [0, 0, 0, 0], [1, 1, 1, 1]], np.float32),
        "video_values": np.array([[0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 2, 2]], np.float32),
        "title_available": np.array([True, False, True]),
        "video_available": np.array([False, False, True]),
    }
    model = simple_model(3, features=features)
    with torch.no_grad():
        model.title_projection.weight.fill_(0.2)
        model.title_projection.bias.fill_(3)
        model.video_projection.weight.fill_(0.3)
        model.video_projection.bias.fill_(5)
        model.item_mlp.fc1.weight.zero_()
        model.item_mlp.fc1.bias.fill_(7)
    values = model.item_vectors(torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(values[0], torch.zeros(8))
    torch.testing.assert_close(values[2], torch.zeros(8))
    torch.testing.assert_close(values[1, :2], torch.full((2,), 12.0))
    torch.testing.assert_close(values[1, 2:], torch.full((6,), 7.0))
    assert torch.all(values[3] != 0)
    # A missing video projection contributes zero to the pre-MLP input, even with bias.
    seen = []
    handle = model.item_mlp.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach()))
    try:
        model.item_vectors(torch.tensor([1]))
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0][0, 2:], torch.zeros(6))


def test_sasrec_causal_prefix():
    model = simple_model().eval()
    a = model.encode(torch.tensor([[1, 2, 3, 4]]))
    b = model.encode(torch.tensor([[1, 2, 8, 9]]))
    torch.testing.assert_close(a[:, :2], b[:, :2], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(a[:, 2:], b[:, 2:])


def test_role_fusion_matches_reference():
    encoder = SceneRoleEncoder(feature_dim=2, hidden=1)
    with torch.no_grad():
        for module in (
            encoder.title_projection,
            encoder.entity_projection,
            encoder.action_projection,
            encoder.context_projection,
        ):
            module.bias.zero_()
        encoder.title_projection.weight.copy_(torch.tensor([[1.0, 0.0]]))
        encoder.entity_projection.weight.copy_(torch.tensor([[1.0, 0.0]]))
        encoder.action_projection.weight.copy_(torch.tensor([[0.0, 1.0]]))
        encoder.context_projection.weight.copy_(torch.tensor([[1.0, 1.0]]))
        encoder.role_fusion.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))
        encoder.role_fusion.bias.zero_()
        encoder.role_states.copy_(
            torch.tensor([[[11.0], [21.0]], [[12.0], [22.0]], [[13.0], [23.0]], [[14.0], [24.0]]])
        )
        encoder.scene_mlp.fc1.weight.zero_()
        encoder.scene_mlp.fc1.bias.zero_()
    batch = dict(
        features=torch.tensor([[2.0, 0.0], [3.0, 0.0], [0.0, 5.0]]),
        types=torch.tensor([0, 0, 1]),
        entity_ids=torch.tensor([0, 1]),
        action_ids=torch.tensor([2]),
        role_entities=torch.tensor([[0, 1, -1, -1]]),
        role_states=torch.tensor([[0, 0, 0, 1]]),
        node_scenes=torch.tensor([0, 0, 0]),
        scene_videos=torch.tensor([0]),
        contexts=torch.tensor([[1.0, 2.0]]),
        titles=torch.tensor([[7.0, 0.0]]),
        title_available=torch.tensor([1]),
        available=torch.tensor([1]),
        offsets=[0, 1],
    )
    output = encoder(batch)
    # action = 5 + ReLU(5 + 2*2 + 3*3 + 4*13 + 5*24) = 195
    torch.testing.assert_close(output, torch.tensor([[7.0, 2.5, 195.0, 3.0]]))
    output.sum().backward()
    assert encoder.role_states.grad[2, 0, 0] != 0
    assert encoder.role_states.grad[3, 1, 0] != 0


def test_mean_aggregation_matches_reference(context):
    arrays, _, _, _ = graph_store(context)
    batch = PackedGraphStore(arrays).batch([0, 1], "cpu")
    encoder = SceneRoleEncoder(hidden=2)
    captured = []
    handle = encoder.scene_mlp.register_forward_hook(lambda m, args, out: captured.append(out))
    try:
        actual = encoder(batch)
    finally:
        handle.remove()
    scenes = captured[0]
    expected = torch.stack([(scenes[0] + scenes[1]) / 2, scenes[2]])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    swapped = PackedGraphStore(arrays).batch([1, 0], "cpu")
    torch.testing.assert_close(encoder(swapped), expected.flip(0), rtol=1e-5, atol=1e-6)
