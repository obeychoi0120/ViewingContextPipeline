
import pytest
import torch

from helpers import simple_model
from validation.graph_model import SceneRoleEncoder

pytestmark = pytest.mark.torch




def _check_sasrec_causal_prefix():
    model = simple_model().eval()
    a = model.encode(torch.tensor([[1, 2, 3, 4]]))
    b = model.encode(torch.tensor([[1, 2, 8, 9]]))
    torch.testing.assert_close(a[:, :2], b[:, :2], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(a[:, 2:], b[:, 2:])


def _check_role_fusion_matches_reference():
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


def test_sasrec_causal_prefix_and_role_fusion():
    # Sasrec causal prefix.
    _check_sasrec_causal_prefix()
    # Role fusion matches reference.
    _check_role_fusion_matches_reference()
