"""Contract and learning checks for the final diagram's three item towers."""

import numpy as np
import pytest

from validation.text_features import encode_components, validate_arrays


def test_component_dedup_truncation_and_roundtrip(tmp_path):
    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, texts):
            self.calls.append(texts)
            self.last_truncated_flags = [t == "long" for t in texts]
            return np.arange(len(texts) * 8, dtype=np.float32).reshape(-1, 8)

    encoder, cache = Encoder(), {}
    docs = [
        {"title_text": "same", "video_text": "long"},
        {"title_text": "", "video_text": "same"},
        {"title_text": "", "video_text": ""},
    ]
    arrays, stats = encode_components(docs, 8, encoder, cache)
    other, _ = encode_components(docs, 8, encoder, cache)
    assert encoder.calls == [["same", "long"]]
    assert stats == {
        "title": {"text_count": 1, "truncated_count": 0},
        "video": {"text_count": 2, "truncated_count": 1},
    }
    np.testing.assert_array_equal(arrays["title_values"][0], arrays["video_values"][1])
    path = tmp_path / "features.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path) as stored:
        validate_arrays(stored, 3, 8, docs)
        for key in stored.files:
            np.testing.assert_array_equal(stored[key], other[key])
    with pytest.raises(ValueError, match="rerun"):
        validate_arrays({"values": arrays["title_values"]}, 3, 8)
    arrays["title_available"][2] = True
    with pytest.raises(ValueError, match="availability"):
        validate_arrays(arrays, 3, 8, docs)


@pytest.mark.parametrize("arm", ["metadata", "graph", "desc"])
def test_component_order_masks_and_gradients(arm):
    torch = pytest.importorskip("torch")
    from validation.model import SASRec

    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    features = dict(
        title_values=rng.normal(size=(4, 6)).astype("float32"),
        video_values=rng.normal(size=(4, 6)).astype("float32"),
        title_available=np.array([True, True, False, False]),
        video_available=np.array([True, False, True, False]),
    )
    model = SASRec(4, 10, 8, 2, 2, 0.0, arm=arm, item_features=features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    with torch.no_grad():
        model.item_mlp.fc1.bias.fill_(0.4)
        if arm != "metadata":
            model.title_projection.bias.fill_(1)
    captured = []
    hook = model.item_mlp.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach())
    )
    values = model.item_vectors(torch.arange(5))
    hook.remove()
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in model.item_mlp.modules())
    assert sum(isinstance(m, torch.nn.Linear) for m in model.item_mlp.modules()) == 1
    assert sum(isinstance(m, torch.nn.Linear) for m in model.user_mlp.modules()) == 2
    assert not model.title_features.requires_grad and not model.video_features.requires_grad
    if arm != "metadata":
        assert not captured[0][3, :2].any() and not captured[0][2, 2:].any()
        torch.testing.assert_close(
            captured[0][1, :2], model.title_projection(model.title_features[1])
        )
    users = model.user_vectors(torch.tensor([[1, 2, 0], [2, 1, 3]]))
    loss = torch.nn.functional.cross_entropy(users @ values[1:4].T, torch.tensor([0, 2]))
    loss.backward()
    assert model.item_mlp.fc1.weight.grad is not None
    if arm == "metadata":
        assert model.item_projection.weight.grad is not None
    else:
        assert model.title_projection.weight.grad is not None
        assert model.video_projection.weight.grad is not None
        assert model.video_projection.weight.grad.abs().sum() > 0
        torch.testing.assert_close(
            captured[0][1, 2:], model.video_projection(model.video_features[1]).detach()
        )
    optimizer.step()
    assert not model.item_vectors(torch.tensor([0, 4])).any()
    if arm == "metadata":
        assert not model.item_vectors(torch.tensor([3])).any()


def test_component_boundaries_are_in_semantic_identity():
    from validation.cache_identity import semantic_document_hash
    shared = dict(content_id='1', text='a\n\nb\n\nc',
                  composition_policy='title-video-separate-bge/v1')
    left = {**shared, 'title_text': 'a', 'video_text': 'b\n\nc'}
    right = {**shared, 'title_text': 'a\n\nb', 'video_text': 'c'}
    assert semantic_document_hash(left) != semantic_document_hash(right)


def test_single_layer_residual_formula_gradient_and_legacy_rejection():
    torch = pytest.importorskip("torch")
    from validation.model import SingleLayerResidualMLP

    block = SingleLayerResidualMLP(2)
    weight = np.array([[2.0, -1.0], [-1.0, 0.5]], dtype=np.float32)
    bias = np.array([0.25, -0.25], dtype=np.float32)
    inputs = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    with torch.no_grad():
        block.fc1.weight.copy_(torch.from_numpy(weight))
        block.fc1.bias.copy_(torch.from_numpy(bias))
    values = torch.tensor(inputs, requires_grad=True)
    pre_activation = inputs @ weight.T + bias
    expected = inputs + np.maximum(pre_activation, 0)
    output = block(values)
    np.testing.assert_allclose(output.detach().numpy(), expected)
    output.sum().backward()
    active = (pre_activation > 0).astype(np.float32)
    np.testing.assert_allclose(values.grad.numpy(), 1 + active @ weight)
    np.testing.assert_allclose(block.fc1.weight.grad.numpy(), active.T @ inputs)
    legacy = {**block.state_dict(), "fc2.weight": torch.eye(2), "fc2.bias": torch.zeros(2)}
    with pytest.raises(RuntimeError, match="Unexpected key"):
        block.load_state_dict(legacy)
