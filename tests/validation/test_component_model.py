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
        model.item_norm.bias.fill_(0.4)
        if arm != "metadata":
            model.title_projection.bias.fill_(1)
    captured = []
    hook = model.item_norm.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach())
    )
    values = model.item_vectors(torch.arange(5))
    hook.remove()
    assert not hasattr(model, "item_mlp")
    assert not model.title_features.requires_grad and not model.video_features.requires_grad
    if arm != "metadata":
        assert not captured[0][3, :2].any() and not captured[0][2, 2:].any()
        torch.testing.assert_close(
            captured[0][1, :2], model.title_projection(model.title_features[1])
        )
    users = model.user_vectors(torch.tensor([[1, 2, 0], [2, 1, 3]]))
    loss = torch.nn.functional.cross_entropy(users @ values[1:4].T, torch.tensor([0, 2]))
    loss.backward()
    assert model.item_norm.weight.grad is not None
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


def test_metadata_array_and_component_adapters_agree():
    torch = pytest.importorskip("torch")
    from validation.model import SASRec

    features = np.random.default_rng(1).normal(size=(3, 16)).astype("float32")
    features[1] = 0
    kwargs = dict(
        item_count=3,
        max_length=10,
        embedding_dim=8,
        num_blocks=2,
        num_heads=2,
        dropout=0.0,
        arm="metadata",
    )
    graph_meta = SASRec(**kwargs, item_features=features)
    text_meta = SASRec(
        **kwargs,
        item_features=dict(
            title_values=features,
            video_values=np.zeros_like(features),
            title_available=np.array([True, False, True]),
            video_available=np.zeros(3, dtype=bool),
        ),
    )
    text_meta.load_state_dict(graph_meta.state_dict())
    graph_meta.eval()
    text_meta.eval()
    torch.testing.assert_close(graph_meta.catalog_vectors(), text_meta.catalog_vectors())
    history = torch.tensor([[1, 3, 0]])
    torch.testing.assert_close(graph_meta.score_catalog(history), text_meta.score_catalog(history))


def test_component_boundaries_are_in_semantic_identity():
    from validation.cache_identity import semantic_document_hash
    shared = dict(content_id='1', text='a\n\nb\n\nc',
                  composition_policy='title-video-separate-bge/v1')
    left = {**shared, 'title_text': 'a', 'video_text': 'b\n\nc'}
    right = {**shared, 'title_text': 'a\n\nb', 'video_text': 'c'}
    assert semantic_document_hash(left) != semantic_document_hash(right)
