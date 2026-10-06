"""V7 role gathering/scene pooling parity with a loop reference, including real backward passes."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import graph_v7_reference as reference  # noqa: E402
from graph_fixture import synthetic_store  # noqa: E402
from validation.graph_batching import PackedGraphStore  # noqa: E402
from validation.graph_model import GraphSASRec, SceneRoleEncoder, graph_batch  # noqa: E402
from validation.config import GraphExecutionConfig  # noqa: E402
from validation.rolling_execution import negative_mask  # noqa: E402


@pytest.fixture(
    params=["cpu"] + ([os.environ["GRAPH_TEST_CUDA"]] if os.environ.get("GRAPH_TEST_CUDA") else [])
)
def device(request):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch.device(request.param)


def test_ragged_gathers_preserve_order_duplicates_and_empty_videos(device):
    store = synthetic_store(16, feature_dim=32)
    for items in ([0], [11, 2, 1, 11, 2, 7], list(range(16))):
        old, new = reference.graph_batch(store, items, device), graph_batch(store, items, device)
        for name in old:
            if name == "offsets":
                assert old[name] == new[name]
            else:
                torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)


@pytest.mark.parametrize("pool", ["mean"])
def test_encoder_outputs_and_all_parameter_gradients(pool, device):
    store = synthetic_store(20, feature_dim=32)
    old = reference.SceneRoleEncoder(feature_dim=32, hidden=16, aggregation=pool).to(device)
    new = SceneRoleEncoder(feature_dim=32, hidden=16, aggregation=pool).to(device)
    new.load_state_dict(old.state_dict())
    items = [0, 5, 2, 11, 7, 9, 1]
    inputs = [
        reference.graph_batch(store, items, device),
        PackedGraphStore(store).batch(items, device),
    ]
    outputs = [model(batch) for model, batch in zip((old, new), inputs, strict=True)]
    torch.testing.assert_close(*outputs, atol=2e-6, rtol=1e-5)
    for out in outputs:
        (
            out * torch.linspace(-1, 1, out.numel(), device=device).reshape(out.shape)
        ).sum().backward()
    for a, b in zip(old.parameters(), new.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-4, rtol=2e-4)


def make_model(cls, store, pool, device, checkpoint="auto"):
    kwargs = dict(
        store=store,
        aggregation=pool,
        item_count=len(store),
        max_length=10,
        embedding_dim=32,
        num_blocks=2,
        num_heads=2,
        dropout=0.1,
        arm="graph",
    )
    if cls is GraphSASRec:
        kwargs["execution"] = GraphExecutionConfig(
            chunk_items=7, chunk_nodes=1000, checkpoint=checkpoint
        )
    return cls(**kwargs).to(device)


def batch_loss(model, device, seed=101):
    rng = np.random.default_rng(seed)
    sequences = torch.as_tensor(rng.integers(1, model.item_count + 1, size=(16, 10)), device=device)
    sequences[:, -2:] = 0
    targets = torch.as_tensor(rng.integers(1, model.item_count + 1, size=16), device=device)
    model.prepare_items(torch.cat([sequences.flatten(), targets]))
    users = model.user_vectors(sequences)
    logits = users @ model.item_vectors(targets).T
    logits = logits.masked_fill(negative_mask(sequences, targets), -1e4)
    return torch.nn.functional.cross_entropy(logits, torch.arange(16, device=device))


@pytest.mark.parametrize("pool", ["mean"])
@pytest.mark.parametrize("checkpoint", ["always", "never"])
def test_recommendation_loss_gradients_updates_and_reload(pool, checkpoint, device):
    store = synthetic_store(24, feature_dim=32)
    old = make_model(reference.GraphSASRec, store, pool, device)
    new = make_model(GraphSASRec, store, pool, device, checkpoint)
    new.load_state_dict(old.state_dict())
    optimizers = [torch.optim.SGD(m.parameters(), lr=1e-4) for m in (old, new)]
    for step in range(2):
        losses = []
        for model, optimizer in zip((old, new), optimizers, strict=True):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(123 + step)
            loss = batch_loss(model, device, 101 + step)
            loss.backward()
            losses.append(loss.detach())
        torch.testing.assert_close(*losses, atol=2e-5, rtol=2e-5)
        for a, b in zip(old.parameters(), new.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-4, rtol=5e-4)
        for model, optimizer in zip((old, new), optimizers, strict=True):
            optimizer.step()
            model.clear_item_cache()
    for m in (old, new):
        m.eval()
    with torch.no_grad():
        old_catalog, new_catalog = old.catalog_vectors(), new.catalog_vectors()
        torch.testing.assert_close(old_catalog, new_catalog, atol=2e-5, rtol=1e-4)
        histories = torch.tensor([[1, 2, 3, 0], [7, 8, 0, 0]], device=device)
        a, b = (
            old.user_vectors(histories) @ old_catalog.T,
            new.user_vectors(histories) @ new_catalog.T,
        )
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(a.argsort(dim=1, stable=True), b.argsort(dim=1, stable=True))
        restored = make_model(GraphSASRec, store, pool, device).eval()
        restored.load_state_dict(new.state_dict())
        torch.testing.assert_close(new.catalog_vectors(), restored.catalog_vectors())


def test_chunk_feature_cache_and_catalog_outputs_and_gradients_agree(device):
    store = synthetic_store(12, feature_dim=32)
    models = [make_model(GraphSASRec, store, "mean", device, "never") for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    models[0].execution.chunk_items = 1
    models[1].execution.chunk_items = 12
    models[0].packed.feature_cache_mb = 0
    models[1].packed.feature_cache_mb = 2048
    outputs = []
    for model in models:
        output = model.catalog_vectors()
        outputs.append(output.detach())
        (output * torch.linspace(-1, 1, output.numel(), device=device).reshape(output.shape)).sum().backward()
    torch.testing.assert_close(*outputs, atol=2e-6, rtol=1e-5)
    for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-4, rtol=5e-4)
    if device.type == "cuda":
        assert models[0].packed.feature_cache is None
        assert models[1].packed.feature_cache is not None
    for model in models:
        model.eval()
        with torch.no_grad():
            cached = model.catalog_vectors()
            assert model.catalog_vectors() is not None
            torch.testing.assert_close(model.item_vectors(torch.arange(1, 13, device=device)), cached)
            model.clear_item_cache()
            torch.testing.assert_close(model.catalog_vectors(), cached)
