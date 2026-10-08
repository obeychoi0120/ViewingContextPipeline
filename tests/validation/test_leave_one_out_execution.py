"""Loss and evaluation parity with an independent scalar implementation."""
from copy import deepcopy
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import leave_one_out_reference as reference  # noqa: E402
from validation import leave_one_out_recommendation as optimized  # noqa: E402
from validation.model import SASRec, seed_everything  # noqa: E402
from validation.leave_one_out_data import DAY, EventTable  # noqa: E402
from validation.leave_one_out_execution import masked_ranks  # noqa: E402
from validation.metrics import metrics_from_rank  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(params=["cpu"] + (["cuda:0"] if os.environ.get("ROLLING_TEST_CUDA") == "1" else []))
def device(request):
    return torch.device(request.param)


def fixture_table(users=8, count=32):
    return EventTable([
        {"event_id": user * count + i, "user_id": str(user),
         "item_id": str((i * 7 + user * 3) % 31 + 1), "timestamp": (i // 2) * DAY}
        for user in range(users) for i in range(count)
    ])


def fixture_model(table, device, seed=42):
    seed_everything(seed)
    features = np.random.default_rng(17).normal(size=(len(table.items), 12)).astype(np.float32)
    features[::7] = 0
    return SASRec(len(table.items), 10, 16, 1, 2, 0.1,
                  arm="graph", item_features=features).to(device)


def test_rank_parity_full_history_ties_repeated_targets(device):
    table = fixture_table()
    ids = table.select()
    scores = np.random.default_rng(31).integers(-2, 3, (len(ids), len(table.items))).astype(np.float32)
    scores[::3] = 0
    expected = []
    for event, values in zip(ids, scores, strict=True):
        masked = torch.tensor(values, device=device)
        history = torch.tensor(table.history(int(event)), dtype=torch.long, device=device) - 1
        masked[history] = -torch.inf
        order = torch.argsort(masked, descending=True)
        expected.append(int((order == int(table.targets[event]) - 1).long().argmax()) + 1)
    actual = masked_ranks(torch.tensor(scores, device=device), table, ids)
    assert actual.cpu().tolist() == expected
    for bad in (np.nan, np.inf, -np.inf):
        broken = torch.tensor(scores, device=device)
        broken[0, 0] = bad
        with pytest.raises(RuntimeError, match="nonfinite catalog"):
            masked_ranks(broken, table, ids)


@pytest.mark.parametrize("catalog_size,history,target,tied", [
    (25, list(range(1, 13)), 1, False),  # Repeat outside the last 10 inputs.
    (25, list(range(2, 14)), 1, False),
    (25, list(range(2, 14)), 1, True),
    (4, [1, 2, 3, 4], 1, True),  # All masked, including the target, inside top 10.
])
def test_reference_topk_with_masked_targets_and_small_catalog(
    device, catalog_size, history, target, tied,
):
    rows = [dict(event_id=i, user_id="eval", item_id=str(item), timestamp=i)
            for i, item in enumerate([*history, target])]
    event = len(rows) - 1
    for item in range(1, catalog_size + 1):
        rows.append(dict(event_id=len(rows), user_id="catalog", item_id=str(item), timestamp=item))
    table = EventTable(rows)
    values = (torch.ones(catalog_size, device=device) if tied else
              torch.arange(catalog_size, 0, -1, device=device).float())
    reference_scores = values.clone()
    reference_scores[torch.tensor(history, dtype=torch.long, device=device) - 1] = -torch.inf
    # IISAN/DFF metrics_topK: argsort -> reorder one-hot label -> weighted rank.
    order = torch.argsort(reference_scores, descending=True)
    labels = torch.zeros(catalog_size, device=device)
    labels[target - 1] = 1
    item_rank = torch.arange(1, catalog_size + 1, device=device)
    expected = int((torch.take(labels, order) * item_rank).sum())
    actual = int(masked_ranks(values[None, :].clone(), table, [event])[0])
    assert actual == expected
    assert metrics_from_rank(actual) == metrics_from_rank(expected)
    if catalog_size == 4:
        assert metrics_from_rank(actual)["HR@10"] == 1
    if not tied and target in history:
        assert actual > 10 and metrics_from_rank(actual)["HR@10"] == 0


def test_loss_gradient_and_multiple_updates(device):
    table = fixture_table()
    ids = table.select()
    probabilities = np.bincount(table.targets[ids], minlength=len(table.items) + 1) / len(ids)
    old = fixture_model(table, device)
    new = deepcopy(old)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4) for m in (old, new)]
    for step, batch in enumerate((ids[:37], ids[37:74], ids[-1:])):
        grads, losses = [], []
        for model, optimizer, implementation in zip((old, new), optimizers, (reference, optimized), strict=True):
            seed_everything(42)
            optimizer.zero_grad(set_to_none=True)
            loss = implementation.transition_loss(model, table, batch, probabilities, device)
            loss.backward()
            losses.append(loss.detach().item())
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        assert abs(losses[1] - losses[0]) <= 1e-6 + 1e-5 * abs(losses[0])
        assert all(torch.isfinite(g).all() for g in grads)
        assert ((grads[1] - grads[0]).norm() / grads[0].norm().clamp_min(1e-12)).item() <= 1e-4


def test_training_epoch_and_validation_parity(device):
    table = fixture_table()
    ids = table.select()
    probabilities = np.bincount(table.targets[ids], minlength=len(table.items) + 1) / len(ids)
    config = SimpleNamespace(model=SimpleNamespace(batch_size=37),
                             evaluation=SimpleNamespace(cutoffs=[4, 8, 10, 20, 30]))
    results, models = [], []
    for implementation in (reference, optimized):
        model = fixture_model(table, device)
        results.append(implementation.train_epoch(model, torch.optim.AdamW(model.parameters()),
                       table, ids, probabilities, np.random.default_rng(42), config, device))
        models.append(model)
    assert results[0] == pytest.approx(results[1], abs=1e-6, rel=1e-5)
    old_rows = list(reference.evaluate(models[0], table, ids, config, device))
    new_rows = list(optimized.evaluate(models[1], table, ids, config, device))
    assert old_rows == new_rows
    assert optimized.validation_hr10(models[1], table, ids, config, device) == (
        sum(r["HR@10"] for r in new_rows) / len(ids)
    )


def test_nonfinite_loss_prevents_optimizer_update(device, monkeypatch):
    table = fixture_table()
    model = fixture_model(table, device)
    monkeypatch.setattr(model, "item_vectors", lambda ids: torch.full((*ids.shape, 16), torch.nan, device=device))
    optimizer = torch.optim.AdamW(model.parameters())
    monkeypatch.setattr(optimizer, "step", lambda: pytest.fail("must fail before update"))
    with pytest.raises(RuntimeError, match="nonfinite transition loss"):
        optimized.train_epoch(model, optimizer, table, table.select(),
            np.ones(len(table.items) + 1), np.random.default_rng(42),
            SimpleNamespace(model=SimpleNamespace(batch_size=32)), device)
