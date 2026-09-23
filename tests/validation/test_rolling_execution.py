"""Semantic parity with the frozen pre-optimization implementation."""
from copy import deepcopy
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import rolling_reference as reference  # noqa: E402
from validation import rolling_recommendation as optimized  # noqa: E402
from validation.model import SASRec, pad_sequences, seed_everything  # noqa: E402
from validation.rolling_data import DAY, EventTable  # noqa: E402
from validation.rolling_execution import execution_for, masked_ranks, negative_mask  # noqa: E402
from validation.scoring import mask_history, rank_of_target  # noqa: E402

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
    features = np.random.default_rng(17).normal(size=(len(table.items), 16)).astype(np.float32)
    features[::7] = 0
    return SASRec(len(table.items), 10, 16, 1, 2, 0.1,
                  arm="graph", item_features=features).to(device)


def test_prepared_sequences_and_vector_mask_exact(device):
    table = fixture_table()
    ids = np.arange(len(table.rows))
    runtime = execution_for(table)
    original = pad_sequences([table.history(int(i), 10) for i in ids], 10, device)
    actual = runtime.inputs(ids, 10, device)
    assert torch.equal(actual, original)
    assert runtime.padded(10) is runtime.padded(10)
    for batch in (ids, ids[-1:], np.array([2, 2, 1, 32])):
        targets = torch.as_tensor(table.targets[batch], device=device)
        mask = negative_mask(runtime.inputs(batch, 10, device), targets)
        expected = [
            [j != k and int(target) in set(table.history(int(i), 10)) | {int(table.targets[i])}
             for k, target in enumerate(table.targets[batch])]
            for j, i in enumerate(batch)
        ]
        assert mask.cpu().tolist() == expected


def test_popularity_cache_validation_and_bounded_lifetime(device):
    runtime = execution_for(fixture_table())
    values = np.array([0, 0.2, 0.8, 0, np.nan, np.inf, -1, 1e-99])
    logs = runtime.log_probabilities(values, [1, 2], device, torch.float32)
    assert runtime.log_probabilities(values, [1], device, torch.float32) is logs
    for target in [0, 3, 4, 5, 6, 7]:
        with pytest.raises(RuntimeError, match="invalid positive popularity"):
            runtime.log_probabilities(values, [target], device, torch.float32)
    for _ in range(4):
        runtime.log_probabilities(values.copy(), [1], device, torch.float32)
    assert len(runtime.popularity) == 2


def test_rank_parity_full_history_ties_repeated_targets(device):
    table = fixture_table()
    ids = table.select()
    scores = np.random.default_rng(31).integers(-2, 3, (len(ids), len(table.items))).astype(np.float32)
    scores[::3] = 0
    expected = [rank_of_target(mask_history(s, [x - 1 for x in table.history(int(i))],
                                          table.targets[i] - 1), table.targets[i] - 1)
                for i, s in zip(ids, scores, strict=True)]
    actual = masked_ranks(torch.tensor(scores, device=device), table, ids)
    assert actual.cpu().tolist() == expected
    for bad in (np.nan, np.inf, -np.inf):
        broken = torch.tensor(scores, device=device)
        broken[0, 0] = bad
        with pytest.raises(RuntimeError, match="nonfinite catalog"):
            masked_ranks(broken, table, ids)


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
    assert optimized.validation_ndcg10(models[1], table, ids, config, device) == (
        sum(r["NDCG@10"] for r in new_rows) / len(ids)
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


def test_selection_refit_test_and_old_bundle_compatibility(tmp_path, monkeypatch, device):
    arm, seed = "graph_qwen_meta", 42
    from pipeline_runtime import read_json
    from validation.recommendation_cache import valid_bundle, cache_for
    from validation.recommendation_contracts import (
        ARCHITECTURE_VERSION, TRAINING_IMPLEMENTATION_VERSION,
    )

    table = fixture_table()
    split = table.splits()[0]
    config = SimpleNamespace(
        model=SimpleNamespace(max_sequence_length=10, max_epochs=4, patience=2, seeds=[seed],
                              batch_size=37, learning_rate=1e-4),
        evaluation=SimpleNamespace(cutoffs=[4, 8, 10, 20, 30]),
    )
    features_dir = tmp_path / "representations"
    features_dir.mkdir()
    features = np.random.default_rng(len(arm)).normal(size=(len(table.items), 16)).astype(np.float32)
    features[::7] = 0
    np.savez(features_dir / f"{arm}_embeddings.npz", values=features)

    def tiny(config, *, item_count, branch, features, device):
        return SASRec(item_count, 10, 16, 1, 2, 0.1,
                      arm="metadata" if branch == "meta" else "graph", item_features=features).to(device)

    monkeypatch.setattr(reference, "_new_model", tiny)
    monkeypatch.setattr(optimized, "_new_model", tiny)
    monkeypatch.setattr("validation.recommendation_cache.eligible", lambda *a: False)
    identity = dict(run_id="comparison", evaluation_date=split["evaluation_date"],
                    seed=seed, arm=arm, training_input_hash="same-input")
    bundles, contexts = [], []
    for name, implementation in (("reference", reference), ("optimized", optimized)):
        ctx = SimpleNamespace(representations_dir=features_dir,
                              recommendations_dir=tmp_path / name,
                              run_root=tmp_path / "artifacts" / "runs" / name,
                              run_id="comparison", config={"experiment_config_version": "v4",
                                                           "protocol": {"arms": [arm]}})
        contexts.append(ctx)
        current = EventTable(table.rows)
        implementation.run_combination(ctx, config, current, split, identity, arm,
                                       implementation.prepare_split(current, split), device)
        bundles.append(optimized.combination_dir(ctx, split["evaluation_date"], seed, arm))
    before, after = (read_json(p / "training.json") for p in bundles)
    assert "execution" not in before and "execution" in after
    assert before["best_epoch"] == after["best_epoch"]
    for phase in ("selection", "refit"):
        assert len(before[phase]) == len(after[phase])
        for old, new in zip(before[phase], after[phase], strict=True):
            assert old.keys() == new.keys()
            assert old["optimizer_updates"] == new["optimizer_updates"]
            assert old == pytest.approx(new, abs=1e-6, rel=1e-5)
    # Exact ranks are stronger than the specified per-combination/mean metric gates.
    assert (bundles[0] / "per_event_metrics.jsonl").read_bytes() == (
        bundles[1] / "per_event_metrics.jsonl").read_bytes()
    original = {p.name: p.read_bytes() for p in bundles[0].iterdir()}
    assert all(valid_bundle(p, identity, table, split) for p in bundles)
    assert original == {p.name: p.read_bytes() for p in bundles[0].iterdir()}
    assert cache_for(contexts[0], identity).key == cache_for(contexts[1], identity).key
    assert ARCHITECTURE_VERSION == "sasrec-content-v3"
    assert TRAINING_IMPLEMENTATION_VERSION == "shared-scenes-training-evaluation/v3"

    # Actually resume a pre-optimization bundle without preparing tensors or devices.
    def forbid(*args, **kwargs):
        pytest.fail("a completed combination must not prepare tensors or train")

    monkeypatch.setattr("validation.steps.validation_config", lambda _: config)
    monkeypatch.setattr(optimized, "load_validation_cohort", lambda _: {
        "events": table.rows, "manifest": {"policy": "full-catalog"}, "plan": {"splits": [split]},
    })
    monkeypatch.setattr(optimized, "verify_representations", lambda *a, **kw: None)
    monkeypatch.setattr(optimized, "training_signature", lambda *a: "same-input")
    monkeypatch.setattr("validation.representation_provenance.recommendation_identity", lambda *a: {})
    monkeypatch.setattr(optimized, "execution_for", forbid)
    monkeypatch.setattr(optimized, "worker_devices", forbid)
    monkeypatch.setattr(optimized, "run_combination", forbid)
    assert optimized.run_rolling(contexts[0], target=[arm])["skipped"] == 1
    assert original == {p.name: p.read_bytes() for p in bundles[0].iterdir()}

    # The same old bundle can be shared, with no execution metadata retrofit.
    from validation.recommendation_cache import publish
    publish(contexts[0], bundles[0], identity, table, split)
    shared_context = SimpleNamespace(**{**vars(contexts[0]), "run_id": "shared",
                                       "recommendations_dir": tmp_path / "shared"})
    monkeypatch.setattr("validation.recommendation_cache.eligible", lambda *a: True)
    assert optimized.run_rolling(shared_context, target=[arm])["skipped"] == 1
    assert read_json(shared_context.recommendations_dir / "reuse.json")["shared"] == 1
    shared_bundle = optimized.combination_dir(shared_context, split["evaluation_date"], seed, arm)
    assert "execution" not in read_json(shared_bundle / "training.json")
    assert original == {p.name: p.read_bytes() for p in bundles[0].iterdir()}
