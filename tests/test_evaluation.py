from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from artifact_io import read_json
from helpers import events, prepare_inputs, result_identity, selected_table, simple_model
from validation.early_stopping import EarlyStopping
from validation.leave_one_out_data import EventTable
from validation.leave_one_out_execution import execution_for, masked_ranks, negative_mask
from validation.leave_one_out_recommendation import (
    phase_ids,
    prepare_split,
    run_combination,
    transition_loss,
)
from validation.metrics import metrics_from_rank
from validation.steps import validation_config

pytestmark = pytest.mark.torch


def test_history_inputs_keep_latest_ten():
    table = EventTable(events((13,)))
    execution = execution_for(table)
    actual = execution.inputs([0, 3, 12], 10, "cpu")
    assert actual.tolist() == [[0] * 10, [1, 2, 3] + [0] * 7, list(range(3, 13))]
    assert table.targets[12] == 13
    assert 13 not in actual[2].tolist()
    assert table.history(12) == list(range(1, 13))


def test_training_popularity_excludes_holdouts():
    table = EventTable(events())
    split = table.splits()[0]
    ids, probabilities, frequencies = prepare_split(table, split)
    assert ids["train"].tolist() == [1, 2, 6, 7, 8] + list(range(12, 22))
    assert phase_ids(table, split, "test").tolist() == [4, 10, 23]
    expected_frequency = np.zeros(25, dtype=np.int64)
    expected_frequency[[1, 2, 3, 6, 7, 8, 9] + list(range(12, 23))] = 1
    np.testing.assert_array_equal(frequencies, expected_frequency)
    expected_probability = np.zeros(25)
    expected_probability[[2, 3, 7, 8, 9] + list(range(13, 23))] = 1 / 15
    np.testing.assert_allclose(probabilities["train"], expected_probability)
    for event in ids["validation"].tolist() + ids["test"].tolist():
        assert frequencies[table.targets[event]] == 0
        assert probabilities["train"][table.targets[event]] == 0


def test_inbatch_negative_mask():
    inputs = torch.tensor([[1, 3, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0]])
    targets = torch.tensor([2, 2, 3, 4])
    expected = torch.tensor(
        [
            [False, True, True, False],
            [True, False, False, False],
            [True, True, False, False],
            [False, False, False, False],
        ]
    )
    assert torch.equal(negative_mask(inputs, targets), expected)


def test_transition_loss_matches_reference():
    table = EventTable(events((5, 5)))
    model = simple_model(item_count=10)
    ids = np.array([2, 7, 3])
    probabilities = np.full(11, 0.1)
    probabilities[3], probabilities[8] = 0.2, 0.4
    inputs = execution_for(table).inputs(ids, 10, "cpu")
    model.eval()
    with torch.no_grad():
        users = model.user_vectors(inputs).numpy()
        candidates = model.item_vectors(torch.tensor([3, 8, 4])).numpy()
    logits = users @ candidates.T - np.log([0.2, 0.4, 0.1])[None, :]
    targets = [3, 8, 4]
    for row, event in enumerate(ids):
        for column, target in enumerate(targets):
            if column != row and (
                target in table.history(int(event), 10) or target == targets[row]
            ):
                logits[row, column] = -1e4
    expected = np.mean(
        [np.log(np.exp(row - row.max()).sum()) + row.max() - row[i] for i, row in enumerate(logits)]
    )
    loss = transition_loss(model, table, ids, probabilities, torch.device("cpu"))
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-5, abs=1e-6)
    loss.backward()
    assert model.video_projection.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_ranking_masks_full_preceding_history():
    rows = events((13,))
    rows[-1]["item_id"] = "1"  # Repeated target, outside the model's latest ten inputs.
    table = EventTable(rows)
    scores = torch.arange(12, dtype=torch.float32).reshape(1, -1)
    reference = scores[0].clone()
    reference[:] = -torch.inf
    expected = int((torch.argsort(reference, descending=True) == 0).nonzero()[0, 0]) + 1
    ranks = masked_ranks(scores, table, np.array([12]))
    assert ranks.tolist() == [expected]
    assert torch.isneginf(scores).all()
    # Also exercise an unmasked target and a nontrivial unmasked candidate set.
    table = EventTable(events((13,)))
    scores = torch.tensor([[99.0] + [0.0] * 11 + [-1.0]])
    assert masked_ranks(scores, table, np.array([12])).tolist() == [1]
    assert torch.isneginf(scores[0, :12]).all()


def test_hr_ndcg_cutoff_boundaries():
    for cutoff in (4, 8, 10, 20, 30):
        assert metrics_from_rank(1, [cutoff]) == {f"HR@{cutoff}": 1.0, f"NDCG@{cutoff}": 1.0}
        at = metrics_from_rank(cutoff, [cutoff])
        assert at[f"HR@{cutoff}"] == 1
        assert at[f"NDCG@{cutoff}"] == pytest.approx(1 / math.log2(cutoff + 1))
        assert metrics_from_rank(cutoff + 1, [cutoff]) == {
            f"HR@{cutoff}": 0.0,
            f"NDCG@{cutoff}": 0.0,
        }


def test_early_stopping_keeps_first_best_epoch():
    stopping = EarlyStopping(patience=2)
    assert not stopping.update(1, 0.2)
    assert not stopping.update(2, 0.7)
    assert not stopping.update(3, 0.7)
    assert stopping.update(4, 0.6)
    assert stopping.best_epoch == 2
    assert stopping.best_score == 0.7


def test_best_checkpoint_is_tested_once(context, monkeypatch):
    import validation.leave_one_out_recommendation as execution
    import validation.graph_model as graph_model

    prepare_inputs(context, monkeypatch)
    cohort, table, split = selected_table(context)
    config = validation_config(context)
    config.model.max_epochs = 4
    config.model.patience = 2
    identity = result_identity(context, cohort)
    real_train, real_evaluate, real_create = (
        execution.train_epoch,
        execution.evaluate,
        graph_model.new_graph_model,
    )
    calls = {"create": 0, "train": 0, "test": 0}
    best_state = {}

    def create(*args, **kwargs):
        calls["create"] += 1
        return real_create(*args, **kwargs)

    def train(model, *args, **kwargs):
        result = real_train(model, *args, **kwargs)
        calls["train"] += 1
        if calls["train"] == 2:
            best_state.update(deepcopy(model.state_dict()))
        return result

    def evaluate(model, *args, **kwargs):
        calls["test"] += 1
        assert best_state
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, best_state[name], rtol=0, atol=0)
        return real_evaluate(model, *args, **kwargs)

    scores = iter([0.2, 0.9, 0.9, 0.1])
    monkeypatch.setattr(graph_model, "new_graph_model", create)
    monkeypatch.setattr(execution, "train_epoch", train)
    monkeypatch.setattr(execution, "validation_hr10", lambda *a, **kw: next(scores))
    monkeypatch.setattr(execution, "evaluate", evaluate)
    run_combination(
        context,
        config,
        table,
        split,
        identity,
        "base",
        prepare_split(table, split),
        torch.device("cpu"),
    )
    assert calls == {"create": 1, "train": 4, "test": 1}
    directory = context.recommendations_dir / "seed_42" / "base"
    assert read_json(directory / "training.json")["best_epoch"] == 2
    checkpoint = torch.load(directory / "sasrec.pt", map_location="cpu", weights_only=True)
    for name, tensor in checkpoint["state_dict"].items():
        torch.testing.assert_close(tensor, best_state[name], rtol=0, atol=0)
