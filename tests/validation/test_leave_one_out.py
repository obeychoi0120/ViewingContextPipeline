from types import SimpleNamespace

import numpy as np
import pytest

from artifact_io import read_json, write_json, write_jsonl
from validation.leave_one_out_data import EventTable
from validation.leave_one_out_diagnosis import cluster_bootstrap, weighted_user_mean
from validation.leave_one_out_recommendation import phase_ids, prepare_split
from validation.selection import build_selection
from validation.recommendation_contracts import EVALUATION_CONTRACT


def replace_source(context, rows):
    write_jsonl(context.cohort_dir / "events.jsonl", rows)
    path = context.cohort_dir / "cohort_plan.json"
    plan = read_json(path)
    plan.update(user_count=len({r["user_id"] for r in rows}), interaction_count=len(rows))
    write_json(path, plan)


def test_preprocessing_boundaries_ties_duplicates_and_source_mapping(ready_context, tmp_path):
    records = []
    for user, count in enumerate((4, 5, 13, 14), 1):
        for i in reversed(range(count)):
            records.append(dict(event_id=len(records), user_id=str(user),
                                item_id=str(i % 4 + 1), timestamp=i // 2))
    records[-3].update(item_id=records[-4]["item_id"], timestamp=records[-4]["timestamp"])
    from preparation.cohort import load_csv

    source_path = tmp_path / "source.csv"
    source_path.write_text("user,item,timestamp\n" + "".join(
        f"{r['user_id']},{r['item_id']},{r['timestamp']}\n" for r in records))
    source_table, duplicates = load_csv(source_path)
    assert source_table.rows == records and duplicates == 1
    replace_source(ready_context, records)
    selected = build_selection(ready_context)
    table = EventTable(selected["events"])
    assert [len(table.by_user[u]) for u in table.users] == [5, 13, 13]
    assert table.users == ["2", "3", "4"]
    assert selected["plan"]["short_history_event_count"] == 4
    assert selected["plan"]["trimmed_event_count"] == 1
    assert [r["event_id"] for r in table.rows] == list(range(31))
    assert {len(records) - 3, len(records) - 4} <= {r["source_event_id"] for r in table.rows}
    for row in table.rows:
        source = records[row["source_event_id"]]
        assert all(row[k] == source[k] for k in ("user_id", "item_id", "timestamp"))
    split = selected["plan"]["splits"][0]
    for user in split["users"]:
        ids = table.by_user[user["user_id"]]
        assert user["train"] == ids[:-2]
        assert user["validation"] == ids[-2] and user["test"] == ids[-1]
        assert table.history(ids[-1]) == table.targets[ids[:-1]].tolist()
        assert table.history(ids[-2]) == table.targets[ids[:-2]].tolist()
        assert len(table.history(ids[-1], 10)) == min(10, len(ids) - 1)
    assert max(r["timestamp"] for r in table.rows) == max(r["timestamp"] for r in records)


def test_held_out_targets_are_not_training_inputs_or_popularity(ready_context):
    rows = [dict(event_id=i, user_id="1", item_id=str(item), timestamp=7)
            for i, item in enumerate((1, 2, 3, 4, 4))]
    replace_source(ready_context, rows)
    cohort = build_selection(ready_context)
    table = EventTable(cohort["events"])
    split = cohort["plan"]["splits"][0]
    ids, probabilities, frequencies = prepare_split(table, split)
    assert ids["train"].tolist() == [1, 2]
    assert phase_ids(table, split, "validation").tolist() == [3]
    assert phase_ids(table, split, "test").tolist() == [4]
    assert table.history(3) == [1, 2, 3]
    assert table.history(4) == [1, 2, 3, 4]
    assert frequencies.tolist() == [0, 1, 1, 1, 0]
    assert probabilities["train"].tolist() == [0, 0, 0.5, 0.5, 0]
    assert [r["item_id"] for r in cohort["catalog"]] == ["1", "2", "3", "4"]


def test_catalog_excludes_short_user_and_trimmed_only_items(ready_context):
    items = [4] + [i % 3 + 1 for i in range(13)]
    rows = [dict(event_id=i, user_id="1", item_id=str(item), timestamp=i)
            for i, item in enumerate(items)]
    rows.extend(dict(event_id=len(items) + i, user_id="2", item_id="4", timestamp=i)
                for i in range(4))
    replace_source(ready_context, rows)
    cohort = build_selection(ready_context)
    assert [r["item_id"] for r in cohort["catalog"]] == ["1", "2", "3"]
    assert [r["item_id"] for r in cohort["metadata_titles"]] == ["1", "2", "3"]
    assert cohort["excluded"][0]["reason"] == "outside_retained_sequences"
    assert {r["source_event_id"] for r in cohort["events"]} == set(range(1, 14))
    assert cohort["plan"]["excluded_users"] == [
        {"user_id": "2", "reason": "short_history", "event_count": 4}]
    assert cohort["manifest"]["statistics"]["removed_event_count"] == 5


def test_equal_user_mean_and_paired_seed_averages():
    counts = np.ones(2)
    sums = np.array([[1, 0.5], [0, 0]])
    np.testing.assert_allclose(weighted_user_mean(sums, counts, np.ones((1, 2))), [[0.5, 0.25]])
    observed, draws, report = cluster_bootstrap(sums, counts, samples=200, seed=4)
    np.testing.assert_allclose(observed, [0.5, 0.25])
    np.testing.assert_allclose(draws[:, 1], draws[:, 0] / 2)
    assert set(draws[:, 0]) == {0, 0.5, 1}
    assert report["working_bytes_upper_bound"] <= 128 * 1024**2
    with pytest.raises(ValueError, match="one test event"):
        cluster_bootstrap(sums, np.array([1, 2]), samples=200)


@pytest.mark.torch
@pytest.mark.parametrize("mode", ["text", "mean", "gnn"])
@pytest.mark.parametrize("ranks,expected_epoch", [
    ([[1, 100, 100], [10, 10, 100], [100, 100, 100]], 2),
    ([[1, 100, 100], [2, 100, 100], [100, 100, 100]], 1),
])
def test_test_uses_best_checkpoint_without_refit(ready_context, monkeypatch, ranks, expected_epoch, mode):
    import torch
    from validation import leave_one_out_recommendation as runner
    from validation.selection import prepare_validation_cohort
    from validation.steps import validation_config
    from validation.graph_context import graph_context
    from artifact_io import read_json

    cohort = prepare_validation_cohort(ready_context)
    table = EventTable(cohort["events"])
    split = cohort["plan"]["splits"][0]
    config = validation_config(ready_context)
    config.model.max_epochs = 3
    config.model.patience = 5
    context = ready_context if mode == "text" else graph_context(ready_context, mode)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def clear_item_cache(self):
            self.cached = None

        def catalog_vectors(self):
            if self.cached is None:
                self.cached = self.weight.expand(len(table.items), 1).detach().clone()
            return self.cached

    model = Model()
    ready_context.representations_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ready_context.representations_dir / "meta_embeddings.npz", values=np.zeros(1))
    monkeypatch.setattr(runner, "_new_model", lambda *a, **k: model)
    monkeypatch.setattr("validation.graph_model.new_graph_model", lambda *a: model)
    monkeypatch.setattr("validation.recommendation_cache.eligible", lambda *a: False)
    monkeypatch.setattr(runner, "_optimizer", lambda *a: None)
    monkeypatch.setattr(runner, "execution_for", lambda *a: SimpleNamespace(
        padded=lambda *a: None, log_probabilities=lambda *a: None))
    trained = []

    def epoch(model, optimizer, table, ids, probabilities, rng, config, device, **kwargs):
        trained.append(kwargs["epoch"])
        model.weight.data.fill_(kwargs["epoch"])
        model.cached = "stale"
        return {"loss": 0, "positive_count": len(ids), "optimizer_updates": 1}

    monkeypatch.setattr(runner, "train_epoch", epoch)
    monkeypatch.setattr(runner, "rank_batches", lambda model, table, ids, *a, **k:
                        iter([(ids, ranks[k["epoch"] - 1])]))

    def evaluate(model, table, ids, *args, **kwargs):
        assert model.weight.item() == expected_epoch
        assert model.cached is None
        assert (model.catalog_vectors() == expected_epoch).all()
        for i in ids:
            yield {**table.rows[int(i)], "rank": 1, "candidate_count": len(table.items)}

    monkeypatch.setattr(runner, "evaluate", evaluate)
    identity = {"run_id": ready_context.run_id, "protocol": "leave_one_out",
                "evaluation_contract": dict(EVALUATION_CONTRACT),
                "split_hash": split["split_hash"], "seed": 42, "arm": "meta"}
    if mode != "text":
        identity["representation_mode"] = "graph"
    runner.run_combination(context, config, table, split, identity, "meta",
                           prepare_split(table, split), torch.device("cpu"))
    assert trained == [1, 2, 3]
    directory = runner.combination_dir(context, 42, "meta")
    assert read_json(directory / "training.json")["best_epoch"] == expected_epoch
    training = read_json(directory / "training.json")
    assert training["checkpoint_policy"] == "best_validation_hr10"
    assert [row["validation_hr10"] for row in training["train"]] == [
        sum(rank <= 10 for rank in epoch) / len(epoch) for epoch in ranks
    ]
    assert runner.combination_complete(directory, identity, len(table.users))
    assert not runner.combination_complete(
        directory, {**identity, "evaluation_contract": {**EVALUATION_CONTRACT,
                    "checkpoint_metric": "NDCG@10"}}, len(table.users)
    )
    assert torch.load(directory / "sasrec.pt", weights_only=True)["state_dict"]["weight"].item() == expected_epoch
    assert runner.combination_complete(directory, identity, len(table.users))
    assert not runner.combination_complete(directory, {**identity, "protocol": "rolling"}, len(table.users))
    assert not runner.combination_complete(directory, {**identity, "split_hash": "stale"}, len(table.users))
    if mode != "text":
        assert (np.load(directory / "catalog_vectors.npy") == expected_epoch).all()
