from copy import deepcopy
from datetime import datetime, timezone
import numpy as np
import pytest
from validation.rolling_data import DAY, EventTable, load_csv
from validation.rolling_diagnosis import cluster_bootstrap, weighted_day_mean

def event_table(records):
    return EventTable(
        [
            dict(event_id=i, user_id=str(u), item_id=str(item), timestamp=t)
            for i, (u, item, t) in enumerate(records)
        ]
    )


def test_all_transitions_strict_ties_and_full_history():
    table = event_table([(1, i + 1, i) for i in range(30)] + [(1, 31, 29), (2, 1, 29)])
    assert table.history(30) == list(range(1, 30))
    assert table.history(29, 10) == list(range(20, 30))
    assert table.history(31) == []
    assert len(table.select()) == 30
    assert len(table.select(end=15)) == 14
    # An earlier event appearing later in the source file is still in context.
    table = event_table([(1, 3, 3), (1, 1, 1), (1, 2, 2), (1, 4, 3)])
    assert table.history(0) == [1, 2] == table.history(3)


def test_csv_retains_duplicates_and_rejects_fractional_timestamps(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("user,item,timestamp\n1,2,100\n1,2,100\n1,3,101\n")
    table, duplicates = load_csv(path)
    assert duplicates == 1 and len(table.rows) == 3
    assert table.history(2) == [1, 1]
    path.write_text("user,item,timestamp\n1,2,100.0\n")
    with pytest.raises(ValueError, match="integer milliseconds"):
        load_csv(path)
    path.write_text("user,item,timestamp\n1,2,100\n\n1,3,101\n")
    with pytest.raises(ValueError, match="malformed CSV"):
        load_csv(path)


def test_utc_rolling_boundaries_and_same_day_context():
    origin = int(datetime(2022, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
    table = event_table([(1, i % 4 + 1, origin + i * DAY // 2) for i in range(24)])
    splits = table.splits()
    assert [s["evaluation_date"] for s in splits] == [f"2022-09-{i:02}" for i in range(5, 12)]
    first = splits[0]["phases"]
    assert first["selection"]["end_ms"] == origin + 3 * DAY
    assert first["test"]["raw_count"] == 2
    assert table.history(9)[-1] == int(table.targets[8])
    assert all(
        table.timestamps[i] < first["refit"]["end_ms"]
        for i in table.select(end=first["refit"]["end_ms"])
    )


def test_equal_date_mean_and_paired_user_multiplicity():
    counts = np.array([[1, 3], [1, 1]], dtype=float)
    sums = np.array([[[1, 0.5], [0, 0]], [[0, 0], [1, 0.5]]])
    np.testing.assert_allclose(weighted_day_mean(sums, counts, np.ones((1, 2))), [[0.375, 0.1875]])
    observed, draws, report = cluster_bootstrap(sums, counts, samples=200, seed=4)
    np.testing.assert_allclose(observed, [0.375, 0.1875])
    np.testing.assert_allclose(draws[:, 1], draws[:, 0] / 2)
    # Draws select both events of a user as a cluster, across both dates.
    assert set(draws[:, 0]) <= {0.375, 0.5}
    assert report["working_bytes_upper_bound"] <= 128 * 1024**2
    with pytest.raises(ValueError, match="empty evaluation date"):
        weighted_day_mean(sums, np.zeros_like(counts), np.ones((1, 2)))


def test_100k_bootstrap_arrays_are_bounded():
    counts = np.ones((100_000, 7))
    sums = np.full((100_000, 7, 7), 0.2)
    observed, draws, report = cluster_bootstrap(sums, counts, samples=70)
    np.testing.assert_allclose(observed, 0.2)
    np.testing.assert_allclose(draws, 0.2)
    assert report["working_bytes_upper_bound"] <= report["memory_limit_bytes"]


@pytest.mark.torch
def test_evaluation_masks_history_older_than_the_ten_item_context():
    import torch
    from types import SimpleNamespace
    from validation.rolling_recommendation import evaluate

    table = event_table([(1, i + 1, i) for i in range(31)])

    class FixedScores(torch.nn.Module):
        max_length = 10

        def catalog_vectors(self):
            return torch.arange(31, 0, -1, dtype=torch.float32).reshape(-1, 1)

        def user_vectors(self, sequences):
            assert sequences.shape[1] == 10
            return torch.ones((len(sequences), 1))

    config = SimpleNamespace(
        model=SimpleNamespace(batch_size=2), evaluation=SimpleNamespace(cutoffs=[10, 30])
    )
    row = next(evaluate(FixedScores(), table, np.array([30]), config, torch.device("cpu")))
    assert row["rank"] == 1 and row["candidate_count"] == 31


@pytest.mark.torch
def test_evaluation_keeps_parameters_and_caches_catalog():
    import torch
    from types import SimpleNamespace
    from validation.model import SASRec
    from validation.rolling_recommendation import evaluate

    table = event_table([(1, i % 4 + 1, i) for i in range(15)])
    model = SASRec(4, 10, 8, 1, 2, 0, arm="metadata", item_features=np.ones((4, 8)))
    before = deepcopy(model.state_dict())
    calls = []
    original = model.catalog_vectors

    def catalog():
        calls.append(1)
        return original()

    model.catalog_vectors = catalog
    config = SimpleNamespace(
        model=SimpleNamespace(batch_size=2), evaluation=SimpleNamespace(cutoffs=[10, 30])
    )
    rows = list(evaluate(model, table, table.select(), config, torch.device("cpu")))
    assert len(calls) == 1 and len(rows) == 14
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)
    assert all(r["rank"] == 1 for r in rows[3:])
