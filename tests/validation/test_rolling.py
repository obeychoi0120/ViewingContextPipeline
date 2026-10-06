from datetime import datetime, timezone
import numpy as np
import pytest
from validation.rolling_data import DAY, EventTable
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
