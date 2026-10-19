from artifact_io import read_jsonl
from helpers import events, make_context, ready_cohort
from preparation.cohort import load_csv
from validation.leave_one_out_data import EventTable
from validation.selection import build_selection
from visual_sampling import build_fixed_windows, timestamp_stem


def test_csv_preserves_duplicate_events(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("user,item,timestamp\n1,2,1000\n1,2,1000\n1,3,999\n")
    table, duplicates = load_csv(path)
    assert duplicates == 1
    assert [r["event_id"] for r in table.rows] == [0, 1, 2]
    assert [r["item_id"] for r in table.rows] == ["2", "2", "3"]
    assert table.by_user["1"] == [2, 0, 1]


def test_full_cohort_roundtrip(context):
    cohort = ready_cohort(context)
    assert len(cohort["catalog"]) == 24
    assert cohort["eligibility"]["status"] == "ready"
    assert cohort["plan"]["user_count"] == 3
    assert read_jsonl(context.cohort_dir / "events.jsonl") == events()
    assert cohort["required_items"] == [
        {"item_id": r["item_id"], "content_id": r["content_id"]} for r in cohort["catalog"]
    ]
    assert [r["title"] for r in cohort["metadata_titles"]] == [
        f"TITLE_SECRET_{i}" for i in range(1, 25)
    ]


def test_fixed_window_sampling():
    windows = build_fixed_windows(32.2, scene_duration=30, num_keyframes=6)
    assert windows == [
        {
            "scene_start": 0,
            "scene_end": 30.0,
            "duration": 30.0,
            "shot_change_timestamps": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            "keyframe_timestamps": [2.5, 7.5, 12.5, 17.5, 22.5, 27.5],
        },
        {
            "scene_start": 30,
            "scene_end": 32.2,
            "duration": 2.2,
            "shot_change_timestamps": [30.0],
            "keyframe_timestamps": [31.1],
        },
    ]
    assert len(build_fixed_windows(30, scene_duration=30, num_keyframes=6)) == 1
    assert timestamp_stem(31.199999) == "0031_1"


def test_selection_filters_and_trims(tmp_path):
    context = make_context(tmp_path / "boundary", lengths=(4, 5, 14))
    ready_cohort(context)
    selected = build_selection(context)
    retained = selected["events"]
    assert len(retained) == 18
    assert {r["user_id"] for r in retained} == {"2", "3"}
    assert [r["source_event_id"] for r in retained] == list(range(4, 9)) + list(range(10, 23))
    assert {r["item_id"] for r in selected["catalog"]} == {r["item_id"] for r in retained}
    assert selected["manifest"]["statistics"]["excluded_by_arm"] == {}
    assert selected["plan"]["trimmed_event_count"] == 1


def test_leave_one_out_partitions():
    rows = events((5,))
    rows[0].update(timestamp=3000, source_event_id=30)
    rows[1].update(timestamp=1000, source_event_id=20)
    rows[2].update(timestamp=1000, source_event_id=10)
    rows[3].update(timestamp=4000, source_event_id=40)
    rows[4].update(timestamp=5000, source_event_id=50)
    table = EventTable(rows)
    split = table.splits()[0]
    assert split["users"] == [{"user_id": "1", "train": [2, 1, 0], "validation": 3, "test": 4}]
    assert table.history(3) == [3, 2, 1]
    assert table.history(4) == [3, 2, 1, 4]
    assert split["phases"]["train"]["eligible_count"] == 2
    assert split["phases"]["train"]["no_history_count"] == 1
    assert set(split["phases"]["train"]["event_ids"]).isdisjoint({3, 4})
