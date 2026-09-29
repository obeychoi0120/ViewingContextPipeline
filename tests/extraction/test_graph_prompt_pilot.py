from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.graph_prompt_pilot import compare, prepare, select_cohort
from extraction.scene_storage import write_scene_records


def test_first100_is_stable_and_does_not_modify_shared_cohort():
    cohort = {"catalog": [{"content_id": f"video_{i:05d}"} for i in reversed(range(110))],
              "metadata_titles": [{"content_id": "video_00001", "title": "kept"},
                                  {"content_id": "video_00109", "title": "excluded"}]}
    selected = select_cohort(cohort)
    assert len(selected["catalog"]) == 100
    assert selected["catalog"][0]["content_id"] == "video_00000"
    assert selected["catalog"][-1]["content_id"] == "video_00099"
    assert len(cohort["catalog"]) == 110
    assert len(selected["metadata_titles"]) == 1
    with pytest.raises(ValueError, match="requires 100"):
        select_cohort({"catalog": []})


def test_cannot_use_baseline_run_as_pilot():
    context = SimpleNamespace(run_root=Path("baseline"))
    with pytest.raises(ValueError, match="separate run"):
        prepare(context, context)


def test_comparison_joins_scene_ids_and_counts_raw_invalid_missing(tmp_path):
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old = SimpleNamespace(scene_arm_dir=lambda _: old_dir)
    new = SimpleNamespace(scene_arm_dir=lambda _: new_dir)
    write_scene_records(old_dir / "video.jsonl", [
        {"scene_idx": 0, "graph": {"entities": [], "relations": []}},
        {"scene_idx": 1, "graph": "raw"},
    ])
    write_scene_records(new_dir / "video.jsonl", [
        {"scene_idx": 1, "graph": {"invalid": "object"}},
    ])
    manifest = {"content_ids": ["video"], "evidence": [
        {"content_id": "video", "scene_idx": index} for index in range(3)]}
    rows, metrics = compare(new, old, manifest)
    assert [r["scene_idx"] for r in rows] == [0, 1, 2]
    assert metrics["baseline"]["structured"] == 1
    assert metrics["baseline"]["raw"] == 1
    assert metrics["candidate"]["structured"] == 0
    assert metrics["candidate"]["raw"] == 1
    assert metrics["candidate"]["missing"] == 2
    assert all(value is None for value in rows[0]["review"].values())
