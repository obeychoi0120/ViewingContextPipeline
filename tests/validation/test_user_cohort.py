from __future__ import annotations

import json
import hashlib
from fractions import Fraction

import pytest

import validation.cohort as cohort_module
from validation.cohort import CohortError, load_ready_cohort, prepare_cohort
from validation.cohort_selection import (
    CATALOG_SCOPE,
    cohort_statistics,
    select_users,
)
from validation.config import ValidationConfig
from pipeline_runtime import read_json, read_jsonl, write_json, write_jsonl

from conftest import config_data


SELECTION = {"seed": 42, "boundaries": [5, 10, 20, 50], "min_length": 5, "max_length": 13}


def _config(tmp_path, users=2):
    config = ValidationConfig.model_validate(config_data(tmp_path, users=users))
    config.dataset.pairs_tsv.write_text("u1\t1 2 4 3 5\nu2\t1 2 3 5 4\n", encoding="utf-8")
    return config


def _assets(config, items=range(1, 6)):
    config.dataset.videos_dir.mkdir(exist_ok=True)
    for item in items:
        (config.dataset.videos_dir / f"{item}.mp4").write_bytes(b"video")
    config.dataset.titles_csv.write_text(
        "".join(f"{item},Title {item}\n" for item in items), encoding="utf-8"
    )


def _ready(config):
    return load_ready_cohort(
        config.output_dir / "data/cohort",
        run_id=config.run_id,
        settings=config.cohort.model_dump(mode="json"),
        inputs={key: str(path.resolve()) for key, path in config.dataset.model_dump().items()},
    )


def test_stratified_prefixes_are_nested_through_100k_and_input_order_independent():
    lengths = [5] * 51001 + [10] * 30999 + [20] * 17999 + [50]
    histories = {length: [str(item) for item in range(1, length + 1)] for length in set(lengths)}
    pairs = [(f"u{index:06d}", histories[length]) for index, length in enumerate(lengths)]
    complete = select_users(pairs, count=100000, **SELECTION)
    for count in (1000, 10000):
        assert select_users(list(reversed(pairs)), count=count, **SELECTION) == complete[:count]
    assert len({row["user_id"] for row in complete}) == 100000
    assert [row["cohort_rank"] for row in complete] == list(range(1, 100001))
    assert select_users(pairs, count=1000, **{**SELECTION, "seed": 43}) != complete[:1000]


def test_equal_priorities_break_ties_by_stratum_lower_bound():
    pairs = [("long", ["1"] * 10), ("short", ["1"] * 5)]
    assert [row["user_id"] for row in select_users(pairs, count=2, **SELECTION)] == [
        "short",
        "long",
    ]


def test_integer_heap_order_matches_an_independent_rational_priority_reference():
    pairs = []
    expected = []
    for lower, count in ((5, 7), (10, 4), (20, 3), (50, 1)):
        users = [f"group-{lower}-user-{index}" for index in range(count)]
        users.sort(key=lambda user: (hashlib.sha256(f"42:{user}".encode()).hexdigest(), user))
        for rank, user in enumerate(users):
            pairs.append((user, ["1"] * lower))
            expected.append((Fraction(2 * rank + 1, 2 * count), lower, user))
    actual = select_users(list(reversed(pairs)), count=len(pairs), **SELECTION)
    assert [row["user_id"] for row in actual] == [user for _, _, user in sorted(expected)]


def test_boundary_lengths_and_repeated_items_are_not_filtered_or_deduplicated():
    lengths = [4, 5, 9, 10, 19, 20, 49, 50]
    pairs = [
        (str(length), ["1", "2", "1"] * (length // 3) + ["3"] * (length % 3)) for length in lengths
    ]
    selected = select_users(pairs, count=7, **SELECTION)
    by_user = {row["user_id"]: row for row in selected}
    assert "4" not in by_user
    assert [by_user[str(length)]["stratum"] for length in lengths[1:]] == [
        "5-9",
        "5-9",
        "10-19",
        "10-19",
        "20-49",
        "20-49",
        "50+",
    ]
    for user, original in pairs[1:]:
        assert by_user[user]["sequence"] == original[-13:]
        assert by_user[user]["train"] == original[-13:][:-2]
        assert by_user[user]["valid_target"] == original[-2]
        assert by_user[user]["test_target"] == original[-1]
    with pytest.raises(CohortError, match="only 7 are candidates"):
        select_users(pairs, count=8, **SELECTION)


def test_plan_only_requires_no_media_title_probe_or_ml_dependencies(tmp_path, monkeypatch):
    config = _config(tmp_path)

    def forbidden(*_args, **_kwargs):
        pytest.fail("plan-only tried to access an asset or launch a process")

    monkeypatch.setattr(cohort_module, "build_item_inventory", forbidden)
    monkeypatch.setattr(cohort_module, "load_metadata_titles", forbidden)
    monkeypatch.setattr(cohort_module.subprocess, "run", forbidden)
    result = prepare_cohort(config, plan_only=True, probe=forbidden)
    output = config.output_dir / "data/cohort"
    assert result["status"] == "planned"
    assert result["catalog_scope"] == CATALOG_SCOPE
    assert set(path.name for path in output.iterdir()) == {
        "cohort_plan.json",
        "selected_users.jsonl",
        "required_items.jsonl",
        "eligibility_summary.json",
    }
    plan = read_json(output / "cohort_plan.json")
    assert plan["required_item_count"] == 5
    assert plan["scale_statistics"]["1000"]["status"] == "insufficient_candidates"
    eligibility = read_json(output / "eligibility_summary.json")
    assert all(value is None for value in eligibility["media"].values())
    with pytest.raises(CohortError, match="not ready"):
        _ready(config)


def test_missing_assets_block_without_substitution_and_resume_the_same_users(tmp_path):
    config = _config(tmp_path)
    output = config.output_dir / "data/cohort"
    prepare_cohort(config, plan_only=True)
    frozen = {
        name: (output / name).read_bytes()
        for name in ("cohort_plan.json", "selected_users.jsonl", "required_items.jsonl")
    }
    _assets(config, range(1, 5))
    with pytest.raises(CohortError, match="selection is unchanged"):
        prepare_cohort(config, probe=lambda _: 30.1)
    assert read_json(output / "eligibility_summary.json")["status"] == "blocked"
    assert read_jsonl(output / "failures.jsonl") == [
        {"item_id": "5", "reason": "missing_video"},
        {"item_id": "5", "reason": "missing_metadata_title"},
    ]
    assert prepare_cohort(config, plan_only=True)["status"] == "blocked"
    with pytest.raises(CohortError, match="not ready"):
        _ready(config)
    _assets(config)
    assert prepare_cohort(config, probe=lambda _: 30.1)["status"] == "ready"
    for name, value in frozen.items():
        assert (output / name).read_bytes() == value
    assert not (output / "failures.jsonl").exists()
    ready = _ready(config)
    assert len(ready["catalog"]) == 5
    assert ready["eligibility"]["media"]["scene_count"] == 10
    assert ready["eligibility"]["media"]["keyframe_count"] == 20
    before = (output / "eligibility_summary.json").read_bytes()
    assert prepare_cohort(config, plan_only=True)["status"] == "ready"
    assert (output / "eligibility_summary.json").read_bytes() == before


def test_failed_refresh_keeps_final_files_but_blocks_their_use(tmp_path):
    config = _config(tmp_path)
    _assets(config)
    prepare_cohort(config, probe=lambda _: 30.0)
    output = config.output_dir / "data/cohort"
    completed = {
        name: (output / name).read_bytes()
        for name in ("catalog.jsonl", "sequences.jsonl", "metadata_titles.jsonl")
    }
    config.dataset.titles_csv.write_text("1,Only one title\n", encoding="utf-8")
    with pytest.raises(CohortError):
        prepare_cohort(config, force=True, probe=lambda _: 30.0)
    assert all((output / name).read_bytes() == data for name, data in completed.items())
    with pytest.raises(CohortError, match="not ready"):
        _ready(config)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True, "invalid"])
def test_invalid_duration_never_becomes_a_ready_inventory(tmp_path, duration):
    config = _config(tmp_path)
    _assets(config)
    with pytest.raises(CohortError, match="unavailable videos"):
        prepare_cohort(config, probe=lambda _: duration)
    rows = read_jsonl(config.output_dir / "data/cohort/item_inventory.jsonl")
    assert all(row["duration_seconds"] is None and not row["eligible"] for row in rows)
    with pytest.raises(CohortError, match="not ready"):
        _ready(config)


def test_required_union_excludes_unselected_users_and_truncated_history(tmp_path):
    config = _config(tmp_path, users=1)
    config.dataset.pairs_tsv.write_text(
        "only\t99 98 " + " ".join(str(index) for index in range(1, 14)) + "\nshort\t100\n",
        encoding="utf-8",
    )
    _assets(config, range(1, 14))
    prepare_cohort(config, probe=lambda _: 2.0)
    ready = _ready(config)
    assert [row["item_id"] for row in ready["catalog"]] == [str(index) for index in range(1, 14)]
    assert ready["sequences"][0]["valid_target"] == "12"
    assert ready["sequences"][0]["test_target"] == "13"
    assert ready["plan"]["statistics"]["truncated_user_count"] == 1
    assert all(row["content_id"].endswith(f"{int(row['item_id']):05d}") for row in ready["catalog"])


def test_statistics_distinguish_selection_and_refit_cold_targets(tmp_path):
    config = _config(tmp_path)
    prepare_cohort(config, plan_only=True)
    selected = read_jsonl(config.output_dir / "data/cohort/selected_users.jsonl")
    stats = cohort_statistics(selected)
    assert stats["selection"]["interaction_count"] == 6
    assert stats["refit"]["interaction_count"] == 8
    assert stats["selection"]["unique_item_count"] == 4
    assert stats["refit"]["unique_item_count"] == 5
    assert stats["selection"]["valid_target_unseen"]["fraction"] == 0.5
    assert stats["selection"]["test_target_unseen"]["fraction"] == 0.5
    assert stats["refit"]["test_target_unseen"]["fraction"] == 0.0


@pytest.mark.parametrize("change", ["seed", "user_count", "sequence", "path"])
@pytest.mark.parametrize("force", [False, True])
def test_selection_changes_require_a_new_run_even_with_force(tmp_path, change, force):
    config = _config(tmp_path)
    prepare_cohort(config, plan_only=True)
    output = config.output_dir / "data/cohort"
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    if change == "seed":
        config.cohort.seed = 43
    elif change == "user_count":
        config.cohort.user_count = 1
    elif change == "sequence":
        config.dataset.pairs_tsv.write_text("u1\t1 2 3 4 6\nu2\t1 2 3 5 4\n", encoding="utf-8")
    else:
        other = tmp_path / "other.tsv"
        other.write_bytes(config.dataset.pairs_tsv.read_bytes())
        config.dataset.pairs_tsv = other
    with pytest.raises(CohortError, match="new run_id"):
        prepare_cohort(config, plan_only=True, force=force)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize("file", ["catalog.jsonl", "eligibility_summary.json"])
def test_legacy_cohort_is_not_migrated(tmp_path, file):
    config = _config(tmp_path)
    output = config.output_dir / "data/cohort"
    if file.endswith(".jsonl"):
        write_jsonl(output / file, [{"item_id": "1"}])
    else:
        write_json(output / file, {"schema_version": "microlens-cohort-eligibility/v2"})
    with pytest.raises(CohortError, match="new run_id"):
        prepare_cohort(config, plan_only=True, force=True)
    assert not (output / "cohort_plan.json").exists()


def test_interrupted_plan_commit_can_resume_identical_partial_files(tmp_path, monkeypatch):
    config = _config(tmp_path)
    original_write = cohort_module.atomic_write_json

    def interrupt(path, value):
        if path.name == "cohort_plan.json":
            raise KeyboardInterrupt
        original_write(path, value)

    monkeypatch.setattr(cohort_module, "atomic_write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prepare_cohort(config, plan_only=True)
    output = config.output_dir / "data/cohort"
    assert (output / "selected_users.jsonl").is_file()
    monkeypatch.setattr(cohort_module, "atomic_write_json", original_write)
    assert prepare_cohort(config, plan_only=True)["status"] == "planned"


@pytest.mark.parametrize(
    "artifact,change",
    [
        ("sequences.jsonl", lambda rows: rows[0].update(test_target="1")),
        ("catalog.jsonl", lambda rows: rows.pop()),
        ("selected_users.jsonl", lambda rows: rows[0].update(cohort_rank=99)),
        ("required_items.jsonl", lambda rows: rows.append({"item_id": "6", "content_id": "c6"})),
        ("metadata_titles.jsonl", lambda rows: rows[0].update(title="")),
        ("item_inventory.jsonl", lambda rows: rows[0].update(duration_seconds=float("nan"))),
        ("cohort_plan.json", lambda doc: doc.update(required_item_count=999)),
        ("cohort_plan.json", lambda doc: doc.update(scale_statistics=None)),
        ("eligibility_summary.json", lambda doc: doc.update(candidate_user_count=999)),
    ],
)
def test_ready_gate_reads_actual_artifacts_and_rejects_mismatch(tmp_path, artifact, change):
    config = _config(tmp_path)
    _assets(config)
    prepare_cohort(config, probe=lambda _: 30.0)
    path = config.output_dir / "data/cohort" / artifact
    jsonl = artifact.endswith(".jsonl")
    value = read_jsonl(path) if jsonl else read_json(path)
    change(value)
    if jsonl:
        write_jsonl(path, value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(CohortError):
        _ready(config)
