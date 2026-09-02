import json

import pytest

from conftest import config_data
from validation.cohort import (
    CohortError,
    largest_remainder_quotas,
    load_metadata_titles,
    prepare_cohort,
    select_users,
    split_record,
)
from validation.config import ValidationConfig


def test_largest_remainder_is_exact() -> None:
    quotas = largest_remainder_quotas({"5-9": 51, "10-19": 31, "20-49": 18}, 33)
    assert quotas == {"5-9": 17, "10-19": 10, "20-49": 6}


def test_user_selection_is_deterministic_and_split_has_no_target_leakage() -> None:
    pairs = [(f"u{i:03d}", [str(value) for value in range(1, 6 + i % 20)]) for i in range(80)]
    eligible = {str(value) for value in range(1, 30)}
    first, quotas = select_users(
        pairs, eligible, count=32, seed=42, boundaries=[5, 10, 20, 50], min_length=5, max_length=13
    )
    second, _ = select_users(
        pairs, eligible, count=32, seed=42, boundaries=[5, 10, 20, 50], min_length=5, max_length=13
    )
    assert first == second and len(first) == sum(quotas.values()) == 32
    for row in first:
        split = split_record(row)
        assert split["train"] == split["sequence"][:-2]


def _write_titles(path, item_ids: range) -> None:
    path.write_text(
        "".join(f"{item},Title {item}\n" for item in item_ids),
        encoding="utf-8",
    )


def test_prepare_cohort_requires_complete_metadata_titles(tmp_path) -> None:
    data = config_data(tmp_path, users=10)
    videos = data["dataset"]["videos_dir"]
    videos.mkdir()
    for item in range(1, 21):
        (videos / f"{item}.mp4").write_bytes(b"video")
    pairs = data["dataset"]["pairs_tsv"]
    pairs.write_text(
        "".join(
            f"u{user:02d}\t" + " ".join(str((user + step) % 20 + 1) for step in range(8)) + "\n"
            for user in range(12)
        ),
        encoding="utf-8",
    )
    _write_titles(data["dataset"]["titles_csv"], range(1, 21))
    config = ValidationConfig.model_validate(data)
    stale_failure = config.output_dir / "data/cohort/failures.jsonl"
    stale_failure.parent.mkdir(parents=True)
    stale_failure.write_text('{"reason":"stale"}\n', encoding="utf-8")
    manifest = prepare_cohort(config, probe=lambda _: 2.0)
    assert manifest["user_count"] == 10
    catalog = [
        json.loads(line)
        for line in (config.output_dir / "data/cohort/catalog.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all("title" not in row for row in catalog)
    metadata_titles = [
        json.loads(line)
        for line in (config.output_dir / "data/cohort/metadata_titles.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(metadata_titles) == manifest["catalog_size"] == 19
    assert set(metadata_titles[0]) == {"item_id", "content_id", "title"}
    assert not (config.output_dir / "data/cohort/failures.jsonl").exists()
    assert not (config.output_dir / "data/cohort/vce_smoke_selection.jsonl").exists()


def test_prepare_cohort_uses_full_eligible_catalog_not_selected_user_union(tmp_path) -> None:
    data = config_data(tmp_path, users=1)
    videos = data["dataset"]["videos_dir"]
    videos.mkdir()
    for item in range(1, 11):
        (videos / f"{item}.mp4").write_bytes(b"video")
    data["dataset"]["pairs_tsv"].write_text(
        "u01\t1 2 3 4 5\nu02\t6 7 8 9 10\n",
        encoding="utf-8",
    )
    _write_titles(data["dataset"]["titles_csv"], range(1, 11))
    config = ValidationConfig.model_validate(data)

    prepare_cohort(config, probe=lambda _: 2.0)

    cohort_dir = config.output_dir / "data/cohort"
    sequences = [
        json.loads(line)
        for line in (cohort_dir / "sequences.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    catalog = [
        json.loads(line)
        for line in (cohort_dir / "catalog.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len({item for row in sequences for item in row["sequence"]}) == 5
    assert [row["item_id"] for row in catalog] == [str(item) for item in range(1, 11)]


def test_prepare_cohort_preserves_eligibility_diagnostics_on_failure(tmp_path) -> None:
    data = config_data(tmp_path, users=2)
    videos = data["dataset"]["videos_dir"]
    videos.mkdir()
    (videos / "1.mp4").write_bytes(b"video")
    pairs = data["dataset"]["pairs_tsv"]
    pairs.write_text(
        "u01\t1 2 3 4 5\nu02\t1 2 3 4 5\n",
        encoding="utf-8",
    )
    _write_titles(data["dataset"]["titles_csv"], range(1, 2))
    config = ValidationConfig.model_validate(data)

    with pytest.raises(CohortError, match="eligibility_summary.json"):
        prepare_cohort(config, probe=lambda _: 2.0)

    cohort_dir = config.output_dir / "data/cohort"
    summary = json.loads((cohort_dir / "eligibility_summary.json").read_text(encoding="utf-8"))
    assert summary["requested_users"] == 2
    assert summary["eligible_users"] == 0
    assert summary["item_exclusions"] == {"missing_video": 4}
    assert (cohort_dir / "item_inventory.jsonl").exists()
    assert (cohort_dir / "failures.jsonl").exists()


def test_title_csv_accepts_bom_and_splits_only_the_first_comma(tmp_path) -> None:
    path = tmp_path / "titles.csv"
    path.write_text("\ufeff1,A title, with commas\n2,Second title\n", encoding="utf-8")

    assert load_metadata_titles(path) == {
        "1": "A title, with commas",
        "2": "Second title",
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1,First\n01,Duplicate\n", "duplicate metadata title"),
        ("1,   \n01,Duplicate\n", "duplicate metadata title"),
        ("1,First\n01,   \n", "duplicate metadata title"),
        ("1,\n01,\n", "duplicate metadata title"),
        ("not-an-id,Title\n", "invalid metadata title row"),
        ("not-an-id,   \n", "invalid metadata title row"),
        ("1 Title\n", "missing comma"),
    ],
)
def test_title_csv_rejects_invalid_rows(tmp_path, text: str, message: str) -> None:
    path = tmp_path / "titles.csv"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(CohortError, match=message):
        load_metadata_titles(path)


def test_title_csv_omits_blank_titles_for_catalog_coverage_check(tmp_path) -> None:
    path = tmp_path / "titles.csv"
    path.write_text("\ufeff1,A title, with commas\n2,\n1849, \t\n", encoding="utf-8")

    assert load_metadata_titles(path) == {"1": "A title, with commas"}


@pytest.mark.parametrize("outside_item", ["1849", "6"])
def test_blank_title_outside_eligible_catalog_preserves_cohort(tmp_path, outside_item) -> None:
    data = config_data(tmp_path, users=1)
    videos = data["dataset"]["videos_dir"]
    videos.mkdir()
    for item in range(1, 6):
        (videos / f"{item}.mp4").write_bytes(b"video")
    # Item 6 is referenced but has no video; item 1849 is not referenced at all.
    sequence = "1 2 3 4 5 6" if outside_item == "6" else "1 2 3 4 5"
    data["dataset"]["pairs_tsv"].write_text(f"u01\t{sequence}\n", encoding="utf-8")
    titles_path = data["dataset"]["titles_csv"]
    _write_titles(titles_path, range(1, 6))
    config = ValidationConfig.model_validate(data)
    baseline_result = prepare_cohort(config, probe=lambda _: 2.0)
    cohort_dir = config.output_dir / "data/cohort"
    baseline_files = {path.name: path.read_bytes() for path in cohort_dir.iterdir()}

    titles_path.write_text(
        titles_path.read_text(encoding="utf-8") + f"{outside_item}, \t\n", encoding="utf-8"
    )
    result = prepare_cohort(config, probe=lambda _: 2.0)

    assert result == baseline_result
    assert {path.name: path.read_bytes() for path in cohort_dir.iterdir()} == baseline_files
    catalog = [json.loads(line) for line in (cohort_dir / "catalog.jsonl").read_text().splitlines()]
    assert [row["item_id"] for row in catalog] == [str(item) for item in range(1, 6)]


@pytest.mark.parametrize("last_title_row", ["", "5,\n", "5, \t\n"])
def test_missing_title_fails_without_shrinking_catalog(tmp_path, last_title_row) -> None:
    data = config_data(tmp_path, users=1)
    videos = data["dataset"]["videos_dir"]
    videos.mkdir()
    for item in range(1, 6):
        (videos / f"{item}.mp4").write_bytes(b"video")
    data["dataset"]["pairs_tsv"].write_text(
        "u01\t1 2 3 4 5\n",
        encoding="utf-8",
    )
    _write_titles(data["dataset"]["titles_csv"], range(1, 5))
    titles_path = data["dataset"]["titles_csv"]
    titles_path.write_text(
        titles_path.read_text(encoding="utf-8") + last_title_row, encoding="utf-8"
    )
    config = ValidationConfig.model_validate(data)

    with pytest.raises(CohortError, match="without metadata titles"):
        prepare_cohort(config, probe=lambda _: 2.0)

    cohort_dir = config.output_dir / "data/cohort"
    eligibility = json.loads((cohort_dir / "eligibility_summary.json").read_text(encoding="utf-8"))
    assert eligibility["eligible_items"] == 5
    assert eligibility["metadata_title_coverage"] == {
        "schema_version": "metadata-title/v1",
        "catalog_item_count": 5,
        "covered_item_count": 4,
        "missing_item_count": 1,
    }
    assert not (cohort_dir / "metadata_titles.jsonl").exists()
    failures = [
        json.loads(line)
        for line in (cohort_dir / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert failures == [{"item_id": "5", "reason": "missing_metadata_title"}]
    inventory = [
        json.loads(line)
        for line in (cohort_dir / "item_inventory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(inventory) == 5 and all(row["eligible"] for row in inventory)
