import json

from conftest import config_data
from viewing_context_pipeline.validation.cohort import largest_remainder_quotas, prepare_cohort, select_users, split_record
from viewing_context_pipeline.validation.config import ValidationConfig


def test_largest_remainder_is_exact() -> None:
    quotas = largest_remainder_quotas({"5-9": 51, "10-19": 31, "20-49": 18}, 33)
    assert quotas == {"5-9": 17, "10-19": 10, "20-49": 6}


def test_user_selection_is_deterministic_and_split_has_no_target_leakage() -> None:
    pairs = [(f"u{i:03d}", [str(value) for value in range(1, 6 + i % 20)]) for i in range(80)]
    eligible = {str(value) for value in range(1, 30)}
    first, quotas = select_users(pairs, eligible, count=32, seed=42, boundaries=[5, 10, 20, 50], min_length=5, max_length=13)
    second, _ = select_users(pairs, eligible, count=32, seed=42, boundaries=[5, 10, 20, 50], min_length=5, max_length=13)
    assert first == second and len(first) == sum(quotas.values()) == 32
    for row in first:
        split = split_record(row)
        assert split["train"] == split["sequence"][:-2]


def test_prepare_cohort_needs_pairs_and_mp4_not_titles(tmp_path) -> None:
    data = config_data(tmp_path, users=10)
    videos = data["dataset"]["videos_dir"]
    videos.mkdir()
    for item in range(1, 21):
        (videos / f"{item}.mp4").write_bytes(b"video")
    pairs = data["dataset"]["pairs_tsv"]
    pairs.write_text("".join(f"u{user:02d}\t" + " ".join(str((user + step) % 20 + 1) for step in range(8)) + "\n" for user in range(12)), encoding="utf-8")
    config = ValidationConfig.model_validate(data)
    manifest = prepare_cohort(config, probe=lambda _: 2.0)
    assert manifest["user_count"] == 10
    catalog = [json.loads(line) for line in (config.output_dir / "data/cohort/catalog.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("title" not in row for row in catalog)
    assert not (config.output_dir / "data/cohort/vce_smoke_selection.jsonl").exists()
