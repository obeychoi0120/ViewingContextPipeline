from __future__ import annotations

import csv
from pathlib import Path
from unittest import mock

import pytest

from viewing_context_pipeline.extraction.common.manifest import read_manifest_rows
from viewing_context_pipeline.extraction.data_preparation.microlens import prepare_catalog
from viewing_context_pipeline.extraction.data_preparation.raw_pipeline import normalize_shot_interval


def _write_values(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "value"])
        writer.writerows(rows)


def test_sampling_accepts_fixed_30s_only() -> None:
    assert normalize_shot_interval("fixed_30s") == "fixed_30s"
    with pytest.raises(ValueError, match="fixed_30s"):
        normalize_shot_interval("fixed_15s")


def test_prepare_catalog_processes_exact_cohort(tmp_path: Path) -> None:
    titles = tmp_path / "titles.csv"
    tags = tmp_path / "tags.csv"
    _write_values(titles, [("1", "one"), ("2", "two")])
    _write_values(tags, [("1", "tag-a"), ("2", "tag-b")])
    catalog = [
        {"item_id": "1", "content_id": "microlens_100k_00001", "source_video_path": str(tmp_path / "1.mp4"), "duration_seconds": 30.0},
        {"item_id": "2", "content_id": "microlens_100k_00002", "source_video_path": str(tmp_path / "2.mp4"), "duration_seconds": 31.0},
    ]

    with mock.patch(
        "viewing_context_pipeline.extraction.data_preparation.microlens.process_local_source",
        side_effect=lambda **kwargs: {"content_id": kwargs["name"]},
    ) as process:
        result = prepare_catalog(
            catalog,
            titles_csv=titles,
            tags_csv=tags,
            assets_root=tmp_path / "assets",
            output_root=tmp_path / "run",
            processing_config=tmp_path / "processing.json",
        )

    assert result["succeeded"] == 2 and result["failed"] == 0
    assert [call.kwargs["source_video_path"].name for call in process.call_args_list] == ["1.mp4", "2.mp4"]
    assert read_manifest_rows(result["manifest"]) == [
        {"content_id": "microlens_100k_00001"},
        {"content_id": "microlens_100k_00002"},
    ]
