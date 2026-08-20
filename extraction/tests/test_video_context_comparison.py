from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.scene_context_extraction.graph_core.scoring import CONTENT_AXIS_ORDER
from src.validation.compare_video_contexts import main, parse_args
from src.validation.video_context_comparison import (
    PER_CONTENT_FILENAME,
    REPORT_FILENAME,
    jaccard_similarity,
)


TOP_LIST_FIELDS = (
    "top_styles",
    "top_moods",
    "top_scene_functions",
    "top_entities",
    "top_motifs",
)


def _profile(
    content_id: str,
    *,
    subject_sociality: float = 0.0,
    subject_distribution: dict[str, float] | None = None,
    top_styles: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, object]:
    axes = {axis: 0.0 for axis in CONTENT_AXIS_ORDER}
    axes["subject_sociality"] = subject_sociality
    distributions = {
        axis: {"neutral": 1.0}
        for axis in CONTENT_AXIS_ORDER
    }
    if subject_distribution is not None:
        distributions["subject_sociality"] = subject_distribution
    top_ids = {
        field: [f"{field}:shared"]
        for field in TOP_LIST_FIELDS
    }
    if top_styles is not None:
        top_ids["top_styles"] = top_styles
    profile = {
        "content_axes_4d": axes,
        "content_axis_distribution": distributions,
        **{
            field: [
                {"id": item_id, "count": index + 1}
                for index, item_id in enumerate(ids)
            ]
            for field, ids in top_ids.items()
        },
    }
    return {
        "content_id": content_id,
        "source_scene_context_path": f"output/{content_id}.jsonl",
        "context": profile,
        "aggregation_warnings": warnings or [],
    }


def _write_manifest(path: Path, content_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["content_id", "url"])
        writer.writeheader()
        for index, content_id in enumerate(content_ids):
            writer.writerow(
                {
                    "content_id": content_id,
                    "url": f"https://example.test/{index}",
                }
            )


def _write_pair(
    context_dir: Path,
    context_ref_dir: Path,
    content_id: str,
    ondevice: dict[str, object],
    reference: dict[str, object],
) -> tuple[Path, Path]:
    ondevice_path = (
        context_dir / f"{content_id}_context_graph_ond.json"
    )
    reference_path = (
        context_ref_dir / f"{content_id}_context_graph_ref.json"
    )
    ondevice_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    ondevice_path.write_text(
        json.dumps(ondevice, ensure_ascii=False),
        encoding="utf-8",
    )
    reference_path.write_text(
        json.dumps(reference, ensure_ascii=False),
        encoding="utf-8",
    )
    return ondevice_path, reference_path


def test_cli_writes_metrics_to_default_report_dir_without_mutating_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    output_root = tmp_path / "output"
    context_dir = (
        output_root
        / "viewing_context"
        / "img_only"
        / "shot_wise"
        / "video_context_graph_gaussa_gemma4_e2b_v0_3"
    )
    context_ref_dir = output_root / "viewing_context" / "img_only" / "shot_wise" / "video_context_graph_ref"
    content_ids = ["News_Manual_001_a", "Tech_Manual_002_b"]
    _write_manifest(manifest_path, content_ids)
    monkeypatch.setenv("OUTPUT_SAVE_PATH", str(output_root))

    first_ondevice = _profile(
        content_ids[0],
        subject_sociality=0.5,
        subject_distribution={"positive": 1.0},
        top_styles=["style:a", "style:b"],
        warnings=["graph warning"],
    )
    first_reference = _profile(
        content_ids[0],
        subject_distribution={"neutral": 1.0},
        top_styles=["style:b", "style:c"],
        warnings=["ref warning 1", "ref warning 2"],
    )
    paths = list(
        _write_pair(
            context_dir,
            context_ref_dir,
            content_ids[0],
            first_ondevice,
            first_reference,
        )
    )
    paths.extend(
        _write_pair(
            context_dir,
            context_ref_dir,
            content_ids[1],
            _profile(content_ids[1]),
            _profile(content_ids[1]),
        )
    )
    input_bytes = {path: path.read_bytes() for path in paths}

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--context-dir",
            str(context_dir),
            "--context-ref-dir",
            str(context_ref_dir),
        ]
    )

    assert exit_code == 0
    report_dir = output_root / "custom" / "reports" / "viewing_context" / "comparisons" / "shot_wise"
    report = json.loads(
        (report_dir / REPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert report["expected_content_count"] == 2
    assert report["paired_content_count"] == 2
    assert report["context_dir"] == str(context_dir.resolve())
    assert report["context_ref_dir"] == str(context_ref_dir.resolve())
    assert report["metrics"]["overall"]["axis_mae"] == {
        "mean": 0.0625,
        "median": 0.0625,
        "max": 0.125,
    }
    assert report["metrics"]["axes"]["subject_sociality"] == {
        "signed_delta": {
            "mean": 0.25,
            "median": 0.25,
            "max": 0.5,
        },
        "absolute_error": {
            "mean": 0.25,
            "median": 0.25,
            "max": 0.5,
        },
        "distribution_tvd": {
            "mean": 0.5,
            "median": 0.5,
            "max": 1.0,
        },
    }
    assert report["metrics"]["top_lists"]["top_styles"] == {
        "mean": 0.666667,
        "median": 0.666667,
        "max": 1.0,
    }
    assert (
        report["categories"]["News"]["metrics"]["overall"]["axis_mae"][
            "max"
        ]
        == 0.125
    )
    assert (
        report["categories"]["Tech"]["metrics"]["overall"]["axis_mae"][
            "max"
        ]
        == 0.0
    )
    assert report["categories"]["News"]["warning_counts"] == {
        "graph": 1,
        "reference": 2,
    }
    assert report["warning_counts"] == {
        "graph": 1,
        "reference": 2,
    }
    assert report["worst_contents_by_axis_mae"][0]["content_id"] == (
        content_ids[0]
    )

    with (report_dir / PER_CONTENT_FILENAME).open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["subject_sociality_graph"] == "0.5"
    assert rows[0]["axis_mae"] == "0.125"
    assert rows[0]["subject_sociality_distribution_tvd"] == "1.0"
    assert rows[0]["top_styles_jaccard"] == "0.333333"
    assert rows[0]["mean_top_list_jaccard"] == "0.866667"
    assert {path: path.read_bytes() for path in paths} == input_bytes


@pytest.mark.parametrize(
    ("failure", "expected_group"),
    [
        ("graph_missing", "graph_missing=1"),
        ("reference_missing", "reference_missing=1"),
        ("invalid_json", "graph_invalid=1"),
        ("content_id_mismatch", "reference_invalid=1"),
        ("invalid_structure", "graph_invalid=1"),
    ],
)
def test_cli_preflight_failure_returns_one_and_preserves_existing_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: str,
    expected_group: str,
) -> None:
    content_id = "Game_Manual_003_demo"
    manifest_path = tmp_path / "manifest.csv"
    output_root = tmp_path / "output"
    context_dir = output_root / "viewing_context" / "img_only" / "fixed_15s" / "video_context_graph_qwen"
    context_ref_dir = output_root / "viewing_context" / "img_only" / "fixed_15s" / "video_context_graph_ref"
    report_dir = tmp_path / "custom-reports"
    _write_manifest(manifest_path, [content_id])
    ondevice = _profile(content_id)
    reference = _profile(content_id)

    if failure == "content_id_mismatch":
        reference["content_id"] = "other"
    if failure == "invalid_structure":
        invalid_profile = deepcopy(ondevice["context"])
        invalid_profile["content_axis_distribution"]["subject_sociality"] = {}
        ondevice["context"] = invalid_profile

    ondevice_path, reference_path = _write_pair(
        context_dir,
        context_ref_dir,
        content_id,
        ondevice,
        reference,
    )
    if failure == "graph_missing":
        ondevice_path.unlink()
    elif failure == "reference_missing":
        reference_path.unlink()
    elif failure == "invalid_json":
        ondevice_path.write_text("{", encoding="utf-8")

    report_dir.mkdir()
    report_path = report_dir / REPORT_FILENAME
    csv_path = report_dir / PER_CONTENT_FILENAME
    report_path.write_text("old json report", encoding="utf-8")
    csv_path.write_text("old csv report", encoding="utf-8")

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--context-dir",
            str(context_dir),
            "--context-ref-dir",
            str(context_ref_dir),
            "--report-dir",
            str(report_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert expected_group in captured.err
    assert content_id in captured.err
    assert report_path.read_text(encoding="utf-8") == "old json report"
    assert csv_path.read_text(encoding="utf-8") == "old csv report"


def test_jaccard_empty_set_rules() -> None:
    assert jaccard_similarity(set(), set()) == 1.0
    assert jaccard_similarity({"style:a"}, set()) == 0.0
    assert jaccard_similarity(set(), {"style:a"}) == 0.0


def test_cli_requires_explicit_profile_directories() -> None:
    args = parse_args(
        ["--context-dir", "graph", "--context-ref-dir", "reference"]
    )

    assert args.context_dir == Path("graph")
    assert args.context_ref_dir == Path("reference")
