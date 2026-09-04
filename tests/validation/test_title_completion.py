from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.complete_titles import (
    REPORT_SCHEMA_VERSION,
    TitleCompletionError,
    complete_required_titles,
    main,
)


def _required(path: Path, item_ids: list[int]) -> None:
    path.write_text(
        "".join(
            json.dumps({"item_id": str(item_id), "content_id": f"microlens_100k_{item_id:05d}"})
            + "\n"
            for item_id in item_ids
        ),
        encoding="utf-8",
    )


def test_completion_fills_only_required_blank_titles_and_writes_provenance(tmp_path) -> None:
    primary = tmp_path / "MicroLens-100k_title_en.csv"
    supplement = tmp_path / "MicroLens-50k_titles.csv"
    required = tmp_path / "required_items.jsonl"
    output = tmp_path / "MicroLens-100k_title_en_completed.csv"
    primary.write_text("\ufeff1,Primary one\n2, \n3,Primary three\n4,\n", encoding="utf-8")
    supplement.write_text(
        'item,title\n1,"Different, primary title"\n2,"Completed, title"\n4,"Unused title"\n',
        encoding="utf-8",
    )
    _required(required, [1, 2, 3])

    report = complete_required_titles(
        primary_path=primary,
        supplement_path=supplement,
        required_items_path=required,
        output_path=output,
    )

    assert output.read_text(encoding="utf-8") == (
        "1,Primary one\n2,Completed, title\n3,Primary three\n4,\n"
    )
    assert primary.read_text(encoding="utf-8").startswith("\ufeff1,Primary one")
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["supplemented_item_ids"] == ["2"]
    assert report["primary_blank_item_count"] == 2
    assert report["remaining_blank_item_count"] == 1
    assert report["unresolved_required_item_count"] == 0
    saved = json.loads(Path(f"{output}.report.json").read_text(encoding="utf-8"))
    assert saved == report
    assert saved["sources"]["primary"]["sha256"]
    assert saved["output"]["sha256"]


def test_completion_adds_required_item_missing_from_primary(tmp_path) -> None:
    primary = tmp_path / "primary.csv"
    supplement = tmp_path / "supplement.csv"
    required = tmp_path / "required.jsonl"
    output = tmp_path / "completed.csv"
    primary.write_text("1,One\n3,Three\n", encoding="utf-8")
    supplement.write_text("item,title\n2,Two\n", encoding="utf-8")
    _required(required, [1, 2])

    report = complete_required_titles(
        primary_path=primary,
        supplement_path=supplement,
        required_items_path=required,
        output_path=output,
    )

    assert output.read_text(encoding="utf-8") == "1,One\n3,Three\n2,Two\n"
    assert report["output"]["row_count"] == 3
    assert report["supplemented_item_ids"] == ["2"]


def test_completion_refuses_unresolved_required_title_without_output(tmp_path) -> None:
    primary = tmp_path / "primary.csv"
    supplement = tmp_path / "supplement.csv"
    required = tmp_path / "required.jsonl"
    output = tmp_path / "completed.csv"
    primary.write_text("1,\n", encoding="utf-8")
    supplement.write_text("item,title\n1, \n", encoding="utf-8")
    _required(required, [1])

    with pytest.raises(TitleCompletionError, match="does not resolve.*1"):
        complete_required_titles(
            primary_path=primary,
            supplement_path=supplement,
            required_items_path=required,
            output_path=output,
        )

    assert not output.exists()
    assert not Path(f"{output}.report.json").exists()


@pytest.mark.parametrize(
    ("required_text", "message"),
    [
        ('{"item_id":"1","content_id":"wrong"}\n', "invalid required content_id"),
        (
            '{"item_id":"2","content_id":"microlens_100k_00002"}\n'
            '{"item_id":"1","content_id":"microlens_100k_00001"}\n',
            "numeric item_id order",
        ),
    ],
)
def test_completion_validates_required_item_contract(
    tmp_path, required_text: str, message: str
) -> None:
    primary = tmp_path / "primary.csv"
    supplement = tmp_path / "supplement.csv"
    required = tmp_path / "required.jsonl"
    primary.write_text("1,One\n2,Two\n", encoding="utf-8")
    supplement.write_text("item,title\n1,One\n2,Two\n", encoding="utf-8")
    required.write_text(required_text, encoding="utf-8")

    with pytest.raises(TitleCompletionError, match=message):
        complete_required_titles(
            primary_path=primary,
            supplement_path=supplement,
            required_items_path=required,
            output_path=tmp_path / "output.csv",
        )


def test_completion_refuses_to_overwrite_a_source(tmp_path) -> None:
    primary = tmp_path / "primary.csv"
    supplement = tmp_path / "supplement.csv"
    required = tmp_path / "required.jsonl"
    primary.write_text("1,\n", encoding="utf-8")
    supplement.write_text("item,title\n1,One\n", encoding="utf-8")
    _required(required, [1])

    with pytest.raises(TitleCompletionError, match="must differ"):
        complete_required_titles(
            primary_path=primary,
            supplement_path=supplement,
            required_items_path=required,
            output_path=primary,
        )


def test_completion_cli_returns_nonzero_for_unresolved_title(tmp_path, capsys) -> None:
    primary = tmp_path / "primary.csv"
    supplement = tmp_path / "supplement.csv"
    required = tmp_path / "required.jsonl"
    primary.write_text("1,\n", encoding="utf-8")
    supplement.write_text("item,title\n1,\n", encoding="utf-8")
    _required(required, [1])

    status = main(
        [
            "--primary",
            str(primary),
            "--supplement",
            str(supplement),
            "--required-items",
            str(required),
            "--output",
            str(tmp_path / "output.csv"),
        ]
    )

    assert status == 1
    assert "[FAILED] complete-metadata-titles" in capsys.readouterr().err
