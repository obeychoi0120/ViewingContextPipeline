from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.complete_titles import (
    TitleCompletionError,
    complete_required_titles,
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
