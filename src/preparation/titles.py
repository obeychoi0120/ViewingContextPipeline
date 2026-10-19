from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from preparation.catalog import content_id_for_item, normalize_item_id
from artifact_io import read_jsonl


REPORT_SCHEMA_VERSION = "metadata-title-completion/v1"


class TitleCompletionError(RuntimeError):
    pass


def _decode_supplement_title(raw: str, *, line_number: int) -> str:
    value = raw.strip()
    if not value.startswith('"'):
        return value
    try:
        parsed = next(csv.reader([f"0,{value}"], strict=True))
    except csv.Error as exc:
        raise TitleCompletionError(
            f"invalid quoted supplement title at row {line_number}: {exc}"
        ) from exc
    if len(parsed) != 2:
        raise TitleCompletionError(f"invalid quoted supplement title at row {line_number}")
    return parsed[1].strip()


def _load_titles(
    path: Path,
    *,
    allow_header: bool,
    decode_quoted_title: bool,
) -> tuple[dict[str, str], list[str]]:
    titles: dict[str, str] = {}
    order: list[str] = []
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise TitleCompletionError(f"failed to read title source {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            if allow_header and line_number == 1 and raw.strip().lower() == "item,title":
                continue
            if "," not in raw:
                raise TitleCompletionError(
                    f"invalid title row {line_number} in {path}: missing comma"
                )
            raw_item_id, raw_title = raw.split(",", 1)
            try:
                item_id = normalize_item_id(raw_item_id)
            except RuntimeError as exc:
                raise TitleCompletionError(
                    f"invalid title row {line_number} in {path}: {exc}"
                ) from exc
            if item_id in titles:
                raise TitleCompletionError(f"duplicate title item {item_id} in {path}")
            title = (
                _decode_supplement_title(raw_title, line_number=line_number)
                if decode_quoted_title
                else raw_title.strip()
            )
            if "\r" in title or "\n" in title:
                raise TitleCompletionError(f"multiline title is not supported for item {item_id}")
            titles[item_id] = title
            order.append(item_id)
    if not titles:
        raise TitleCompletionError(f"title source is empty: {path}")
    return titles, order


def _load_required_items(path: Path) -> list[str]:
    try:
        rows = read_jsonl(path)
    except (OSError, TypeError, ValueError) as exc:
        raise TitleCompletionError(f"failed to read required items {path}: {exc}") from exc
    item_ids: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or set(row) != {"item_id", "content_id"}:
            raise TitleCompletionError(f"invalid required item fields at row {index}")
        try:
            item_id = normalize_item_id(row["item_id"])
        except RuntimeError as exc:
            raise TitleCompletionError(f"invalid required item at row {index}: {exc}") from exc
        if row["content_id"] != content_id_for_item(item_id):
            raise TitleCompletionError(f"invalid required content_id at row {index}")
        if item_id in seen:
            raise TitleCompletionError(f"duplicate required item {item_id}")
        seen.add(item_id)
        item_ids.append(item_id)
    if not item_ids:
        raise TitleCompletionError("required items must not be empty")
    if item_ids != sorted(item_ids, key=int):
        raise TitleCompletionError("required items must be in numeric item_id order")
    return item_ids


def resolve_required_titles(
    *,
    primary_path: Path,
    supplement_path: Path,
    required_items_path: Path,
    unresolved_policy: str = "zero-vector",
) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    """Resolve titles without creating a completed CSV or report artifact."""
    if unresolved_policy not in {"error", "zero-vector"}:
        raise TitleCompletionError("unresolved_policy must be error or zero-vector")
    primary_path = primary_path.resolve()
    supplement_path = supplement_path.resolve()
    required_items_path = required_items_path.resolve()
    primary, primary_order = _load_titles(
        primary_path, allow_header=False, decode_quoted_title=False
    )
    supplement, _ = _load_titles(supplement_path, allow_header=True, decode_quoted_title=True)
    required = _load_required_items(required_items_path)
    missing = [item_id for item_id in required if not primary.get(item_id, "").strip()]
    unresolved = [item_id for item_id in missing if not supplement.get(item_id, "").strip()]
    if unresolved and unresolved_policy == "error":
        raise TitleCompletionError(
            "official supplement does not resolve required metadata titles: " + ",".join(unresolved)
        )

    completed = dict(primary)
    unresolved_ids = set(unresolved)
    supplemented = [item_id for item_id in missing if item_id not in unresolved_ids]
    for item_id in missing:
        completed[item_id] = supplement.get(item_id, "").strip()
    appended = sorted((set(completed) - set(primary_order)), key=int)
    output_order = [*primary_order, *appended]

    report = {
        "schema_version": (
            REPORT_SCHEMA_VERSION
            if unresolved_policy == "error"
            else "metadata-title-completion/v2"
        ),
        "policy": "required_blank_or_missing_from_official_supplement",
        "sources": {
            "primary": {"path": str(primary_path)},
            "supplement": {"path": str(supplement_path)},
            "required_items": {
                "path": str(required_items_path),
            },
        },
        "primary_blank_item_count": sum(not title.strip() for title in primary.values()),
        "required_item_count": len(required),
        "required_missing_or_blank_in_primary_count": len(missing),
        "supplemented_required_item_count": len(supplemented),
        "supplemented_item_ids": supplemented,
        "unresolved_required_item_count": len(unresolved),
        "remaining_blank_item_count": sum(not title.strip() for title in completed.values()),
    }
    if unresolved_policy == "zero-vector":
        report.update(unresolved_policy="zero_vector", unresolved_item_ids=unresolved)
    return completed, output_order, report


def missing_metadata_report(titles):
    missing = [
        {"item_id": row["item_id"], "content_id": row["content_id"], "embedding_row": index}
        for index, row in enumerate(titles)
        if not row["title"].strip()
    ]
    return {
        "schema_version": "metadata-missing/v1",
        "policy": "zero_vector",
        "catalog_size": len(titles),
        "missing_count": len(missing),
        "items": missing,
    }


def metadata_titles_match_catalog(path, catalog, *, allow_blank=False):
    try:
        titles = read_jsonl(path)
        return len(titles) == len(catalog) and all(
            (
                set(t) == {"item_id", "content_id", "title"}
                and str(t["item_id"]) == str(c["item_id"])
                and (str(t["content_id"]) == str(c["content_id"]))
                and isinstance(t["title"], str)
                and (allow_blank or bool(t["title"].strip()))
                for t, c in zip(titles, catalog, strict=True)
            )
        )
    except (OSError, ValueError):
        return False
