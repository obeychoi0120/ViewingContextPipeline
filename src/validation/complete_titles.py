from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from validation.cohort_selection import content_id_for_item, normalize_item_id
from validation.io import atomic_write_json, read_jsonl


REPORT_SCHEMA_VERSION = "metadata-title-completion/v1"


class TitleCompletionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def complete_required_titles(
    *,
    primary_path: Path,
    supplement_path: Path,
    required_items_path: Path,
    output_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    primary_path = primary_path.resolve()
    supplement_path = supplement_path.resolve()
    required_items_path = required_items_path.resolve()
    output_path = output_path.resolve()
    report_path = (
        report_path.resolve()
        if report_path is not None
        else Path(f"{output_path}.report.json").resolve()
    )
    paths = [primary_path, supplement_path, required_items_path, output_path, report_path]
    if len(set(paths)) != len(paths):
        raise TitleCompletionError(
            "primary, supplement, required-items, output, and report must differ"
        )

    primary, primary_order = _load_titles(
        primary_path, allow_header=False, decode_quoted_title=False
    )
    supplement, _ = _load_titles(supplement_path, allow_header=True, decode_quoted_title=True)
    required = _load_required_items(required_items_path)
    missing = [item_id for item_id in required if not primary.get(item_id, "").strip()]
    unresolved = [item_id for item_id in missing if not supplement.get(item_id, "").strip()]
    if unresolved:
        raise TitleCompletionError(
            "official supplement does not resolve required metadata titles: " + ",".join(unresolved)
        )

    completed = dict(primary)
    for item_id in missing:
        completed[item_id] = supplement[item_id].strip()
    appended = sorted((set(completed) - set(primary_order)), key=int)
    output_order = [*primary_order, *appended]
    text = "".join(f"{item_id},{completed[item_id]}\n" for item_id in output_order)
    _atomic_write_text(output_path, text)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "policy": "required_blank_or_missing_from_official_supplement",
        "sources": {
            "primary": {"path": str(primary_path), "sha256": _sha256(primary_path)},
            "supplement": {"path": str(supplement_path), "sha256": _sha256(supplement_path)},
            "required_items": {
                "path": str(required_items_path),
                "sha256": _sha256(required_items_path),
            },
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "row_count": len(output_order),
        },
        "primary_blank_item_count": sum(not title.strip() for title in primary.values()),
        "required_item_count": len(required),
        "required_missing_or_blank_in_primary_count": len(missing),
        "supplemented_required_item_count": len(missing),
        "supplemented_item_ids": missing,
        "unresolved_required_item_count": 0,
        "remaining_blank_item_count": sum(not title.strip() for title in completed.values()),
    }
    atomic_write_json(report_path, report)
    print(
        f"[METADATA TITLES] required={len(required)} supplemented={len(missing)} "
        f"output={output_path} report={report_path}",
        flush=True,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Complete required blank MicroLens-100K titles from an official supplement."
    )
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--required-items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        complete_required_titles(
            primary_path=args.primary,
            supplement_path=args.supplement,
            required_items_path=args.required_items,
            output_path=args.output,
            report_path=args.report,
        )
    except KeyboardInterrupt:
        print("[INTERRUPTED] complete-metadata-titles", file=sys.stderr)
        return 130
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] complete-metadata-titles: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
