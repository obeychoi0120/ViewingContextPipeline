"""Canonical two-column manifest contract."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Iterable


MANIFEST_FIELDS = ("content_id", "url")
CANONICAL_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "contracts" / "manifest.csv"
)


class ManifestContractError(ValueError):
    """Raised when a manifest violates the repository contract."""


def read_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ManifestContractError(
            f"failed to read manifest {manifest_path}: {exc}"
        ) from exc
    return parse_manifest_text(text, source=str(manifest_path))


def parse_manifest_text(
    text: str,
    *,
    source: str = "manifest",
) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if reader.fieldnames != list(MANIFEST_FIELDS):
        raise ManifestContractError(
            f"{source} columns must be exactly {','.join(MANIFEST_FIELDS)}"
        )
    return validate_manifest_rows(list(reader), source=source)


def validate_manifest_rows(
    rows: Iterable[dict[str, Any]],
    *,
    source: str = "manifest",
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen_content_ids: set[str] = set()
    seen_urls: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        if not isinstance(row, dict) or set(row) != set(MANIFEST_FIELDS):
            raise ManifestContractError(
                f"{source} line {line_number} must contain exactly "
                f"{','.join(MANIFEST_FIELDS)}"
            )
        content_id = str(row.get("content_id") or "").strip()
        url = str(row.get("url") or "").strip()
        if not content_id:
            raise ManifestContractError(
                f"{source} line {line_number} has an empty content_id"
            )
        if not url:
            raise ManifestContractError(
                f"{source} line {line_number} has an empty url"
            )
        if content_id in seen_content_ids:
            raise ManifestContractError(
                f"{source} line {line_number} duplicates content_id {content_id}"
            )
        if url in seen_urls:
            raise ManifestContractError(
                f"{source} line {line_number} duplicates url {url}"
            )
        seen_content_ids.add(content_id)
        seen_urls.add(url)
        normalized.append({"content_id": content_id, "url": url})
    if not normalized:
        raise ManifestContractError(f"{source} contains no rows")
    return normalized
