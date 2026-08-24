from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .raw_pipeline import process_local_source, write_manifest


class MicroLensPreparationError(RuntimeError):
    pass


def _item_values(path: Path, label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle), start=1):
                if not row or not any(cell.strip() for cell in row):
                    continue
                item_id = row[0].strip()
                if line_number == 1 and not item_id.isdigit():
                    continue
                if not item_id.isdigit() or int(item_id) <= 0 or len(row) < 2:
                    raise MicroLensPreparationError(f"invalid {label} row {line_number}")
                item_id = str(int(item_id))
                if item_id in values:
                    raise MicroLensPreparationError(f"duplicate {label} item {item_id}")
                value = ",".join(row[1:]).strip()
                if value:
                    values[item_id] = value
    except OSError as exc:
        raise MicroLensPreparationError(f"failed to read {label}: {path}") from exc
    if not values:
        raise MicroLensPreparationError(f"{label} contains no values: {path}")
    return values


def prepare_catalog(
    catalog: list[dict[str, Any]],
    *,
    titles_csv: str | Path,
    tags_csv: str | Path,
    assets_root: str | Path,
    output_root: str | Path,
    processing_config: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare exactly the cohort catalog from caller-owned MicroLens MP4 files."""

    titles = _item_values(Path(titles_csv), "titles")
    tags = _item_values(Path(tags_csv), "tags")
    failures: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    for index, row in enumerate(catalog, start=1):
        item_id = str(row["item_id"])
        content_id = str(row["content_id"])
        source = Path(str(row["source_video_path"]))
        metadata = {
            "title": titles.get(item_id, ""),
            "tags": tags.get(item_id, ""),
            "dataset_id": "microlens-100k",
            "source_item_id": item_id,
            "duration": row.get("duration_seconds"),
        }
        try:
            prepared = process_local_source(
                name=content_id,
                source_video_path=source,
                data_root=assets_root,
                output_root=output_root,
                metadata=metadata,
                config_path=processing_config,
                force=force,
            )
            manifest_rows.append(prepared)
            print(
                f"[PROGRESS] prepare_data {index}/{len(catalog)} {content_id} success",
                flush=True,
            )
        except Exception as exc:
            failures.append({"item_id": item_id, "content_id": content_id, "error": str(exc)})
            print(
                f"[FAILURE] prepare_data {index}/{len(catalog)} {content_id}: {exc}",
                flush=True,
            )
    cohort_root = Path(output_root) / "data" / "cohort"
    cohort_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cohort_root / "extraction_manifest.csv"
    failure_path = cohort_root / "preparation_failures.jsonl"
    failure_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    write_manifest(manifest_rows, manifest_path)
    return {
        "selected": len(catalog),
        "succeeded": len(manifest_rows),
        "failed": len(failures),
        "manifest": str(manifest_path),
        "failures": str(failure_path),
    }
