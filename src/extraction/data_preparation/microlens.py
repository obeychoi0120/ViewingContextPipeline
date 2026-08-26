from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .fixed30 import prepare_visual_item


class MicroLensPreparationError(RuntimeError):
    pass


PREPARATION_WORKERS = 4


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
    image_size: tuple[int, int],
    force: bool = False,
) -> dict[str, Any]:
    """Prepare exactly the cohort catalog from caller-owned MicroLens MP4 files."""

    titles = _item_values(Path(titles_csv), "titles")
    tags = _item_values(Path(tags_csv), "tags")
    results: list[tuple[dict[str, str] | None, dict[str, str] | None]] = [
        (None, None) for _ in catalog
    ]

    def prepare(index: int, row: dict[str, Any]) -> tuple[int, str, dict[str, str] | None, dict[str, str] | None]:
        item_id = str(row.get("item_id", ""))
        content_id = str(row.get("content_id", ""))
        try:
            prepared = prepare_visual_item(
                content_id=content_id,
                source_video_path=Path(str(row["source_video_path"])),
                assets_root=assets_root,
                output_root=output_root,
                metadata={
                    "title": titles.get(item_id, ""),
                    "tags": tags.get(item_id, ""),
                    "dataset_id": "microlens-100k",
                    "source_item_id": item_id,
                    "duration": row.get("duration_seconds"),
                },
                image_size=image_size,
                force=force,
            )
            return index, content_id, prepared, None
        except Exception as exc:
            return index, content_id, None, {
                "item_id": item_id,
                "content_id": content_id,
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=PREPARATION_WORKERS) as executor:
        futures = [executor.submit(prepare, index, row) for index, row in enumerate(catalog)]
        with tqdm(
            total=len(catalog),
            desc="Prepare input data",
            unit="content",
        ) as progress:
            for future in as_completed(futures):
                index, content_id, prepared, failure = future.result()
                results[index] = (prepared, failure)
                if failure is not None:
                    tqdm.write(
                        f"[FAILURE] prepare_data {content_id} {failure['error']}",
                        file=progress.fp,
                    )
                progress.update(1)
    manifest_rows = [prepared for prepared, _ in results if prepared is not None]
    failures = [failure for _, failure in results if failure is not None]
    cohort_root = Path(output_root) / "data" / "cohort"
    cohort_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cohort_root / "extraction_manifest.csv"
    failure_path = cohort_root / "preparation_failures.jsonl"
    failure_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    _write_manifest(manifest_rows, manifest_path)
    return {
        "selected": len(catalog),
        "succeeded": len(manifest_rows),
        "failed": len(failures),
        "workers": PREPARATION_WORKERS,
        "manifest": str(manifest_path),
        "failures": str(failure_path),
    }


def _write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content_ids = [str(row.get("content_id") or "").strip() for row in rows]
    if not all(content_ids) or len(content_ids) != len(set(content_ids)):
        raise MicroLensPreparationError("extraction manifest requires unique content IDs")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["content_id"])
        writer.writeheader()
        writer.writerows({"content_id": value} for value in content_ids)
