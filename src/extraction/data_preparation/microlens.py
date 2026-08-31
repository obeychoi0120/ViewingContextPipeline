from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .fixed30 import prepare_visual_item


PREPARATION_WORKERS = 4


def prepare_catalog(
    catalog: list[dict[str, Any]],
    *,
    assets_root: str | Path,
    output_root: str | Path,
    image_size: tuple[int, int],
    force: bool = False,
) -> dict[str, Any]:
    """Prepare exactly the cohort catalog from caller-owned MicroLens MP4 files."""

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
                duration_seconds=row.get("duration_seconds"),
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
    prepared_rows = [prepared for prepared, _ in results if prepared is not None]
    failures = [failure for _, failure in results if failure is not None]
    cohort_root = Path(output_root) / "data" / "cohort"
    cohort_root.mkdir(parents=True, exist_ok=True)
    failure_path = cohort_root / "preparation_failures.jsonl"
    if failures:
        failure_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
            encoding="utf-8",
        )
    else:
        failure_path.unlink(missing_ok=True)
    return {
        "selected": len(catalog),
        "succeeded": len(prepared_rows),
        "failed": len(failures),
        "workers": PREPARATION_WORKERS,
        "failures": str(failure_path),
    }
