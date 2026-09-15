from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import fcntl
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .fixed30 import prepare_visual_item
from .media import resolve_duration


PREPARATION_WORKERS = 8


def prepare_catalog(
    catalog: list[dict[str, Any]],
    *,
    assets_root: str | Path,
    output_root: str | Path,
    failure_path: str | Path,
    image_size: tuple[int, int],
    scene_duration: int = 30,
    num_keyframes: int = 6,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare exactly the cohort catalog from caller-owned MicroLens MP4 files."""

    results: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = [
        (None, None) for _ in catalog
    ]

    def prepare(index: int, row: dict[str, Any]) -> tuple[int, str, dict[str, Any] | None, dict[str, Any] | None]:
        item_id = str(row.get("item_id", ""))
        content_id = str(row.get("content_id", ""))
        try:
            from .fixed30 import _safe_content_id
            item_assets = Path(assets_root) / _safe_content_id(content_id) / "assets"
            item_assets.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(item_assets, os.O_RDONLY | os.O_DIRECTORY)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                from extraction.evidence_reuse import SOURCE_IDENTITY_KEYS
                from pipeline_runtime import read_json
                checkpoint = item_assets / "video_duration.json"
                if checkpoint.exists():
                    saved = read_json(checkpoint)
                    if any(saved.get(key) != row.get(key) for key in SOURCE_IDENTITY_KEYS):
                        raise ValueError(f"shared source identity mismatch: {checkpoint}; use a separate artifacts_root for changed source videos")
                duration = resolve_duration(Path(assets_root), row)
                prepared = prepare_visual_item(
                    content_id=content_id,
                    source_video_path=Path(str(row["source_video_path"])),
                    assets_root=assets_root,
                    output_root=output_root,
                    duration_seconds=duration,
                    image_size=image_size,
                    scene_duration=scene_duration,
                    num_keyframes=num_keyframes,
                    force=force,
                )
            finally:
                os.close(descriptor)
            return index, content_id, prepared, None
        except Exception as exc:
            return index, content_id, None, {
                "item_id": item_id,
                "content_id": content_id,
                "error": str(exc),
            }

    counts = {"reused_frames": 0, "extracted_frames": 0}
    with tqdm(
        total=len(catalog),
        desc="Prepare visual evidence",
        unit="video",
        dynamic_ncols=True,
    ) as progress, ThreadPoolExecutor(max_workers=PREPARATION_WORKERS) as executor:
        futures = [executor.submit(prepare, index, row) for index, row in enumerate(catalog)]
        for future in as_completed(futures):
            index, content_id, prepared, failure = future.result()
            results[index] = (prepared, failure)
            if failure is not None:
                tqdm.write(
                    f"[FAILURE] prepare_data {content_id} {failure['error']}",
                    file=progress.fp,
                )
            if prepared is not None:
                for key in counts:
                    counts[key] += prepared[key]
                progress.set_postfix(reused_frames=counts["reused_frames"], new_frames=counts["extracted_frames"])
            progress.update(1)
    prepared_rows = [prepared for prepared, _ in results if prepared is not None]
    failures = [failure for _, failure in results if failure is not None]
    failure_path = Path(failure_path)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    if failures:
        failure_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
            encoding="utf-8",
        )
    else:
        failure_path.unlink(missing_ok=True)
    return {
        **counts,
        "selected": len(catalog),
        "succeeded": len(prepared_rows),
        "failed": len(failures),
        "workers": PREPARATION_WORKERS,
        "failures": str(failure_path),
    }
