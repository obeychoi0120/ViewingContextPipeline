from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

from src.video_data_collection.microlens_config import MicroLensConfig
from src.video_data_collection.raw_pipeline import process_local_source, write_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MicroLensImportError(RuntimeError):
    """Raised when MicroLens inputs cannot produce a stable import."""


def normalize_item_id(value: object) -> str:
    text = str(value or "").strip()
    if not text.isdigit() or int(text) <= 0:
        raise MicroLensImportError(f"invalid MicroLens item ID: {value!r}")
    return str(int(text))


def content_id_for_item(item_id: str) -> str:
    return f"microlens_100k_{int(item_id):05d}"


def provenance_uri(item_id: str) -> str:
    return f"microlens://100k/{int(item_id)}"


def load_item_csv(path: Path, label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for line_number, row in enumerate(csv.reader(file), start=1):
                if not row or not any(cell.strip() for cell in row):
                    continue
                if not values and row[0].strip().lower() in {
                    "item_id",
                    "itemid",
                    "video_id",
                    "videoid",
                }:
                    continue
                if len(row) < 2:
                    raise MicroLensImportError(
                        f"{label} line {line_number} must contain item ID and value"
                    )
                item_id = normalize_item_id(row[0])
                value = ",".join(row[1:]).strip()
                if not value:
                    continue
                if item_id in values:
                    raise MicroLensImportError(
                        f"{label} duplicates item ID {item_id}"
                    )
                values[item_id] = value
    except OSError as exc:
        raise MicroLensImportError(f"failed to read {label} {path}: {exc}") from exc
    if not values:
        raise MicroLensImportError(f"{label} contains no rows: {path}")
    return values


def load_interaction_popularity(path: Path) -> Counter[str]:
    popularity: Counter[str] = Counter()
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t", 1)
                if len(parts) != 2:
                    raise MicroLensImportError(
                        f"pairs line {line_number} must contain user and item sequence"
                    )
                for raw_item_id in parts[1].split():
                    popularity[normalize_item_id(raw_item_id)] += 1
    except OSError as exc:
        raise MicroLensImportError(f"failed to read pairs {path}: {exc}") from exc
    if not popularity:
        raise MicroLensImportError(f"pairs contains no item interactions: {path}")
    return popularity


def video_files_by_item(videos_dir: Path) -> dict[str, Path]:
    if not videos_dir.is_dir():
        raise MicroLensImportError(f"videos directory not found: {videos_dir}")
    videos: dict[str, Path] = {}
    for path in sorted(videos_dir.glob("*.mp4")):
        item_id = normalize_item_id(path.stem)
        if item_id in videos:
            raise MicroLensImportError(f"duplicate video item ID {item_id}")
        videos[item_id] = path
    if not videos:
        raise MicroLensImportError(f"no MP4 videos found: {videos_dir}")
    return videos


def _ratio(value: object) -> float:
    text = str(value or "").strip()
    if not text or text in {"0:1", "N/A"}:
        return 1.0
    try:
        numerator, denominator = text.split(":", 1)
        result = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError) as exc:
        raise MicroLensImportError(f"invalid sample aspect ratio: {value!r}") from exc
    return result if result > 0 else 1.0


def _rotation(stream: dict[str, Any]) -> int:
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return int(round(float(side_data["rotation"]))) % 360
    tags = stream.get("tags") or {}
    try:
        return int(round(float(tags.get("rotate", 0)))) % 360
    except (TypeError, ValueError):
        return 0


def display_aspect_ratio_from_probe(probe: dict[str, Any]) -> tuple[int, int, int, float]:
    try:
        stream = next(
            item for item in probe["streams"] if item.get("codec_type") == "video"
        )
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        raise MicroLensImportError("probe contains no valid video stream") from exc
    if width <= 0 or height <= 0:
        raise MicroLensImportError("probe video dimensions must be positive")
    rotation = _rotation(stream)
    ratio = width * _ratio(stream.get("sample_aspect_ratio")) / height
    if rotation in {90, 270}:
        ratio = 1.0 / ratio
    return width, height, rotation, ratio


def probe_video(path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        probe = json.loads(completed.stdout)
        width, height, rotation, ratio = display_aspect_ratio_from_probe(probe)
        stream = next(
            item for item in probe["streams"] if item.get("codec_type") == "video"
        )
        duration = float(stream.get("duration") or probe.get("format", {}).get("duration"))
    except Exception as exc:
        if isinstance(exc, MicroLensImportError):
            raise
        raise MicroLensImportError(f"failed to probe video {path}: {exc}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise MicroLensImportError(f"video duration must be positive: {path}")
    return {
        "width": width,
        "height": height,
        "rotation": rotation,
        "display_aspect_ratio": ratio,
        "duration_seconds": duration,
        "file_size": path.stat().st_size,
        "source_mtime_ns": path.stat().st_mtime_ns,
    }


def build_inventory(
    config: MicroLensConfig,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = config.source
    titles = load_item_csv(
        config.resolve_project_path(project_root, source.titles_csv), "titles"
    )
    tags = load_item_csv(
        config.resolve_project_path(project_root, source.tags_csv), "tags"
    )
    popularity = load_interaction_popularity(
        config.resolve_project_path(project_root, source.pairs_tsv)
    )
    videos = video_files_by_item(
        config.resolve_project_path(project_root, source.videos_dir)
    )
    inventory: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_ids = sorted(
        set(titles) | set(tags) | set(popularity) | set(videos), key=int
    )
    for item_id in all_ids:
        reasons = []
        if item_id not in titles:
            reasons.append("missing_title")
        if item_id not in tags:
            reasons.append("missing_tag")
        if item_id not in popularity:
            reasons.append("not_interacted")
        if item_id not in videos:
            reasons.append("missing_video")
        probe: dict[str, Any] = {}
        if not reasons and item_id in videos:
            try:
                probe = probe_video(videos[item_id])
            except MicroLensImportError as exc:
                reasons.append("invalid_video")
                failures.append(
                    {"item_id": item_id, "reason": "invalid_video", "error": str(exc)}
                )
        record = {
            "item_id": item_id,
            "content_id": content_id_for_item(item_id),
            "source_video_path": str(videos[item_id].resolve()) if item_id in videos else None,
            "title": titles.get(item_id),
            "category": tags.get(item_id),
            "interaction_count": popularity.get(item_id, 0),
            **probe,
            "eligible": not reasons,
            "exclusion_reasons": reasons,
        }
        inventory.append(record)
        failures.extend(
            {"item_id": item_id, "reason": reason}
            for reason in reasons
            if reason != "invalid_video"
        )
    return inventory, failures


def _stable_tie(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def category_quotas(
    category_sizes: dict[str, int],
    total: int,
    minimum_per_category: int,
) -> dict[str, int]:
    if total <= 0 or total > sum(category_sizes.values()):
        raise MicroLensImportError("selection size exceeds eligible item count")
    if not category_sizes or any(size <= 0 for size in category_sizes.values()):
        raise MicroLensImportError("category sizes must be positive")
    quotas = {category: 0 for category in sorted(category_sizes)}
    remaining = total
    covered_categories = sorted(
        category_sizes,
        key=lambda category: (-category_sizes[category], category),
    )
    for category in covered_categories:
        allocation = min(minimum_per_category, category_sizes[category], remaining)
        quotas[category] = allocation
        remaining -= allocation
        if not remaining:
            return quotas
    capacities = {
        category: category_sizes[category] - quotas[category]
        for category in quotas
    }
    capacity_total = sum(capacities.values())
    exact = {
        category: remaining * capacities[category] / capacity_total
        for category in quotas
    }
    floors = {category: int(math.floor(value)) for category, value in exact.items()}
    for category, value in floors.items():
        quotas[category] += min(value, capacities[category])
    leftover = total - sum(quotas.values())
    order = sorted(
        quotas,
        key=lambda category: (-(exact[category] - floors[category]), category),
    )
    for category in order:
        if not leftover:
            break
        if quotas[category] < category_sizes[category]:
            quotas[category] += 1
            leftover -= 1
    if leftover:
        raise MicroLensImportError("failed to allocate exact category quotas")
    return quotas


def select_balanced(
    eligible: Iterable[dict[str, Any]],
    *,
    size: int,
    seed: int,
    minimum_per_category: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        grouped[str(record["category"])].append(record)
    quotas = category_quotas(
        {category: len(records) for category, records in grouped.items()},
        size,
        minimum_per_category,
    )
    selected: list[dict[str, Any]] = []
    for category in sorted(grouped):
        ordered = sorted(
            grouped[category],
            key=lambda item: (
                int(item["interaction_count"]),
                _stable_tie(seed, str(item["item_id"])),
            ),
        )
        quota = quotas[category]
        if quota == len(ordered):
            chosen = ordered
        else:
            indexes = [
                min(len(ordered) - 1, math.floor((offset + 0.5) * len(ordered) / quota))
                for offset in range(quota)
            ]
            chosen = [ordered[index] for index in indexes]
        selected.extend(chosen)
    return sorted(selected, key=lambda item: int(item["item_id"]))


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _selection_document(
    config: MicroLensConfig,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [record for record in inventory if record["eligible"]]
    pilot = select_balanced(
        eligible,
        size=config.pilot.size,
        seed=config.pilot.seed,
        minimum_per_category=config.pilot.minimum_per_category,
    )
    smoke = select_balanced(
        pilot,
        size=config.pilot.smoke_size,
        seed=config.pilot.seed,
        minimum_per_category=1,
    )
    selection_contract = {
        "dataset_id": config.dataset_id,
        "pilot": config.pilot.model_dump(mode="json"),
        "sampling": config.sampling.model_dump(mode="json"),
    }
    inventory_fingerprint = _fingerprint(inventory)
    return {
        "schema_version": "microlens-selection-manifest/v1",
        "config_fingerprint": _fingerprint(selection_contract),
        "inventory_fingerprint": inventory_fingerprint,
        "selection_id": _fingerprint(
            {
                "config": selection_contract,
                "inventory_fingerprint": inventory_fingerprint,
                "pilot_item_ids": [record["item_id"] for record in pilot],
                "smoke_item_ids": [record["item_id"] for record in smoke],
            }
        ),
        "eligible_count": len(eligible),
        "excluded_count": len(inventory) - len(eligible),
        "pilot_item_ids": [record["item_id"] for record in pilot],
        "smoke_item_ids": [record["item_id"] for record in smoke],
        "pilot_category_counts": dict(Counter(record["category"] for record in pilot)),
        "smoke_category_counts": dict(Counter(record["category"] for record in smoke)),
    }


def _same_frozen_cohort(
    current: dict[str, Any], expected: dict[str, Any]
) -> bool:
    if "selected_item_ids" in expected:
        return current.get("selected_item_ids") == expected.get("selected_item_ids")
    cohort_keys = (
        "inventory_fingerprint",
        "pilot_item_ids",
        "smoke_item_ids",
    )
    return all(current.get(key) == expected.get(key) for key in cohort_keys)


def load_external_selection(
    config: MicroLensConfig,
    *,
    scope: Literal["smoke", "pilot"],
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    configured = (
        config.source.smoke_selection_jsonl
        if scope == "smoke"
        else config.source.selection_jsonl
    )
    if not configured:
        raise MicroLensImportError(f"no external selection configured for {scope}")
    selection_path = config.resolve_project_path(project_root, configured)
    titles = load_item_csv(
        config.resolve_project_path(project_root, config.source.titles_csv), "titles"
    )
    tags = load_item_csv(
        config.resolve_project_path(project_root, config.source.tags_csv), "tags"
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = selection_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MicroLensImportError(
            f"failed to read external selection {selection_path}: {exc}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MicroLensImportError(
                f"external selection line {line_number} is invalid JSON"
            ) from exc
        required = {"item_id", "content_id", "source_video_path", "url"}
        if not isinstance(row, dict) or set(row) != required:
            raise MicroLensImportError(
                f"external selection line {line_number} must contain exactly "
                + ", ".join(sorted(required))
            )
        item_id = normalize_item_id(row["item_id"])
        if item_id in seen:
            raise MicroLensImportError(f"external selection duplicates item {item_id}")
        seen.add(item_id)
        expected_content_id = content_id_for_item(item_id)
        if row["content_id"] != expected_content_id or row["url"] != provenance_uri(item_id):
            raise MicroLensImportError(
                f"external selection identity mismatch for item {item_id}"
            )
        source_path = Path(str(row["source_video_path"]))
        reasons: list[str] = []
        if item_id not in titles:
            reasons.append("missing_title")
        if not source_path.is_file():
            reasons.append("missing_video")
        probe: dict[str, Any] = {}
        if not reasons:
            try:
                probe = probe_video(source_path)
            except MicroLensImportError as exc:
                reasons.append("invalid_video")
                failures.append(
                    {"item_id": item_id, "reason": "invalid_video", "error": str(exc)}
                )
        failures.extend(
            {"item_id": item_id, "reason": reason}
            for reason in reasons
            if reason != "invalid_video"
        )
        records.append(
            {
                "item_id": item_id,
                "content_id": expected_content_id,
                "source_video_path": str(source_path.resolve()),
                "title": titles.get(item_id),
                "category": tags.get(item_id, "unknown"),
                "interaction_count": 0,
                **probe,
                "eligible": not reasons,
                "exclusion_reasons": reasons,
            }
        )
    if not records:
        raise MicroLensImportError(f"external selection contains no rows: {selection_path}")
    document = {
        "schema_version": "microlens-user-derived-selection/v1",
        "scope": scope,
        "selection_path": str(selection_path.resolve()),
        "selection_fingerprint": _fingerprint(
            [{key: row[key] for key in ("item_id", "content_id", "source_video_path")} for row in records]
        ),
        "selected_item_ids": [row["item_id"] for row in records],
    }
    return records, failures, document


def run_import(
    config: MicroLensConfig,
    *,
    scope: Literal["smoke", "pilot"],
    force: bool = False,
    rebuild_selection: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    use_external = bool(config.source.selection_jsonl)
    if use_external:
        inventory, failures, expected_selection = load_external_selection(
            config, scope=scope, project_root=project_root
        )
    else:
        inventory, failures = build_inventory(config, project_root=project_root)
        expected_selection = _selection_document(config, inventory)
    inventory_path = config.resolve_output_path(
        project_root, config.outputs.inventory_jsonl
    )
    failure_path = config.resolve_output_path(
        project_root, config.outputs.failures_jsonl
    )
    selection_path = config.resolve_output_path(
        project_root, config.outputs.selection_json
    )
    if use_external:
        selection_path = selection_path.with_name(
            f"{selection_path.stem}_{scope}{selection_path.suffix}"
        )
    _write_jsonl(inventory_path, inventory)
    if selection_path.exists() and not rebuild_selection:
        current = json.loads(selection_path.read_text(encoding="utf-8"))
        if current != expected_selection and not _same_frozen_cohort(
            current, expected_selection
        ):
            raise MicroLensImportError(
                "frozen selection differs from current inputs; use --rebuild-selection"
            )
        selection = current
    else:
        selection = expected_selection
        _write_json(selection_path, selection)

    by_id = {record["item_id"]: record for record in inventory}
    selected_ids = (
        selection["selected_item_ids"]
        if use_external
        else selection[f"{scope}_item_ids"]
    )
    selected = [by_id[item_id] for item_id in selected_ids if by_id[item_id]["eligible"]]
    output_root = config.resolve_output_root(project_root)
    assets_root = config.resolve_project_path(
        output_root, config.assets_root
    )
    manifest_path = config.resolve_output_path(
        project_root,
        config.outputs.smoke_manifest_csv
        if scope == "smoke"
        else config.outputs.pilot_manifest_csv,
    )
    categories_path = config.resolve_output_path(
        project_root,
        config.outputs.smoke_categories_jsonl
        if scope == "smoke"
        else config.outputs.pilot_categories_jsonl,
    )
    rows = [
        {"content_id": record["content_id"], "url": provenance_uri(record["item_id"])}
        for record in selected
    ]
    write_manifest(rows, manifest_path)
    _write_jsonl(
        categories_path,
        (
            {
                "content_id": record["content_id"],
                "source_item_id": record["item_id"],
                "category": record["category"],
            }
            for record in selected
        ),
    )

    processing_failures: list[dict[str, Any]] = []
    succeeded = 0
    processing_config_path = config.resolve_project_path(
        project_root, config.processing_config_path
    )
    for record in selected:
        metadata = {
            "title": record["title"],
            "channel": config.metadata_defaults.channel,
            "upload_date": config.metadata_defaults.upload_date,
            "description": config.metadata_defaults.description,
            "duration": record["duration_seconds"],
            "dataset_id": config.dataset_id,
            "source_item_id": record["item_id"],
            "category": record["category"],
        }
        try:
            process_local_source(
                name=record["content_id"],
                source_video_path=record["source_video_path"],
                data_root=assets_root,
                output_root=output_root,
                metadata=metadata,
                config_path=processing_config_path,
                frames_per_window=config.sampling.frames_per_scene,
                force=force,
            )
            succeeded += 1
        except Exception as exc:
            processing_failures.append(
                {
                    "item_id": record["item_id"],
                    "content_id": record["content_id"],
                    "reason": "processing_failed",
                    "error": str(exc),
                }
            )
    _write_jsonl(failure_path, [*failures, *processing_failures])
    return {
        "scope": scope,
        "selected": len(selected),
        "succeeded": succeeded,
        "failed": len(processing_failures),
        "manifest": str(manifest_path),
        "categories": str(categories_path),
        "selection": str(selection_path),
    }
