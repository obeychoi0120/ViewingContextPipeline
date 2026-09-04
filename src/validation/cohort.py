from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .cohort_selection import (
    CATALOG_SCOPE,
    COHORT_SAMPLING,
    ELIGIBILITY_SCHEMA_VERSION,
    METADATA_TITLE_SCHEMA_VERSION,
    CohortError,
    build_cohort_plan,
    content_id_for_item,
    normalize_item_id,
    validate_plan,
)
from .config import ValidationConfig
from .io import atomic_write_json, atomic_write_jsonl, read_jsonl


def load_pairs(path: Path) -> list[tuple[str, list[str]]]:
    users: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2 or not parts[0].strip():
                raise CohortError(f"invalid pairs row {line_number}")
            user_id = parts[0].strip()
            if user_id in seen:
                raise CohortError(f"duplicate user {user_id}")
            seen.add(user_id)
            users.append((user_id, [normalize_item_id(item) for item in parts[1].split()]))
    return users


def load_metadata_titles(path: Path) -> dict[str, str]:
    """Blank titles outside the required catalog do not block preparation."""
    titles: dict[str, str] = {}
    seen: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise CohortError(f"failed to read metadata titles {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            if "," not in raw:
                raise CohortError(f"invalid metadata title row {line_number}: missing comma")
            raw_item_id, raw_title = raw.split(",", 1)
            try:
                item_id = normalize_item_id(raw_item_id)
            except CohortError as exc:
                raise CohortError(f"invalid metadata title row {line_number}: {exc}") from exc
            if item_id in seen:
                raise CohortError(
                    f"duplicate metadata title for item {item_id} at row {line_number}"
                )
            seen.add(item_id)
            title = raw_title.strip()
            if title:
                titles[item_id] = title
    return titles


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def probe_duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    duration = float(json.loads(completed.stdout)["format"]["duration"])
    if not _positive_finite(duration):
        raise ValueError("duration must be positive and finite")
    return duration


def build_item_inventory(
    referenced_items: set[str],
    videos_dir: Path,
    probe: Callable[[Path], float] = probe_duration,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    videos: dict[str, list[Path]] = {}
    for path in sorted(videos_dir.glob("*.mp4")):
        try:
            item_id = normalize_item_id(path.stem)
        except CohortError:
            continue
        if item_id in referenced_items:
            videos.setdefault(item_id, []).append(path)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item_id in sorted(referenced_items, key=int):
        matches = videos.get(item_id, [])
        path = matches[0] if matches else None
        reasons: list[str] = []
        duration = size = mtime = None
        if path is None:
            reasons.append("missing_video")
            failures.append({"item_id": item_id, "reason": "missing_video"})
        else:
            try:
                if len(matches) != 1:
                    raise ValueError(f"multiple video files normalize to item {item_id}")
                stat = path.stat()
                size, mtime = stat.st_size, stat.st_mtime_ns
                if not path.is_file() or size <= 0:
                    raise ValueError("video must be a non-empty file")
                duration = probe(path)
                if not _positive_finite(duration):
                    raise ValueError("duration must be positive and finite")
            except Exception as exc:
                duration = None
                reasons.append("invalid_video")
                failures.append({"item_id": item_id, "reason": "invalid_video", "error": str(exc)})
        rows.append(
            {
                "item_id": item_id,
                "content_id": content_id_for_item(item_id),
                "source_video_path": str(path.resolve()) if path else None,
                "duration_seconds": duration,
                "source_file_size": size,
                "source_mtime_ns": mtime,
                "eligible": not reasons,
                "exclusion_reasons": reasons,
            }
        )
    return rows, failures


def _read(path: Path, *, jsonl: bool = False) -> Any:
    try:
        value = read_jsonl(path) if jsonl else json.loads(path.read_text(encoding="utf-8"))
        if jsonl:
            if not all(isinstance(row, dict) for row in value):
                raise ValueError("JSONL rows must be objects")
        elif not isinstance(value, dict):
            raise ValueError("JSON root must be an object")
        return value
    except (OSError, TypeError, ValueError) as exc:
        raise CohortError(f"missing or invalid cohort artifact {path}: {exc}") from exc


def _media_statistics(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in inventory if row["eligible"]]
    complete = len(available) == len(inventory)
    duration = math.fsum(row["duration_seconds"] for row in available)
    return {
        "available_duration_seconds": duration,
        "total_duration_seconds": duration if complete else None,
        "scene_count": (
            sum((math.ceil(row["duration_seconds"]) + 29) // 30 for row in available)
            if complete
            else None
        ),
        "keyframe_count": (
            sum((math.ceil(row["duration_seconds"]) + 9) // 10 for row in available)
            if complete
            else None
        ),
    }


def _eligibility(plan: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "status": status,
        "cohort_sampling": COHORT_SAMPLING,
        "catalog_scope": CATALOG_SCOPE,
        "inputs": plan["inputs"],
        "requested_users": plan["selected_user_count"],
        "pairs_user_count": plan["pairs_user_count"],
        "candidate_user_count": plan["candidate_user_count"],
        "selected_user_count": plan["selected_user_count"],
        "required_item_count": plan["required_item_count"],
        "available_item_count": None,
        "item_exclusions": {},
        "metadata_title_coverage": {
            "schema_version": METADATA_TITLE_SCHEMA_VERSION,
            "catalog_item_count": plan["required_item_count"],
            "covered_item_count": None,
            "missing_item_count": None,
        },
        "statistics": plan["statistics"],
        "media": {
            "available_duration_seconds": None,
            "total_duration_seconds": None,
            "scene_count": None,
            "keyframe_count": None,
        },
    }


def _freeze_plan(
    output: Path,
    plan: dict[str, Any],
    selected: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> None:
    plan_path = output / "cohort_plan.json"
    eligibility_path = output / "eligibility_summary.json"
    if eligibility_path.exists():
        eligibility = _read(eligibility_path)
        if eligibility.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION or eligibility.get(
            "status"
        ) not in {"planned", "blocked", "ready"}:
            raise CohortError("legacy/invalid cohort artifacts cannot be reused; use a new run_id")
    if not plan_path.exists() and any(
        (output / name).exists()
        for name in ("catalog.jsonl", "sequences.jsonl", "metadata_titles.jsonl")
    ):
        raise CohortError("cohort artifacts have no user-first plan; use a new run_id")
    documents = (
        (output / "selected_users.jsonl", selected, True),
        (output / "required_items.jsonl", items, True),
        (plan_path, plan, False),
    )
    # Check every existing file before any writes. The plan is committed last;
    # identical interrupted partial writes can therefore be resumed safely.
    for path, value, jsonl in documents:
        if path.exists() and _read(path, jsonl=jsonl) != value:
            raise CohortError(f"cohort selection/input changed at {path.name}; use a new run_id")
    for path, value, jsonl in documents:
        if not path.exists():
            (atomic_write_jsonl if jsonl else atomic_write_json)(path, value)
    if not eligibility_path.exists():
        atomic_write_json(eligibility_path, _eligibility(plan, "planned"))


def load_ready_cohort(
    output: Path,
    *,
    run_id: str,
    settings: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    """Shared artifact gate. No original videos, ffprobe, Torch or VLM imports."""
    eligibility = _read(output / "eligibility_summary.json")
    if eligibility.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION:
        raise CohortError("legacy or invalid cohort eligibility; use a new run_id")
    if eligibility.get("status") != "ready":
        raise CohortError(
            f"cohort is not ready (status={eligibility.get('status')!r}); "
            "complete prepare-cohort after supplying the required assets"
        )
    plan = _read(output / "cohort_plan.json")
    selected = _read(output / "selected_users.jsonl", jsonl=True)
    items = _read(output / "required_items.jsonl", jsonl=True)
    validate_plan(plan, selected, items, run_id=run_id, settings=settings, inputs=inputs)
    sequences = _read(output / "sequences.jsonl", jsonl=True)
    if sequences != sorted(selected, key=lambda row: row["user_id"]):
        raise CohortError("cohort sequences differ from the frozen selected users")
    catalog = _read(output / "catalog.jsonl", jsonl=True)
    inventory = _read(output / "item_inventory.jsonl", jsonl=True)
    titles = _read(output / "metadata_titles.jsonl", jsonl=True)
    for label, rows in (
        ("catalog", catalog),
        ("inventory", inventory),
        ("metadata titles", titles),
    ):
        mapping = [
            {"item_id": row.get("item_id"), "content_id": row.get("content_id")} for row in rows
        ]
        if mapping != items:
            raise CohortError(f"{label} does not match the required selected sequence union")
    for item, row, title in zip(catalog, inventory, titles, strict=True):
        if (
            set(item) != {"item_id", "content_id", "source_video_path", "duration_seconds"}
            or any(item[key] != row.get(key) for key in item)
            or row.get("eligible") is not True
            or row.get("exclusion_reasons") != []
            or not _positive_finite(row.get("duration_seconds"))
            or not isinstance(row.get("source_video_path"), str)
            or not row["source_video_path"].strip()
            or type(row.get("source_file_size")) is not int
            or row["source_file_size"] <= 0
            or type(row.get("source_mtime_ns")) is not int
        ):
            raise CohortError("invalid ready cohort video inventory/catalog")
        if (
            set(title) != {"item_id", "content_id", "title"}
            or not isinstance(title["title"], str)
            or not title["title"].strip()
        ):
            raise CohortError("invalid ready cohort metadata title")
    expected = _eligibility(plan, "ready")
    expected.update(
        available_item_count=len(items),
        metadata_title_coverage={
            "schema_version": METADATA_TITLE_SCHEMA_VERSION,
            "catalog_item_count": len(items),
            "covered_item_count": len(items),
            "missing_item_count": 0,
        },
        media=_media_statistics(inventory),
    )
    if eligibility != expected:
        raise CohortError("cohort eligibility does not match the actual plan/catalog/sequences")
    failure_path = output / "failures.jsonl"
    if failure_path.exists() and _read(failure_path, jsonl=True):
        raise CohortError("ready cohort still has unresolved asset failures")
    return {
        "plan": plan,
        "selected_users": selected,
        "required_items": items,
        "catalog": catalog,
        "sequences": sequences,
        "inventory": inventory,
        "metadata_titles": titles,
        "eligibility": eligibility,
    }


def prepare_cohort(
    config: ValidationConfig,
    *,
    output_dir: Path | None = None,
    plan_only: bool = False,
    force: bool = False,
    probe: Callable[[Path], float] | None = None,
) -> dict[str, Any]:
    output = output_dir or config.output_dir / "data" / "cohort"
    settings = config.cohort.model_dump(mode="json")
    inputs = {key: str(path.resolve()) for key, path in config.dataset.model_dump().items()}
    plan, selected, items = build_cohort_plan(
        load_pairs(config.dataset.pairs_tsv),
        run_id=config.run_id,
        settings=settings,
        inputs=inputs,
    )
    _freeze_plan(output, plan, selected, items)
    eligibility_path = output / "eligibility_summary.json"
    eligibility = _read(eligibility_path)
    if plan_only:
        if eligibility.get("status") == "ready":
            load_ready_cohort(output, run_id=config.run_id, settings=settings, inputs=inputs)
        print(
            f"[COHORT PLAN] status={eligibility['status']} "
            f"candidate_users={plan['candidate_user_count']} selected_users={len(selected)} "
            f"required_videos={len(items)}",
            flush=True,
        )
        for size, statistics in plan["scale_statistics"].items():
            print(
                f"[COHORT SCALE] users={size} {json.dumps(statistics, sort_keys=True)}", flush=True
            )
    else:
        # Preparation revalidates the assets even on resume. --force never redraws
        # users or permits a changed plan, and does not invalidate downstream files.
        eligibility = _eligibility(plan, "blocked")
        atomic_write_json(eligibility_path, eligibility)
        inventory, failures = build_item_inventory(
            {row["item_id"] for row in items}, config.dataset.videos_dir, probe or probe_duration
        )
        titles: dict[str, str] = {}
        try:
            titles = load_metadata_titles(config.dataset.titles_csv)
        except (CohortError, UnicodeError) as exc:
            failures.append({"reason": "invalid_metadata_titles", "error": str(exc)})
        missing = [row["item_id"] for row in items if row["item_id"] not in titles]
        failures.extend({"item_id": item, "reason": "missing_metadata_title"} for item in missing)
        eligibility.update(
            available_item_count=sum(row["eligible"] for row in inventory),
            item_exclusions=dict(
                sorted(
                    Counter(
                        reason for row in inventory for reason in row["exclusion_reasons"]
                    ).items()
                )
            ),
            metadata_title_coverage={
                "schema_version": METADATA_TITLE_SCHEMA_VERSION,
                "catalog_item_count": len(items),
                "covered_item_count": len(items) - len(missing),
                "missing_item_count": len(missing),
            },
            media=_media_statistics(inventory),
        )
        atomic_write_jsonl(output / "item_inventory.jsonl", inventory)
        failure_path = output / "failures.jsonl"
        if failures:
            atomic_write_jsonl(failure_path, failures)
        else:
            failure_path.unlink(missing_ok=True)
        atomic_write_json(eligibility_path, eligibility)
        if failures:
            raise CohortError(
                f"required catalog has {len(missing)} items without metadata titles and "
                f"{len(items) - eligibility['available_item_count']} unavailable videos; "
                f"selection is unchanged; check {eligibility_path}"
            )
        catalog = [
            {
                key: row[key]
                for key in ("item_id", "content_id", "source_video_path", "duration_seconds")
            }
            for row in inventory
        ]
        metadata_titles = [{**item, "title": titles[item["item_id"]]} for item in items]
        atomic_write_jsonl(output / "catalog.jsonl", catalog)
        atomic_write_jsonl(
            output / "sequences.jsonl", sorted(selected, key=lambda row: row["user_id"])
        )
        atomic_write_jsonl(output / "metadata_titles.jsonl", metadata_titles)
        eligibility["status"] = "ready"
        atomic_write_json(eligibility_path, eligibility)
        print(
            f"[COHORT] status=ready selected_users={len(selected)} catalog_size={len(items)} "
            f"scenes={eligibility['media']['scene_count']}",
            flush=True,
        )
    return {
        "run_id": config.run_id,
        "status": eligibility["status"],
        "user_count": len(selected),
        "catalog_size": len(items),
        "catalog_scope": CATALOG_SCOPE,
    }
