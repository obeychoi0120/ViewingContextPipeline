from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .config import ValidationConfig
from .io import atomic_write_json, atomic_write_jsonl, file_fingerprint, fingerprint


class CohortError(RuntimeError):
    pass


def normalize_item_id(value: object) -> str:
    text = str(value).strip()
    if not text.isdigit() or int(text) <= 0:
        raise CohortError(f"invalid item id: {value!r}")
    return str(int(text))


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


def probe_duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    duration = float(json.loads(completed.stdout)["format"]["duration"])
    if duration <= 0:
        raise ValueError("duration must be positive")
    return duration


def build_item_inventory(referenced_items: set[str], videos_dir: Path, probe: Callable[[Path], float] = probe_duration) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    videos = {normalize_item_id(path.stem): path for path in sorted(videos_dir.glob("*.mp4"))}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item_id in sorted(referenced_items, key=int):
        path = videos.get(item_id)
        reasons: list[str] = []
        duration: float | None = None
        if path is None:
            reasons.append("missing_video")
            failures.append({"item_id": item_id, "reason": "missing_video"})
        else:
            try:
                duration = probe(path)
            except Exception as exc:
                reasons.append("invalid_video")
                failures.append({"item_id": item_id, "reason": "invalid_video", "error": str(exc)})
        rows.append({
            "item_id": item_id, "content_id": f"microlens_100k_{int(item_id):05d}",
            "source_video_path": str(path.resolve()) if path else None,
            "duration_seconds": duration, "source_file_size": path.stat().st_size if path else None,
            "source_mtime_ns": path.stat().st_mtime_ns if path else None,
            "eligible": not reasons, "exclusion_reasons": reasons,
        })
    return rows, failures


def history_stratum(length: int, boundaries: list[int]) -> str:
    for start, end in zip(boundaries, boundaries[1:]):
        if start <= length < end:
            return f"{start}-{end - 1}"
    return f"{boundaries[-1]}+"


def largest_remainder_quotas(sizes: dict[str, int], total: int) -> dict[str, int]:
    available = sum(sizes.values())
    if total > available:
        raise CohortError(f"requested {total} users but only {available} are eligible")
    exact = {key: total * size / available for key, size in sizes.items()}
    quotas = {key: min(sizes[key], int(value)) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(sizes, key=lambda key: (-(exact[key] - int(exact[key])), key))
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < sizes[key] and remaining:
                quotas[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise CohortError("unable to allocate user quotas")
    return quotas


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_users(pairs: list[tuple[str, list[str]]], eligible_items: set[str], *, count: int, seed: int, boundaries: list[int], min_length: int, max_length: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for user_id, original in pairs:
        filtered = [item for item in original if item in eligible_items]
        if len(filtered) < min_length:
            continue
        stratum = history_stratum(len(original), boundaries)
        grouped[stratum].append({"user_id": user_id, "original_length": len(original), "filtered_length": len(filtered), "sequence": filtered[-max_length:], "stratum": stratum})
    sizes = {key: len(rows) for key, rows in grouped.items()}
    quotas = largest_remainder_quotas(sizes, count)
    selected: list[dict[str, Any]] = []
    for stratum in sorted(grouped):
        rows = sorted(grouped[stratum], key=lambda row: _stable_key(seed, row["user_id"]))
        selected.extend(rows[:quotas[stratum]])
    return sorted(selected, key=lambda row: row["user_id"]), quotas


def split_record(row: dict[str, Any]) -> dict[str, Any]:
    sequence = row["sequence"]
    return {**row, "train": sequence[:-2], "valid_target": sequence[-2], "test_target": sequence[-1]}


def prepare_cohort(
    config: ValidationConfig,
    *,
    output_dir: Path | None = None,
    probe: Callable[[Path], float] = probe_duration,
) -> dict[str, Any]:
    output = output_dir or config.output_dir / "data" / "cohort"
    pairs = load_pairs(config.dataset.pairs_tsv)
    referenced_items = {item for _, sequence in pairs for item in sequence}
    inventory, failures = build_item_inventory(referenced_items, config.dataset.videos_dir, probe)
    eligible = {row["item_id"] for row in inventory if row["eligible"]}
    eligible_user_count = sum(
        1
        for _, sequence in pairs
        if sum(item in eligible for item in sequence) >= config.cohort.min_sequence_length
    )
    exclusion_counts = Counter(
        reason
        for row in inventory
        for reason in row["exclusion_reasons"]
    )
    eligibility = {
        "schema_version": "microlens-cohort-eligibility/v1",
        "requested_users": config.cohort.user_count,
        "pairs_users": len(pairs),
        "referenced_items": len(referenced_items),
        "eligible_items": len(eligible),
        "eligible_users": eligible_user_count,
        "min_sequence_length": config.cohort.min_sequence_length,
        "item_exclusions": dict(sorted(exclusion_counts.items())),
        "pairs_tsv": str(config.dataset.pairs_tsv),
        "videos_dir": str(config.dataset.videos_dir),
    }
    atomic_write_jsonl(output / "item_inventory.jsonl", inventory)
    atomic_write_jsonl(output / "failures.jsonl", failures)
    atomic_write_json(output / "eligibility_summary.json", eligibility)
    print(
        "[COHORT] "
        f"pairs_users={len(pairs)} referenced_items={len(referenced_items)} "
        f"eligible_items={len(eligible)} eligible_users={eligible_user_count} "
        f"requested_users={config.cohort.user_count}",
        flush=True,
    )
    if eligible_user_count < config.cohort.user_count:
        raise CohortError(
            f"requested {config.cohort.user_count} users but only {eligible_user_count} are eligible; "
            f"check pairs_tsv/videos_dir and {output / 'eligibility_summary.json'}"
        )
    selected, quotas = select_users(pairs, eligible, count=config.cohort.user_count, seed=config.cohort.seed, boundaries=config.cohort.history_strata, min_length=config.cohort.min_sequence_length, max_length=config.cohort.max_sequence_length)
    sequences = [split_record(row) for row in selected]
    catalog_ids = sorted({item for row in sequences for item in row["sequence"]}, key=int)
    by_item = {row["item_id"]: row for row in inventory}
    catalog = [{key: by_item[item][key] for key in ("item_id", "content_id", "source_video_path", "duration_seconds")} for item in catalog_ids]
    source_fingerprints = [file_fingerprint(config.dataset.pairs_tsv)]
    inventory_fp = fingerprint([{key: by_item[item][key] for key in ("item_id", "source_video_path", "duration_seconds", "source_file_size", "source_mtime_ns")} for item in catalog_ids])
    cohort_fp = fingerprint({"config": config.cohort.model_dump(), "sources": source_fingerprints, "inventory_fingerprint": inventory_fp, "sequences": sequences})
    manifest = {"schema_version": "microlens-user-cohort/v2", "run_id": config.run_id, "user_count": len(sequences), "catalog_size": len(catalog), "stratum_quotas": quotas, "source_fingerprints": source_fingerprints, "cohort_fingerprint": cohort_fp, "inventory_fingerprint": inventory_fp}
    atomic_write_jsonl(output / "sequences.jsonl", sequences)
    atomic_write_jsonl(output / "catalog.jsonl", catalog)
    atomic_write_json(output / "cohort_manifest.json", manifest)
    return manifest
