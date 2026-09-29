"""Shared source cohort preparation and integrity checks."""

from collections import Counter
from pathlib import Path
import csv
import json
import numpy as np
from artifact_io import read_json, read_jsonl, write_json, write_jsonl
from preparation.catalog import (
    _positive_finite,
    build_item_inventory,
    load_metadata_titles,
    load_pairs,
    content_id_for_item,
    normalize_item_id,
)
from preparation.titles import missing_metadata_report
from validation.rolling_data import EventTable

SCHEMA = "microlens-full-rolling/v1"


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_csv(path):
    rows = []
    duplicates = 0
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        if next(reader, None) != ["user", "item", "timestamp"]:
            raise ValueError("pairs CSV header must be user,item,timestamp")
        for event_id, values in enumerate(reader):
            if len(values) != 3:
                raise ValueError(f"malformed CSV event {event_id}")
            raw = dict(zip(("user", "item", "timestamp"), values, strict=True))
            user = normalize_item_id(raw["user"])
            item = normalize_item_id(raw["item"])
            timestamp = raw["timestamp"]
            if not timestamp or not timestamp.isdecimal():
                raise ValueError(f"timestamp must be integer milliseconds: event {event_id}")
            key = (user, item, int(timestamp))
            duplicates += key in seen
            seen.add(key)
            rows.append(
                dict(event_id=event_id, user_id=user, item_id=item, timestamp=int(timestamp))
            )
    if not rows:
        raise ValueError("empty pairs CSV")
    return EventTable(rows), duplicates


def prepare_full_cohort(context, *, plan_only=False):
    settings = context.config["validation"]["cohort"]
    source = context.path("data", "pairs_csv")
    print("[COHORT] Loading interaction CSV and checking source counts...", flush=True)
    table, duplicates = load_csv(source)
    observed = {
        "user_count": len(table.users),
        "interaction_count": len(table.rows),
        "item_count": len(table.items),
    }
    if any(observed[key] != settings[key] for key in observed):
        raise ValueError(f"full source cardinality mismatch: {observed}")
    tsv = context.path("data", "pairs_tsv")
    if tsv.is_file():
        print("[COHORT] Checking CSV/TSV consistency...", flush=True)
        pairs = load_pairs(tsv)
        expected = {
            user: Counter(table.rows[i]["item_id"] for i in ids)
            for user, ids in table.by_user.items()
        }
        if {u: Counter(v) for u, v in pairs} != expected:
            raise ValueError("CSV/TSV user interaction multisets differ")
    directory = context.cohort_dir
    directory.mkdir(parents=True, exist_ok=True)
    print("[COHORT] Building rolling splits and saving the cohort plan...", flush=True)
    plan = {
        "schema_version": SCHEMA,
        "metadata_missing_policy": settings["metadata_missing_policy"],
        "experiment_config_version": context.config["experiment_config_version"],
        **observed,
        "duplicate_rows_preserved": duplicates,
        "no_history_count": int(np.count_nonzero(table.history_ends == 0)),
        "splits": table.splits(settings["evaluation_days"]),
    }
    start = plan["splits"][0]["phases"]["test"]["start_ms"]
    end = plan["splits"][-1]["phases"]["test"]["end_ms"]
    plan["outside_evaluation"] = {
        "before_window": len(table.select(end=start, eligible=False)),
        "final_observed_day": len(table.select(start=end, eligible=False)),
    }
    plan["eligible_test_count"] = sum(s["phases"]["test"]["eligible_count"] for s in plan["splits"])
    if any(p["eligible_count"] == 0 for s in plan["splits"] for p in s["phases"].values()):
        raise ValueError("rolling partition has no eligible events")
    # Invalidate the shared ready marker before replacing any cohort files.
    write_json(directory / "eligibility.json", {"schema_version": SCHEMA, "status": "blocked"})
    write_jsonl(directory / "events.jsonl", table.rows)
    required = [{"item_id": item, "content_id": content_id_for_item(item)} for item in table.items]
    write_jsonl(directory / "required_items.jsonl", required)
    write_json(directory / "cohort_plan.json", plan)
    result = {
        "stage": "prepare-cohort",
        "user_count": len(table.users),
        "catalog_size": len(table.items),
        "content_count": len(table.items),
        "catalog_scope": "full_source_catalog",
        "status": "planned",
    }
    print(
        f"[Full cohort] users={len(table.users)} events={len(table.rows)} items={len(table.items)} "
        f"dates={plan['splits'][0]['evaluation_date']}..{plan['splits'][-1]['evaluation_date']} "
        f"eligible_test_events={plan['eligible_test_count']}",
        flush=True,
    )
    if plan_only:
        print(f"[COHORT] Plan saved: {directory} (plan-only complete)", flush=True)
        return result
    print(
        f"[COHORT] Checking {len(table.items)} video files (existence, size, duplicates)...",
        flush=True,
    )
    inventory, failures = build_item_inventory(
        set(table.items),
        context.path("data", "videos_dir"),
        probe=None,
    )
    # Preserve durations from older runs without probing when the source is unchanged.
    try:
        previous_rows = read_jsonl(directory / "item_inventory.jsonl")
        previous = {row["item_id"]: row for row in previous_rows}
        if len(previous) != len(previous_rows):
            previous = {}
    except (OSError, KeyError, TypeError, ValueError):
        previous = {}
    for row in inventory:
        old = previous.get(row["item_id"], {})
        if (
            row["eligible"]
            and old.get("eligible") is True
            and _positive_finite(old.get("duration_seconds"))
            and all(row[key] == old.get(key) for key in row if key != "duration_seconds")
        ):
            row["duration_seconds"] = old["duration_seconds"]
    print("[COHORT] Checking metadata titles...", flush=True)
    titles_path = context.path("data", "titles_csv")
    if "titles_supplement_csv" in context.config["data"]:
        from preparation.titles import resolve_required_titles

        titles, _, completion = resolve_required_titles(
            primary_path=titles_path,
            supplement_path=context.path("data", "titles_supplement_csv"),
            required_items_path=directory / "required_items.jsonl",
            unresolved_policy="zero-vector",
        )
        plan["title_completion"] = completion
        write_json(directory / "cohort_plan.json", plan)
        print(
            f"[METADATA TITLES] supplemented={completion['supplemented_required_item_count']} "
            f"unresolved={completion['unresolved_required_item_count']} policy=zero-vector",
            flush=True,
        )
    else:
        titles = load_metadata_titles(titles_path, keep_blank=True) if titles_path.is_file() else {}
    failures += [{"item_id": i, "reason": "missing_title"} for i in table.items if i not in titles]
    if failures:
        write_jsonl(directory / "preparation_failures.jsonl", failures)
    else:
        (directory / "preparation_failures.jsonl").unlink(missing_ok=True)
    write_jsonl(directory / "item_inventory.jsonl", inventory)
    if failures:
        raise RuntimeError(
            f"{len(failures)} unresolved assets; see {directory / 'preparation_failures.jsonl'}"
        )
    print(
        "[COHORT] Saving catalog; duration probing is deferred to prepare-input-data...", flush=True
    )
    catalog = [
        {
            key: row[key]
            for key in ("item_id", "content_id", "source_video_path", "duration_seconds")
        }
        for row in inventory
    ]
    write_jsonl(directory / "catalog.jsonl", catalog)
    metadata_titles = [{**r, "title": titles[r["item_id"]]} for r in required]
    write_jsonl(directory / "metadata_titles.jsonl", metadata_titles)
    missing_metadata = missing_metadata_report(metadata_titles)
    print(
        f"[Metadata] zero-vector items={missing_metadata['missing_count']}: "
        + ",".join(r["item_id"] for r in missing_metadata["items"]),
        flush=True,
    )
    write_json(
        directory / "eligibility.json",
        {
            "schema_version": SCHEMA,
            "status": "ready",
        },
    )
    print(
        f"[COHORT] Ready: videos={len(catalog)}; output={directory}",
        flush=True,
    )
    return {**result, "status": "ready"}


def load_cohort(directory):
    eligibility = read_json(directory / "eligibility.json")
    if (
        eligibility.get("schema_version"),
        eligibility.get("status"),
    ) != (SCHEMA, "ready"):
        raise RuntimeError("full rolling cohort is not ready")
    cohort = {
        "eligibility": eligibility,
        "plan": read_json(directory / "cohort_plan.json"),
        "catalog": read_jsonl(directory / "catalog.jsonl"),
        "inventory": read_jsonl(directory / "item_inventory.jsonl"),
        "required_items": read_jsonl(directory / "required_items.jsonl"),
        "metadata_titles": read_jsonl(directory / "metadata_titles.jsonl"),
    }
    from preparation.titles import metadata_titles_match_catalog

    plan, catalog = cohort["plan"], cohort["catalog"]
    expected = [{"item_id": r["item_id"], "content_id": r["content_id"]} for r in catalog]
    if (
        len(catalog) != plan["item_count"]
        or len({r["item_id"] for r in catalog}) != len(catalog)
        or cohort["required_items"] != expected
        or not metadata_titles_match_catalog(
            directory / "metadata_titles.jsonl", catalog, allow_blank=True
        )
    ):
        raise RuntimeError("invalid full cohort catalog or metadata mapping")
    users, items = set(), set()
    count = 0
    for index, row in enumerate(iter_jsonl(directory / "events.jsonl")):
        if (
            type(row.get("event_id")) is not int
            or row["event_id"] != index
            or type(row.get("timestamp")) is not int
            or row["timestamp"] < 0
        ):
            raise ValueError("invalid source event ID or timestamp")
        users.add(row["user_id"])
        items.add(row["item_id"])
        count += 1
    if (
        count != plan["interaction_count"]
        or len(users) != plan["user_count"]
        or items != {r["item_id"] for r in catalog}
    ):
        raise RuntimeError("full cohort cardinality mismatch")
    return cohort
