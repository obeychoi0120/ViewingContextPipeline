"""Lossless source events and strictly causal UTC rolling partitions."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from pipeline_runtime import read_json, read_jsonl, write_json, write_jsonl
from validation.cohort import build_item_inventory, load_metadata_titles, load_pairs
from validation.cohort_selection import content_id_for_item, normalize_item_id
from validation.metadata import missing_metadata_report
from visual_sampling import build_fixed_windows

DAY = 86_400_000
SCHEMA = "microlens-full-rolling/v1"


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class EventTable:
    def __init__(self, rows):
        self.rows = list(rows)
        self.users = sorted({r["user_id"] for r in self.rows})
        self.items = sorted({r["item_id"] for r in self.rows}, key=int)
        self.item_index = {item: i for i, item in enumerate(self.items)}
        self.by_user = defaultdict(list)
        for i, row in enumerate(self.rows):
            if row["event_id"] != i:
                raise ValueError("event IDs must preserve source row order")
            self.by_user[row["user_id"]].append(i)
        self.timestamps = np.asarray([r["timestamp"] for r in self.rows], dtype=np.int64)
        self.targets = np.asarray([self.item_index[r["item_id"]] + 1 for r in self.rows])
        self.history_ends = np.zeros(len(self.rows), dtype=np.int64)
        for ids in self.by_user.values():
            ids.sort(key=lambda i: (self.timestamps[i], i))
            times = self.timestamps[ids]
            self.history_ends[ids] = np.searchsorted(times, times, side="left")

    def history(self, event, limit=None):
        ids = self.by_user[self.rows[event]["user_id"]]
        end = int(self.history_ends[event])
        start = max(0, end - limit) if limit else 0
        return self.targets[ids[start:end]].tolist()

    def select(self, start=None, end=None, *, eligible=True):
        mask = np.ones(len(self.rows), dtype=bool)
        if start is not None:
            mask &= self.timestamps >= start
        if end is not None:
            mask &= self.timestamps < end
        if eligible:
            mask &= self.history_ends > 0
        ids = np.flatnonzero(mask)
        return ids[np.lexsort((ids, self.timestamps[ids]))]

    def splits(self, days=7):
        final_day = int(self.timestamps.max()) // DAY * DAY
        result = []
        for start in range(final_day - days * DAY, final_day, DAY):
            ranges = {
                "selection": (None, start - DAY),
                "validation": (start - DAY, start),
                "refit": (None, start),
                "test": (start, start + DAY),
            }
            phases = {}
            for name, (lower, upper) in ranges.items():
                raw = len(self.select(lower, upper, eligible=False))
                count = len(self.select(lower, upper))
                phases[name] = {
                    "start_ms": lower,
                    "end_ms": upper,
                    "raw_count": raw,
                    "eligible_count": count,
                    "no_history_count": raw - count,
                }
            result.append(
                {
                    "evaluation_date": datetime.fromtimestamp(start / 1000, timezone.utc)
                    .date()
                    .isoformat(),
                    "phases": phases,
                }
            )
        return result


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
        pairs = load_pairs(tsv)
        expected = {
            user: Counter(table.rows[i]["item_id"] for i in ids)
            for user, ids in table.by_user.items()
        }
        if {u: Counter(v) for u, v in pairs} != expected:
            raise ValueError("CSV/TSV user interaction multisets differ")
    directory = context.cohort_dir
    directory.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": SCHEMA,
        "metadata_missing_policy": settings["metadata_missing_policy"],
        "run_id": context.run_id,
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
        return result
    write_json(directory / "eligibility.json", {"schema_version": SCHEMA, "status": "blocked"})
    inventory, failures = build_item_inventory(set(table.items), context.path("data", "videos_dir"))
    titles_path = context.path("data", "titles_csv")
    titles = load_metadata_titles(titles_path, keep_blank=True) if titles_path.is_file() else {}
    failures += [{"item_id": i, "reason": "missing_title"} for i in table.items if i not in titles]
    write_jsonl(directory / "preparation_failures.jsonl", failures)
    write_jsonl(directory / "item_inventory.jsonl", inventory)
    if failures:
        raise RuntimeError(
            f"{len(failures)} unresolved assets; see {directory / 'preparation_failures.jsonl'}"
        )
    sampling = context.config["extraction"]["visual_evidence"]
    scene_count = frame_count = 0
    for row in inventory:
        windows = build_fixed_windows(
            row["duration_seconds"],
            scene_duration=sampling["scene_duration"],
            num_keyframes=sampling["num_keyframes"],
        )
        scene_count += len(windows)
        frame_count += sum(len(w["keyframe_timestamps"]) for w in windows)
    write_json(
        directory / "media_preflight.json",
        {
            "video_count": len(inventory),
            "duration_seconds": sum(r["duration_seconds"] for r in inventory),
            "source_bytes": sum(r["source_file_size"] for r in inventory),
            "scene_count": scene_count,
            "keyframe_count": frame_count,
            "uncompressed_rgb_bytes": frame_count * int(np.prod(sampling["image_resolution"])) * 3,
            "storage_note": "RGB payload estimate; PNG size and extraction outputs are additional/variable",
            "sampling": sampling,
        },
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
    write_json(directory / "metadata_missing.json", missing_metadata)
    print(
        f"[Metadata] zero-vector items={missing_metadata['missing_count']}: "
        + ",".join(r["item_id"] for r in missing_metadata["items"]), flush=True,
    )
    write_json(
        directory / "eligibility.json",
        {
            "schema_version": SCHEMA,
            "status": "ready",
            "run_id": context.run_id,
        },
    )
    return {**result, "status": "ready"}


def load_cohort(directory, run_id):
    eligibility = read_json(directory / "eligibility.json")
    if (
        eligibility.get("schema_version"),
        eligibility.get("status"),
        eligibility.get("run_id"),
    ) != (SCHEMA, "ready", run_id):
        raise RuntimeError("full rolling cohort is not ready")
    cohort = {
        "eligibility": eligibility,
        "plan": read_json(directory / "cohort_plan.json"),
        "catalog": read_jsonl(directory / "catalog.jsonl"),
        "inventory": read_jsonl(directory / "item_inventory.jsonl"),
        "required_items": read_jsonl(directory / "required_items.jsonl"),
        "metadata_titles": read_jsonl(directory / "metadata_titles.jsonl"),
    }
    from validation.steps import _metadata_titles_match_catalog

    plan, catalog = cohort["plan"], cohort["catalog"]
    expected = [{"item_id": r["item_id"], "content_id": r["content_id"]} for r in catalog]
    if (
        len(catalog) != plan["item_count"]
        or len({r["item_id"] for r in catalog}) != len(catalog)
        or cohort["required_items"] != expected
        or not _metadata_titles_match_catalog(
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
