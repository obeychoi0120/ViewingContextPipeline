"""Run-local validation cohort shared by every configured arm."""

from collections import Counter
from copy import deepcopy

from arm_registry import active_arms, select_arms
from artifact_io import atomic_write_json, atomic_write_jsonl
from extraction.recovery import fingerprint
from extraction.summary_executor import summary_failure_rows
from pipeline_runtime import read_json, read_jsonl
from validation.rolling_data import EventTable

POLICY = "complete-summary-intersection/v1"


def cohort_directory(context):
    return context.run_root / "validation" / "cohort"


def validation_arms(context, target=None):
    configured = active_arms(context.config)
    if target is not None and set(target) - set(configured):
        raise ValueError("--target must be a subset of protocol.arms")
    return select_arms(context.config, target)


def recount_splits(table, splits):
    result = deepcopy(splits)
    for split in result:
        for phase in split["phases"].values():
            bounds = dict(start=phase["start_ms"], end=phase["end_ms"])
            raw = len(table.select(**bounds, eligible=False))
            count = len(table.select(**bounds))
            phase.update(raw_count=raw, eligible_count=count, no_history_count=raw - count)
    return result


def build_selection(context, summary_source):
    from validation.representation_inputs import documents_for_arm

    if summary_source not in {"qwen", "gemini"}:
        raise ValueError("summary_source must be qwen or gemini")
    source = context.require_ready_cohort()
    events = read_jsonl(context.cohort_dir / "events.jsonl")
    arms = validation_arms(context)
    failures = {
        name: summary_failure_rows(
            context.summary_dir(arm.representation, arm.model, summary_source)
        )
        for name, arm in arms.items()
        if name != "metadata"
    }
    excluded, included, evidence = [], [], []
    counts = {name: Counter() for name in failures}
    for item in source["catalog"]:
        cid = str(item["content_id"])
        reasons = []
        for name, arm in arms.items():
            if name == "metadata":
                continue
            path = (
                context.summary_dir(arm.representation, arm.model, summary_source) / f"{cid}.json"
            )
            reason = None
            if cid in failures[name]:
                reason = "failed"
            elif not path.is_file():
                reason = "missing"
            else:
                try:
                    doc = read_json(path)
                    if isinstance(doc, dict) and doc.get("status") == "raw_fallback":
                        reason = "raw_fallback"
                    else:
                        docs = documents_for_arm(
                            context,
                            {"catalog": [item]},
                            arm,
                            summary_source=summary_source,
                            strict=True,
                            failure_rows=failures[name],
                        )
                        evidence.append([name, cid, docs[0]["document_hash"]])
                except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError):
                    reason = "invalid"
            if reason:
                reasons.append({"arm": name, "reason": reason})
                counts[name][reason] += 1
        if reasons:
            excluded.append({**item, "reasons": reasons})
        else:
            included.append(item)
    if not included:
        raise ValueError("summary intersection is empty; no validation items")
    item_ids = {r["item_id"] for r in included}
    retained = [
        {**row, "event_id": i, "source_event_id": row["event_id"]}
        for i, row in enumerate(r for r in events if r["item_id"] in item_ids)
    ]
    table = EventTable(retained)
    splits = recount_splits(table, source["plan"]["splits"])
    if any(p["eligible_count"] == 0 for s in splits for p in s["phases"].values()):
        raise ValueError("summary intersection leaves a rolling partition with no eligible events")
    original_table = EventTable(events)
    lost_history = 0
    for split in splits:
        phase = split["phases"]["test"]
        for event in table.select(phase["start_ms"], phase["end_ms"], eligible=False):
            if (
                table.history_ends[event] == 0
                and original_table.history_ends[retained[event]["source_event_id"]] > 0
            ):
                lost_history += 1
    plan = {
        **source["plan"],
        "schema_version": POLICY,
        "user_count": len(table.users),
        "interaction_count": len(retained),
        "item_count": len(included),
        "splits": splits,
        "no_history_count": int((table.history_ends == 0).sum()),
        "duplicate_rows_preserved": len(retained)
        - len({(r["user_id"], r["item_id"], r["timestamp"]) for r in retained}),
        "eligible_test_count": sum(s["phases"]["test"]["eligible_count"] for s in splits),
    }
    start = splits[0]["phases"]["test"]["start_ms"]
    end = splits[-1]["phases"]["test"]["end_ms"]
    plan["outside_evaluation"] = {
        "before_window": len(table.select(end=start, eligible=False)),
        "final_observed_day": len(table.select(start=end, eligible=False)),
    }
    payload = {
        "catalog": included,
        "metadata_titles": [r for r in source["metadata_titles"] if r["item_id"] in item_ids],
        "events": retained,
        "plan": plan,
        "excluded": excluded,
    }
    manifest = {
        "policy": POLICY,
        "arms": list(arms),
        "summary_source": summary_source,
        "included_item_ids": [r["item_id"] for r in included],
        "source_hash": fingerprint({"cohort": source, "events": events}),
        "summary_inputs_hash": fingerprint(evidence),
        "data_hash": fingerprint(payload),
        "statistics": {
            "original_item_count": len(source["catalog"]),
            "included_item_count": len(included),
            "excluded_item_count": len(excluded),
            "removed_event_count": len(events) - len(retained),
            "lost_history_test_event_count": lost_history,
            "excluded_by_arm": {name: dict(values) for name, values in counts.items()},
        },
        "excluded_details_path": "validation/cohort/excluded.jsonl",
    }
    manifest["selection_hash"] = fingerprint(manifest)
    return {**payload, "manifest": manifest}


def prepare_validation_cohort(context, summary_source):
    cohort = build_selection(context, summary_source)
    directory = cohort_directory(context)
    directory.mkdir(parents=True, exist_ok=True)
    # Publish the manifest last; a partial write never validates against an old manifest.
    for key in ("catalog", "metadata_titles", "events", "excluded"):
        atomic_write_jsonl(directory / f"{key}.jsonl", cohort[key], durable=True)
    atomic_write_json(directory / "plan.json", cohort["plan"], durable=True)
    atomic_write_json(directory / "manifest.json", cohort["manifest"], durable=True)
    stats = cohort["manifest"]["statistics"]
    print(
        f"[Validation cohort] included={stats['included_item_count']} excluded={stats['excluded_item_count']}",
        flush=True,
    )
    return cohort


def load_validation_cohort(context, *, verify_current=True):
    directory = cohort_directory(context)
    try:
        manifest = read_json(directory / "manifest.json")
        cohort = {
            key: read_jsonl(directory / f"{key}.jsonl")
            for key in ("catalog", "metadata_titles", "events", "excluded")
        }
        cohort["plan"] = read_json(directory / "plan.json")
        if (
            manifest["policy"] != POLICY
            or fingerprint(cohort) != manifest["data_hash"]
            or fingerprint({k: v for k, v in manifest.items() if k != "selection_hash"})
            != manifest["selection_hash"]
        ):
            raise ValueError("invalid selection manifest")
        if (
            verify_current
            and build_selection(context, manifest["summary_source"])["manifest"] != manifest
        ):
            raise ValueError("selection inputs changed")
        return {**cohort, "manifest": manifest}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"missing, invalid, or stale validation cohort; rerun embed-representations: {exc}"
        ) from exc


def training_signature(context, cohort, config):
    return fingerprint(
        {
            "events": cohort["events"],
            "selection_hash": cohort["manifest"]["selection_hash"],
            "model": context.config["validation"]["model"],
            "cutoffs": config.evaluation.cutoffs,
        }
    )
