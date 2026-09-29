"""Run-local validation cohort shared by every configured arm."""

from copy import deepcopy
from arm_registry import active_arms, select_arms
from artifact_io import atomic_write_json, atomic_write_jsonl
from artifact_io import fingerprint
from artifact_io import read_json, read_jsonl
from validation.rolling_data import EventTable

POLICY = "full-catalog-zero-vector/v2"


def cohort_directory(context):
    return context.run_root / "validation" / "cohort"


def validation_arms(context, target=None):
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


def data_identity(cohort):
    return {
        "policy": POLICY,
        "catalog": [{k: row[k] for k in ("item_id", "content_id")} for row in cohort["catalog"]],
        "events": cohort["events"],
        "splits": cohort["plan"]["splits"],
    }


def build_selection(context):
    source = context.require_ready_cohort()
    events = read_jsonl(context.cohort_dir / "events.jsonl")
    table = EventTable(events)
    if [str(r["item_id"]) for r in source["catalog"]] != table.items:
        raise ValueError("catalog must match the ordered event item universe")
    plan = {
        **source["plan"],
        "schema_version": POLICY,
        "splits": recount_splits(table, source["plan"]["splits"]),
    }
    plan["eligible_test_count"] = sum(
        (s["phases"]["test"]["eligible_count"] for s in plan["splits"])
    )
    payload = {
        "catalog": source["catalog"],
        "metadata_titles": source["metadata_titles"],
        "events": events,
        "plan": plan,
        "excluded": [],
    }
    count = len(source["catalog"])
    from arm_registry import arm_contract

    manifest = {
        "arm_contract": arm_contract(context.config),
        "arms": list(active_arms(context.config)),
        "policy": POLICY,
        "data_hash": fingerprint(payload),
        "selection_hash": fingerprint(data_identity(payload)),
        "included_item_ids": [r["item_id"] for r in source["catalog"]],
        "statistics": {
            "original_item_count": count,
            "included_item_count": count,
            "excluded_item_count": 0,
            "removed_event_count": 0,
            "lost_history_test_event_count": 0,
            "excluded_by_arm": {},
        },
    }
    return {**payload, "manifest": manifest}


def prepare_validation_cohort(context):
    cohort = build_selection(context)
    directory = cohort_directory(context)
    directory.mkdir(parents=True, exist_ok=True)
    # Publish the manifest last so partially written data cannot pass validation.
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
        if manifest["policy"] != POLICY or fingerprint(cohort) != manifest["data_hash"]:
            raise ValueError("invalid selection manifest")
        expected = fingerprint(data_identity(cohort))
        if expected != manifest["selection_hash"]:
            raise ValueError("invalid selection manifest")
        if verify_current:
            current = build_selection(context)
            if current["manifest"] != manifest:
                raise ValueError("selection inputs changed")
        return {**cohort, "manifest": manifest}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"missing, invalid, or stale validation cohort; rerun embed-representations: {exc}"
        ) from exc


def training_signature(context, cohort, config):
    from validation.recommendation_contracts import TRAINING_IMPLEMENTATION_VERSION

    return fingerprint(
        {
            "implementation": TRAINING_IMPLEMENTATION_VERSION,
            "events": cohort["events"],
            "selection_hash": cohort["manifest"]["selection_hash"],
            "model": {
                k: v for k, v in context.config["validation"]["model"].items() if k != "seeds"
            },
            "cutoffs": config.evaluation.cutoffs,
        }
    )
