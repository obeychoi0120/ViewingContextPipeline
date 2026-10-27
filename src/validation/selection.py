"""Run-local validation cohort shared by every configured arm."""

from collections import Counter, defaultdict
import hashlib
import heapq
import json

from arm_registry import active_arms, select_arms
from artifact_io import atomic_write_json, atomic_write_jsonl
from artifact_io import fingerprint
from artifact_io import read_json, read_jsonl
from validation.leave_one_out_data import EventTable, PROTOCOL, SCHEMA
from preparation.cohort import iter_jsonl

POLICY = "leave-one-out-sequence-catalog/v1"


def cohort_directory(context):
    return context.validation_dir / "cohort"


def validation_arms(context, target=None):
    return select_arms(context.config, target)


def data_identity(cohort):
    return {
        "policy": POLICY,
        "preprocessing": cohort["plan"]["preprocessing"],
        "catalog": [{k: row[k] for k in ("item_id", "content_id")} for row in cohort["catalog"]],
        "events": cohort["events"],
        "splits": cohort["plan"]["splits"],
    }


def build_selection(context):
    source = context.require_ready_cohort()
    settings = context.config["validation"]["cohort"]
    # Bound retained memory by users * 13 while preserving the legacy JSON-list hash.
    counts, histories, source_items = Counter(), defaultdict(list), set()
    digest = hashlib.sha256(b"[")
    raw_count = 0
    for row in iter_jsonl(context.cohort_dir / "events.jsonl"):
        if raw_count:
            digest.update(b",")
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False).encode())
        raw_count += 1
        user = row["user_id"]
        counts[user] += 1
        source_items.add(row["item_id"])
        entry = (row["timestamp"], row.get("source_event_id", row["event_id"]), row["event_id"], row)
        history = histories[user]
        if len(history) < settings["retained_sequence_length"]:
            heapq.heappush(history, entry)
        else:
            heapq.heappushpop(history, entry)
    digest.update(b"]")
    if [str(r["item_id"]) for r in source["catalog"]] != sorted(source_items, key=int):
        raise ValueError("catalog must match the ordered event item universe")
    retained = []
    excluded_users = []
    trimmed_events = 0
    for user in sorted(counts):
        count = counts[user]
        if count < settings["min_user_interactions"]:
            excluded_users.append({"user_id": user, "reason": "short_history", "event_count": count})
            continue
        retained.extend(entry[-1] for entry in histories[user])
        trimmed_events += count - len(histories[user])
    if not retained:
        raise ValueError("leave-one-out preprocessing retained no users")
    selected_events = [
        {**row, "event_id": i, "source_event_id": row["event_id"]}
        for i, row in enumerate(sorted(retained, key=lambda row: row["event_id"]))
    ]
    table = EventTable(selected_events)
    items = set(table.items)
    catalog = [row for row in source["catalog"] if row["item_id"] in items]
    titles = [row for row in source["metadata_titles"] if row["item_id"] in items]
    excluded = [
        {**row, "reason": "outside_retained_sequences"}
        for row in source["catalog"] if row["item_id"] not in items
    ]
    plan = {
        "schema_version": SCHEMA,
        "protocol": PROTOCOL,
        "preprocessing": {
            "min_user_interactions": settings["min_user_interactions"],
            "retained_sequence_length": settings["retained_sequence_length"],
            "order": "timestamp,source_event_id",
            "catalog_scope": "retained_sequence_catalog",
        },
        "source_counts": {k: source["plan"][k] for k in ("user_count", "interaction_count", "item_count")},
        "source_data_hash": digest.hexdigest(),
        "user_count": len(table.users),
        "interaction_count": len(table.rows),
        "item_count": len(table.items),
        "metadata_missing_policy": settings["metadata_missing_policy"],
        "excluded_users": excluded_users,
        "trimmed_event_count": trimmed_events,
        "short_history_event_count": sum(row["event_count"] for row in excluded_users),
        "splits": table.splits(),
    }
    plan["eligible_test_count"] = sum(
        (s["phases"]["test"]["eligible_count"] for s in plan["splits"])
    )
    payload = {
        "catalog": catalog,
        "metadata_titles": titles,
        "events": selected_events,
        "plan": plan,
        "excluded": excluded,
    }
    count = len(source["catalog"])
    from arm_registry import arm_contract

    manifest = {
        "arm_contract": arm_contract(context.config),
        "arms": list(active_arms(context.config)),
        "policy": POLICY,
        "data_hash": fingerprint(payload),
        "selection_hash": fingerprint(data_identity(payload)),
        "included_item_ids": [r["item_id"] for r in catalog],
        "statistics": {
            "original_item_count": count,
            "included_item_count": len(catalog),
            "excluded_item_count": len(excluded),
            "removed_event_count": raw_count - len(selected_events),
            "excluded_user_count": len(excluded_users),
            "trimmed_event_count": trimmed_events,
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
    reports = context.run_root / "reports"
    report = {
        "run_id": context.run_id,
        "dataset": context.dataset,
        "protocol": PROTOCOL,
        "source_data_hash": cohort["plan"]["source_data_hash"],
        "selection_hash": cohort["manifest"]["selection_hash"],
        "preprocessing": cohort["plan"]["preprocessing"],
        "source_counts": cohort["plan"]["source_counts"],
        "selected_counts": {k: cohort["plan"][k] for k in ("user_count", "interaction_count", "item_count")},
        "statistics": cohort["manifest"]["statistics"],
        "short_history_event_count": cohort["plan"]["short_history_event_count"],
        "split_hash": cohort["plan"]["splits"][0]["split_hash"],
        "status": "prepared_not_evaluated",
    }
    source, selected = report["source_counts"], report["selected_counts"]
    rows = "\n".join(
        f"| {label} | {source[key]} | {selected[key]} | {source[key] - selected[key]} |"
        for label, key in (("사용자", "user_count"), ("상호작용", "interaction_count"), ("아이템", "item_count"))
    )
    report_path = reports / "leave_one_out_cohort.md"
    semantics = ""
    if context.dataset == "m3l_10m":
        from preparation.m3l import INTERPRETATION

        source_plan = read_json(context.cohort_dir / "cohort_plan.json")
        report["m3l_source"] = {key: source_plan[key] for key in
                              ("source_counts", "rating_policy", "interpretation", "sources",
                               "excluded_item_count", "excluded_interaction_count")}
        semantics = (f"{INTERPRETATION}. 모든 평점을 사용하며 영상 미확보 영화와 상호작용은 제외합니다. "
                     f"영상 제외 영화: {source_plan['excluded_item_count']}; "
                     f"제외 상호작용: {source_plan['excluded_interaction_count']}.\n\n")
    atomic_write_json(reports / "leave_one_out_cohort.json", report, durable=True)
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(
        f"# {context.run_id} / {context.dataset}: leave-one-out cohort\n\n"
        + semantics +
        "원본 이력 5개 이상 사용자에서 최근 13개를 보존하고, 마지막 1개는 test, 직전 1개는 validation입니다.\n\n"
        f"| 대상 | 원본 | 선택 | 제외 |\n| --- | ---: | ---: | ---: |\n{rows}\n\n"
        f"짧은 이력 제외 이벤트: {report['short_history_event_count']}; 오래된 이벤트 절단: {cohort['plan']['trimmed_event_count']}.\n\n"
        f"원본 fingerprint: `{report['source_data_hash']}`\n\n분할 fingerprint: `{report['split_hash']}`\n\n"
        "준비 단계의 데이터 점검 결과이며 추천 성능은 아직 평가하지 않았습니다. 재생성: 같은 Run에서 embed-representations 실행.\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
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
        table = EventTable(cohort["events"])
        if cohort["plan"]["protocol"] != PROTOCOL or table.splits() != cohort["plan"]["splits"]:
            raise ValueError("invalid leave-one-out split manifest")
        if [row["item_id"] for row in cohort["catalog"]] != table.items:
            raise ValueError("invalid leave-one-out catalog")
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
    from validation.recommendation_contracts import TRAINING_IMPLEMENTATION_VERSION, EVALUATION_CONTRACT

    return fingerprint(
        {
            "implementation": TRAINING_IMPLEMENTATION_VERSION,
            "evaluation_contract": EVALUATION_CONTRACT,
            "events": cohort["events"],
            "selection_hash": cohort["manifest"]["selection_hash"],
            "model": {
                k: v for k, v in config.model.model_dump().items() if k != "seeds"
            },
            "cutoffs": config.evaluation.cutoffs,
        }
    )
