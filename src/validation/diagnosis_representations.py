"""Compact diagnosis summaries; full per-item evidence stays in .inputs files."""

import json
from collections import Counter

from arm_registry import registry
from validation.diagnosis_support import MAX_ERROR_EXAMPLES
from validation.representation_provenance import read_state, state_path


def _distribution(rows, field):
    counts = Counter(json.dumps(row.get(field), sort_keys=True, ensure_ascii=False)
                     for row in rows)
    values = counts.most_common(MAX_ERROR_EXAMPLES)
    return {
        "distinct_count": len(counts),
        "values": [{"value": json.loads(value), "count": count} for value, count in values],
        "omitted_count": sum(counts.values()) - sum(count for _, count in values),
    }


def representation_report(context, arms):
    representations, fallbacks = {}, {}
    registered = registry(context.config)
    for name in arms:
        state = read_state(context, name)
        rows = state.get("sources", [])
        details_path = state_path(context, name).relative_to(context.run_root).as_posix()
        fallback_count = issue_count = 0
        fallback_examples, issue_examples = [], []
        for index, row in enumerate(rows):
            example = {"source_index": index, "content_id": row.get("content_id")}
            if row.get("actual_arm") and row["actual_arm"] != name:
                fallback_count += 1
                if len(fallback_examples) < MAX_ERROR_EXAMPLES:
                    fallback_examples.append({**example, "actual_arm": row["actual_arm"]})
            reasons = [key for key in ("violations", "correction_count") if row.get(key)]
            if row.get("status") not in (None, "complete"):
                reasons.append("status")
            if reasons:
                issue_count += 1
                if len(issue_examples) < MAX_ERROR_EXAMPLES:
                    issue_examples.append({**example, "fields": reasons})
        summary = {
            "count": len(rows),
            "generation_record_count": sum(row.get("generation") is not None for row in rows),
            "issue_count": issue_count,
            "issue_examples": issue_examples,
            "issue_examples_omitted_count": issue_count - len(issue_examples),
        }
        if registered[name].model is not None:
            summary["distributions"] = {
                field: _distribution(rows, field)
                for field in ("actual_arm", "source_arm", "status", "summary_schema",
                              "summary_policy", "source_provenance")
            }
            lengths = [row["word_count"] for row in rows if row.get("word_count") is not None]
            summary["word_count"] = {
                "count": len(lengths),
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
                "mean": sum(lengths) / len(lengths) if lengths else None,
            }
        representations[name] = {
            **{key: state[key] for key in (
                "input_hash", "embedding_hash", "recommendation_hash", "truncation"
            ) if key in state},
            "details_path": details_path,
            "sources_summary": summary,
        }
        if registered[name].model == "gemini":
            fallbacks[name] = {
                "count": fallback_count,
                "examples": fallback_examples,
                "omitted_count": fallback_count - len(fallback_examples),
                "details_path": details_path,
            }
    return representations, fallbacks
