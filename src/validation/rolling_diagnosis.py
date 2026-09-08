"""Streaming event validation and paired user-cluster, equal-day statistics."""

from __future__ import annotations

import numpy as np

from pipeline_runtime import read_json, write_json
from validation.diagnosis_scenes import _scene_coverage
from validation.diagnosis_statistics import multiple_comparison_policy
from validation.metrics import metrics_from_rank
from validation.recommendation_contracts import RECOMMENDATION_ARMS
from validation.rolling_data import EventTable, iter_jsonl
from validation.rolling_recommendation import (
    combination_complete,
    combination_dir,
    phase_ids,
    recommendation_dependencies,
)

MEMORY_LIMIT = 128 * 1024**2


def weighted_day_mean(sums, counts, weights):
    denominators = weights @ counts
    if np.any(denominators <= 0):
        raise ValueError("bootstrap has an empty evaluation date denominator")
    numerators = weights @ sums.reshape(len(counts), -1)
    return (numerators.reshape(-1, *sums.shape[1:]) / denominators[:, :, None]).mean(axis=1)


def cluster_bootstrap(sums, counts, *, samples, seed=42, memory_limit=MEMORY_LIMIT):
    """sums[user,date,arm] contains seed-averaged event sums, counts[user,date]."""
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    users, days, arms = sums.shape
    if counts.shape != (users, days) or not users or not np.isfinite(sums).all():
        raise ValueError("invalid user/date aggregates")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("invalid event counts")
    # Reserve space for NumPy's sampler/matmul workspace and Python array metadata.
    fixed = sums.nbytes + counts.nbytes + samples * arms * 8 + users * 8 + 8 * 1024**2
    # Include integer multinomial draws, float weights, matrix outputs and division temporaries.
    per_draw = users * 16 + days * (arms * 3 + 2) * 8
    batch = min(64, (memory_limit - fixed) // per_draw)
    if batch < 1:
        raise ValueError("bootstrap input and one draw exceed the working memory limit")
    observed = weighted_day_mean(sums, counts, np.ones((1, users)))[0]
    rng = np.random.default_rng(seed)
    probabilities = np.full(users, 1 / users)
    draws = np.empty((samples, arms))
    for start in range(0, samples, batch):
        size = min(batch, samples - start)
        weights = rng.multinomial(users, probabilities, size=size).astype(np.float64)
        draws[start : start + size] = weighted_day_mean(sums, counts, weights)
        del weights
    return (
        observed,
        draws,
        {
            "unit": "user",
            "seed": seed,
            "samples": samples,
            "batch_size": int(batch),
            "working_bytes_upper_bound": int(fixed + batch * per_draw),
            "memory_limit_bytes": memory_limit,
            "aggregation": "seed mean then equal date mean",
        },
    )


def comparisons(observed, draws, settings):
    alpha = settings.familywise_alpha
    names = list(RECOMMENDATION_ARMS)
    result = {}
    for index in (1, 2, 3):
        delta = draws[:, index] - draws[:, 0]
        lo, hi = np.quantile(delta, [alpha / 3 / 2, 1 - alpha / 3 / 2])
        result[f"{names[index]}-{names[0]}"] = {
            "family": "metadata_baseline_superiority",
            "difference": float(observed[index] - observed[0]),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "superior": bool(lo > 0),
        }
    if observed[3] <= 0 or np.any(draws[:, 3] <= 0):
        raise ValueError("Description relative-effect denominator is zero; NI is undefined")
    for index in (1, 2):
        relative = (draws[:, index] - draws[:, 3]) / draws[:, 3]
        lo = float(np.quantile(relative, alpha / 2))
        result[f"{names[index]}-{names[3]}"] = {
            "family": "graph_vs_description_non_inferiority",
            "relative_difference": float((observed[index] - observed[3]) / observed[3]),
            "one_sided_relative_ci_low": lo,
            "margin": -settings.non_inferiority_margin,
            "non_inferior": lo > -settings.non_inferiority_margin,
        }
    if observed[1] <= 0 or np.any(draws[:, 1] <= 0):
        raise ValueError("Qwen relative-effect denominator is zero; exploratory CI is undefined")
    relative = (draws[:, 2] - draws[:, 1]) / draws[:, 1]
    lo, hi = np.quantile(relative, [alpha / 2, 1 - alpha / 2])
    result[f"{names[2]}-{names[1]}"] = {
        "family": "qwen_vs_gemini",
        "role": "exploratory",
        "relative_difference": float((observed[2] - observed[1]) / observed[1]),
        "relative_ci_low": float(lo),
        "relative_ci_high": float(hi),
    }
    return result


def collect_metrics(context, config, cohort):
    table = EventTable(iter_jsonl(context.cohort_dir / "events.jsonl"))
    users = {user: i for i, user in enumerate(table.users)}
    splits = cohort["plan"]["splits"]
    if (len(table.users), len(table.rows), len(table.items)) != (
        config.cohort.user_count,
        config.cohort.interaction_count,
        config.cohort.item_count,
    ) or splits != table.splits():
        raise ValueError("full source cardinality or rolling split manifest mismatch")
    arms = list(RECOMMENDATION_ARMS)
    fingerprint = recommendation_dependencies(context, cohort)
    sums = np.zeros((len(users), len(splits), len(arms)))
    counts = np.zeros((len(users), len(splits)))
    metrics = list(metrics_from_rank(1, config.evaluation.cutoffs))
    daily = []
    total = 0
    for day, split in enumerate(splits):
        ids = phase_ids(table, split, "test")
        expected = set(ids.tolist())
        for event in ids:
            counts[users[table.rows[event]["user_id"]], day] += 1
        raw_refit = table.select(end=split["phases"]["refit"]["end_ms"], eligible=False)
        frequencies = np.bincount(table.targets[raw_refit], minlength=len(table.items) + 1)
        for seed in config.model.seeds:
            for index, arm in enumerate(arms):
                identity = {
                    "run_id": context.run_id,
                    "evaluation_date": split["evaluation_date"],
                    "seed": seed,
                    "arm": arm,
                    "fingerprint": fingerprint,
                }
                directory = combination_dir(context, split["evaluation_date"], seed, arm)
                if not combination_complete(directory, identity, len(ids)):
                    raise ValueError(f"incomplete/corrupt combination: {directory}")
                training = read_json(directory / "training.json")
                best = training["best_epoch"]
                if training["split"] != split or len(training["refit"]) != best:
                    raise ValueError("training split/refit epoch mismatch")
                for phase in ("selection", "refit"):
                    count = split["phases"][phase]["eligible_count"]
                    for row in training[phase]:
                        if (
                            row["positive_count"] != count
                            or row["optimizer_updates"]
                            != (count + config.model.batch_size - 1) // config.model.batch_size
                        ):
                            raise ValueError("training transition/update count mismatch")
                seen = set()
                totals = dict.fromkeys(metrics, 0.0)
                for row in iter_jsonl(directory / "per_event_metrics.jsonl"):
                    event = row["event_id"]
                    if type(event) is not int or event not in expected or event in seen:
                        raise ValueError("duplicate or unexpected test event")
                    seen.add(event)
                    source = table.rows[event]
                    if any(row.get(k) != v for k, v in {**source, **identity}.items()):
                        raise ValueError("event identity mismatch")
                    rank = row["rank"]
                    if type(rank) is not int or not 1 <= rank <= len(table.items):
                        raise ValueError("invalid event rank")
                    if row["candidate_count"] != len(table.items):
                        raise ValueError("candidate catalog mismatch")
                    if row["refit_item_frequency"] != int(frequencies[table.targets[event]]):
                        raise ValueError("refit target frequency mismatch")
                    calculated = metrics_from_rank(rank, config.evaluation.cutoffs)
                    if any(
                        not np.isfinite(row[m]) or abs(row[m] - calculated[m]) > 1e-12
                        for m in metrics
                    ):
                        raise ValueError("event metric/rank mismatch")
                    for metric in metrics:
                        totals[metric] += row[metric]
                    sums[users[source["user_id"]], day, index] += row["NDCG@10"] / len(
                        config.model.seeds
                    )
                    total += 1
                if seen != expected:
                    raise ValueError("missing eligible test events")
                daily.append(
                    {
                        "evaluation_date": split["evaluation_date"],
                        "seed": seed,
                        "arm": arm,
                        "event_count": len(ids),
                        **{k: v / len(ids) for k, v in totals.items()},
                    }
                )
    expected_total = cohort["plan"]["eligible_test_count"] * len(arms) * len(config.model.seeds)
    if total != expected_total:
        raise ValueError("event grid cardinality mismatch")
    means = {
        arm: {m: float(np.mean([r[m] for r in daily if r["arm"] == arm])) for m in metrics}
        for arm in arms
    }
    return (
        sums,
        counts,
        {
            "expected_event_count": expected_total,
            "actual_event_count": total,
            "combination_count": len(daily),
            "daily": daily,
            "means": means,
        },
    )


def diagnose(context):
    from validation.steps import validation_config

    context.initialize()
    config = validation_config(context)
    errors = []
    document = {
        "schema_version": "rolling-diagnosis/v1",
        "run_id": context.run_id,
        "statistics": {"status": "not_computed"},
    }
    try:
        cohort = context.require_ready_cohort()
        document["cohort"] = cohort["plan"]
        scene = _scene_coverage(
            context.run_root,
            [r["content_id"] for r in cohort["catalog"]],
            errors,
            context.config["extraction"]["visual_evidence"]["scene_duration"],
            config.evaluation.model_dump(),
            True,
            True,
        )
        document["scene_coverage"] = scene[0]
        document["gemini_summary_fallbacks"] = read_json(
            context.representations_dir / "graph_gemini_fallbacks.json"
        )
        sums, counts, report = collect_metrics(context, config, cohort)
        document["recommendations"] = report
        if not errors:
            observed, draws, bootstrap = cluster_bootstrap(
                sums, counts, samples=config.evaluation.bootstrap_samples
            )
            document["statistics"] = {
                "status": "computed",
                "bootstrap": bootstrap,
                "comparisons": comparisons(observed, draws, config.evaluation),
                "policy": multiple_comparison_policy(
                    config.evaluation.model_dump(), True, "NDCG@10"
                ),
            }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        errors.append({"code": "invalid_rolling_evidence", "message": str(exc)})
    document["runtime_decision"] = {"status": "fail" if errors else "pass", "errors": errors}
    document["paper_reference"] = {
        "metadata_hr10": 0.046,
        "interpretation": "Reference only: full data/rolling does not establish every unspecified paper setting.",
    }
    if "recommendations" in document:
        value = document["recommendations"]["means"]["SASRec_METADATA"]["HR@10"]
        document["paper_reference"]["hr10_difference"] = value - 0.046
    write_json(context.diagnosis_path, document)
    if errors or document["statistics"]["status"] != "computed":
        raise RuntimeError(f"rolling diagnosis failed; see {context.diagnosis_path}")
    return {"stage": "run-diagnosis", "status": "pass"}
