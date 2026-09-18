"""Streaming event validation and paired user-cluster, equal-day statistics."""

from __future__ import annotations

import numpy as np

from pipeline_runtime import read_json, write_json
from extraction.recovery import fingerprint
from validation.diagnosis_scenes import _scene_coverage
from validation.diagnosis_statistics import multiple_comparison_policy
from validation.metrics import metrics_from_rank
from validation.recommendation_contracts import resolve_target_arms, target_scope
from arm_registry import registry
from validation.representation_checks import verify_representations
from validation.rolling_data import EventTable, iter_jsonl
from validation.rolling_recommendation import (
    combination_complete,
    combination_dir,
    phase_ids,
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


def comparisons(observed, draws, settings, *, arms=None, config=None):
    from validation.diagnosis_statistics import comparison_families
    from validation.recommendation_contracts import DEFAULT_PROTOCOL
    configured = registry(config or DEFAULT_PROTOCOL)
    names = list(resolve_target_arms(config=config) if arms is None else arms)
    indices = {name: index for index, name in enumerate(names)}
    if len(observed) != len(names) or draws.shape[1] != len(names):
        raise ValueError("arm/metric dimension mismatch")
    result = {}
    for family, pairs in comparison_families(config).items():
        alpha = settings.familywise_alpha / len(pairs)
        for left, right in pairs:
            if left not in indices or right not in indices:
                continue
            a, b = indices[left], indices[right]
            delta = draws[:, a] - draws[:, b]
            lo, hi = np.quantile(delta, [alpha / 2, 1 - alpha / 2])
            result[f"{left}-{right}"] = {
                "family": family, "role": "confirmatory", "family_size": len(pairs),
                "difference": float(observed[a] - observed[b]),
                "ci_low": float(lo), "ci_high": float(hi), "confidence_level": 1 - alpha,
                "superior": bool(lo > 0), "inferior": bool(hi < 0),
                "relative_difference": (float((observed[a] - observed[b]) / observed[b])
                                        if observed[b] > 0 else None),
            }
    by_kind = {(arm.representation, arm.model): name for name, arm in configured.items()}
    for control in ("description",):
        terms = [by_kind["graph", "gemini"], by_kind[control, "gemini"],
                 by_kind["graph", "qwen"], by_kind[control, "qwen"]]
        if not set(terms) <= indices.keys():
            continue
        a, b, c, d = [indices[name] for name in terms]
        delta = draws[:, a] - draws[:, b] - draws[:, c] + draws[:, d]
        lo, hi = np.quantile(delta, [0.025, 0.975])
        result[f"interaction_graph_vs_{control}"] = {
            "family": "interaction", "role": "exploratory", "arms": terms,
            "difference": float(observed[a] - observed[b] - observed[c] + observed[d]),
            "ci_low": float(lo), "ci_high": float(hi), "confidence_level": 0.95,
        }
    return result


def collect_metrics(context, config, cohort, *, arms=None):
    selected = resolve_target_arms(config=context.config) if arms is None else arms
    table = EventTable(iter_jsonl(context.cohort_dir / "events.jsonl"))
    training_input_hash = fingerprint({"events": table.rows, "model": context.config["validation"]["model"],
                                       "cutoffs": config.evaluation.cutoffs})
    users = {user: i for i, user in enumerate(table.users)}
    splits = cohort["plan"]["splits"]
    if (len(table.users), len(table.rows), len(table.items)) != (
        config.cohort.user_count,
        config.cohort.interaction_count,
        config.cohort.item_count,
    ) or splits != table.splits():
        raise ValueError("full source cardinality or rolling split manifest mismatch")
    arms = list(selected)
    verify_representations(context, cohort, arms=selected)
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
                    "training_input_hash": training_input_hash,
                }
                from validation.representation_provenance import recommendation_identity
                identity.update(recommendation_identity(context, selected[arm]))
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


def diagnose(context, *, target=None, compare_run_id=None):
    from validation.steps import validation_config

    arms = resolve_target_arms(target, config=context.config)
    context.initialize()
    config = validation_config(context)
    errors = []
    document = {
        "schema_version": "rolling-diagnosis/v3",
        "run_id": context.run_id,
        **target_scope(arms, config=context.config),
        "statistics": {"status": "not_computed"},
    }
    try:
        cohort = context.require_ready_cohort()
        document["cohort"] = cohort["plan"]
        from validation.metadata import verify_missing_metadata
        if "metadata" in arms.values():
            document["metadata_missing"] = verify_missing_metadata(context, cohort)
        scene = _scene_coverage(
            context.run_root,
            [r["content_id"] for r in cohort["catalog"]],
            errors,
            context.config["extraction"]["visual_evidence"]["scene_duration"],
            config.evaluation.model_dump(),
            True,
            True,
            branches=set(arms.values()), config=context.config,
        )
        document["scene_coverage"] = scene[0]
        from extraction.recovery_report import recovery_report
        document["generation_recovery"] = recovery_report(context, branches=arms)
        from validation.diagnosis_representations import representation_report
        document["representations"], document["gemini_summary_fallbacks"] = (
            representation_report(context, arms)
        )
        sums, counts, report = collect_metrics(context, config, cohort, arms=arms)
        document["recommendations"] = report
        if not errors:
            observed, draws, bootstrap = cluster_bootstrap(
                sums, counts, samples=config.evaluation.bootstrap_samples
            )
            document["statistics"] = {
                "status": "computed",
                "bootstrap": bootstrap,
                "comparisons": comparisons(observed, draws, config.evaluation, arms=arms, config=context.config),
                "policy": multiple_comparison_policy(
                    config.evaluation.model_dump(), True, "NDCG@10", arms=arms, config=context.config,
                ),
            }
            if compare_run_id is not None:
                from validation.run_comparison import compare_graph_runs
                document["run_comparison"] = compare_graph_runs(context, compare_run_id, target=arms)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        errors.append({"code": "invalid_rolling_evidence", "message": str(exc)})
    document["runtime_decision"] = {"status": "fail" if errors else "pass", "errors": errors}
    document["paper_reference"] = {
        "metadata_hr10": 0.046,
        "interpretation": "Reference only: full data/rolling does not establish every unspecified paper setting.",
    }
    if "recommendations" in document and "metadata" in arms:
        value = document["recommendations"]["means"]["metadata"]["HR@10"]
        document["paper_reference"]["hr10_difference"] = value - 0.046
    write_json(context.diagnosis_path, document)
    if errors or document["statistics"]["status"] != "computed":
        raise RuntimeError(f"rolling diagnosis failed; see {context.diagnosis_path}")
    return {"stage": "run-diagnosis", "status": "pass"}
