from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def metrics_from_rank(rank: int, cutoffs: Iterable[int] = (4, 8, 10, 20)) -> dict[str, float]:
    result: dict[str, float] = {}
    for cutoff in cutoffs:
        result[f"HR@{cutoff}"] = float(rank <= cutoff)
        result[f"NDCG@{cutoff}"] = 1.0 / math.log2(rank + 1) if rank <= cutoff else 0.0
    return result


def paired_bootstrap_ci(values: np.ndarray, *, samples: int = 10_000, seed: int = 42) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires a finite non-empty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = max(1, min(samples, 256))
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_users": int(len(values)),
        "bootstrap_samples": samples,
    }


def paired_relative_bootstrap_ci(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    treatment = np.asarray(treatment, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if treatment.ndim != 1 or treatment.shape != control.shape or not len(treatment):
        raise ValueError("relative paired bootstrap requires equal non-empty vectors")
    if not np.isfinite(treatment).all() or not np.isfinite(control).all() or control.mean() <= 0:
        raise ValueError("relative paired bootstrap requires finite values and positive control mean")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk = max(1, min(samples, 256))
    accepted = 0
    draws = 0
    zero_control_resamples = 0
    max_draws = max(samples * 10, 10_000)
    while accepted < samples:
        remaining = samples - accepted
        count = min(4096, max(chunk, remaining * 2))
        indices = rng.integers(0, len(treatment), size=(count, len(treatment)))
        treatment_means = treatment[indices].mean(axis=1)
        control_means = control[indices].mean(axis=1)
        valid = control_means > 0
        zero_control_resamples += int(np.count_nonzero(~valid))
        valid_estimates = (
            (treatment_means[valid] - control_means[valid]) / control_means[valid]
        )
        take = min(remaining, len(valid_estimates))
        estimates[accepted:accepted + take] = valid_estimates[:take]
        accepted += take
        draws += count
        if draws >= max_draws and accepted < samples:
            raise ValueError(
                "relative paired bootstrap could not collect enough positive-control resamples"
            )
    point = (treatment.mean() - control.mean()) / control.mean()
    return {
        "relative_delta": float(point),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "n_users": int(len(treatment)),
        "bootstrap_samples": samples,
        "bootstrap_draws": draws,
        "zero_control_resamples": zero_control_resamples,
        "control_nonzero_users": int(np.count_nonzero(control)),
        "conditional_on_positive_control": zero_control_resamples > 0,
    }
