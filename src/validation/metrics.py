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
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, len(treatment), size=(count, len(treatment)))
        treatment_means = treatment[indices].mean(axis=1)
        control_means = control[indices].mean(axis=1)
        if np.any(control_means <= 0):
            raise ValueError("bootstrap resample has zero control mean")
        estimates[start:start + count] = (treatment_means - control_means) / control_means
    point = (treatment.mean() - control.mean()) / control.mean()
    return {
        "relative_delta": float(point),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "n_users": int(len(treatment)),
        "bootstrap_samples": samples,
    }
