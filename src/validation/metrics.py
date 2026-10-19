from __future__ import annotations

import math
from typing import Iterable


def metrics_from_rank(rank: int, cutoffs: Iterable[int] = (4, 8, 10, 20)) -> dict[str, float]:
    result: dict[str, float] = {}
    for cutoff in cutoffs:
        result[f"HR@{cutoff}"] = float(rank <= cutoff)
        result[f"NDCG@{cutoff}"] = 1.0 / math.log2(rank + 1) if rank <= cutoff else 0.0
    return result
