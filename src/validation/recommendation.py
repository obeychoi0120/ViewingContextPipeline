from __future__ import annotations
import numpy as np
from validation.config import ValidationConfig
from validation.model import SASRec, torch

def popularity_probabilities(
    sequences: list[list[int]],
    item_count: int,
    power: float,
) -> np.ndarray:
    counts = np.zeros(item_count + 1, dtype=np.float64)
    for sequence in sequences:
        for item_id in sequence:
            if not 1 <= item_id <= item_count:
                raise ValueError(f"item id outside catalog: {item_id}")
            counts[item_id] += 1
    powered = np.power(counts[1:], power)
    denominator = float(powered.sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("cannot build popularity probabilities from empty training data")
    probabilities = np.zeros(item_count + 1, dtype=np.float32)
    probabilities[0] = 1.0
    probabilities[1:] = (powered / denominator).astype(np.float32)
    return probabilities


def _arm_kind(branch: str) -> str:
    if branch in {"meta", "metadata"}:
        return "metadata"
    if branch.startswith("graph_"):
        return "graph"
    if branch.startswith("desc_"):
        return "desc"
    raise ValueError(f"unsupported recommendation branch: {branch}")


def _new_model(
    config: ValidationConfig,
    *,
    item_count: int,
    branch: str,
    features: np.ndarray,
    device: "torch.device",
) -> SASRec:
    return SASRec(
        item_count,
        config.model.max_sequence_length,
        config.model.embedding_dim,
        config.model.num_blocks,
        config.model.num_heads,
        config.model.dropout,
        arm=_arm_kind(branch),
        item_features=features,
    ).to(device)


def _optimizer(model: SASRec, config: ValidationConfig) -> "torch.optim.Optimizer":
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=0.1,
    )
