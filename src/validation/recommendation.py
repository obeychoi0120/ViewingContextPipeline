from __future__ import annotations
import numpy as np
from validation.config import ValidationConfig
from validation.model import SASRec, torch


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
    features: dict[str, np.ndarray],
    device: "torch.device",
) -> SASRec:
    for component in ("title_values", "video_values"):
        if features[component].shape != (item_count, config.encoder.embedding_dim):
            raise ValueError("text feature dimension mismatch; rerun embed-representations")
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
