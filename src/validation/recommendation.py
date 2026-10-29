"""Optimizer shared by every recommendation Arm."""

from validation.config import ValidationConfig
from validation.model import SASRec, torch


def _optimizer(model: SASRec, config: ValidationConfig) -> "torch.optim.Optimizer":
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=0.1,
    )
