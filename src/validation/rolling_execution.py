"""Ephemeral worker-local acceleration; nothing here participates in cache identity."""
from collections import OrderedDict

import numpy as np

from validation.model import torch

EXECUTION_VERSION = "rolling-vectorized/v1"


class RollingExecution:
    def __init__(self, table):
        self.table = table
        self.sequences = {}
        self.popularity = OrderedDict()

    def padded(self, max_length):
        if max_length not in self.sequences:
            values = np.zeros((len(self.table.rows), max_length), dtype=np.int64)
            for event in range(len(values)):
                history = self.table.history(event, max_length)
                values[event, :len(history)] = history
            self.sequences[max_length] = values
        return self.sequences[max_length]

    def inputs(self, ids, max_length, device):
        return torch.as_tensor(self.padded(max_length)[ids], dtype=torch.long, device=device)

    def log_probabilities(self, probabilities, targets, device, dtype):
        key = (id(probabilities), str(device), dtype)
        if key not in self.popularity:
            values = torch.as_tensor(probabilities, dtype=dtype, device=device)
            valid = (torch.isfinite(values) & values.gt(0)).cpu().numpy()
            # Retain the source array so its Python identity cannot be recycled.
            self.popularity[key] = (probabilities, values.log(), valid)
            # Only selection/refit for the current date need to stay resident.
            while len(self.popularity) > 2:
                self.popularity.popitem(last=False)
        self.popularity.move_to_end(key)
        _, logs, valid = self.popularity[key]
        if not valid[targets].all():
            raise RuntimeError("invalid positive popularity probabilities")
        return logs


def execution_for(table):
    if not hasattr(table, "_rolling_execution"):
        table._rolling_execution = RollingExecution(table)
    return table._rolling_execution


def negative_mask(inputs, targets):
    mask = (inputs[:, :, None] == targets[None, None, :]).any(dim=1)
    mask |= targets[:, None] == targets[None, :]
    mask.fill_diagonal_(False)
    return mask


def masked_ranks(scores, table, batch):
    """Full strictly-earlier history, with ascending item-index tie breaking."""
    if not torch.isfinite(scores).all():
        raise RuntimeError("nonfinite catalog scores")
    target_rows = table.targets[batch] - 1
    targets = torch.as_tensor(target_rows, dtype=torch.long, device=scores.device)
    row_ids = torch.arange(len(batch), device=scores.device)
    target_scores = scores[row_ids, targets].clone()
    rows, columns = [], []
    for row, event in enumerate(batch):
        seen = np.unique(table.history(int(event))) - 1
        seen = seen[seen != target_rows[row]]
        rows.extend([row] * len(seen))
        columns.extend(seen)
    if rows:
        coordinates = torch.as_tensor(
            np.asarray([rows, columns], dtype=np.int64), device=scores.device,
        )
        # Coordinates are unique, including under deterministic CUDA execution.
        scores[coordinates[0], coordinates[1]] = -torch.inf
    item_rows = torch.arange(scores.shape[1], device=scores.device)
    return (
        1 + (scores > target_scores[:, None]).sum(dim=1)
        + ((scores == target_scores[:, None]) & (item_rows < targets[:, None])).sum(dim=1)
    )
