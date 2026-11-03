"""Existing cosine-similarity scene propagation, shared by all scene-based Arms."""

import numpy as np

from validation.model import torch, nn
from validation.profiling import span, workload


def scene_groups(offsets):
    """Group videos with equal scene counts without connecting different videos."""
    offsets = np.asarray(offsets)
    counts = np.diff(offsets)
    return {
        f"scene_group_{size}": offsets[:-1][counts == size, None] + np.arange(size)
        for size in np.unique(counts)
        if size > 0
    }


def normalized_similarity(features):
    """Nonnegative cosine, unit self-loops and symmetric degree normalization."""
    unit = nn.functional.normalize(features, dim=-1, eps=1e-8)
    weights = (unit @ unit.transpose(-1, -2)).clamp_min(0)
    diagonal = torch.eye(features.shape[-2], dtype=torch.bool, device=features.device)
    weights = torch.where(diagonal, torch.ones_like(weights), weights)
    scale = weights.sum(dim=-1).rsqrt()
    return weights * scale.unsqueeze(-1) * scale.unsqueeze(-2)


class SceneGCN(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, scenes, similarity_features, groups):
        output = torch.zeros_like(scenes)
        for indices in groups:
            with span("scene_similarity", cuda=True):
                adjacency = normalized_similarity(similarity_features[indices])
            with span("scene_gcn", cuda=True):
                values = scenes[indices]
                updated = values + nn.functional.relu(self.projection(adjacency @ values))
                output = output.index_copy(0, indices.reshape(-1), updated.flatten(0, 1))
            workload(scene_pairs=indices.shape[0] * indices.shape[1] ** 2)
        return output
