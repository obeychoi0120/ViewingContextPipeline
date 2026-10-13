"""DCGN-inspired dense similarity propagation within each video's scenes."""

import numpy as np

from validation.model import torch, nn
from validation.profiling import span, workload


def scene_groups(offsets):
    """Static CPU indices for batches of videos having the same scene count."""
    offsets = np.asarray(offsets, dtype=np.int64)
    counts = np.diff(offsets)
    return {
        f"scene_group_{size}": offsets[:-1][counts == size, None] + np.arange(size)
        for size in np.unique(counts)
        if size > 0
    }


def normalized_similarity(features):
    """Nonnegative cosine with unit self-loops and symmetric degree scaling."""
    unit = nn.functional.normalize(features, dim=-1, eps=1e-8)
    weights = (unit @ unit.transpose(-1, -2)).clamp_min(0)
    diagonal = torch.eye(features.shape[-2], dtype=torch.bool, device=features.device)
    weights = torch.where(diagonal, torch.ones_like(weights), weights)
    scale = weights.sum(dim=-1).rsqrt()
    return weights * scale.unsqueeze(-1) * scale.unsqueeze(-2)


class SceneGCN(nn.Module):
    adjacency_span = "scene_similarity"

    def __init__(self, dimension):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension)

    def adjacency(self, features):
        return normalized_similarity(features)

    def forward(self, scenes, similarity_features, groups):
        output = torch.zeros_like(scenes)
        for indices in groups:
            with span(self.adjacency_span, cuda=True):
                adjacency = self.adjacency(similarity_features[indices])
            with span("scene_gcn", cuda=True):
                values = scenes[indices]
                updated = values + nn.functional.relu(self.projection(adjacency @ values))
                output = output.index_copy(0, indices.reshape(-1), updated.flatten(0, 1))
            workload(scene_pairs=indices.shape[0] * indices.shape[1] ** 2)
        return output
