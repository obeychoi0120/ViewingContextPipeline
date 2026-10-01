"""V4 architecture with the old loop-based packing/pooling for numerical checks.

The historical v2 reference remains untouched. This reference substitutes v4's
one-layer, concatenation-only readout/fusion, not v2 predictions or weights.
"""

import numpy as np

import graph_reference as legacy
from validation.graph_model import GraphSASRec as CurrentGraphSASRec
from validation.graph_model import RoleGraphEncoder as CurrentEncoder
from validation.model import nn, torch

# Independently pack raw graph topology with the original nested-loop algorithm.
graph_batch = legacy.graph_batch


class RoleGraphEncoder(CurrentEncoder):
    def __init__(self, feature_dim=1024, hidden=128, aggregation="mean"):
        super().__init__(feature_dim, hidden, aggregation)
        self.layers = nn.ModuleList([legacy.RoleLayer(hidden)])

    forward = legacy.RoleGraphEncoder.forward


class GraphSASRec(CurrentGraphSASRec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_encoder = RoleGraphEncoder(
            self.store["features"].shape[1],
            hidden=self.title_dim,
            aggregation=kwargs["aggregation"],
        )

    def _calculate(self, ids):
        # Multiple videos in an ordinary raw batch exercise loop pooling and
        # contrast with production's smaller packed chunks in these tests.
        if not len(ids):
            return self.title_projection.weight.new_empty((0, self.embedding_dim))
        device = self.title_projection.weight.device
        indices = np.asarray(ids, dtype=np.int64) - 1
        batch = graph_batch(self.store, indices, device)
        video = self.graph_encoder(batch)
        title_ids = self.store["titles"][indices]
        titles = torch.as_tensor(np.array(self.store["features"][title_ids]), device=device)
        titles = self.title_projection(titles)
        titles = titles.masked_fill(torch.as_tensor(title_ids == 0, device=device)[:, None], 0)
        values = self.item_norm(torch.cat([titles, video], dim=-1))
        available = (
            self.store["video_offsets"][indices + 1] > self.store["video_offsets"][indices]
        ) | (title_ids != 0)
        return values.masked_fill(~torch.as_tensor(available, device=device)[:, None], 0)
