"""Role-aware scene encoding and differentiable video pooling for SASRec."""

import numpy as np

from validation.model import SASRec, torch, nn
from validation.graph_inputs import GraphStore


def segment_mean(values, indices, count):
    output = values.new_zeros((count, values.shape[-1]))
    output.index_add_(0, indices, values)
    sizes = values.new_zeros(count)
    sizes.index_add_(0, indices, values.new_ones(len(values)))
    return output / sizes.clamp_min(1).unsqueeze(-1)


class RoleLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.relations = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(10)])
        self.states = nn.Parameter(torch.empty(5, 2, hidden))
        nn.init.normal_(self.states, std=0.02)
        self.norm = nn.LayerNorm(hidden)
        self.self_projection = nn.Linear(hidden, hidden)

    def forward(self, h, edges, missing):
        messages = torch.zeros_like(h)
        counts = h.new_zeros(len(h))
        for role, transform in enumerate(self.relations):
            selected = edges[edges[:, 2] == role]
            messages.index_add_(0, selected[:, 1], transform(h[selected[:, 0]]))
            counts.index_add_(0, selected[:, 1], h.new_ones(len(selected)))
        for role in range(5):
            selected = missing[missing[:, role + 1] >= 0]
            messages.index_add_(0, selected[:, 0], self.states[role, selected[:, role + 1]])
            counts.index_add_(0, selected[:, 0], h.new_ones(len(selected)))
        return self.norm(
            h
            + torch.nn.functional.gelu(
                self.self_projection(h) + messages / counts.clamp_min(1).unsqueeze(-1)
            )
        )


class RoleGraphEncoder(nn.Module):
    def __init__(self, feature_dim=1024, output_dim=512, hidden=256, aggregation="mean"):
        super().__init__()
        if aggregation not in ("mean", "attention"):
            raise ValueError("unknown scene aggregation")
        self.aggregation = aggregation
        self.output_dim = output_dim
        self.projection = nn.Linear(feature_dim, hidden)
        self.node_type = nn.Embedding(2, hidden)
        self.layers = nn.ModuleList([RoleLayer(hidden) for _ in range(2)])
        self.readout = nn.Sequential(nn.Linear(hidden * 3, output_dim), nn.LayerNorm(output_dim))
        self.attention = (
            nn.Sequential(nn.Linear(output_dim, 128), nn.Tanh(), nn.Linear(128, 1))
            if aggregation == "attention"
            else None
        )

    def forward(self, batch):
        h = self.projection(batch["features"]) + self.node_type(batch["types"])
        for layer in self.layers:
            h = layer(h, batch["edges"], batch["missing"])
        count = len(batch["contexts"])
        pooled = [
            segment_mean(h[batch["types"] == t], batch["node_scenes"][batch["types"] == t], count)
            for t in (0, 1)
        ]
        context = self.projection(batch["contexts"])
        scenes = self.readout(torch.cat([*pooled, context], dim=-1))
        videos = []
        for start, end in zip(batch["offsets"][:-1], batch["offsets"][1:]):
            part = scenes[start:end]
            if not len(part):
                videos.append(scenes.new_zeros(self.output_dim))
            elif self.attention is None:
                videos.append(part.mean(dim=0))
            else:
                weights = torch.softmax(self.attention(part).squeeze(-1), dim=0)
                videos.append((part * weights[:, None]).sum(dim=0))
        return torch.stack(videos)


def graph_batch(store, item_ids, device):
    """Pack only requested videos; no padding, truncation, or cross-scene links."""
    nodes, types, node_scenes, contexts, edges, missing = [], [], [], [], [], []
    offsets = [0]
    node_count = scene_count = 0
    for item in item_ids:
        start, end = store["video_offsets"][item : item + 2]
        for scene in range(start, end):
            first, last = store["scene_offsets"][scene : scene + 2]
            nodes.extend(store["nodes"][first:last])
            types.extend(store["types"][first:last])
            node_scenes.extend([scene_count] * (last - first))
            contexts.append(store["contexts"][scene])
            a, b = store["edge_offsets"][scene : scene + 2]
            edge = np.array(store["edges"][a:b], copy=True)
            edge[:, :2] += node_count - first
            edges.extend(edge)
            a, b = store["missing_offsets"][scene : scene + 2]
            states = np.array(store["missing"][a:b], copy=True)
            states[:, 0] += node_count - first
            missing.extend(states)
            node_count += last - first
            scene_count += 1
        offsets.append(scene_count)

    def ints(values, width=None):
        array = np.asarray(values, dtype=np.int64)
        if width:
            array = array.reshape(-1, width)
        return torch.as_tensor(array, device=device)

    def features(ids):
        return torch.as_tensor(
            np.array(store["features"][np.asarray(ids, dtype=np.int64)], copy=True), device=device
        )

    return {
        "features": features(nodes),
        "contexts": features(contexts),
        "types": ints(types),
        "node_scenes": ints(node_scenes),
        "edges": ints(edges, 3),
        "missing": ints(missing, 6),
        "offsets": offsets,
    }


class GraphSASRec(SASRec):
    def __init__(
        self,
        *,
        store,
        aggregation,
        item_count,
        max_length,
        embedding_dim,
        num_blocks,
        num_heads,
        dropout,
        arm,
    ):
        super().__init__(
            item_count,
            max_length,
            embedding_dim,
            num_blocks,
            num_heads,
            dropout,
            arm=arm,
            item_features=np.zeros((item_count, 1), dtype=np.float32),
        )
        self.store = store
        self.graph_encoder = RoleGraphEncoder(
            store["features"].shape[1], embedding_dim, aggregation=aggregation
        )
        self.item_projection = nn.Linear(store["features"].shape[1] + embedding_dim, embedding_dim)
        self._vectors = self._indices = None
        self.embedding_dim = embedding_dim

    def train(self, mode=True):
        self.clear_item_cache()
        return super().train(mode)

    def clear_item_cache(self):
        self._vectors = self._indices = None

    def _calculate_chunk(self, items):
        device = self.item_projection.weight.device
        indices = np.asarray(items, dtype=np.int64) - 1
        batch = graph_batch(self.store, indices, device)
        video = self.graph_encoder(batch)
        title_ids = self.store["titles"][indices]
        titles = torch.as_tensor(
            np.array(self.store["features"][title_ids], copy=True), device=device
        )
        values = self.item_mlp(self.item_projection(torch.cat([titles, video], dim=-1)))
        available = (
            self.store["video_offsets"][indices + 1] > self.store["video_offsets"][indices]
        ) | (title_ids != 0)
        return values.masked_fill(~torch.as_tensor(available, device=device)[:, None], 0.0)

    def _calculate(self, ids):
        from torch.utils.checkpoint import checkpoint

        vectors = []
        # Bound graph workspace by both video and node counts. Oversized individual
        # videos are retained intact. Recompute chunk activations during backward.
        chunk, nodes = [], 0
        chunks = []
        for item in ids:
            first, last = self.store["video_offsets"][item - 1 : item + 1]
            count = int(self.store["scene_offsets"][last] - self.store["scene_offsets"][first])
            if chunk and (len(chunk) >= 64 or nodes + count > 8192):
                chunks.append(chunk)
                chunk, nodes = [], 0
            chunk.append(item)
            nodes += count
        if chunk:
            chunks.append(chunk)
        for items in chunks:
            if self.training and torch.is_grad_enabled():
                values = checkpoint(self._calculate_chunk, items, use_reentrant=False)
            else:
                values = self._calculate_chunk(items)
            vectors.append(values)
        return (
            torch.cat(vectors)
            if vectors
            else self.item_projection.weight.new_empty((0, self.embedding_dim))
        )

    def prepare_items(self, ids):
        unique = sorted(set(ids.detach().cpu().reshape(-1).tolist()) - {0})
        self._vectors = torch.cat(
            [
                self.item_projection.weight.new_zeros((1, self.embedding_dim)),
                self._calculate(unique),
            ]
        )
        self._indices = torch.full(
            (self.item_count + 1,), -1, dtype=torch.long, device=self._vectors.device
        )
        self._indices[0] = 0
        self._indices[torch.tensor(unique, dtype=torch.long, device=self._vectors.device)] = (
            torch.arange(1, len(unique) + 1, device=self._vectors.device)
        )

    def item_vectors(self, item_ids):
        if self._vectors is None:
            # Standalone calls, e.g. smoke tests. Training explicitly prepares the union.
            self.prepare_items(item_ids)
        indices = self._indices[item_ids]
        if (indices < 0).any():
            raise RuntimeError("graph item cache does not cover this batch")
        return self._vectors[indices]

    def catalog_vectors(self):
        ids = torch.arange(1, self.item_count + 1, device=self.item_projection.weight.device)
        self.prepare_items(ids)
        return self.item_vectors(ids)


def new_graph_model(context, config, branch, device):
    store = GraphStore(context.representations_dir / f"{branch}_embeddings")
    kwargs = dict(
        item_count=len(store),
        max_length=config.model.max_sequence_length,
        embedding_dim=config.model.embedding_dim,
        num_blocks=config.model.num_blocks,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
    )
    if branch == "meta":
        features = np.array(store["features"][store["titles"]], copy=True)
        return SASRec(**kwargs, arm="metadata", item_features=features).to(device)
    return GraphSASRec(
        **kwargs, store=store, aggregation=context.scene_aggregation, arm="graph"
    ).to(device)
