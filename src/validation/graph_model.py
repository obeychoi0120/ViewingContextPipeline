"""Role-aware scene encoding and differentiable video pooling for SASRec."""

import numpy as np

from validation.model import SASRec, torch, nn
from validation.graph_inputs import GraphStore
from validation.graph_context import GRAPH_MODEL
from validation.graph_batching import PackedGraphStore, GRAPH_EXECUTION_VERSION
from validation.config import GraphExecutionConfig
from validation.profiling import span, timed, workload, is_profiling, checkpoint_contexts


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

    def forward(self, h, edges=None, missing=None, *, roles=None, states=None, counts=None):
        # Compatibility for standalone callers; training supplies CPU-prepared indices.
        if roles is None:
            roles = [edges[edges[:, 2] == role, :2] for role in range(10)]
            states = [missing[missing[:, role + 1] >= 0][:, [0, role + 1]] for role in range(5)]
        messages = torch.zeros_like(h)
        sizes = h.new_zeros(len(h)) if counts is None else counts.to(h.dtype)
        for transform, pairs in zip(self.relations, roles, strict=True):
            messages.index_add_(0, pairs[:, 1], transform(h[pairs[:, 0]]))
            if counts is None:
                sizes.index_add_(0, pairs[:, 1], h.new_ones(len(pairs)))
        for role, pairs in enumerate(states):
            messages.index_add_(0, pairs[:, 0], self.states[role, pairs[:, 1]])
            if counts is None:
                sizes.index_add_(0, pairs[:, 0], h.new_ones(len(pairs)))
        return self.norm(
            h
            + torch.nn.functional.gelu(
                self.self_projection(h) + messages / sizes.clamp_min(1).unsqueeze(-1)
            )
        )


class RoleGraphEncoder(nn.Module):
    def __init__(self, feature_dim=1024, hidden=GRAPH_MODEL["hidden_dim"], aggregation="mean"):
        super().__init__()
        if aggregation not in ("mean", "attention"):
            raise ValueError("unknown scene aggregation")
        self.aggregation = aggregation
        self.output_dim = output_dim = hidden * 3
        self.projection = nn.Linear(feature_dim, hidden)
        self.node_type = nn.Embedding(2, hidden)
        self.layers = nn.ModuleList([RoleLayer(hidden) for _ in range(GRAPH_MODEL["layers"])])
        self.readout = nn.Identity()
        self.attention = (
            nn.Sequential(nn.Linear(output_dim, 128), nn.Tanh(), nn.Linear(128, 1))
            if aggregation == "attention"
            else None
        )

    def forward(self, batch):
        if not len(batch["contexts"]):
            # Match the old constant-zero branch: unused parameters must keep
            # grad=None, so AdamW does not decay them on fully missing batches.
            return batch["features"].new_zeros((len(batch["offsets"]) - 1, self.output_dim))
        with span("node_projection", cuda=True):
            with span("node_linear", cuda=True):
                h = self.projection(batch["features"])
            h = h + self.node_type(batch["types"])
        prepared = "roles" in batch
        with span("message_passing", cuda=True):
            for layer in self.layers:
                if prepared:
                    h = layer(
                        h,
                        roles=batch["roles"],
                        states=batch["states"],
                        counts=batch["message_counts"],
                    )
                else:
                    h = layer(h, batch["edges"], batch["missing"])
        with span("scene_readout", cuda=True):
            count = len(batch["contexts"])
            groups = (
                [batch["entity_ids"], batch["action_ids"]]
                if prepared
                else [torch.where(batch["types"] == t)[0] for t in (0, 1)]
            )
            pooled = [
                segment_mean(h[index], batch["node_scenes"][index], count) for index in groups
            ]
            with span("context_projection", cuda=True):
                context = self.projection(batch["contexts"])
            scenes = self.readout(torch.cat([*pooled, context], dim=-1))
        with span("video_pooling", cuda=True):
            video_ids, video_count = batch["scene_videos"], len(batch["offsets"]) - 1
            if self.attention is None:
                return segment_mean(scenes, video_ids, video_count)
            scores = self.attention(scenes).squeeze(-1)
            maxima = scores.new_full((video_count,), -torch.inf)
            maxima.scatter_reduce_(0, video_ids, scores.detach(), reduce="amax", include_self=True)
            weights = torch.exp(scores - maxima[video_ids])
            totals = weights.new_zeros(video_count).index_add_(0, video_ids, weights)
            weights = weights / totals[video_ids].clamp_min(torch.finfo(weights.dtype).tiny)
            return scenes.new_zeros((video_count, self.output_dim)).index_add_(
                0, video_ids, scenes * weights[:, None]
            )


def graph_batch(store, item_ids, device):
    """Standalone mutable batch API; model training reuses a PackedGraphStore."""
    return PackedGraphStore(store, feature_cache_mb=0).batch(item_ids, device, prepared=False)


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
        execution=None,
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
        if embedding_dim % 4:
            raise ValueError("Graph recommendation dimension must be divisible by 4")
        self.store = store
        self.title_dim = embedding_dim // 4
        self.graph_encoder = RoleGraphEncoder(
            store["features"].shape[1], hidden=self.title_dim, aggregation=aggregation
        )
        # Production: [title 128; video 384] already matches SASRec's 512.
        del self.video_projection
        self.title_projection = nn.Linear(store["features"].shape[1], self.title_dim)
        self._vectors = self._indices = None
        self.embedding_dim = embedding_dim
        self.execution = execution or GraphExecutionConfig()
        self.packed = PackedGraphStore(store, self.execution.feature_cache_mb)
        self.execution_counts = {"checkpoint_batches": 0, "direct_batches": 0, "chunks": 0}
        self._catalog_cached = False

    def train(self, mode=True):
        self.clear_item_cache()
        return super().train(mode)

    def clear_item_cache(self):
        self._vectors = self._indices = None
        self._catalog_cached = False

    def _calculate_chunk(self, batch):
        # Only differentiable operations are recomputed by checkpoint backward.
        with span("graph_encoder", cuda=True):
            video = self.graph_encoder(batch)
        with span("item_fusion", cuda=True):
            with span("title_projection", cuda=True):
                title = self.title_projection(batch["titles"])
            # Biases and LayerNorm must not invent a title for a missing input.
            title = title.masked_fill(~batch["title_available"].bool()[:, None], 0.0)
            values = self.item_norm(torch.cat([title, video], dim=-1))
            return values.masked_fill(~batch["available"].bool()[:, None], 0.0)

    def _use_checkpoint(self, items):
        if not self.training or not torch.is_grad_enabled():
            return False
        if self.execution.checkpoint != "auto":
            return self.execution.checkpoint == "always"
        device = self.title_projection.weight.device
        if device.type != "cuda":
            return False
        indices = np.asarray(items, dtype=np.int64) - 1
        nodes = int(self.packed.node_counts[indices].sum())
        edges = int(self.packed.edge_counts[indices].sum())
        scenes = int(self.packed.scene_counts[indices].sum())
        hidden = self.graph_encoder.projection.out_features
        feature = self.store["features"].shape[1]
        layers = len(self.graph_encoder.layers)
        # Conservative whole-batch activation estimate, not merely one chunk.
        estimate = 4 * (
            nodes * (feature + hidden * 24 * layers)
            + edges * hidden * 4 * layers
            + scenes * (feature + self.embedding_dim * 4)
        )
        free, _ = torch.cuda.mem_get_info(device)
        return estimate > min(8 * 1024**3, int(free * 0.35))

    def _calculate(self, ids):
        from torch.utils.checkpoint import checkpoint

        vectors, chunks = [], []
        chunk, nodes = [], 0
        for item in ids:
            count = int(self.packed.node_counts[item - 1])
            if chunk and (
                len(chunk) >= self.execution.chunk_items
                or nodes + count > self.execution.chunk_nodes
            ):
                chunks.append(chunk)
                chunk, nodes = [], 0
            chunk.append(item)
            nodes += count
        if chunk:
            chunks.append(chunk)
        use_checkpoint = self._use_checkpoint(ids)
        if is_profiling():
            selected = np.asarray(ids, dtype=np.int64) - 1
            workload(
                unique_videos=len(ids),
                scenes=self.packed.scene_counts[selected].sum(),
                nodes=self.packed.node_counts[selected].sum(),
                edges=self.packed.edge_counts[selected].sum(),
                chunks=len(chunks),
                checkpoint_batches=int(use_checkpoint),
            )
        if self.training:
            self.execution_counts["checkpoint_batches" if use_checkpoint else "direct_batches"] += 1
        device = self.title_projection.weight.device
        for items in chunks:
            batch = self.packed.batch(np.asarray(items) - 1, device)
            self.execution_counts["chunks"] += 1
            if use_checkpoint:
                options = {"context_fn": checkpoint_contexts} if is_profiling() else {}
                values = checkpoint(self._calculate_chunk, batch, use_reentrant=False, **options)
            else:
                values = self._calculate_chunk(batch)
            vectors.append(values)
        return (
            torch.cat(vectors)
            if vectors
            else self.title_projection.weight.new_empty((0, self.embedding_dim))
        )

    @timed("graph_items")
    def prepare_items(self, ids):
        with span("unique_ids", cuda=False):
            if isinstance(ids, torch.Tensor):
                ids = ids.detach().cpu().numpy()
            unique = np.unique(np.asarray(ids, dtype=np.int64))
            unique = unique[unique != 0]
            if len(unique) and (unique[0] < 1 or unique[-1] > self.item_count):
                raise ValueError("graph item ID outside catalog")
        self._vectors = torch.cat(
            [
                self.title_projection.weight.new_zeros((1, self.embedding_dim)),
                self._calculate(unique),
            ]
        )
        # Build the lookup in CPU memory once; transfer it in a single operation.
        with span("item_lookup", cuda=True):
            indices = np.full(self.item_count + 1, -1, dtype=np.int64)
            indices[0] = 0
            indices[unique] = np.arange(1, len(unique) + 1)
            self._indices = torch.as_tensor(indices, device=self._vectors.device)
            self._catalog_cached = len(unique) == self.item_count

    def item_vectors(self, item_ids):
        if self._vectors is None:
            self.prepare_items(item_ids)
        indices = self._indices[item_ids]
        if not self._catalog_cached:
            # Check on device, avoiding a host synchronization for every history/target lookup.
            torch._assert_async((indices >= 0).all(), "graph item cache does not cover this batch")
        return self._vectors[indices]

    def catalog_vectors(self):
        if not self._catalog_cached:
            self.prepare_items(np.arange(1, self.item_count + 1))
        return self._vectors[1:]

    def execution_report(self):
        return {
            "version": GRAPH_EXECUTION_VERSION,
            **self.execution.model_dump(),
            **self.execution_counts,
            "resident_feature_bytes": (
                self.store["features"].nbytes if self.packed.feature_cache is not None else 0
            ),
        }


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
        return SASRec(
            **kwargs,
            arm="metadata",
            item_features={
                "title_values": features,
                "video_values": np.zeros_like(features),
                "title_available": np.asarray(store["titles"]) != 0,
                "video_available": np.zeros(len(store), dtype=bool),
            },
        ).to(device)
    return GraphSASRec(
        **kwargs,
        store=store,
        aggregation=context.scene_aggregation,
        arm="graph",
        execution=config.graph_execution,
    ).to(device)
