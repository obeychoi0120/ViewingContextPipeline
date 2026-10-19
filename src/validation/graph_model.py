"""Role-aware scene encoding and differentiable video pooling for SASRec."""

import numpy as np

from validation.model import SASRec, SingleLayerResidualMLP, torch, nn
from validation.graph_inputs import GraphStore
from validation.graph_context import (
    ATTENTION_AGGREGATIONS, GRAPH_MODEL, SCENE_AGGREGATIONS, SCENE_EXCHANGES, SCENE_TRANSFORMER,
    TRANSFORMER_VARIANTS,
)
from validation.graph_batching import PackedGraphStore, GRAPH_EXECUTION_VERSION
from validation.config import GraphExecutionConfig
from validation.scene_attention import SceneAttentionGCN, SceneAttentionPool, recorded_weights
from validation.scene_gcn import SceneGCN, scene_groups
from validation.scene_transformer import SceneTransformer
from validation.profiling import span, timed, workload, is_profiling, checkpoint_contexts


def segment_mean(values, indices, count):
    output = values.new_zeros((count, values.shape[-1]))
    output.index_add_(0, indices, values)
    sizes = values.new_zeros(count)
    sizes.index_add_(0, indices, values.new_ones(len(values)))
    return output / sizes.clamp_min(1).unsqueeze(-1)


class SceneRoleEncoder(nn.Module):
    """Entity→Action role gather, scene MLP, optional scene exchange, and video pooling.

    Exchange: cosine GCN, attention GCN or Transformer; pooling: mean or attention.
    """

    ROLE_INDICES = (0, 1, 2, 4)  # actor, target, tool, place; receiver is ignored.

    def __init__(self, feature_dim=1024, hidden=GRAPH_MODEL["hidden_dim"], aggregation="mean",
                 dropout=0.0):
        super().__init__()
        if aggregation not in SCENE_AGGREGATIONS:
            raise ValueError(
                "Scene role encoder supports " + ", ".join(SCENE_AGGREGATIONS) + " pooling"
            )
        self.aggregation = aggregation
        self.hidden_dim = hidden
        self.output_dim = hidden * 4
        self.title_projection = nn.Linear(feature_dim, hidden)
        self.entity_projection = nn.Linear(feature_dim, hidden)
        self.action_projection = nn.Linear(feature_dim, hidden)
        self.context_projection = nn.Linear(feature_dim, hidden)
        self.role_fusion = nn.Linear(hidden * 5, hidden)
        self.role_states = nn.Parameter(torch.empty(4, 2, hidden))
        nn.init.normal_(self.role_states, std=0.02)
        self.scene_mlp = SingleLayerResidualMLP(self.output_dim)
        if aggregation == "gnn":
            self.gcn = SceneGCN(self.output_dim)
        elif aggregation in TRANSFORMER_VARIANTS:
            self.scene_transformer = SceneTransformer(
                self.output_dim, dropout=dropout,
                **SCENE_TRANSFORMER, **TRANSFORMER_VARIANTS[aggregation],
            )
        elif aggregation == "gnn_attn":
            # Same attribute as gnn: the checkpoints differ only by gcn.query/gcn.key.
            self.gcn = SceneAttentionGCN(self.output_dim, hidden * 3, hidden)
        elif aggregation == "attn_pool":
            self.attention_pool = SceneAttentionPool(self.output_dim, hidden)

    def _role_inputs(self, batch, action_ids):
        if "role_entities" in batch:
            return batch["role_entities"], batch["role_states"]
        # Standalone raw-batch API; production prepares these indices on CPU.
        lookup = action_ids.new_full((len(batch["types"]),), -1)
        lookup[action_ids] = torch.arange(len(action_ids), device=action_ids.device)
        entities = action_ids.new_full((len(action_ids), 4), -1)
        states = action_ids.new_zeros((len(action_ids), 4))
        for slot, role in enumerate(self.ROLE_INDICES):
            pairs = batch["edges"][batch["edges"][:, 2] == role, :2]
            entities[lookup[pairs[:, 1]], slot] = pairs[:, 0]
            missing = batch["missing"]
            valid = missing[:, role + 1] >= 0
            states[lookup[missing[valid, 0]], slot] = missing[valid, role + 1]
        return entities, states

    def forward(self, batch):
        video_count = len(batch["offsets"]) - 1
        scene_count = len(batch["contexts"])
        with span("title_projection", cuda=True):
            title = self.title_projection(batch["titles"])
            title = title.masked_fill(~batch["title_available"].bool()[:, None], 0.0)
        scenes = title.new_empty((0, self.output_dim))
        if scene_count:
            entity_ids = batch.get("entity_ids")
            action_ids = batch.get("action_ids")
            if entity_ids is None:
                entity_ids = torch.where(batch["types"] == 0)[0]
                action_ids = torch.where(batch["types"] == 1)[0]
            with span("entity_projection", cuda=True):
                entities = self.entity_projection(batch["features"][entity_ids])
            with span("action_projection", cuda=True):
                actions = self.action_projection(batch["features"][action_ids])
            if len(action_ids):
                with span("role_fusion", cuda=True):
                    role_entities, role_states = self._role_inputs(batch, action_ids)
                    # Node IDs are local to the batch; compact Entity rows differ.
                    lookup = action_ids.new_full((len(batch["types"]),), -1)
                    lookup[entity_ids] = torch.arange(len(entity_ids), device=entity_ids.device)
                    padded_entities = torch.cat([entities.new_zeros((1, self.hidden_dim)), entities])
                    role_vectors = []
                    for role in range(4):
                        ids = role_entities[:, role]
                        rows = lookup[ids.clamp_min(0)] + 1
                        rows = rows.masked_fill(ids < 0, 0)
                        linked = padded_entities[rows]
                        with span("state_lookup", cuda=True):
                            state = nn.functional.embedding(role_states[:, role], self.role_states[role])
                        role_vectors.append(torch.where((ids >= 0)[:, None], linked, state))
                    actions = actions + nn.functional.relu(
                        self.role_fusion(torch.cat([actions, *role_vectors], dim=-1))
                    )
            with span("scene_readout", cuda=True):
                entity_mean = segment_mean(entities, batch["node_scenes"][entity_ids], scene_count)
                action_mean = segment_mean(actions, batch["node_scenes"][action_ids], scene_count)
                with span("context_projection", cuda=True):
                    context = self.context_projection(batch["contexts"])
                scenes = torch.cat([title[batch["scene_videos"]], entity_mean, action_mean, context], dim=-1)
                if self.aggregation in ("gnn", "gnn_attn"):
                    similarity_features = torch.cat([entity_mean, action_mean, context], dim=-1)
            with span("scene_mlp", cuda=True):
                scenes = self.scene_mlp(scenes)
            if self.aggregation in SCENE_EXCHANGES:
                groups = [value for key, value in batch.items() if key.startswith("scene_group_")]
                if not groups:
                    groups = [torch.as_tensor(value, device=scenes.device)
                              for value in scene_groups(batch["offsets"]).values()]
                if self.aggregation in TRANSFORMER_VARIANTS:
                    scenes = self.scene_transformer(scenes, groups)
                else:
                    scenes = self.gcn(scenes, similarity_features, groups)
        with span("video_pooling", cuda=True):
            if self.aggregation == "attn_pool" and scene_count:
                output = self.attention_pool(scenes, batch["scene_videos"], video_count)
            else:
                output = segment_mean(scenes, batch["scene_videos"], video_count)
        # Only videos with no valid scenes use the title-only scene path.
        missing = [i for i in range(video_count) if batch["offsets"][i] == batch["offsets"][i + 1]]
        if missing:
            ids = torch.as_tensor(missing, device=title.device)
            fallback = torch.cat([title[ids], title.new_zeros((len(ids), self.hidden_dim * 3))], dim=-1)
            output = output.index_copy(0, ids, self.scene_mlp(fallback))
        return output.masked_fill(~batch["available"].bool()[:, None], 0.0)


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
            arm="metadata",
            item_features=np.zeros((item_count, 1), dtype=np.float32),
        )
        # The graph tower supplies its own title/video vectors; initialize only
        # the common recommender with the minimal placeholder feature above.
        del self.item_projection
        del self.item_mlp
        self.arm = arm
        if embedding_dim % 4:
            raise ValueError("Graph recommendation dimension must be divisible by 4")
        self.store = store
        self.title_dim = embedding_dim // 4
        self.graph_encoder = SceneRoleEncoder(
            store["features"].shape[1], hidden=self.title_dim, aggregation=aggregation,
            dropout=dropout,
        )
        self.graph_encoder.apply(self._reset_parameters)
        self._reset_attention(self.graph_encoder)
        # Undo the shared Linear reset where attention needs its own start.
        if aggregation == "attn_pool":
            # Uniform scene weights start exactly as mean pooling.
            self.graph_encoder.attention_pool.start_uniform()
        elif aggregation == "gnn_attn":
            self.graph_encoder.gcn.reset_query_key()
        self._vectors = self._indices = None
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
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

    @property
    def title_projection(self):
        return self.graph_encoder.title_projection

    def _calculate_chunk(self, batch):
        # Checkpoint backward recomputes only neural operations on static inputs.
        with span("graph_encoder", cuda=True):
            return self.graph_encoder(batch)

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
        hidden = self.graph_encoder.hidden_dim
        feature = self.store["features"].shape[1]
        # Conservative whole-batch activation estimate, not merely one chunk.
        estimate = 4 * (
            nodes * (feature + hidden * 32)
            + edges * hidden * 4
            + scenes * (feature + self.embedding_dim * 12)
        )
        if self.aggregation in SCENE_EXCHANGES:
            pairs = int(np.square(self.packed.scene_counts[indices]).sum())
        if self.aggregation == "gnn":
            estimate += 4 * (pairs * 8 + scenes * self.embedding_dim * 8)
        elif self.aggregation == "gnn_attn":
            # gnn activations plus query/key projections; scores reuse the adjacency budget.
            estimate += 4 * (pairs * 8 + scenes * (self.embedding_dim * 8 + hidden * 4))
        elif self.aggregation == "attn_pool":
            # Score hidden layer and tanh, weighted scene copies; no pairwise term.
            estimate += 4 * scenes * (hidden * 3 + self.embedding_dim * 2)
        elif self.aggregation in TRANSFORMER_VARIANTS:
            # Per layer: score/softmax/dropout per head; LN, QKV, residual and FFN activations.
            layers, heads, ratio = (SCENE_TRANSFORMER[k] for k in ("layers", "heads", "ffn_ratio"))
            estimate += 4 * layers * (
                pairs * heads * 4 + scenes * self.embedding_dim * (12 + 3 * ratio)
            )
        free, _ = torch.cuda.mem_get_info(device)
        return estimate > min(8 * 1024**3, int(free * 0.35))

    def _calculate(self, ids):
        from torch.utils.checkpoint import checkpoint

        vectors, chunks = [], []
        chunk, nodes, scene_pairs = [], 0, 0
        for item in ids:
            count = int(self.packed.node_counts[item - 1])
            pairs = int(self.packed.scene_counts[item - 1]) ** 2
            if chunk and (
                len(chunk) >= self.execution.chunk_items
                or nodes + count > self.execution.chunk_nodes
                or (self.aggregation in SCENE_EXCHANGES
                    and scene_pairs + pairs > self.execution.chunk_scene_pairs)
            ):
                chunks.append(chunk)
                chunk, nodes, scene_pairs = [], 0, 0
            chunk.append(item)
            nodes += count
            scene_pairs += pairs
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
            batch = self.packed.batch(np.asarray(items) - 1, device,
                                      group_scenes=self.aggregation in SCENE_EXCHANGES)
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
        report = {
            "version": GRAPH_EXECUTION_VERSION,
            **self.execution.model_dump(),
            **self.execution_counts,
            "resident_feature_bytes": (
                self.store["features"].nbytes if self.packed.feature_cache is not None else 0
            ),
        }
        if self.aggregation in TRANSFORMER_VARIANTS:
            # Learned [attention, FFN] scales per layer; 0 means that sublayer stayed unused.
            report["scene_transformer_residual_scales"] = (
                self.graph_encoder.scene_transformer.residual_scales()
            )
        if self.aggregation in ATTENTION_AGGREGATIONS:
            report["scene_attention_summary"] = self.scene_attention_summary()
        return report

    @torch.no_grad()
    def scene_attention_summary(self):
        """Learned scene weights over the whole catalog; all statistics are 1 for uniform weights.

        A separate no-grad pass (no dropout in these encoders) that leaves the item
        cache and execution counters untouched.
        """
        encoder = self.graph_encoder
        module = encoder.attention_pool if self.aggregation == "attn_pool" else encoder.gcn
        device = self.title_projection.weight.device
        items, size = np.arange(self.item_count), self.execution.chunk_items
        with recorded_weights(module) as summary:
            for start in range(0, len(items), size):
                encoder(self.packed.batch(items[start:start + size], device,
                                          group_scenes=self.aggregation in SCENE_EXCHANGES))
        return summary


def new_graph_model(context, config, branch, device):
    store = GraphStore(context.representations_dir / f"{branch}_embeddings")
    if store["features"].shape[1] != config.encoder.embedding_dim:
        raise ValueError("graph feature dimension mismatch; rerun embed-representations")
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
