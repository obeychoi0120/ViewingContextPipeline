"""Worker-local static topology and vectorized ragged gathers for graph training."""

import numpy as np

from validation.model import torch
from validation.profiling import span, timed, workload

GRAPH_EXECUTION_VERSION = "packed-role-graph/v1"


def ranges(starts, stops):
    """Flatten ragged ranges without Python loops over videos, scenes or nodes."""
    lengths = stops - starts
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    owners = np.repeat(np.arange(len(starts)), lengths)
    indices = np.arange(offsets[-1]) + np.repeat(starts - offsets[:-1], lengths)
    return indices, owners, offsets


class PackedGraphStore:
    def __init__(self, store, feature_cache_mb=2048):
        self.store = store
        self.feature_cache_mb = feature_cache_mb
        self.feature_cache = None
        self.cache_device = None
        self.cache_attempted = False
        self.video_scenes = np.asarray(store["video_offsets"])
        self.video_nodes = np.asarray(store["scene_offsets"][self.video_scenes])
        self.video_edges = np.asarray(store["edge_offsets"][self.video_scenes])
        self.video_missing = np.asarray(store["missing_offsets"][self.video_scenes])
        self.node_scenes = np.repeat(
            np.arange(len(store["contexts"])), np.diff(store["scene_offsets"])
        )
        self.node_counts = np.diff(self.video_nodes)
        self.scene_counts = np.diff(self.video_scenes)
        self.edge_counts = np.diff(self.video_edges)
        # Preserve original order within each relation, as in the old boolean gathers.
        edges, missing = store["edges"], store["missing"]
        self.roles, self.states = [], []
        for role in range(10):
            indices = np.flatnonzero(edges[:, 2] == role)
            self.roles.append(
                (np.array(edges[indices, :2]), np.searchsorted(indices, self.video_edges))
            )
        for role in range(5):
            indices = np.flatnonzero(missing[:, role + 1] >= 0)
            self.states.append(
                (
                    np.array(missing[indices][:, [0, role + 1]]),
                    np.searchsorted(indices, self.video_missing),
                )
            )

    def _features(self, ids, device):
        """Resident fixed vocabulary if affordable; otherwise one bulk gather/copy."""
        device = torch.device(device)
        if self.cache_device != device:
            self.feature_cache = None
            self.cache_device, self.cache_attempted = device, False
        if not self.cache_attempted:
            with span("feature_cache_init"):
                self.cache_attempted = True
                if device.type == "cuda" and self.feature_cache_mb:
                    free, _ = torch.cuda.mem_get_info(device)
                    limit = min(self.feature_cache_mb * 1024**2, int(free * 0.2))
                    if self.store["features"].nbytes <= limit:
                        try:
                            self.feature_cache = torch.as_tensor(
                                np.array(self.store["features"]), device=device
                            )
                        except torch.OutOfMemoryError:
                            self.feature_cache = None
        if self.feature_cache is not None:
            workload(feature_gpu_rows=len(ids))
            with span("feature_gpu_gather"):
                return self.feature_cache[torch.as_tensor(ids, device=device)]
        workload(feature_cpu_rows=len(ids))
        with span("feature_cpu_gather", cuda=False):
            values = np.array(self.store["features"][ids])
        with span("feature_transfer"):
            return torch.as_tensor(values, device=device)

    @timed("graph_batch")
    def batch(self, item_ids, device, *, prepared=True):
        with span("cpu_pack", cuda=False):
            items = np.asarray(item_ids, dtype=np.int64)
            nodes, node_videos, node_offsets = ranges(
                self.video_nodes[items], self.video_nodes[items + 1]
            )
            scenes, scene_videos, offsets = ranges(
                self.video_scenes[items], self.video_scenes[items + 1]
            )
            shift = node_offsets[:-1] - self.video_nodes[items]
            scene_shift = offsets[:-1] - self.video_scenes[items]
            types = np.asarray(self.store["types"][nodes])
            arrays = {
                "types": types,
                "node_scenes": self.node_scenes[nodes] + scene_shift[node_videos],
                "scene_videos": scene_videos,
            }
            if prepared:
                counts = np.zeros(len(nodes), dtype=np.int64)
                for role, (values, bounds) in enumerate(self.roles):
                    indices, owners, _ = ranges(bounds[items], bounds[items + 1])
                    pairs = values[indices] + shift[owners, None]
                    arrays[f"role_{role}"] = pairs
                    counts += np.bincount(pairs[:, 1], minlength=len(nodes))
                for role, (values, bounds) in enumerate(self.states):
                    indices, owners, _ = ranges(bounds[items], bounds[items + 1])
                    pairs = values[indices].copy()
                    pairs[:, 0] += shift[owners]
                    arrays[f"state_{role}"] = pairs
                    counts += np.bincount(pairs[:, 0], minlength=len(nodes))
                arrays.update(
                    message_counts=counts.clip(min=1),
                    entity_ids=np.flatnonzero(types == 0),
                    action_ids=np.flatnonzero(types == 1),
                )
            else:
                indices, owners, _ = ranges(self.video_edges[items], self.video_edges[items + 1])
                edges = np.array(self.store["edges"][indices])
                edges[:, :2] += shift[owners, None]
                indices, owners, _ = ranges(
                    self.video_missing[items], self.video_missing[items + 1]
                )
                missing = np.array(self.store["missing"][indices])
                missing[:, 0] += shift[owners]
                arrays.update(edges=edges, missing=missing)
            titles = self.store["titles"][items]
            arrays["title_available"] = (titles != 0).astype(np.int64)
            arrays["available"] = ((np.diff(offsets) > 0) | (titles != 0)).astype(np.int64)
            # One integer transfer, with views for the small relation/index tensors.
            flat = np.concatenate([a.reshape(-1) for a in arrays.values()])
        with span("index_transfer", cuda=True):
            tensor = torch.as_tensor(flat, device=device)
            batch, offset = {}, 0
            for name, array in arrays.items():
                batch[name] = tensor[offset : offset + array.size].reshape(array.shape)
                offset += array.size
        with span("feature_ids", cuda=False):
            feature_ids = np.concatenate(
                (self.store["nodes"][nodes], self.store["contexts"][scenes], titles)
            )
        values = self._features(feature_ids, device)
        batch.update(
            features=values[: len(nodes)],
            contexts=values[len(nodes) : len(nodes) + len(scenes)],
            titles=values[len(nodes) + len(scenes) :],
            offsets=offsets.tolist(),
        )
        if prepared:
            batch["roles"] = [batch.pop(f"role_{r}") for r in range(10)]
            batch["states"] = [batch.pop(f"state_{r}") for r in range(5)]
        return batch
