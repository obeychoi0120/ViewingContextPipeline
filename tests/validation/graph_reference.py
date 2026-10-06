"""Independent raw graph packing oracle for current encoder tests."""

import numpy as np
from validation.model import torch


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
