"""Deterministic ragged graphs for correctness tests and isolated benchmarks."""

import numpy as np


class ArrayStore:
    def __init__(self, arrays):
        self.arrays = arrays

    def __getitem__(self, name):
        return self.arrays[name]

    def __len__(self):
        return len(self["titles"])


def synthetic_store(videos=64, feature_dim=1024, seed=42):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(max(videos, 256), feature_dim)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    features[0] = 0
    data = {k: [] for k in ("nodes", "types", "contexts", "edges", "missing", "titles")}
    data.update(
        {k: [0] for k in ("scene_offsets", "video_offsets", "edge_offsets", "missing_offsets")}
    )
    for video in range(videos):
        data["titles"].append(0 if video % 7 == 0 else int(rng.integers(1, len(features))))
        for scene in range(0 if video % 11 == 0 else int(rng.integers(1, 10))):
            base = len(data["nodes"])
            entity_count = int(rng.integers(1, 9))
            action_count = 0 if scene % 5 == 0 else int(rng.integers(1, 6))
            data["nodes"].extend(rng.integers(1, len(features), size=entity_count + action_count))
            data["types"].extend([0] * entity_count + [1] * action_count)
            for action in range(action_count):
                node = base + entity_count + action
                states = [node]
                for role in range(5):
                    state = -1 if role == 0 else int(rng.integers(-1, 2))
                    states.append(state)
                    if state == -1:
                        entity = base + int(rng.integers(entity_count))
                        data["edges"].extend([[entity, node, role], [node, entity, role + 5]])
                data["missing"].append(states)
            data["contexts"].append(int(rng.integers(1, len(features))))
            data["scene_offsets"].append(len(data["nodes"]))
            data["edge_offsets"].append(len(data["edges"]))
            data["missing_offsets"].append(len(data["missing"]))
        data["video_offsets"].append(len(data["contexts"]))
    arrays = {k: np.asarray(v, dtype=np.int64) for k, v in data.items()}
    arrays["edges"] = arrays["edges"].reshape(-1, 3)
    arrays["missing"] = arrays["missing"].reshape(-1, 6)
    return ArrayStore({"features": features, **arrays})
