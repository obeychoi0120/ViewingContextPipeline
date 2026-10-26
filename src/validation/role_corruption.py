"""A deterministic read-only-source view with scene-local role endpoint corruption."""

import hashlib
import json

import numpy as np

CORRUPTION = {
    "version": "scene-local-alternative-entity/v1",
    "seed": 42,
    "roles": ["actor", "target", "tool", "place"],
    "missing": "preserve",
    "schedule": "fixed_across_training_seeds_epochs_and_phases",
}


class CorruptedGraphStore:
    def __init__(self, source):
        self.arrays = dict(source.arrays)
        edges = self.arrays["edges"] = np.array(source["edges"], copy=True)
        total = changed = impossible = 0
        for scene, key in enumerate(source["scene_keys"]):
            start, stop = source["scene_offsets"][scene:scene + 2]
            entities = np.arange(start, stop)[source["types"][start:stop] == 0]
            first, last = source["edge_offsets"][scene:scene + 2]
            for index in range(first, last):
                entity, action, role = source["edges"][index]
                if role not in (0, 1, 2, 4):
                    continue
                total += 1
                choices = entities[entities != entity]
                if not len(choices):
                    impossible += 1
                    continue
                identity = json.dumps(
                    [CORRUPTION["seed"], str(key), str(source["node_keys"][action]), int(role)]
                ).encode()
                seed = int.from_bytes(hashlib.sha256(identity).digest(), "big")
                replacement = choices[np.random.default_rng(seed).integers(len(choices))]
                # The source builder publishes a forward/reverse pair together.
                if index + 1 >= last or not np.array_equal(
                    source["edges"][index + 1], [action, entity, role + 5]
                ):
                    raise ValueError("invalid forward/reverse role pair")
                edges[index, 0] = replacement
                edges[index + 1, 1] = replacement
                changed += 1
        self.report = {
            **CORRUPTION,
            "total_links": total,
            "changed_links": changed,
            "changed_ratio": changed / total if total else 0.0,
            "impossible_links": impossible,
        }

    def __getitem__(self, name):
        return self.arrays[name]

    def __len__(self):
        return len(self["titles"])
