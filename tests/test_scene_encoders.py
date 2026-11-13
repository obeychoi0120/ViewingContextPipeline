"""Mean/hypergraph Arms, with Role Fusion only in G2."""

from copy import deepcopy

import numpy as np
import torch

from test_scene_extensions import make_model, synthetic_store
from validation.config import GraphExecutionConfig
from validation.graph_batching import PackedGraphStore
from validation.graph_model import SceneRoleEncoder


def _check_scene_order_and_video_isolation_with_hypergraph():
    store = synthetic_store(videos=12)
    packed = PackedGraphStore(store)
    encoder = SceneRoleEncoder(16, hidden=4, extensions=("segments",))
    with torch.no_grad():
        encoder.segment_gates.fill_(0.5)
    batch = packed.batch(np.arange(len(store)), "cpu", scene_segments=True)
    expected = encoder(batch)
    separate = torch.cat([encoder(packed.batch([video], "cpu", scene_segments=True))
                          for video in range(len(store))])
    torch.testing.assert_close(separate, expected, rtol=1e-5, atol=1e-6)
    permutation = torch.cat([torch.arange(stop - 1, start - 1, -1)
                             for start, stop in zip(batch["offsets"][:-1], batch["offsets"][1:])])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    reordered = deepcopy(batch)
    reordered["contexts"] = batch["contexts"][permutation]
    reordered["node_scenes"] = inverse[batch["node_scenes"]]
    reordered["segment_scenes"] = inverse[batch["segment_scenes"]]
    torch.testing.assert_close(encoder(reordered), expected, rtol=1e-5, atol=1e-6)
    modified = deepcopy(batch)
    modified["titles"][1] += 100
    changed = encoder(modified)
    others = torch.arange(len(store)) != 1
    torch.testing.assert_close(changed[others], expected[others], rtol=0, atol=0)
    assert not torch.allclose(changed[1], expected[1])


def _check_legacy_scene_pair_limit_is_ignored_and_checkpoint_preserves_gradients():
    store = synthetic_store(videos=12)
    reference = make_model(store, ("segments",), items=12, role_fusion=False).train()
    with torch.no_grad():
        reference.graph_encoder.segment_gates.fill_(0.5)
    limited = make_model(store, ("segments",), items=12, role_fusion=False).train()
    limited.load_state_dict(reference.state_dict())
    limited.execution = GraphExecutionConfig(chunk_items=3, chunk_nodes=100000,
                                            chunk_scene_pairs=1, checkpoint="always", feature_cache_mb=0)
    expected, actual = reference.catalog_vectors(), limited.catalog_vectors()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for (name, left), (_, right) in zip(reference.named_parameters(), limited.named_parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, rtol=1e-4, atol=1e-4, msg=name)
    assert limited.execution_counts["chunks"] == 4
    assert reference.execution_counts["chunks"] == 1
    assert limited.execution_counts["checkpoint_batches"] == 1


def test_scene_encoder_isolation_and_chunked_checkpoint():
    # Scene order and video isolation with hypergraph.
    _check_scene_order_and_video_isolation_with_hypergraph()
    # Legacy scene pair limit is ignored and checkpoint preserves gradients.
    _check_legacy_scene_pair_limit_is_ignored_and_checkpoint_preserves_gradients()
