from copy import deepcopy

import numpy as np
import pytest
import torch

from arm_registry import registry
from extraction.scene_storage import write_scene_records
from helpers import FakeEncoder, graph, graph_store, scene_row
from validation.graph_batching import PackedGraphStore
from validation.graph_inputs import build_input
from validation.graph_model import DescriptionSceneEncoder, SceneRoleEncoder
from validation.role_corruption import CORRUPTION, CorruptedGraphStore
from validation.scene_attention import SceneAttentionGCN


class ArrayStore:
    def __init__(self, arrays):
        self.arrays = arrays

    def __getitem__(self, name):
        return self.arrays[name]

    def __len__(self):
        return len(self["titles"])


def test_g1a_ignores_roles_but_keeps_entity_action_and_context(context):
    arrays, _, _, _ = graph_store(context)
    batch = PackedGraphStore(arrays).batch([0, 1], "cpu")
    encoder = SceneRoleEncoder(hidden=4, aggregation="gnn_attn", role_fusion=False)
    assert encoder.role_fusion is None and not hasattr(encoder, "role_states")
    modified = deepcopy(batch)
    modified["role_entities"] = modified["role_entities"].flip(1)
    modified["role_states"] = 1 - modified["role_states"]
    torch.testing.assert_close(encoder(batch), encoder(modified), rtol=0, atol=0)
    batch["features"].requires_grad_()
    batch["contexts"].requires_grad_()
    encoder(batch).square().sum().backward()
    assert batch["features"].grad[batch["entity_ids"]].abs().sum() > 0
    assert batch["features"].grad[batch["action_ids"]].abs().sum() > 0
    assert batch["contexts"].grad.abs().sum() > 0


def test_g1b_corruption_is_fixed_local_and_preserves_missing(context):
    arrays, cohort, _, _ = graph_store(context, counts=(2, 1, 0))
    one = graph()
    one["entities"] = [one["entities"][0]]
    one["actions"][0].update(target="unknown", tool=None, place=None)
    cid = cohort["catalog"][1]["content_id"]
    write_scene_records(context.scene_arm_dir("graph_qwen3vl") / f"{cid}.jsonl", [scene_row(cid, value=one)])
    texts, arrays, _ = build_input(context, cohort, registry(context.config)["g1"])
    arrays["features"] = FakeEncoder().encode(texts)
    original = deepcopy(arrays)
    first = CorruptedGraphStore(ArrayStore(arrays))
    np.random.seed(999)
    torch.manual_seed(888)
    second = CorruptedGraphStore(ArrayStore(arrays))
    np.testing.assert_array_equal(first["edges"], second["edges"])
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], original[name])
    assert first.report["seed"] == 42
    assert first.report["changed_links"] == 6
    assert first.report["impossible_links"] == 1
    np.testing.assert_array_equal(first["missing"], arrays["missing"])
    for scene in range(len(arrays["contexts"])):
        start, stop = arrays["scene_offsets"][scene:scene + 2]
        a, b = arrays["edge_offsets"][scene:scene + 2]
        for index in range(a, b, 2):
            entity, action, role = first["edges"][index]
            assert start <= entity < stop and start <= action < stop
            assert arrays["types"][entity] == 0 and arrays["types"][action] == 1
            np.testing.assert_array_equal(first["edges"][index + 1], [action, entity, role + 5])
            if scene == 0 or scene == 1:
                assert entity != arrays["edges"][index, 0]
    encoder = SceneRoleEncoder(hidden=4, aggregation="gnn_attn")
    packed = PackedGraphStore(first, feature_cache_mb=0)
    together = encoder(packed.batch([0, 1, 2], "cpu"))
    separate = torch.cat([encoder(packed.batch([i], "cpu")) for i in (0, 1, 2)])
    reordered = encoder(packed.batch([2, 0, 1], "cpu"))
    torch.testing.assert_close(together, separate, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reordered, together[[2, 0, 1]], rtol=1e-5, atol=1e-6)


def test_g1b_policy_changes_recommendation_identity(context, monkeypatch):
    from helpers import prepare_inputs
    from validation.representation_provenance import recommendation_identity

    prepare_inputs(context, monkeypatch)
    original = recommendation_identity(context, "g1b")
    g1 = recommendation_identity(context, "g1")
    monkeypatch.setitem(CORRUPTION, "seed", 43)
    changed = recommendation_identity(context, "g1b")
    assert original != changed and original["embedding_hash"] == changed["embedding_hash"]
    assert recommendation_identity(context, "g1") == g1


def test_g1c_matches_self_transform_and_blocks_exchange():
    global_module = SceneAttentionGCN(4, 3, 2)
    local = SceneAttentionGCN(4, 3, 2, self_only=True)
    local.load_state_dict(global_module.state_dict())
    scenes = torch.randn(3, 4)
    features = torch.randn(3, 3)
    groups = [torch.tensor([[0, 1, 2]])]
    actual = local(scenes, features, groups)
    torch.testing.assert_close(actual, scenes + torch.relu(local.projection(scenes)))
    changed = scenes.clone()
    changed[1] += 20
    changed_features = features.clone()
    changed_features[1] -= 30
    torch.testing.assert_close(local(changed, changed_features, groups)[[0, 2]], actual[[0, 2]])
    assert not torch.allclose(global_module(changed, changed_features, groups)[0],
                              global_module(scenes, features, groups)[0])
    actual.square().sum().backward()
    assert local.projection.weight.grad.abs().sum() > 0
    assert local.query.weight.grad is local.key.weight.grad is None


def test_g1c_single_scene_matches_g1(context):
    arrays, _, _, _ = graph_store(context, counts=(1,))
    batch = PackedGraphStore(arrays).batch([0], "cpu")
    g1 = SceneRoleEncoder(hidden=4, aggregation="gnn_attn")
    g1c = SceneRoleEncoder(hidden=4, aggregation="self_only")
    g1c.load_state_dict(g1.state_dict())
    torch.testing.assert_close(g1(batch), g1c(batch), rtol=0, atol=0)


@pytest.mark.parametrize("aggregation", ["mean", "gnn_attn"])
def test_description_mean_and_missing_components(aggregation):
    encoder = DescriptionSceneEncoder(4, hidden=2, aggregation=aggregation)
    batch = {
        "titles": torch.randn(3, 4), "title_available": torch.tensor([1, 1, 0]),
        "contexts": torch.randn(2, 4), "scene_videos": torch.tensor([0, 0]),
        "available": torch.tensor([1, 1, 0]), "offsets": [0, 2, 2, 2],
    }
    scenes = []
    hook = encoder.scene_mlp.register_forward_hook(lambda module, args, out: scenes.append(out))
    try:
        output = encoder(batch)
    finally:
        hook.remove()
    if aggregation == "mean":
        torch.testing.assert_close(output[0], scenes[0].mean(0))
    # Missing scenes use title + zeros before the scene MLP; fully absent inputs stay zero.
    title = encoder.title_projection(batch["titles"])[1]
    expected = encoder.scene_mlp(torch.cat([title, torch.zeros(6)]))
    torch.testing.assert_close(output[1], expected)
    torch.testing.assert_close(output[2], torch.zeros(8), rtol=0, atol=0)
    output.square().sum().backward()
    assert encoder.description_projection.weight.grad.abs().sum() > 0


def test_worker_consumes_mixed_arm_jobs(context, monkeypatch):
    import pickle
    from queue import Queue
    from helpers import prepare_inputs, result_identity, selected_table
    from validation.leave_one_out_recommendation import combination_complete
    from validation.leave_one_out_workers import _consume_combinations
    from validation.steps import embed_representations

    prepare_inputs(context, monkeypatch)
    embed_representations(context, target=["d1"])
    # The same serializable context is used for every Arm in a spawned worker.
    worker_context = pickle.loads(pickle.dumps(context))
    cohort, _, split = selected_table(context)
    jobs, results = Queue(), Queue()
    for seed, arm in ((42, "d1"), (43, "g1c")):
        jobs.put((split, result_identity(context, cohort, seed, arm), arm))
    jobs.put(None)
    _consume_combinations(worker_context, "cpu", jobs, results)
    responses = [results.get_nowait() for _ in range(results.qsize())]
    assert [status for status, _ in responses] == ["complete", "complete"]
    for _, identity in responses:
        directory = context.recommendations_dir / f"seed_{identity['seed']}" / identity["arm"]
        assert combination_complete(directory, identity, 3)
