from copy import deepcopy

import numpy as np
import torch

from helpers import graph_store
from validation.graph_batching import PackedGraphStore
from validation.graph_model import DescriptionSceneEncoder, SceneRoleEncoder


def description_batch():
    return {
        "titles": torch.randn(4, 4), "title_available": torch.tensor([1, 1, 0, 0]),
        "contexts": torch.randn(3, 4), "scene_videos": torch.tensor([0, 0, 2]),
        "available": torch.tensor([1, 1, 1, 0]), "offsets": [0, 2, 2, 3, 3],
    }


def test_d_title_description_residual_mlp_mean_and_missing_components():
    encoder = DescriptionSceneEncoder(4, hidden=2, aggregation="mean")
    batch = description_batch()
    scene_inputs = []
    hook = encoder.scene_mlp.register_forward_pre_hook(lambda module, args: scene_inputs.append(args[0]))
    try:
        output = encoder(batch)
    finally:
        hook.remove()
    title_vectors = encoder.title_projection(batch["titles"]).masked_fill(
        ~batch["title_available"].bool()[:, None], 0)
    torch.testing.assert_close(scene_inputs[0][:, :2], title_vectors[batch["scene_videos"]])
    torch.testing.assert_close(scene_inputs[0][:, 2:], encoder.description_projection(batch["contexts"]))
    scenes = scene_inputs[0] + torch.relu(encoder.scene_mlp.fc1(scene_inputs[0]))
    torch.testing.assert_close(output[0], scenes[:2].mean(0))
    title = encoder.title_projection(batch["titles"])[1]
    torch.testing.assert_close(output[1], encoder.scene_mlp(torch.cat([title, torch.zeros(6)])))
    torch.testing.assert_close(output[3], torch.zeros(8), rtol=0, atol=0)
    # A missing title does not erase a video's available Description scenes.
    assert output[2].abs().sum() > 0
    output.square().sum().backward()
    for module in (encoder.description_projection, encoder.title_projection, encoder.scene_mlp.fc1):
        assert module.weight.grad is not None
        assert module.weight.grad.abs().sum() > 0


def test_worker_consumes_mixed_arm_jobs(context, monkeypatch):
    import pickle
    from queue import Queue
    from helpers import prepare_inputs, result_identity, selected_table
    from validation.leave_one_out_recommendation import combination_complete
    from validation.leave_one_out_workers import _consume_combinations
    from validation.steps import embed_representations

    prepare_inputs(context, monkeypatch)
    embed_representations(context, target=["d"])
    worker_context = pickle.loads(pickle.dumps(context))
    cohort, _, split = selected_table(context)
    jobs, results = Queue(), Queue()
    for seed, arm in ((42, "d"), (43, "g1")):
        jobs.put((split, result_identity(context, cohort, seed, arm), arm))
    jobs.put(None)
    _consume_combinations(worker_context, "cpu", jobs, results)
    responses = [results.get_nowait() for _ in range(results.qsize())]
    assert [status for status, _ in responses] == ["complete", "complete"]
    for _, identity in responses:
        directory = context.recommendations_dir / f"seed_{identity['seed']}" / identity["arm"]
        assert combination_complete(directory, identity, 3)
        assert np.isfinite(np.load(directory / "catalog_vectors.npy")).all()


def test_g0_ignores_roles_and_retains_all_feature_gradients(context):
    arrays, _, _, _ = graph_store(context)
    batch = PackedGraphStore(arrays).batch([0, 1], "cpu")
    encoder = SceneRoleEncoder(hidden=4, role_fusion=False)
    assert encoder.role_fusion is None and not hasattr(encoder, "role_states")
    modified = deepcopy(batch)
    modified["role_entities"] = modified["role_entities"].flip(1)
    modified["role_states"] = 1 - modified["role_states"]
    torch.testing.assert_close(encoder(batch), encoder(modified), rtol=0, atol=0)
    # Neither role tensor is needed by the encoder.
    del modified["role_entities"], modified["role_states"]
    torch.testing.assert_close(encoder(batch), encoder(modified), rtol=0, atol=0)
    for key in ("features", "contexts", "titles"):
        batch[key].requires_grad_()
    encoder(batch).square().sum().backward()
    for ids in (batch["entity_ids"], batch["action_ids"]):
        assert batch["features"].grad[ids].abs().sum() > 0
    assert batch["contexts"].grad.abs().sum() > 0
    assert batch["titles"].grad.abs().sum() > 0
    assert encoder.scene_mlp.fc1.weight.grad.abs().sum() > 0
