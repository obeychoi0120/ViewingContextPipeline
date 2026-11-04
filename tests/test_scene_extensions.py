"""G1/G2 activity hypergraph (A): numerical behavior, batching and source sidecars."""

import json

import numpy as np
import pytest
import torch

from artifact_io import write_json
from extraction.scene_storage import write_scene_records
from helpers import graph, scene_row
from validation.config import GraphExecutionConfig
from arm_registry import registry
from validation.graph_context import EXTENSION_MODELS
from validation.graph_inputs import load_scene_segments, scene_segments_paths, write_scene_segments
from validation.graph_model import GraphSASRec, SceneRoleEncoder
from validation.representation_provenance import recommendation_identity
from validation.scene_segments import activity_key, attach, segments

pytestmark = pytest.mark.torch


class ArrayStore:
    def __init__(self, arrays):
        self.arrays = arrays
        self.scene_segments = None

    def __getitem__(self, name):
        return self.arrays[name]

    def __len__(self):
        return len(self["titles"])


def synthetic_store(videos=24, feature_dim=16, seed=3):
    """Ragged role graphs with title-only and scene-free videos, plus random segment runs."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(64, feature_dim)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    features[0] = 0
    data = {k: [] for k in ("nodes", "types", "contexts", "edges", "missing", "titles")}
    data.update({k: [0] for k in ("scene_offsets", "video_offsets", "edge_offsets", "missing_offsets")})
    for video in range(videos):
        data["titles"].append(0 if video % 7 == 0 else int(rng.integers(1, len(features))))
        for _ in range(0 if video % 11 == 0 else int(rng.integers(1, 6))):
            base = len(data["nodes"])
            entities, actions = int(rng.integers(1, 5)), int(rng.integers(0, 4))
            data["nodes"].extend(rng.integers(1, len(features), size=entities + actions))
            data["types"].extend([0] * entities + [1] * actions)
            for action in range(actions):
                node = base + entities + action
                states = [node]
                for role in range(5):
                    state = -1 if role == 0 else int(rng.integers(-1, 2))
                    states.append(state)
                    if state == -1:
                        entity = base + int(rng.integers(entities))
                        data["edges"].extend([[entity, node, role], [node, entity, role + 5]])
                data["missing"].append(states)
            data["contexts"].append(int(rng.integers(1, len(features))))
            for key, source in (("scene_offsets", "nodes"), ("edge_offsets", "edges"), ("missing_offsets", "missing")):
                data[key].append(len(data[source]))
        data["video_offsets"].append(len(data["contexts"]))
    arrays = {k: np.asarray(v, dtype=np.int64) for k, v in data.items()}
    arrays["edges"] = arrays["edges"].reshape(-1, 3)
    arrays["missing"] = arrays["missing"].reshape(-1, 6)
    store = ArrayStore({"features": features, **arrays})
    offsets, pairs, segment = arrays["video_offsets"], [], 0
    for video in range(videos):
        start, stop = offsets[video], offsets[video + 1]
        for _ in range(int(rng.integers(0, 3)) if stop > start else 0):
            first = int(rng.integers(start, stop))
            pairs += [[s, segment] for s in range(first, int(rng.integers(first, stop)) + 1)]
            segment += 1
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    store.scene_segments = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return store


def make_model(store, extensions=(), items=8, aggregation="mean", role_fusion=True):
    torch.manual_seed(0)
    return GraphSASRec(
        store=store, aggregation=aggregation, item_count=len(store), max_length=10, embedding_dim=32,
        num_blocks=1, num_heads=2, dropout=0.0, arm="g2", role_fusion=role_fusion, extensions=extensions,
        execution=GraphExecutionConfig(chunk_items=items, chunk_nodes=100000, chunk_scene_pairs=1048576,
                                       checkpoint="never", feature_cache_mb=0),
    ).eval()




@pytest.mark.parametrize("role_fusion", [False, True])
def test_segments_start_as_matching_backbone_and_leave_the_rng_alone(role_fusion):
    store = synthetic_store()
    reference = make_model(store, role_fusion=role_fusion)
    expected_rng = torch.rand(3)
    model = make_model(store, ("segments",), role_fusion=role_fusion)
    assert torch.equal(torch.rand(3), expected_rng)
    shared = reference.state_dict()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in shared.items())
    with torch.no_grad():
        assert torch.equal(reference.catalog_vectors(), model.catalog_vectors())
    assert model.execution_report()["segment_gates"] == {"scene_update": 0.0, "video_readout": 0.0}


def test_open_segment_gates_change_only_videos_with_segments_and_membership_matters():
    store = synthetic_store()
    reference, model = make_model(store), make_model(store, ("segments",))
    with torch.no_grad():
        model.graph_encoder.segment_gates.copy_(torch.tensor([0.8, 0.6]))
        expected, actual = reference.catalog_vectors(), model.catalog_vectors()
    offsets = np.asarray(store["video_offsets"])
    scene_videos = np.repeat(np.arange(len(store)), np.diff(offsets))
    segmented = set(scene_videos[store.scene_segments[:, 0]].tolist())
    for video in range(len(store)):
        assert torch.equal(actual[video], expected[video]) == (video not in segmented)
    moved = store.scene_segments.copy()
    video = scene_videos[moved[:, 0]]
    moved[:, 0] = offsets[video] + offsets[video + 1] - 1 - moved[:, 0]  # reverse scenes inside each video
    store.scene_segments = moved[np.lexsort((moved[:, 1], moved[:, 0]))]
    permuted = make_model(store, ("segments",))
    permuted.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert not torch.allclose(permuted.catalog_vectors(), actual, atol=1e-6)


def test_segment_readout_weights_each_segment_equally():
    store = synthetic_store()
    offsets = np.asarray(store["video_offsets"])
    video = next(v for v in range(len(store)) if offsets[v + 1] - offsets[v] >= 3)
    first = int(offsets[video])
    store.scene_segments = np.array([[first, 0], [first + 1, 0], [first + 2, 0], [first + 2, 1]], np.int64)
    model = make_model(store, ("segments",))
    encoder = model.graph_encoder
    batch = model.packed.batch(np.array([video]), "cpu", scene_segments=True)
    scenes = torch.randn(int(offsets[video + 1] - offsets[video]), encoder.output_dim)
    with torch.no_grad():
        encoder.segment_gates.copy_(torch.tensor([0.0, 1.0]))
        _, readout = encoder._segments(batch, scenes, 1)
        units = []
        for members in ([0, 1, 2], [2]):
            length = torch.tensor([np.log(len(members)), len(members) / len(scenes)], dtype=torch.float32)
            hidden = torch.cat([scenes[members].mean(0), length])
            units.append(torch.relu(encoder.segment_weight @ hidden + encoder.segment_bias))
        expected = encoder.segment_video_weight @ torch.stack(units).mean(0)
    torch.testing.assert_close(readout[0], expected, rtol=1e-5, atol=1e-5)




def test_extensions_train_and_ignore_chunking():
    extensions, aggregation = ("segments",), "mean"
    store = synthetic_store(videos=30)
    model = make_model(store, extensions, items=30, aggregation=aggregation)
    if "segments" in extensions:
        with torch.no_grad():
            model.graph_encoder.segment_gates.copy_(torch.tensor([0.5, 0.5]))
    small = make_model(store, extensions, items=3, aggregation=aggregation)
    small.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(small.catalog_vectors(), model.catalog_vectors(), rtol=1e-5, atol=1e-5)
    model.train()
    model.catalog_vectors().sum().backward()
    for name in ("segment_weight", "segment_scene_weight", "segment_video_weight", "segment_gates"):
        grad = getattr(model.graph_encoder, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_segments_require_the_sidecar():
    store = synthetic_store(videos=4)
    store.scene_segments = None
    with pytest.raises(ValueError, match="scene_segments"):
        make_model(store, ("segments",))
    with pytest.raises(ValueError, match="unknown"):
        SceneRoleEncoder(hidden=4, aggregation="mean", extensions=("video_node",))


def test_sidecar_variants_round_trip_validation_and_staleness(tmp_path):
    store = synthetic_store(videos=8)
    for name, array in store.arrays.items():
        np.save(tmp_path / f"{name}.npy", array)
    (tmp_path / "manifest.json").write_text(json.dumps({"input_hash": "abc"}))
    pairs = store.scene_segments
    with pytest.raises(ValueError, match="missing"):
        load_scene_segments(tmp_path)
    write_scene_segments(tmp_path, pairs[::-1], statistics={"note": "test"})
    assert np.array_equal(load_scene_segments(tmp_path), pairs)
    assert scene_segments_paths(tmp_path)[0].name == "scene_segments.npy"
    with pytest.raises(ValueError, match="only once"):
        write_scene_segments(tmp_path, np.vstack([pairs, pairs[:1]]))
    offsets = np.asarray(store["video_offsets"])
    filled = [v for v in range(len(offsets) - 1) if offsets[v + 1] > offsets[v]]
    with pytest.raises(ValueError, match="one video"):
        write_scene_segments(tmp_path, np.array([[offsets[filled[0]], 9], [offsets[filled[1]], 9]]))
    (tmp_path / "manifest.json").write_text(json.dumps({"input_hash": "changed"}))
    with pytest.raises(ValueError, match="stale"):
        load_scene_segments(tmp_path)


def test_activity_keys_and_runs():
    forms = {"attacking": "attack", "attacks": "attack", "being attacked": "attack", "looking at p2": "look",
             "running": "run", "making": "mak", "makes": "mak", "cries": "cry", "lying": "li", "unknown": None}
    assert {text: activity_key(text) for text in forms} == forms
    scenes = [(0, {"cut", "talk"}), (1, {"cut"}), (3, {"cut"}), (4, {"talk"}), (5, {"talk"})]
    assert segments(scenes) == [("cut", [0, 1]), ("cut", [2]), ("talk", [0]), ("talk", [3, 4])]


def test_attach_builds_segments_and_identity_records_them(context, monkeypatch):
    from validation.graph_inputs import build_input

    catalog = [{"item_id": "1", "content_id": "video_0"}, {"item_id": "2", "content_id": "video_1"}]
    cohort = {"catalog": catalog, "metadata_titles": [{**r, "title": f"title {i}"} for i, r in enumerate(catalog)]}
    verbs = [["cutting"], ["cuts", "talking"], ["talking"], ["eating"]]
    for item, indices in zip(catalog, ([0, 1, 2], [0])):
        rows = []
        for index in indices:
            value = graph()
            phrases = verbs[index if item["content_id"] == "video_0" else 3]
            value["actions"] = [dict(value["actions"][0], action=p) for p in phrases]
            rows.append(scene_row(item["content_id"], index, value=value))
        write_scene_records(context.scene_arm_dir("graph_qwen3vl") / f"{item['content_id']}.jsonl", rows)
    texts, arrays, stats = build_input(context, cohort, registry(context.config)["g2"])
    bundle = context.representations_dir / "graph_qwen3vl"
    bundle.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(bundle / f"{name}.npy", array)
    write_json(bundle / "manifest.json", {"input_hash": "h", "version": "v", "catalog": catalog})
    write_json(bundle / "statistics.json", stats)
    result = attach(context, "g2")
    pairs = load_scene_segments(bundle)
    # video_0: cut [0, 1], talk [1, 2]; video_1: eat [0]  (bundle scene rows 0, 1, 2, 3)
    assert result["segments"] == 3 and sorted(map(tuple, pairs.tolist())) == [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2)]
    with pytest.raises(ValueError, match="no segments extension"):
        attach(context, "g0")
    g1, g2 = (recommendation_identity(context, name) for name in ("g1", "g2"))
    assert g1["experiment_arm"]["extensions"] == ["segments"]
    assert g1["experiment_arm"]["role_fusion"] is False
    assert g2["experiment_arm"]["role_fusion"] is True
    assert g1["scene_segments"] == g2["scene_segments"]
    assert g2["experiment_arm"]["extensions"] == ["segments"]
    assert g2["item_model"]["extensions"]["segments"] == EXTENSION_MODELS["segments"]
    assert g2["scene_aggregation"] == "mean"
    assert "gcn" not in g2["item_model"]
    assert g2["item_model"]["extensions"]["segments"]["applied_after"] == "scene_mlp_without_scene_exchange"
    assert g2["item_model"]["scene_mlp"] is True
