"""Scene extensions (Arms g2-g4, g3a, g5, g5a): segments and video_node on gnn_attn, segments on mean."""

import json

import numpy as np
import pytest
import torch

from artifact_io import write_json
from extraction.scene_storage import write_scene_records
from helpers import graph, scene_row
from validation.config import GraphExecutionConfig
from arm_registry import registry
from validation.graph_context import EXTENSION_MODELS, check_extensions
from validation.graph_inputs import load_scene_segments, scene_segments_paths, write_scene_segments
from validation.graph_model import GraphSASRec, SceneRoleEncoder
from validation.representation_provenance import recommendation_identity
from validation.scene_segments import activity_key, attach, segments, shuffled

pytestmark = pytest.mark.torch
EXTENDED = {"g2": ("video_node",), "g3": ("segments",), "g4": ("segments", "video_node")}
ATTENTION_FREE = {"g5": "true", "g5a": "shuffled"}


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


def make_model(store, extensions=(), items=8, aggregation="gnn_attn"):
    torch.manual_seed(0)
    return GraphSASRec(
        store=store, aggregation=aggregation, item_count=len(store), max_length=10, embedding_dim=32,
        num_blocks=1, num_heads=2, dropout=0.0, arm="g1", role_fusion=True, extensions=extensions,
        execution=GraphExecutionConfig(chunk_items=items, chunk_nodes=100000, chunk_scene_pairs=1048576,
                                       checkpoint="never", feature_cache_mb=0),
    ).eval()


def test_arms_and_extension_checks():
    arms = registry({})
    for name, extensions in EXTENDED.items():
        arm = arms[name]
        assert arm.extensions == extensions and arm.aggregation == "gnn_attn" and arm.role_fusion
        assert (arm.representation, arm.uses_title, arm.scene_arm) == ("graph", True, "graph_qwen3vl")
    assert arms["g3a"].extensions == ("segments",) and arms["g3a"].segments == "shuffled"
    assert all(not arms[name].extensions for name in ("base", "d0", "d1", "g0", "g1", "g1a", "g1b", "g1c"))
    for name, variant in ATTENTION_FREE.items():
        arm = arms[name]
        assert (arm.extensions, arm.segments, arm.aggregation, arm.role_fusion) == (("segments",), variant, "mean", True)
        assert (arm.representation, arm.uses_title, arm.scene_arm) == ("graph", True, "graph_qwen3vl")
    assert check_extensions(("video_node",), "gnn_attn") == {"video_node"}
    assert check_extensions(("segments",), "mean") == {"segments"}
    for extensions, aggregation in ((("video_node",), "mean"), (("segments",), "gnn"), (("segments",), "self_only")):
        with pytest.raises(ValueError, match="gnn_attn"):
            check_extensions(extensions, aggregation)
    with pytest.raises(ValueError, match="unknown"):
        check_extensions(("bogus",), "gnn_attn")
    assert set(EXTENSION_MODELS) == {"segments", "video_node"}


def test_segments_start_as_gnn_attn_and_leave_the_rng_alone():
    store = synthetic_store()
    reference = make_model(store)
    expected_rng = torch.rand(3)
    model = make_model(store, ("segments",))
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
    batch = model.packed.batch(np.array([video]), "cpu", group_scenes=True, scene_segments=True)
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


def test_video_node_without_messages_is_the_title_start():
    store = synthetic_store()
    model = make_model(store, ("video_node",))
    encoder = model.graph_encoder
    with torch.no_grad():
        for layer in (encoder.video_up, encoder.video_down):
            layer.weight.zero_()
            layer.bias.zero_()
        actual = model.catalog_vectors()
        titles = torch.as_tensor(np.asarray(store["features"])[np.asarray(store["titles"])])
        title = encoder.title_projection(titles).masked_fill(
            ~torch.as_tensor(np.asarray(store["titles"]) != 0)[:, None], 0.0)
        start = encoder.scene_mlp(torch.cat([title, title.new_zeros((len(store), encoder.hidden_dim * 3))], -1))
    available = (np.diff(np.asarray(store["video_offsets"])) > 0) | (np.asarray(store["titles"]) != 0)
    for video in range(len(store)):
        torch.testing.assert_close(actual[video], start[video] if available[video] else torch.zeros_like(start[video]))


@pytest.mark.parametrize("extensions,aggregation", [*((e, "gnn_attn") for e in EXTENDED.values()),
                                                     (("segments",), "mean")])
def test_extensions_train_and_ignore_chunking(extensions, aggregation):
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
    if aggregation == "gnn_attn":
        assert model.graph_encoder.gcn.query.weight.grad.abs().sum() > 0
        assert "scene_attention_summary" in model.eval().execution_report()
    else:
        assert model.graph_encoder.segment_scene_weight.grad.abs().sum() > 0


def test_segments_require_the_sidecar():
    store = synthetic_store(videos=4)
    store.scene_segments = None
    with pytest.raises(ValueError, match="scene_segments"):
        make_model(store, ("segments",))
    make_model(store, ("video_node",))
    with pytest.raises(ValueError, match="gnn_attn"):
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
    write_scene_segments(tmp_path, pairs[:3], variant="shuffled")
    assert np.array_equal(load_scene_segments(tmp_path), pairs)
    assert np.array_equal(load_scene_segments(tmp_path, "shuffled"), pairs[:3])
    assert scene_segments_paths(tmp_path, "shuffled")[0].name == "scene_segments_shuffled.npy"
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
    moved = shuffled(len(scenes), segments(scenes), "video_x")
    assert [(a, len(m)) for a, m in moved] == [(a, len(m)) for a, m in segments(scenes)]
    assert moved == shuffled(len(scenes), segments(scenes), "video_x")


def test_attach_builds_both_variants_and_identity_records_them(context, monkeypatch):
    from validation.graph_inputs import build_input
    import validation.representation_provenance as provenance

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
    texts, arrays, stats = build_input(context, cohort, registry(context.config)["g3"])
    bundle = context.representations_dir / "graph_qwen3vl"
    bundle.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(bundle / f"{name}.npy", array)
    write_json(bundle / "manifest.json", {"input_hash": "h", "version": "v", "catalog": catalog})
    write_json(bundle / "statistics.json", stats)
    result = attach(context, "g3")
    pairs = load_scene_segments(bundle)
    # video_0: cut [0, 1], talk [1, 2]; video_1: eat [0]  (bundle scene rows 0, 1, 2, 3)
    assert result["segments"] == 3 and sorted(map(tuple, pairs.tolist())) == [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2)]
    shuffled_result = attach(context, "g3a")
    assert shuffled_result["variant"] == "shuffled" and shuffled_result["memberships"] == len(pairs)
    with pytest.raises(ValueError, match="no segments extension"):
        attach(context, "g1")
    monkeypatch.setattr(provenance, "local_model_identity", lambda path: {"model": "stub"}, raising=False)
    import model_provenance
    monkeypatch.setattr(model_provenance, "local_model_identity", lambda path: {"model": "stub"})
    g1, g3, g3a = (recommendation_identity(context, name) for name in ("g1", "g3", "g3a"))
    assert "extensions" not in g1["experiment_arm"] and "scene_segments" not in g1
    assert tuple(g3["experiment_arm"]["extensions"]) == ("segments",) and g3["scene_segments"]["variant"] == "true"
    assert g3a["scene_segments"]["variant"] == "shuffled" and "segments" in g3["item_model"]["extensions"]
    # g5/g5a read the same sidecar variants; only where the segments stage runs differs.
    g5, g5a = (recommendation_identity(context, name) for name in ("g5", "g5a"))
    assert g5["scene_segments"] == g3["scene_segments"] and g5a["scene_segments"] == g3a["scene_segments"]
    assert g3["item_model"]["extensions"]["segments"] == EXTENSION_MODELS["segments"]
    assert g5["item_model"]["extensions"]["segments"] == {
        **EXTENSION_MODELS["segments"], "applied_after": "scene_mlp_without_scene_exchange"}
    assert g5["scene_aggregation"] == "mean" and g5["item_model"]["attention"] is None
    assert g3["scene_aggregation"] == "gnn_attn" and g3["item_model"]["attention"] is not None


def test_segments_on_mean_start_as_mean_and_are_the_only_path_between_scenes():
    store = synthetic_store()
    reference = make_model(store, aggregation="mean")
    expected_rng = torch.rand(3)
    model = make_model(store, ("segments",), aggregation="mean")
    assert torch.equal(torch.rand(3), expected_rng)
    assert not hasattr(model.graph_encoder, "gcn") and not hasattr(model.graph_encoder, "scene_transformer")
    shared = reference.state_dict()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in shared.items())
    with torch.no_grad():
        assert torch.equal(reference.catalog_vectors(), model.catalog_vectors())
        model.graph_encoder.segment_gates.copy_(torch.tensor([0.8, 0.6]))
        model.clear_item_cache()
        expected, actual = reference.catalog_vectors(), model.catalog_vectors()
    offsets = np.asarray(store["video_offsets"])
    scene_videos = np.repeat(np.arange(len(store)), np.diff(offsets))
    segmented = set(scene_videos[store.scene_segments[:, 0]].tolist())
    multi_scene = {v for v in range(len(store)) if offsets[v + 1] - offsets[v] >= 2}
    assert multi_scene - segmented  # videos whose scenes can only meet in the final mean
    for video in range(len(store)):
        assert torch.equal(actual[video], expected[video]) == (video not in segmented)
