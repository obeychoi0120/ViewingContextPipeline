"""ASHA ablation: segment variants change only the scene incidence, beside an untouched default sidecar."""

from collections import Counter

import numpy as np
import pytest
import torch

from rec_experiments import CONFIG_KEY, parse_experiments
from validation.segment_variants import (VARIANTS, action_bag, contiguous_starts, overlap, random_permutation,
                                         time_window, variant_segments)
from validation.scene_segments import segments

# Eight valid scenes: cut cut stir {cut, stir} - cut pour pour (scene 4 has no action).
SCENES = [(0, {"cut"}, 1), (1, {"cut"}, 1), (2, {"stir"}, 1), (3, {"cut", "stir"}, 2),
          (4, set(), 0), (5, {"cut"}, 1), (6, {"pour"}, 1), (7, {"pour"}, 1)]


def shape(groups):
    """Segment sizes and per-scene membership counts, which random and time_window must keep."""
    degree = Counter(p for _, members in groups for p in members)
    return sorted(len(m) for _, m in groups), sorted(degree.values())


def test_default_segments_of_the_example():
    assert segments(SCENES) == [("cut", [0, 1]), ("cut", [3]), ("cut", [5]),
                                ("pour", [6, 7]), ("stir", [2, 3])]


def test_action_bag_merges_reappearances_and_ignores_gaps():
    assert action_bag(SCENES, segments(SCENES)) == [("cut", [0, 1, 3, 5]), ("pour", [6, 7]), ("stir", [2, 3])]
    gapped = [(0, {"run"}, 1), (2, {"run"}, 1)]
    assert segments(gapped) == [("run", [0]), ("run", [1])]
    assert action_bag(gapped, segments(gapped)) == [("run", [0, 1])]


def test_contiguous_starts_respect_missing_windows():
    scenes = [(0, set(), 0), (1, set(), 0), (2, set(), 0), (4, set(), 0), (5, set(), 0)]
    assert contiguous_starts(scenes, 1) == [0, 1, 2, 3, 4]
    assert contiguous_starts(scenes, 2) == [0, 1, 3]
    assert contiguous_starts(scenes, 3) == [0]


def test_time_window_keeps_lengths_and_contiguity():
    groups = segments(SCENES)
    for seed in range(20):
        moved = time_window(SCENES, groups, np.random.default_rng(seed))
        assert [len(m) for _, m in moved] == [len(m) for _, m in groups]
        for _, members in moved:
            assert [SCENES[p][0] for p in members] == list(range(SCENES[members[0]][0],
                                                                   SCENES[members[0]][0] + len(members)))


def test_random_keeps_sizes_and_scene_degrees():
    groups = segments(SCENES)
    for seed in range(20):
        shuffled = random_permutation(SCENES, groups, np.random.default_rng(seed))
        assert shape(shuffled) == shape(groups)
        assert all(members == sorted(set(members)) for _, members in shuffled)


@pytest.mark.parametrize("variant", ["time_window", "random"])
def test_random_variants_are_reproducible_per_seed_and_video(variant):
    def draw(seed, video):
        return variant_segments(SCENES, variant, seed, video)[1]

    assert draw(0, 3) == draw(0, 3)
    assert len({str(draw(seed, 3)) for seed in range(10)}) > 1
    assert len({str(draw(0, video)) for video in range(10)}) > 1


def test_overlap_reports_identity_and_jaccard():
    groups = segments(SCENES)
    assert all(same and best == 1.0 for same, best in overlap(groups, groups))
    bag = overlap(groups, action_bag(SCENES, groups))
    assert [same for same, _ in bag] == [False, True, True]
    assert bag[0][1] == pytest.approx(2 / 4)


def experiments(*rows):
    return parse_experiments({"version": 1, "experiments": list(rows)})


def test_segments_config_is_optional_and_validated():
    default = {"label": "asha", "source": "graph_qwen3vl_2b", "modules": ["ASHA"]}
    assert "segments" not in experiments(default)["experiments"][0]
    resolved = experiments({**default, "segments": "random", "segments_seed": 2})["experiments"][0]
    assert resolved["segments"] == {"variant": "random", "seed": 2}
    assert experiments({**default, "segments": "action_bag"})["experiments"][0]["segments"]["seed"] == 0
    for change in ({"segments": "shuffle"}, {"segments": None}, {"segments_seed": 1},
                   {"segments": "random", "segments_seed": -1}, {"segments": "random", "segments_seed": True},
                   {"segments": "random", "segments_seed": "1"},
                   {"segments": "random", "modules": ["REAF"]}, {"segments": "random", "modules": []}):
        with pytest.raises(ValueError):
            experiments({**default, **change})


def test_variant_arms_prepare_separate_sidecars_and_identities(context, monkeypatch):
    from arm_registry import registry
    from helpers import prepare_inputs
    from validation.graph_inputs import load_scene_segments, scene_segments_identity, scene_segments_paths
    from validation.graph_model import new_graph_model
    from validation.representation_provenance import recommendation_identity
    from validation.steps import prepare_module, validation_config

    graph = {"source": "graph_qwen3vl_2b"}
    context.config[CONFIG_KEY] = experiments(
        {"label": "base", "source": "metadata"},
        {"label": "graph-activity", **graph, "modules": ["ASHA"]},
        {"label": "graph-role-activity", **graph, "modules": ["REAF", "ASHA"]},
        *({"label": f"graph-{v}", **graph, "modules": ["REAF", "ASHA"], "segments": v} for v in VARIANTS),
        {"label": "graph-random-s1", **graph, "modules": ["REAF", "ASHA"], "segments": "random",
         "segments_seed": 1},
    )
    prepare_inputs(context, monkeypatch, modules=False)
    directory = context.representations_dir / "graph_qwen3vl_2b"
    prepare_module(context, target=["graph-activity"])
    default_files = {path: path.read_bytes() for path in scene_segments_paths(directory)}
    default_pairs = load_scene_segments(directory)

    calls = []
    from validation.segment_variants import attach_variant

    def tracked(ctx, arm):
        calls.append(arm)
        return attach_variant(ctx, arm)

    monkeypatch.setattr("validation.segment_variants.attach_variant", tracked)
    result = prepare_module(context)
    # The default sidecar is shared by both default Arms and rewritten with identical bytes.
    assert {path: path.read_bytes() for path in scene_segments_paths(directory)} == default_files
    assert calls == [f"graph-{v}" for v in VARIANTS] + ["graph-random-s1"]
    arms = registry(context.config)
    for name in calls:
        variant = arms[name].segments
        pairs = load_scene_segments(directory, variant)
        meta = scene_segments_identity(directory, variant)
        assert [meta["variant"], meta["seed"]] == list(variant)
        assert result["statistics"][name]["variant"] == variant[0]
        assert set(np.unique(pairs[:, 0])) <= set(range(int(np.load(directory / "video_offsets.npy")[-1])))
    bag = load_scene_segments(directory, ("action_bag", 0))
    assert len(np.unique(bag[:, 1])) <= len(np.unique(default_pairs[:, 1]))
    random_pairs = load_scene_segments(directory, ("random", 0))
    assert sorted(Counter(random_pairs[:, 1]).values()) == sorted(Counter(default_pairs[:, 1]).values())

    default = recommendation_identity(context, "graph-role-activity")
    assert "segments" not in default["experiment_arm"]
    assert "segments" not in default["experiment_definition"]
    seen = {str(default["scene_segments"])}
    for name in calls:
        identity = recommendation_identity(context, name)
        assert identity["experiment_arm"]["segments"] == list(arms[name].segments)
        assert identity["experiment_definition"]["segments"] == dict(zip(("variant", "seed"), arms[name].segments))
        seen.add(str(identity["scene_segments"]))
    assert len(seen) == 1 + len(calls)

    # Every fixture video has two scenes in one segment, so no variant can move it; the statistics say so
    # before any training is spent on an uninformative comparison.
    assert result["statistics"]["graph-random"]["identical_to_default_fraction"] == 1.0

    config = validation_config(context)
    loaded = []
    import validation.graph_inputs as graph_inputs

    original = graph_inputs.load_scene_segments
    monkeypatch.setattr(graph_inputs, "load_scene_segments",
                        lambda directory, variant=None: loaded.append(variant) or original(directory, variant))
    torch.manual_seed(0)
    model = new_graph_model(context, config, "graph-random", "cpu").eval()
    # The model reads the Arm's variant incidence, not the default one.
    assert loaded == [("random", 0)]
    assert np.array_equal(model.packed.store.scene_segments, random_pairs)
    with torch.no_grad():
        assert torch.isfinite(model.catalog_vectors()).all()

    # A variant sidecar written for another seed is stale for this Arm.
    _, meta_path = scene_segments_paths(directory, ("random", 0))
    from artifact_io import read_json, write_json

    meta = read_json(meta_path)
    write_json(meta_path, {**meta, "seed": 9})
    with pytest.raises(ValueError, match="prepare-module"):
        load_scene_segments(directory, ("random", 0))
