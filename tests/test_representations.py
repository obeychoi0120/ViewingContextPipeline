from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from arm_registry import registry
from artifact_io import write_jsonl
from extraction.graph_warnings import PARSE_FAILED, REPEAT_OCCURRED
from helpers import (
    failure_row,
    graph_store,
    prepare_inputs,
    scene_row,
)
from validation.graph_batching import PackedGraphStore
from validation.graph_inputs import build_input
from helpers import make_context


def _check_scene_sources_and_shared_features(context, monkeypatch):
    from artifact_io import read_json
    from validation.graph_inputs import GraphStore
    from validation.steps import embed_representations

    prepare_inputs(context, monkeypatch)
    result = embed_representations(context, target=["description", "graph", "graph-activity", "graph-role-activity"])
    assert result["generated_sources"] == ["desc_qwen3vl_2b"]
    assert result["reused_sources"] == ["graph_qwen3vl_2b"]
    root = context.representations_dir
    assert {p.name for p in root.iterdir()} == {"metadata", "desc_qwen3vl_2b", "graph_qwen3vl_2b"}
    description = GraphStore(root / "desc_qwen3vl_2b")
    assert len(description["contexts"]) == 48  # two independent scenes per video
    assert len(description["types"]) == 0
    assert description["video_offsets"].tolist() == list(range(0, 49, 2))
    assert read_json(root / "desc_qwen3vl_2b" / "manifest.json")["source"] == "desc_qwen3vl_2b"


def _check_v8_location_reaches_embeddings_and_segments_without_source_rewrite(context, monkeypatch):
    from artifact_io import read_json, read_jsonl
    from extraction.scene_storage import read_scene_records
    from validation.selection import load_validation_cohort
    from validation.steps import embed_representations, prepare_module
    from validation.graph_inputs import GraphStore, load_scene_segments, source_identity

    prepare_inputs(context, monkeypatch)
    cohort = load_validation_cohort(context)
    arm = registry(context.config)["graph-activity"]
    texts_before, arrays_before, stats_before = build_input(context, cohort, arm)
    identity_before = source_identity(context, cohort, arm)
    cid = cohort["catalog"][0]["content_id"]
    path = context.scene_arm_dir(arm.scene_arm) / f"{cid}.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        value = row["scene_graph"]
        for action in value["actions"]:
            action["location"] = action.pop("place")
        row["scene_graph"] = json.dumps(value)
    write_jsonl(path, rows)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert source_identity(context, cohort, arm) != identity_before
    assert all(not row["semantic_warnings"] for row in read_scene_records(path))

    # Reproduce a v1 cache that excluded the same JSON source before compatibility.
    with monkeypatch.context() as old_runtime:
        old_runtime.setattr("validation.graph_inputs.VERSION", "scene-bge-inputs/v1")
        old_runtime.setattr(
            "extraction.structured_output.normalize_graph_fields",
            lambda value: (deepcopy(value), False),
        )
        embed_representations(context, target=["graph-activity"])
    stats_path = context.representations_dir / arm.scene_arm / "statistics.json"
    assert read_json(stats_path)["counts"]["parse_failed"] == len(rows)

    texts, arrays, stats = build_input(context, cohort, arm)
    assert texts == texts_before and stats == stats_before
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], arrays_before[key])
    result = embed_representations(context, target=["graph-activity"])
    assert result["generated_sources"] == [arm.scene_arm]
    prepare_module(context, target=["graph-activity"])
    store = GraphStore(context.representations_dir / arm.scene_arm)
    np.testing.assert_array_equal(store["edges"], arrays_before["edges"])
    assert len(load_scene_segments(store.directory))
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def _check_graph_edges_are_scene_local(context):
    arrays, _, texts, _ = graph_store(context, counts=(2,))
    assert arrays["scene_offsets"].tolist() == [0, 5, 10]
    forward = arrays["edges"][arrays["edges"][:, 2] < 5].tolist()
    assert forward == [[2, 4, 0], [0, 4, 1], [3, 4, 2], [1, 4, 4], [5, 9, 0], [7, 9, 1]]
    # Entities are sorted by ID (f1, l1, p1, t1), not declaration order.
    assert arrays["missing"][:, [0, 1, 2, 3, 5]].tolist() == [
        [4, -1, -1, -1, -1],
        [9, -1, -1, 0, 1],
    ]
    assert all(not any(token in text for token in ("p1", "f1", "t1", "l1")) for text in texts)
    assert arrays["missing"][:, 4].tolist() == [0, 0]  # receiver compatibility slot


def _check_graph_exclusions_preserve_items(context):
    _, cohort, _, _ = graph_store(context, counts=(1, 0))
    cid = cohort["catalog"][0]["content_id"]
    rows = [
        scene_row(cid),
        scene_row(cid, 1, warnings=[PARSE_FAILED]),
        scene_row(cid, 2, value="unrecoverable observation"),
        failure_row(cid, 3),
        # A repeat-stopped scene is used and only counted.
        scene_row(cid, 4, warnings=[REPEAT_OCCURRED]),
    ]
    write_jsonl(context.scene_arm_dir("graph_qwen3vl_2b") / f"{cid}.jsonl", rows)
    _, arrays, stats = build_input(context, cohort, registry(context.config)["graph-activity"])
    assert arrays["video_offsets"].tolist() == [0, 2, 2]
    assert len(arrays["titles"]) == len(cohort["catalog"]) == 2
    assert stats["counts"]["valid_scenes"] == 2
    assert stats["counts"]["repeat_occurred"] == 1
    assert [r["reason"] for r in stats["excluded_scenes"]] == ["parse_failed", "parse_failed", "error"]
    assert stats["counts"]["zero_scene_video"] == 1


def _check_packed_graph_indices_match_reference(context):
    arrays, _, _, _ = graph_store(context, counts=(2, 1, 0))
    batch = PackedGraphStore(arrays, feature_cache_mb=0).batch([1, 2, 0], "cpu")
    assert batch["offsets"] == [0, 1, 1, 3]
    assert batch["node_scenes"].tolist() == [0] * 5 + [1] * 5 + [2] * 5
    assert batch["scene_videos"].tolist() == [0, 2, 2]
    assert batch["action_ids"].tolist() == [4, 9, 14]
    assert batch["role_entities"].tolist() == [[2, 0, 3, 1], [7, 5, 8, 6], [10, 12, -1, -1]]
    assert batch["role_states"].tolist() == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
    expected_nodes = np.concatenate((arrays["nodes"][10:15], arrays["nodes"][:10]))
    torch.testing.assert_close(batch["features"], torch.tensor(arrays["features"][expected_nodes]))
    assert batch["available"].tolist() == [1, 1, 1]


def test_scene_sources_shared_features_and_v8_location(tmp_path):
    # Scene sources and shared features.
    case = tmp_path / "0"
    with pytest.MonkeyPatch.context() as patch:
        _check_scene_sources_and_shared_features(context=make_context(case / "repo"), monkeypatch=patch)
    # V8 location reaches embeddings and segments without source rewrite.
    case = tmp_path / "1"
    with pytest.MonkeyPatch.context() as patch:
        _check_v8_location_reaches_embeddings_and_segments_without_source_rewrite(context=make_context(case / "repo"), monkeypatch=patch)


@pytest.mark.torch
def test_graph_edges_exclusions_and_packed_indices(tmp_path):
    # Graph edges are scene local.
    case = tmp_path / "0"
    _check_graph_edges_are_scene_local(context=make_context(case / "repo"))
    # Graph exclusions preserve items.
    case = tmp_path / "1"
    _check_graph_exclusions_preserve_items(context=make_context(case / "repo"))
    # Packed graph indices match reference.
    case = tmp_path / "2"
    _check_packed_graph_indices_match_reference(context=make_context(case / "repo"))
