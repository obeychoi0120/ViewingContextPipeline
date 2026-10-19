from copy import deepcopy

import numpy as np
import pytest
import torch

from arm_registry import registry
from artifact_io import write_jsonl
from extraction.graph_warnings import PARSE_FAILED
from extraction.summary_executor import prepare_summary_batch
from helpers import (
    FakeEncoder,
    failure_row,
    graph_store,
    prepare_text,
    ready_cohort,
    scene_row,
    summary_options,
)
from validation.graph_batching import PackedGraphStore
from validation.graph_inputs import build_input
from validation.representation_inputs import documents_for_arm
from validation.selection import prepare_validation_cohort
from validation.text_features import encode_components, validate_arrays


def test_text_components_follow_arm_policy(context, monkeypatch):
    prepare_text(context, monkeypatch)
    cohort = prepare_validation_cohort(context)
    arms = registry(context.config)
    for name, title_present, video_present in (
        ("meta", True, False),
        ("graph_qwen3vl", False, True),
        ("graph_qwen3vl_meta", True, True),
    ):
        docs = documents_for_arm(context, cohort, arms[name])
        assert len(docs) == 24
        for index, doc in enumerate(docs, 1):
            assert doc["title_text"] == (f"TITLE_SECRET_{index}" if title_present else "")
            assert bool(doc["video_text"]) is video_present
            assert doc["title_used"] is title_present
            assert doc["summary_used"] is video_present
            assert doc["text"] == "\n\n".join(
                part for part in (doc["title_text"], doc["video_text"]) if part
            )


def test_missing_summary_preserves_catalog(context):
    # Item 2 has neither a title nor a generated summary; item 1 has a failed summary.
    path = context.path("data", "titles_csv")
    path.write_text(path.read_text().replace("2,TITLE_SECRET_2\n", "2,\n"))
    ready_cohort(context)
    cohort = prepare_validation_cohort(context)
    options = summary_options(context, catalog=cohort["catalog"][:1])
    prepare_summary_batch(context, **options)
    docs = documents_for_arm(context, cohort, registry(context.config)["graph_qwen3vl_meta"])
    arrays, _ = encode_components(docs, 1024, FakeEncoder(), {})
    assert len(docs) == len(cohort["catalog"]) == 24
    assert docs[0]["summary_status"] == "failed" and docs[0]["components"] == "title_only"
    assert docs[1]["summary_status"] == "missing" and docs[1]["components"] == "neither"
    assert not arrays["video_available"].any()
    assert not arrays["video_values"].any()
    assert arrays["title_available"][0] and not arrays["title_available"][1]
    assert np.any(arrays["title_values"][0] != 0)
    assert not arrays["title_values"][1].any()


def test_text_arrays_reject_invalid_values():
    docs = [{"title_text": "hello", "video_text": ""}]
    good, _ = encode_components(docs, 1024, FakeEncoder(), {})
    validate_arrays(good, 1, 1024, docs)
    corruptions = [
        ("title_values", np.zeros((2, 1024), np.float32)),
        ("title_values", good["title_values"].astype(np.float64)),
        ("title_values", np.full((1, 1024), np.nan, np.float32)),
        ("video_values", np.ones((1, 1024), np.float32)),
        ("title_available", np.zeros(1, np.bool_)),
        ("title_available", np.ones(1, np.int64)),
    ]
    for name, value in corruptions:
        bad = deepcopy(good)
        bad[name] = value
        with pytest.raises(ValueError):
            validate_arrays(bad, 1, 1024, docs)
    with pytest.raises(ValueError, match="legacy/invalid"):
        validate_arrays({"values": good["title_values"]}, 1, 1024)


def test_graph_edges_are_scene_local(context):
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


def test_graph_exclusions_preserve_items(context):
    _, cohort, _, _ = graph_store(context, counts=(1, 0))
    cid = cohort["catalog"][0]["content_id"]
    rows = [
        scene_row(cid),
        scene_row(cid, 1, warnings=[PARSE_FAILED]),
        scene_row(cid, 2, value="unrecoverable observation"),
        failure_row(cid, 3),
    ]
    write_jsonl(context.scene_arm_dir("graph_qwen3vl") / f"{cid}.jsonl", rows)
    _, arrays, stats = build_input(context, cohort, registry(context.config)["graph_qwen3vl_meta"])
    assert arrays["video_offsets"].tolist() == [0, 1, 1]
    assert len(arrays["titles"]) == len(cohort["catalog"]) == 2
    assert stats["counts"]["valid_scenes"] == 1
    assert {r["reason"] for r in stats["excluded_scenes"]} == {"warning", "raw", "failed"}
    assert stats["counts"]["zero_scene_video"] == 1


@pytest.mark.torch
def test_packed_graph_indices_match_reference(context):
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
