from copy import deepcopy

import numpy as np
import pytest

from arm_registry import registry, select_arms
from artifact_io import read_json, read_jsonl
from extraction.graph_warnings import PARSE_FAILED
from extraction.scene_storage import write_scene_records
from helpers import failure_row, install_doubles, ready_cohort, scene_row
from validation.graph_context import SCENE_AGGREGATIONS, graph_context, validate_mode
from validation.graph_inputs import GraphStore, build_input, source_identity
from validation.recommendation_contracts import RECOMMENDATION_ARMS, resolve_target_arms
from validation.representation_provenance import recommendation_identity
from validation.selection import prepare_validation_cohort
from validation.steps import embed_representations, run_diagnosis, run_recommendation

TARGETS = ["meta", "graph_minicpm", "graph_minicpm_meta"]


def test_minicpm_graph_mode_and_source_registration(context):
    selected = select_arms(context.config, TARGETS)
    assert list(selected) == TARGETS
    for name, title in (("graph_minicpm", False), ("graph_minicpm_meta", True)):
        arm = selected[name]
        assert arm.model == "minicpm" and arm.scene_arm == "graph_minicpm"
        assert arm.uses_title is title
        assert name in RECOMMENDATION_ARMS
        validate_mode("graph", None, "embed-representations", [name])
        for aggregation in SCENE_AGGREGATIONS:
            for step in ("run-recommendation", "run-diagnosis"):
                validate_mode("graph", aggregation, step, [name])
        with pytest.raises(ValueError, match="require --representation-mode graph"):
            validate_mode("text", None, "embed-representations", [name])
    assert resolve_target_arms(TARGETS, config=context.config) == {name: name for name in TARGETS}
    old = deepcopy(context.config)
    del old["models"]["minicpm"]
    assert not set(TARGETS[1:]) & registry(old).keys()
    with pytest.raises(ValueError):
        select_arms(old, ["graph_minicpm"])


def test_minicpm_graph_inputs_titles_exclusions_and_identity(context):
    ready_cohort(context)
    cohort = prepare_validation_cohort(context)
    cohort = {**cohort, "catalog": cohort["catalog"][:2], "metadata_titles": cohort["metadata_titles"][:2]}
    directory = context.scene_arm_dir("graph_minicpm")
    cid = cohort["catalog"][0]["content_id"]
    write_scene_records(directory / f"{cid}.jsonl", [
        scene_row(cid), scene_row(cid, 1, warnings=[PARSE_FAILED]),
        scene_row(cid, 2, value="unrecoverable observation"), failure_row(cid, 3),
    ])
    arms = registry(context.config)
    texts, plain, plain_stats = build_input(context, cohort, arms["graph_minicpm"])
    titled_texts, titled, titled_stats = build_input(context, cohort, arms["graph_minicpm_meta"])
    assert not any("TITLE_SECRET" in text for text in texts)
    assert any("TITLE_SECRET" in text for text in titled_texts)
    assert plain["titles"].tolist() == [0, 0]
    assert all(titled["titles"] > 0)
    assert plain["video_offsets"].tolist() == titled["video_offsets"].tolist() == [0, 1, 1]
    assert plain_stats["counts"]["valid_scenes"] == titled_stats["counts"]["valid_scenes"] == 1
    assert {row["reason"] for row in plain_stats["excluded_scenes"]} == {"warning", "raw", "failed"}
    assert plain_stats["counts"]["empty_video"] == 1
    assert titled_stats["counts"]["empty_video"] == 0
    assert plain_stats["counts"]["missing_file"] == 1
    plain_id = source_identity(context, cohort, arms["graph_minicpm"])
    titled_id = source_identity(context, cohort, arms["graph_minicpm_meta"])
    assert plain_id != titled_id
    retitled = deepcopy(cohort)
    retitled["metadata_titles"][0]["title"] = "changed title"
    assert source_identity(context, retitled, arms["graph_minicpm"]) == plain_id
    assert source_identity(context, retitled, arms["graph_minicpm_meta"]) != titled_id
    # Identical graph payloads from different extractor families must not share identity.
    write_scene_records(
        context.scene_arm_dir("graph_qwen3vl") / f"{cid}.jsonl",
        read_jsonl(directory / f"{cid}.jsonl"),
    )
    assert source_identity(context, cohort, arms["graph_qwen3vl"]) != plain_id
    changed = scene_row(cid)
    changed["scene_graph"]["actions"][0]["action"] = "washing"
    write_scene_records(directory / f"{cid}.jsonl", [changed])
    assert source_identity(context, cohort, arms["graph_minicpm"]) != plain_id
    assert source_identity(context, cohort, arms["graph_minicpm_meta"]) != titled_id


def test_minicpm_graph_cli_accepts_both_targets(context, monkeypatch):
    from pipeline_runtime import RunContext
    from validation.cli import STEP_HANDLERS, main

    calls = []
    monkeypatch.setattr(RunContext, "load", lambda run_id: context)
    monkeypatch.setitem(STEP_HANDLERS, "embed-representations", lambda ctx, **kw: calls.append(kw))
    assert main([
        "embed-representations", "--run-id", context.run_id,
        "--representation-mode", "graph", "--target", *TARGETS[1:],
    ]) == 0
    assert calls[0]["target"] == TARGETS[1:]


@pytest.mark.torch
def test_minicpm_graph_embedding_recommendation_diagnosis(context, monkeypatch):
    # Real graph storage, embedding composition, training and evaluation; only BGE is fake.
    install_doubles(monkeypatch)
    ready_cohort(context)
    context.config["validation"]["model"]["max_epochs"] = 1
    cohort = context.require_ready_cohort()
    for item in cohort["catalog"]:
        cid = item["content_id"]
        write_scene_records(context.scene_arm_dir("graph_minicpm") / f"{cid}.jsonl", [scene_row(cid)])
    run_contract = (context.run_root / "run_contract.json").read_bytes()
    embedded = embed_representations(context, target=TARGETS, representation_mode="graph")
    assert embedded["generated_arms"] == TARGETS
    graph = graph_context(context, "mean")
    identities = []
    for name in TARGETS[1:]:
        path = graph.representations_dir / f"{name}_embeddings"
        manifest = read_json(path / "manifest.json")
        assert manifest["arm"] == name
        assert read_json(path / "statistics.json")["counts"]["valid_scenes"] == len(cohort["catalog"])
        store = GraphStore(path)
        assert bool(np.any(store["titles"])) is name.endswith("_meta")
        identities.append(recommendation_identity(graph, name)["embedding_hash"])
    assert identities[0] != identities[1]
    reused = embed_representations(context, target=TARGETS, representation_mode="graph")
    assert not reused["generated_arms"] and reused["reused_arms"] == TARGETS
    trained = run_recommendation(
        context, target=TARGETS, representation_mode="graph", scene_aggregation="mean",
    )
    assert trained["completed"] == 9
    diagnosis = run_diagnosis(
        context, target=TARGETS, representation_mode="graph", scene_aggregation="mean",
    )
    assert diagnosis["status"] == "pass"
    report = read_json(graph.diagnosis_path)
    assert report["statistics"]["comparison_count"] == 3
    assert set(report["recommendations"]["means"]) == set(TARGETS)
    assert not (context.run_root / "extraction" / "summaries").exists()
    assert (context.run_root / "run_contract.json").read_bytes() == run_contract
