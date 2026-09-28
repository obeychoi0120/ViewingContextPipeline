"""Current six-arm contracts with fake generation and a small CPU training model."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import shutil

import numpy as np
import pytest

from arm_registry import (
    registry,
    generation_registry,
    select_arms,
    active_arms,
    resolve_generation_arm,
)
from extraction import steps
from pipeline_runtime import read_json, read_jsonl, write_json, write_jsonl
from validation.representation_inputs import documents_for_arm
from validation.representation_provenance import read_state, state_path
from validation.steps import embed_representations

NAMES = [
    "meta", "graph_qwen", "graph_qwen_meta", "graph_gemini_meta",
    "desc_qwen_meta", "desc_gemini_meta",
]


def test_registry_and_config(ready_context):
    from pipeline_runtime import ConfigError, _validate_config

    assert list(registry(ready_context.config)) == NAMES
    _validate_config(ready_context.config)
    invalid = deepcopy(ready_context.config)
    invalid["experiment_config_version"] = "v5"
    with pytest.raises(ConfigError, match="experiment_config_version must be v4"):
        _validate_config(invalid)
    obsolete = deepcopy(ready_context.config)
    obsolete.pop("experiment_config_version")
    obsolete["schema_version"] = "viewing-context-config/v7"
    with pytest.raises(ConfigError, match="accepted only for historical v5/v6"):
        _validate_config(obsolete)
    assert list(generation_registry(ready_context.config)) == [
        "graph_qwen",
        "desc_qwen",
        "graph_gemini",
        "desc_gemini",
    ]
    for name in ["graph_gemini", "desc_gemini", "graph_meta_qwen", "desc_meta_qwen"]:
        with pytest.raises(ValueError):
            select_arms(ready_context.config, [name])
        config = deepcopy(ready_context.config)
        config["protocol"]["arms"] = [name]
        with pytest.raises(ValueError):
            active_arms(config)
    with pytest.raises(ValueError, match="use source --arm graph_qwen"):
        resolve_generation_arm(
            ready_context.config, "graph_qwen_meta", "graph", "gemini", summary=True
        )
    assert (
        resolve_generation_arm(
            ready_context.config, "graph_gemini", "graph", "qwen", summary=True
        ).model
        == "gemini"
    )


@pytest.mark.parametrize("model", ["qwen", "gemini"])
def test_shared_title_free_generation(ready_context, fake_models, model, generate_all):
    generate_all(ready_context, model)
    assert len(fake_models) == 8
    assert {p.name for p in (ready_context.run_root / "extraction/summaries").iterdir()} == set(
        generation_registry(ready_context.config)
    )
    for tasks in fake_models[1::2]:
        assert all(t.max_new_tokens == 1024 for t in tasks)
        assert all(
            "First title" not in t.prompt and "English Title:" not in t.prompt for t in tasks
        )
    for source in generation_registry(ready_context.config):
        doc = read_json(next(ready_context.summary_arm_dir(source).glob("*.json")))
        assert doc["provenance"]["uses_title"] is False
        assert "english_title" not in doc["provenance"]


def test_composition_cases_and_zero_vectors(ready_context, fake_models, generate_all):
    generate_all(ready_context)
    cohort = ready_context.require_ready_cohort()
    titles = cohort["metadata_titles"]
    titles[0]["title"] = "  Title\ninside  "
    titles[1]["title"] = "  "
    titles[3]["title"] = ""
    write_jsonl(ready_context.cohort_dir / "metadata_titles.jsonl", titles)
    for index in (2, 3):
        (
            ready_context.summary_arm_dir("graph_qwen")
            / f"{cohort['catalog'][index]['content_id']}.json"
        ).unlink()
    arm = registry(ready_context.config)["graph_qwen_meta"]
    docs = documents_for_arm(ready_context, cohort, arm)
    assert [d["components"] for d in docs] == ["both", "summary_only", "title_only", "neither"]
    assert docs[0]["text"].startswith("Title\ninside\n\n")
    assert docs[2]["text"] == "Third title" and docs[3]["text"] == ""
    plain = documents_for_arm(ready_context, cohort, registry(ready_context.config)["graph_qwen"])
    assert docs[0]["text"] == "Title\ninside\n\n" + plain[0]["text"]
    assert plain[2]["text"] == plain[3]["text"] == ""
    result = embed_representations(ready_context, target=list(registry(ready_context.config)))
    assert result["generated_arms"] == NAMES
    # Compact scenes are identified by the Summary's actual Scene input hash.
    assert read_state(ready_context, arm.name)["shareable"]
    with np.load(ready_context.representations_dir / f"{arm.name}_embeddings.npz") as data:
        assert data["values"][2].any() and not data["values"][3].any()
    from validation.diagnosis_representations import representation_report

    report, _ = representation_report(ready_context, [arm.name])
    summary = report[arm.name]["sources_summary"]
    assert summary["component_counts"] == dict(both=1, summary_only=1, title_only=1, neither=1)
    assert summary["visual_summary_coverage"] == 0.5
    assert summary["title_fallback_count"] == 1


def test_failed_summary_fallback_and_corruption(ready_context, fake_models, generate_all):
    generate_all(ready_context)
    cohort = ready_context.require_ready_cohort()
    arm = registry(ready_context.config)["graph_qwen_meta"]
    path = (
        ready_context.summary_arm_dir("graph_qwen") / f"{cohort['catalog'][0]['content_id']}.json"
    )
    doc = read_json(path)
    doc.update(status="failed", text="", word_count=0, violations=["max_tokens"])
    write_json(path, doc)
    row = documents_for_arm(ready_context, cohort, arm)[0]
    assert row["text"] == "First title" and row["summary_status"] == "failed"
    path.write_text("{broken")
    with pytest.raises(RuntimeError, match="invalid summary"):
        documents_for_arm(ready_context, cohort, arm)


def test_reject_title_conditioned_summary(ready_context, fake_models, generate_all):
    generate_all(ready_context)
    path = next(ready_context.summary_arm_dir("graph_qwen").glob("*.json"))
    doc = read_json(path)
    doc["provenance"]["english_title"] = "Old title"
    write_json(path, doc)
    with pytest.raises(ValueError, match="title present"):
        documents_for_arm(
            ready_context,
            ready_context.require_ready_cohort(),
            registry(ready_context.config)["graph_qwen_meta"],
        )


def test_cache_invalidation_and_run_rename(ready_context, fake_models, monkeypatch, generate_all):
    generate_all(ready_context)
    embed_representations(ready_context, target=list(registry(ready_context.config)))
    other = replace(
        ready_context, run_id="renamed", run_root=ready_context.run_root.parent / "renamed"
    )
    shutil.copytree(ready_context.run_root / "extraction", other.run_root / "extraction")

    result = embed_representations(other, target=list(registry(other.config)))
    assert result["reuse"]["shared"] == NAMES
    assert result["generated_arms"] == []
    titles = read_jsonl(ready_context.cohort_dir / "metadata_titles.jsonl")
    titles[0]["title"] = "Changed title"
    write_jsonl(ready_context.cohort_dir / "metadata_titles.jsonl", titles)
    result = embed_representations(other, target=list(registry(other.config)))
    assert result["generated_arms"] == [
        "meta",
        "graph_qwen_meta",
        "graph_gemini_meta",
        "desc_qwen_meta",
        "desc_gemini_meta",
    ]
    path = next(other.summary_arm_dir("graph_qwen").glob("*.json"))
    doc = read_json(path)
    doc["text"] += " Additional grounded detail."
    doc["word_count"] = len(doc["text"].split())
    write_json(path, doc)
    result = embed_representations(other, target=list(registry(other.config)))
    assert result["generated_arms"] == ["graph_qwen", "graph_qwen_meta"]


def test_comparisons_and_partial_targets(ready_context):
    from validation.diagnosis_statistics import comparison_families
    from validation.rolling_diagnosis import comparisons

    families = comparison_families(ready_context.config)
    assert {k: len(v) for k, v in families.items()} == {
        "metadata_baseline": 5,
        "representation": 2,
        "title_input": 1,
    }
    settings = SimpleNamespace(familywise_alpha=0.05)
    result = comparisons(
        np.arange(6.0),
        np.tile(np.arange(6.0), (100, 1)),
        settings,
        arms=NAMES,
        config=ready_context.config,
    )
    assert result["graph_qwen_meta-meta"]["primary"]
    assert result["graph_gemini_meta-graph_qwen_meta"]["family"] == "teacher_reference"
    assert not any("interaction" in key for key in result)
    partial = comparisons(
        np.array([0.0, 1.0]),
        np.tile([0.0, 1.0], (100, 1)),
        settings,
        arms=["meta", "graph_qwen_meta"],
        config=ready_context.config,
    )
    assert list(partial) == ["graph_qwen_meta-meta"]
    assert partial["graph_qwen_meta-meta"]["family_size"] == 5
    row = partial["graph_qwen_meta-meta"]
    assert row["difference"] == pytest.approx(1.0)
    assert row["relative_difference"] is None
    assert row["superior"] is True
    assert row["confidence_level"] == pytest.approx(1 - 0.05 / 5)


@pytest.mark.torch
def test_six_arm_training_diagnosis_and_shared_recommendations(
    ready_context, fake_models, monkeypatch, generate_all
):
    torch = pytest.importorskip("torch")
    from validation.model import SASRec
    from validation.rolling_recommendation import run_rolling
    from validation.rolling_diagnosis import diagnose
    from validation.selection import load_validation_cohort

    generate_all(ready_context)
    embed_representations(ready_context, target=list(registry(ready_context.config)))

    def tiny(config, *, item_count, branch, features, device):
        return SASRec(item_count, 10, 8, 1, 2, 0, arm=branch, item_features=features).to(device)

    monkeypatch.setattr("validation.rolling_recommendation._new_model", tiny)
    monkeypatch.setattr("validation.rolling_recommendation.worker_devices", lambda *args: ["cpu"])
    # Old local-only artifacts can be published under the new eligibility rule
    # without changing their recommendation identity or repeating training.
    for arm in NAMES[1:]:
        state = read_state(ready_context, arm)
        state["shareable"] = False
        write_json(state_path(ready_context, arm), state)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert run_rolling(ready_context, target=list(registry(ready_context.config)))["completed"] == 126
        assert embed_representations(ready_context, target=list(registry(ready_context.config)))["reuse"]["local"] == NAMES
        assert run_rolling(ready_context, target=list(registry(ready_context.config))) == {
            "stage": "run-recommendation", "completed": 0, "skipped": 126,
        }
        assert diagnose(ready_context, target=list(registry(ready_context.config)))["status"] == "pass"
        doc = read_json(ready_context.diagnosis_path)
        assert doc["statistics"]["comparisons"]["graph_qwen_meta-meta"]["primary"]
        assert len(doc["statistics"]["comparisons"]) == 10
        assert len(load_validation_cohort(ready_context)["catalog"]) == 4
        assert diagnose(ready_context, target=["meta", "graph_qwen_meta"])["status"] == "pass"
        other = replace(
            ready_context,
            run_id="shared-recommendations",
            run_root=ready_context.run_root.parent / "shared-recommendations",
        )
        shutil.copytree(ready_context.run_root / "extraction", other.run_root / "extraction")

        assert embed_representations(other, target=list(registry(other.config)))["reuse"]["shared"] == NAMES
        monkeypatch.setattr(
            "validation.rolling_recommendation.worker_devices",
            lambda *args: pytest.fail("shared reuse must not initialize training workers"),
        )
        result = run_rolling(other, target=list(registry(other.config)))
        assert result["skipped"] == 126
        assert result["completed"] == 0
        assert read_json(other.recommendations_dir / "reuse.json")["shared"] == 126
        checkpoints = list(ready_context.recommendations_dir.rglob("sasrec.pt"))
        assert len(checkpoints) == 126
        shared_checkpoints = 0
        for path in checkpoints:
            copied = other.recommendations_dir / path.relative_to(ready_context.recommendations_dir)
            original = torch.load(path, map_location="cpu", weights_only=True)
            reused = torch.load(copied, map_location="cpu", weights_only=True)
            assert reused["metadata"]["run_id"] == other.run_id
            if "reused_from" in reused["metadata"]:
                shared_checkpoints += 1
                assert reused["metadata"]["reused_from"]["source_run_id"] == ready_context.run_id
            assert original["state_dict"].keys() == reused["state_dict"].keys()
            assert all(
                torch.equal(value, reused["state_dict"][key])
                for key, value in original["state_dict"].items()
            )
        assert shared_checkpoints == 126
        assert diagnose(other, target=list(registry(other.config)))["status"] == "pass"
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("command", ["summarize", "summarize-graph"])
def test_cli_source_and_target_contract(ready_context, fake_models, monkeypatch, command):
    from extraction.cli import main as extract
    from validation.cli import main as validate

    monkeypatch.setattr("extraction.cli.RunContext.load", lambda _: ready_context)
    monkeypatch.setattr("validation.cli.RunContext.load", lambda _: ready_context)
    assert (
        extract(
            [
                "extract-graph-scenes",
                "--run-id",
                "new",
                "--model",
                "gemini",
                "--arm",
                "graph_gemini",
                "--schema",
                "prompts/scene_graph_v4.md",
            ]
        )
        == 0
    )
    assert (
        extract(
            [
                command,
                "--run-id",
                "new",
                "--model",
                "gemini",
                "--arm",
                "graph_gemini",
                "--schema",
                "prompts/summary_graph_v5.md",
            ]
        )
        == 0
    )
    assert (
        extract(
            [
                command,
                "--run-id",
                "new",
                "--model",
                "gemini",
                "--arm",
                "graph_gemini_meta",
                "--schema",
                "prompts/summary_graph_v5.md",
            ]
        )
        == 1
    )
    assert validate(["embed-representations", "--run-id", "new", "--target", "graph_gemini"]) == 1
    assert (
        validate(["embed-representations", "--run-id", "new", "--target", "graph_gemini_meta"]) == 0
    )


@pytest.mark.parametrize("arm", ["graph_qwen", "desc_qwen", "graph_gemini"])
def test_unified_summary_reuses_alias_output(ready_context, fake_models, arm):
    source = generation_registry(ready_context.config)[arm]
    kind = source.representation
    getattr(steps, f"extract_{kind}_scenes")(
        ready_context,
        arm=arm,
        model=source.model,
        schema=f"prompts/scene_{kind}_v{4 if kind == 'graph' else 2}.md",
    )
    kwargs = dict(arm=arm, model="gemini", schema=f"prompts/summary_{kind}_v5.md")
    steps.summarize(ready_context, **kwargs)
    paths = list(ready_context.summary_arm_dir(arm).glob("*.json"))
    assert paths
    assert all(read_json(path)["status"] == "complete" for path in paths)
    before = {path: path.read_bytes() for path in paths}
    calls = len(fake_models)
    getattr(steps, f"summarize_{kind}")(ready_context, **kwargs)
    steps.summarize(ready_context, **kwargs)
    assert len(fake_models) == calls
    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.parametrize("arm", [None, "meta", "unknown", "desc_qwen_meta"])
def test_unified_summary_rejects_invalid_source(ready_context, fake_models, arm):
    with pytest.raises(ValueError):
        steps.summarize(
            ready_context, arm=arm, model="gemini", schema="prompts/summary_graph_v5.md"
        )
    assert not fake_models


def test_unstructured_graph_reaches_summary_and_embedding(ready_context, fake_models, monkeypatch):
    from contextlib import contextmanager
    import json
    from extraction.scene_storage import read_scene_records

    text = "[Entities]\nperson1: person\n[Relations]\nperson1 -> waving ->"

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            for task in tasks:
                kwargs["runtime"].current_result = {"finish_reason": "stop"}
                callback(task.task_id, text)
            return {}

        yield generate

    context = ready_context
    monkeypatch.setattr(steps, "qwen_generator", generator)
    assert (
        steps.extract_graph_scenes(
            context, arm="graph_qwen", model="qwen", schema="prompts/scene_graph_v4.md"
        )["failure_count"]
        == 0
    )
    paths = list(context.scene_arm_dir("graph_qwen").glob("*.jsonl"))
    assert paths
    for path in paths:
        assert all(
            row["graph"] == text and row["parse_mode"] == "text" for row in read_scene_records(path)
        )
    assert (
        steps.summarize(
            context, arm="graph_qwen", model="gemini", schema="prompts/summary_graph_v5.md"
        )["failure_count"]
        == 0
    )
    assert all(json.dumps(text) in task.prompt for task in fake_models[-1])
    embed_representations(context, target=["graph_qwen"])
    assert read_state(context, "graph_qwen")["zero_vector_count"] == 0
