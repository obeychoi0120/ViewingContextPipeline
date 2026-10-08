"""Current six-arm contracts with fake generation and a small CPU training model."""

from dataclasses import replace
from types import SimpleNamespace
import shutil

import numpy as np
import pytest

from arm_registry import (
    registry,
    generation_registry,
)
from artifact_io import read_json, read_jsonl, write_json, write_jsonl
from validation.representation_inputs import documents_for_arm
from validation.representation_provenance import read_state, state_path
from validation.steps import embed_representations

NAMES = [
    "meta", "graph_qwen", "graph_qwen_meta", "graph_gemini_meta",
    "desc_qwen_meta", "desc_gemini_meta",
]


@pytest.mark.parametrize("model", ["qwen", "gemini"])
def test_shared_title_free_generation(ready_context, fake_models, model, generate_all):
    generate_all(ready_context, model)
    assert len(fake_models) == 8
    assert {p.name for p in (ready_context.run_root / "extraction/summaries").iterdir()} == set(
        generation_registry(ready_context.config)
    )
    for tasks in fake_models[1::2]:
        assert all(t.max_new_tokens == ready_context.config["extraction"]["graph" if "[Entities]" in t.prompt else "description"][model]["summary_max_new_tokens"] for t in tasks)
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
        assert data["title_values"][2].any() and not data["title_values"][3].any()
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
    from validation.leave_one_out_diagnosis import comparisons

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
    from validation.leave_one_out_recommendation import run_leave_one_out
    from validation.leave_one_out_diagnosis import diagnose
    from validation.selection import load_validation_cohort

    legacy = ready_context.run_root / "validation" / "recommendations" / "text" / "2022-09-10" / "seed_42" / "meta"
    legacy.mkdir(parents=True)
    for name in ("sasrec.pt", "training.json", "per_event_metrics.jsonl", "complete.json"):
        (legacy / name).write_bytes(b"preserve original rolling result")
    previous_layout = ready_context.run_root / "validation" / "leave_one_out" / "cohort" / "manifest.json"
    previous_layout.parent.mkdir(parents=True)
    previous_layout.write_bytes(b"preserve previous leave-one-out layout")
    protected = {p: p.read_bytes() for p in [*legacy.iterdir(), previous_layout]}
    assert ready_context.validation_dir == ready_context.run_root / "validation"
    generate_all(ready_context)
    embed_representations(ready_context, target=list(registry(ready_context.config)))

    def tiny(config, *, item_count, branch, features, device):
        features = {**features, "title_values": features["title_values"][:, :6],
                    "video_values": features["video_values"][:, :6]}
        return SASRec(item_count, 10, 8, 1, 2, 0, arm=branch, item_features=features).to(device)

    monkeypatch.setattr("validation.leave_one_out_recommendation._new_model", tiny)
    monkeypatch.setattr("validation.leave_one_out_recommendation.worker_devices", lambda *args: ["cpu"])
    # Old local-only artifacts can be published under the new eligibility rule
    # without changing their recommendation identity or repeating training.
    for arm in NAMES[1:]:
        state = read_state(ready_context, arm)
        state["shareable"] = False
        write_json(state_path(ready_context, arm), state)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert run_leave_one_out(ready_context, target=list(registry(ready_context.config)))["completed"] == 18
        assert embed_representations(ready_context, target=list(registry(ready_context.config)))["reuse"]["local"] == NAMES
        assert run_leave_one_out(ready_context, target=list(registry(ready_context.config))) == {
            "stage": "run-recommendation", "completed": 0, "skipped": 18,
        }
        assert diagnose(ready_context, target=list(registry(ready_context.config)))["status"] == "pass"
        assert protected == {p: p.read_bytes() for p in protected}
        doc = read_json(ready_context.diagnosis_path)
        assert doc["schema_version"] == "leave-one-out-diagnosis/v2"
        assert (ready_context.validation_dir / "cohort" / "manifest.json").is_file()
        assert set(doc) == {"schema_version", "run_id", "representation_mode", "runtime_decision", "recommendations", "statistics"}
        assert set(doc["recommendations"]) == {"means"}
        assert set(doc["statistics"]) == {"status", "comparisons", "metric"}
        from validation.leave_one_out_diagnosis import collect_metrics
        from validation.steps import validation_config

        user_metrics, counts, report = collect_metrics(
            ready_context, validation_config(ready_context), load_validation_cohort(ready_context),
            arms={name: name for name in NAMES})
        assert len(report["per_seed"]) == 18
        assert doc["recommendations"]["means"] == report["means"]
        manual = np.zeros((3, 6))
        for seed in (42, 43, 44):
            for index, arm in enumerate(NAMES):
                for row in read_jsonl(ready_context.recommendations_dir / f"seed_{seed}" / arm / "per_event_metrics.jsonl"):
                    manual[int(row["user_id"]) - 1, index] += row["NDCG@10"] / 3
        np.testing.assert_allclose(user_metrics, manual)
        np.testing.assert_array_equal(counts, np.ones(3))
        for index, arm in enumerate(NAMES):
            assert report["means"][arm]["NDCG@10"] == pytest.approx(manual[:, index].mean())
        assert doc["statistics"]["comparisons"]["graph_qwen_meta-meta"]["primary"]
        from validation.diagnosis_statistics import comparison_families
        expected_pairs = {f"{a}-{b}" for pairs in comparison_families(ready_context.config).values() for a, b in pairs}
        expected_pairs.add("graph_gemini_meta-graph_qwen_meta")
        assert set(doc["statistics"]["comparisons"]) == expected_pairs
        assert {k: len(v) for k, v in comparison_families(ready_context.config).items()} == {"metadata_baseline": 5, "representation": 2, "title_input": 1}
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
            "validation.leave_one_out_recommendation.worker_devices",
            lambda *args: pytest.fail("shared reuse must not initialize training workers"),
        )
        result = run_leave_one_out(other, target=list(registry(other.config)))
        assert result["skipped"] == 18
        assert result["completed"] == 0
        assert read_json(other.recommendations_dir / "reuse.json")["shared"] == 18
        checkpoints = list(ready_context.recommendations_dir.glob("seed_*/*/sasrec.pt"))
        assert len(checkpoints) == 18
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
        assert shared_checkpoints == 18
        assert diagnose(other, target=list(registry(other.config)))["status"] == "pass"
    finally:
        torch.set_num_threads(previous)


def test_embedding_never_sends_combined_title_summary(ready_context, fake_models, generate_all, monkeypatch):
    generate_all(ready_context)
    calls = []

    class Encoder:
        def __init__(self, config):
            pass

        def encode(self, texts):
            calls.extend(texts)
            self.last_truncated_flags = [False] * len(texts)
            return np.ones((len(texts), 1024), dtype=np.float32)

    monkeypatch.setattr('validation.features.BGETextEncoder', Encoder)
    arms = registry(ready_context.config)
    cohort = ready_context.require_ready_cohort()
    docs = [doc for arm in arms.values() for doc in documents_for_arm(ready_context, cohort, arm)]
    expected = {d[f'{part}_text'].strip() for d in docs for part in ('title', 'video')
                if d[f'{part}_text'].strip()}
    embed_representations(ready_context, target=list(arms))
    assert set(calls) == expected
    assert len(calls) == len(expected)
    assert read_state(ready_context, 'graph_qwen_meta')['truncation']['title']['text_count'] > 0
