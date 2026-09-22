from types import SimpleNamespace

import numpy as np
import pytest

from pipeline_runtime import write_jsonl, read_jsonl
from validation.run_comparison import compare_graph_runs


@pytest.fixture
def paired_runs(tmp_path, monkeypatch):
    contexts = []
    for name in ("candidate", "reference"):
        directory = tmp_path / name
        directory.mkdir()
        for filename in ("events.jsonl", "catalog.jsonl"):
            write_jsonl(directory / filename, [{"id": 1}, {"id": 2}])
        contexts.append(
            SimpleNamespace(
                config={"protocol": {"arms": ["metadata", "graph_qwen", "graph_gemini"]}},
                root=tmp_path,
                run_id=name,
                cohort_dir=directory,
                require_ready_cohort=lambda: {"plan": {"splits": ["2026-09-01"]}},
            )
        )
    monkeypatch.setattr("validation.run_comparison.RunContext.load", lambda *a, **k: contexts[1])
    monkeypatch.setattr(
        "validation.run_comparison.read_state", lambda *a: {"sources": [{"prompt": "example"}]}
    )
    monkeypatch.setattr(
        "validation.steps.validation_config",
        lambda _: SimpleNamespace(
            evaluation=SimpleNamespace(bootstrap_samples=1000, familywise_alpha=0.05)
        ),
    )
    monkeypatch.setattr("validation.run_comparison.load_validation_cohort", lambda ctx: {
        **ctx.require_ready_cohort(),
        "events": read_jsonl(ctx.cohort_dir / "events.jsonl"),
        "catalog": read_jsonl(ctx.cohort_dir / "catalog.jsonl"),
    })
    reports = {}
    for ctx in contexts:
        reports[ctx.run_id] = {
            "daily": [
                {"evaluation_date": "2026-09-01", "seed": 42, "arm": arm}
                for arm in ("graph_gemini", "graph_qwen")
            ]
        }

    def collect(ctx, config, cohort, *, arms):
        values = [0.4, 0.25] if ctx.run_id == "candidate" else [0.2, 0.15]
        indices = [("graph_gemini", "graph_qwen").index(name) for name in arms]
        return (
            np.tile(np.array(values)[indices], (2, 1, 1)),
            np.ones((2, 1)),
            {"daily": [r for r in reports[ctx.run_id]["daily"] if r["arm"] in arms]},
        )

    monkeypatch.setattr("validation.rolling_diagnosis.collect_metrics", collect)
    return contexts, reports


def test_paired_run_effect_direction_interaction_and_partial_family(paired_runs):
    contexts, _ = paired_runs
    result = compare_graph_runs(contexts[0], "reference")
    assert result["comparisons"]["graph_gemini"]["difference"] == pytest.approx(0.2)
    assert result["comparisons"]["graph_qwen"]["ci_low"] == pytest.approx(0.1)
    assert result["model_interaction"]["difference"] == pytest.approx(0.1)
    assert result["comparisons"]["graph_qwen"]["confidence_level"] == 0.975
    assert set(result["sources"]) == {"candidate", "reference"}
    partial = compare_graph_runs(contexts[0], "reference", target=["graph_qwen"])
    assert partial["comparisons"]["graph_qwen"] == result["comparisons"]["graph_qwen"]
    assert partial["family_size"] == 2 and partial["model_interaction"] is None


@pytest.mark.parametrize("mismatch", ["events.jsonl", "catalog.jsonl", "seed", "dates"])
def test_cross_run_mismatch_rejected(paired_runs, mismatch):
    contexts, reports = paired_runs
    if mismatch.endswith("jsonl"):
        write_jsonl(contexts[1].cohort_dir / mismatch, [{"id": 99}])
    elif mismatch == "dates":
        contexts[1].require_ready_cohort = lambda: {"plan": {"splits": ["other"]}}
    else:
        reports["reference"]["daily"][0]["seed"] = 43
    with pytest.raises(ValueError, match="identical"):
        compare_graph_runs(contexts[0], "reference")
