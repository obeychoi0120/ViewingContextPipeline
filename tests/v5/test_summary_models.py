"""Summary model selection, failure isolation, and resumable model changes."""
from types import SimpleNamespace

import pytest
import yaml

import extraction.steps as steps
from arm_registry import generated_arm
from artifact_io import atomic_write_json
from extraction.backends import GeminiGenerationOutcome, GeminiWorkerPool
from extraction.cli import main
from extraction.errors import ExtractionStepError
from extraction.failures import FailureLog
from pipeline_runtime import RunContext, read_json, read_jsonl, write_jsonl
from validation.representation_inputs import documents_for_arm, representation_signature
from validation.representation_checks import verify_representations
from validation.representation_provenance import recommendation_identity
from validation.steps import embed_representations


@pytest.mark.parametrize("representation", ["graph", "description"])
@pytest.mark.parametrize("source", ["qwen", "gemini"])
@pytest.mark.parametrize("model", ["qwen", "gemini"])
def test_summary_cli(v5_context, monkeypatch, representation, source, model):
    calls = []
    monkeypatch.setattr("extraction.cli.RunContext.load", lambda _: v5_context)
    monkeypatch.setitem(steps.STEP_HANDLERS, f"summarize-{representation}",
                        lambda context, **kw: calls.append(kw))
    args = [f"summarize-{representation}", "--run-id", "run", "--schema",
            f"prompts/summary_{representation}_v4.md"]
    arm = f"{'graph' if representation == 'graph' else 'desc'}_{source}"
    assert main(args + ["--arm", arm]) == 1
    assert main(args + ["--model", model]) == 1
    assert main(args + ["--arm", arm, "--model", model]) == 0
    assert calls[0]["arm"] == arm and calls[0]["model"] == model


@pytest.fixture(params=[("graph", "qwen"), ("graph", "gemini"),
                        ("description", "qwen"), ("description", "gemini")])
def case(request, ready_context, fake_models):
    representation, source = request.param
    ctx = ready_context
    getattr(steps, f"extract_{representation}_scenes")(
        ctx, model=source,
        schema=f"prompts/scene_{representation}_v{'3' if representation == 'graph' else '2'}.md",
    )
    cohort = ctx.require_ready_cohort()
    arm = generated_arm(ctx.config, representation, source)
    ctx.config["protocol"]["arms"] = [arm.name]
    directory = ctx.summary_dir(representation, source, "gemini")

    def run(model="gemini", **kwargs):
        return getattr(steps, f"summarize_{representation}")(
            ctx, source=source, model=model,
            schema=f"prompts/summary_{representation}_v4.md", **kwargs,
        )

    return SimpleNamespace(ctx=ctx, cohort=cohort, arm=arm, directory=directory, run=run,
                           ids=[r["content_id"] for r in cohort["catalog"]])


def test_gemini_text_only_and_reuse(case, monkeypatch):
    calls, configs = [], []
    case.ctx.config["extraction"]["greedy_decoding"] = False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def no_qwen(**kwargs):
        pytest.fail("Gemini must not start Qwen workers")

    class Backend:
        last_response_diagnostics = None

        def generate(self, images, prompt, max_new_tokens):
            assert images == []
            assert "Scene observations:" in prompt
            assert max_new_tokens == case.ctx.config["extraction"][case.arm.representation]["summary_max_new_tokens"]
            calls.append(prompt)
            return "A person waves."

    def pool(concurrency, **kwargs):
        assert concurrency == case.ctx.config["extraction"]["gemini"]["threads"]
        configs.append(kwargs)
        return GeminiWorkerPool(2, **kwargs, backend_factory=Backend)

    monkeypatch.setattr(steps, "qwen_generator", no_qwen)
    monkeypatch.setattr(steps, "GeminiWorkerPool", pool)
    assert case.run()["failure_count"] == 0
    assert len(calls) == len(case.ids)
    assert configs == [case.ctx.config["models"]["gemini"]]
    doc = read_json(case.directory / f"{case.ids[0]}.json")
    assert doc["provenance"]["summary_model"] == "gemini"
    assert doc["provenance"]["settings"] == {
        "backend": "gemini",
        "max_new_tokens": case.ctx.config["extraction"][case.arm.representation]["summary_max_new_tokens"],
    }
    assert case.run()["content_count"] == len(case.ids)
    assert len(calls) == len(case.ids) and len(configs) == 1


def test_failures_retry_and_preserve_empty_expressions(case, monkeypatch):
    calls = []
    fail = True

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            for task in tasks:
                calls.append(task.task_id)
                index = case.ids.index(task.task_id)
                text, error, diagnostics = "A person waves.", None, None
                if fail and index == 0:
                    text = ""
                elif fail and index == 1:
                    diagnostics = {"candidates": [{"finish_reason": "MAX_TOKENS"}]}
                elif fail and index == 2:
                    text, error = "", "503 unavailable"
                callback(GeminiGenerationOutcome(task.task_id, text, error, diagnostics))

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    with pytest.raises(ExtractionStepError, match="3 Gemini summaries failed"):
        case.run()
    rows = read_jsonl(case.directory / "failures.jsonl")
    assert len(rows) == 3 and all(r["summary_model"] == "gemini" for r in rows)
    assert all(read_json(case.directory / f"{cid}.json")["status"] == "failed"
               for cid in case.ids[:3])
    # A source fallback exists, but must never hide a failed Gemini generation.
    if case.arm.model == "gemini":
        other = case.ctx.summary_dir(case.arm.representation, "qwen", "gemini")
        other.mkdir(parents=True)
        for cid in case.ids:
            atomic_write_json(other / f"{cid}.json", {"text": "must not be read"}, durable=False)
    docs = documents_for_arm(case.ctx, case.cohort, case.arm, summary_source="gemini")
    assert all(d["text"] == "" and d["status"] == "failed" for d in docs[:3])
    before = {p: p.read_bytes() for p in case.directory.iterdir()}
    case.run(model="qwen")
    assert embed_representations(case.ctx, target=[case.arm.name], summary_source="qwen")
    assert before == {p: p.read_bytes() for p in case.directory.iterdir()}
    fail = False
    assert case.run()["failure_count"] == 0
    assert calls == case.ids + case.ids[:3]
    assert not (case.directory / "failures.jsonl").exists()
    assert len(documents_for_arm(case.ctx, case.cohort, case.arm, summary_source="gemini")) == len(case.ids)


def test_models_coexist_interrupt_resume_and_embedding_selection(case, monkeypatch):
    case.run(model="qwen")
    qwen_dir = case.ctx.summary_dir(case.arm.representation, case.arm.model, "qwen")
    # Legacy Qwen provenance remains readable inside the explicit qwen directory.
    for path in qwen_dir.glob("*.json"):
        doc = read_json(path)
        del doc["provenance"]["summary_model"]
        atomic_write_json(path, doc, durable=False)
    saved_qwen = {p: p.read_bytes() for p in qwen_dir.glob("*.json")}
    old = documents_for_arm(case.ctx, case.cohort, case.arm)
    old_signature = representation_signature(case.ctx, case.cohort["catalog"], case.arm, old)
    def embed(model):
        return embed_representations(case.ctx, target=[case.arm.name], summary_source=model)
    assert embed("qwen")["generated_arms"] == [case.arm.name]
    old_recommendation = recommendation_identity(case.ctx, case.arm.name)
    calls = []
    interrupt = True

    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            for task in tasks:
                calls.append(task.task_id)
                callback(GeminiGenerationOutcome(task.task_id, old[0]["text"]))
                if interrupt:
                    raise KeyboardInterrupt

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    with pytest.raises(KeyboardInterrupt):
        case.run()
    assert len(list(case.directory.glob("*.json"))) == 1
    interrupt = False
    case.run()
    assert calls == case.ids
    new = documents_for_arm(case.ctx, case.cohort, case.arm, summary_source="gemini")
    assert representation_signature(case.ctx, case.cohort["catalog"], case.arm, new) != old_signature
    assert all(d["source_provenance"]["model"] == case.ctx.config["models"]["gemini"] for d in new)
    # Adding a different summary model does not invalidate the chosen Qwen embeddings.
    verify_representations(case.ctx, case.cohort, arms=[case.arm.name])
    assert embed("gemini")["generated_arms"] == [case.arm.name]
    assert recommendation_identity(case.ctx, case.arm.name) != old_recommendation
    assert embed("gemini")["generated_arms"] == []
    verify_representations(case.ctx, case.cohort, arms=[case.arm.name])
    # Force applies to Gemini only, including after an interrupted regeneration.
    interrupt = True
    with pytest.raises(KeyboardInterrupt):
        case.run(force=True)
    assert len(list(case.directory.glob("*.json"))) == 1
    assert saved_qwen == {p: p.read_bytes() for p in qwen_dir.glob("*.json")}
    interrupt = False
    case.run()
    assert embed("qwen")["reuse"]["shared"] == [case.arm.name]
    assert recommendation_identity(case.ctx, case.arm.name) == old_recommendation


def test_legacy_failure_mismatch_precedes_migration(case):
    legacy = case.directory / "failures" / f"{case.ids[0]}.jsonl"
    write_jsonl(legacy, [{"content_id": case.ids[0], "error": "empty", "raw_output": ""}])
    before = legacy.read_bytes()
    with pytest.raises(ExtractionStepError, match="--force"):
        case.run()
    assert legacy.read_bytes() == before
    assert not (case.directory / "failures.jsonl").exists()
    case.run(force=True)
    assert not legacy.exists()


def test_failure_model_survives_migration(tmp_path):
    write_jsonl(tmp_path / "failures" / "a.jsonl", [
        {"content_id": "a", "error": "empty", "raw_output": "", "summary_model": "gemini"}
    ])
    failures = FailureLog(tmp_path)
    failures.record("a", None, "network error")
    assert FailureLog(tmp_path).rows[("a", None)]["summary_model"] == "gemini"


@pytest.mark.parametrize("legacy", [None, "qwen", "gemini"])
def test_legacy_config_optional_and_ignored(v5_context, legacy):
    cfg = v5_context.config
    if legacy is not None:
        cfg["protocol"]["graph_summarizer"] = legacy
    (v5_context.root / "config.yaml").write_text(yaml.safe_dump(cfg))
    assert RunContext.load("test", root=v5_context.root).config == cfg


def test_cli_returns_failure_for_gemini_error(case, monkeypatch):
    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            for task in tasks:
                callback(GeminiGenerationOutcome(task.task_id, "", error="503 unavailable"))

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    monkeypatch.setattr("extraction.cli.RunContext.load", lambda _: case.ctx)
    assert main([f"summarize-{case.arm.representation}", "--run-id", case.ctx.run_id,
                 "--arm", case.arm.name, "--model", "gemini", "--schema",
                 f"prompts/summary_{case.arm.representation}_v4.md"]) == 1
    assert len(read_jsonl(case.directory / "failures.jsonl")) == len(case.ids)


def test_embedding_cli_reads_model_from_arm_artifact(v5_context, monkeypatch):
    from validation.cli import main as validate
    from validation.steps import STEP_HANDLERS
    calls = []
    monkeypatch.setattr("validation.cli.RunContext.load", lambda _: v5_context)
    monkeypatch.setitem(STEP_HANDLERS, "embed-representations", lambda context, **kwargs: calls.append(kwargs))
    args = ["embed-representations", "--run-id", "run"]
    assert validate(args) == 0
    assert calls == [{"force": False}]
    with pytest.raises(SystemExit):
        validate(args + ["--summary-source", "qwen"])


def test_embedding_does_not_fall_back_to_other_summary_model(case):
    case.run(model="qwen")
    embed_representations(case.ctx, target=[case.arm.name], summary_source="gemini")
    import numpy as np
    with np.load(case.ctx.representations_dir / f"{case.arm.name}_embeddings.npz") as data:
        assert not data["values"].any()


def test_selected_summary_provenance_and_staleness(case):
    case.run(model="qwen")
    case.run(model="gemini")
    embed_representations(case.ctx, target=[case.arm.name], summary_source="gemini")
    qwen = case.ctx.summary_dir(case.arm.representation, case.arm.model, "qwen")
    qwen_path = qwen / f"{case.ids[0]}.json"
    doc = read_json(qwen_path)
    doc["text"] = "Changed Qwen summary."
    doc["word_count"] = 3
    atomic_write_json(qwen_path, doc, durable=False)
    # Recommendation/diagnosis verify the recorded Gemini selection, not Qwen.
    verify_representations(case.ctx, case.cohort, arms=[case.arm.name])
    gemini_path = case.directory / f"{case.ids[0]}.json"
    doc = read_json(gemini_path)
    doc["text"] = "Changed Gemini summary."
    doc["word_count"] = 3
    atomic_write_json(gemini_path, doc, durable=False)
    with pytest.raises(RuntimeError, match="stale"):
        verify_representations(case.ctx, case.cohort, arms=[case.arm.name])
    doc["provenance"]["summary_model"] = "qwen"
    atomic_write_json(gemini_path, doc, durable=False)
    with pytest.raises(ValueError, match="provenance mismatch"):
        embed_representations(case.ctx, target=[case.arm.name], summary_source="gemini")
