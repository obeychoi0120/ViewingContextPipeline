"""Summary model selection, failure isolation, and resumable model changes."""

from types import SimpleNamespace

import pytest

import extraction.steps as steps
from arm_registry import generated_arm
from extraction.backends import GeminiGenerationOutcome, GeminiWorkerPool
from extraction.cli import main
from extraction.errors import ExtractionStepError
from pipeline_runtime import read_json, read_jsonl
from validation.representation_inputs import documents_for_arm


@pytest.fixture(params=[("graph", "qwen"), ("description", "qwen")])
def case(request, ready_context, fake_models):
    representation, source = request.param
    ctx = ready_context
    getattr(steps, f"extract_{representation}_scenes")(
        ctx,
        arm=f"{'graph' if representation == 'graph' else 'desc'}_{source}",
        model=source,
        schema=f"prompts/scene_{representation}_v{'4' if representation == 'graph' else '2'}.md",
    )
    cohort = ctx.require_ready_cohort()
    arm = generated_arm(ctx.config, representation, source)
    ctx.config["protocol"]["arms"] = [arm.name]
    directory = ctx.summary_arm_dir(arm.name)

    def run(model="gemini", **kwargs):
        return getattr(steps, f"summarize_{representation}")(
            ctx,
            arm=arm.name,
            model=model,
            schema=f"prompts/summary_{representation}_v5.md",
            **kwargs,
        )

    return SimpleNamespace(
        ctx=ctx,
        cohort=cohort,
        arm=arm,
        directory=directory,
        run=run,
        ids=[r["content_id"] for r in cohort["catalog"]],
    )


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
            assert (
                max_new_tokens
                == case.ctx.config["extraction"][case.arm.representation]["summary_max_new_tokens"]
            )
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
        "max_new_tokens": case.ctx.config["extraction"][case.arm.representation][
            "summary_max_new_tokens"
        ],
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
    assert all(
        read_json(case.directory / f"{cid}.json")["status"] == "failed" for cid in case.ids[:3]
    )
    docs = documents_for_arm(case.ctx, case.cohort, case.arm)
    assert all(d["text"] == "" and d["status"] == "failed" for d in docs[:3])
    fail = False
    assert case.run()["failure_count"] == 0
    assert calls == case.ids + case.ids[:3]
    assert not (case.directory / "failures.jsonl").exists()
    assert len(documents_for_arm(case.ctx, case.cohort, case.arm)) == len(case.ids)


def test_cli_returns_failure_for_gemini_error(case, monkeypatch):
    class Pool:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            for task in tasks:
                callback(GeminiGenerationOutcome(task.task_id, "", error="503 unavailable"))

    monkeypatch.setattr(steps, "GeminiWorkerPool", Pool)
    monkeypatch.setattr("extraction.cli.RunContext.load", lambda _: case.ctx)
    assert (
        main(
            [
                f"summarize-{case.arm.representation}",
                "--run-id",
                case.ctx.run_id,
                "--arm",
                case.arm.name,
                "--model",
                "gemini",
                "--schema",
                f"prompts/summary_{case.arm.representation}_v5.md",
            ]
        )
        == 1
    )
    assert len(read_jsonl(case.directory / "failures.jsonl")) == len(case.ids)


def test_summary_model_change_requires_force(case):
    case.run(model="qwen")
    saved = {p: p.read_bytes() for p in case.directory.glob("*.json")}
    with pytest.raises(RuntimeError, match="model mismatch"):
        case.run(model="gemini")
    assert saved == {p: p.read_bytes() for p in saved}
    case.run(model="gemini", force=True)
    assert all(read_json(p)["provenance"]["summary_model"] == "gemini" for p in saved)
