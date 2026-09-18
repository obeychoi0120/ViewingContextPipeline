import pytest
import yaml

from arm_registry import registry, select_arms
from extraction.cli import main as extract
from pipeline_runtime import ConfigError, RunContext


@pytest.mark.parametrize(
    "step,selector",
    [
        ("extract-graph-scenes", "--model"),
        ("extract-description-scenes", "--model"),
        ("summarize-graph", "--source"),
        ("summarize-description", "--source"),
    ],
)
@pytest.mark.parametrize("model", ["qwen", "gemini"])
def test_required_generation_options_and_prompt_paths(
    v5_context, monkeypatch, step, selector, model
):
    calls = []
    monkeypatch.setattr("extraction.cli.RunContext.load", lambda _: v5_context)
    monkeypatch.setitem(
        __import__("extraction.cli", fromlist=["STEP_HANDLERS"]).STEP_HANDLERS,
        step,
        lambda context, **kw: calls.append(kw),
    )
    prefix = [step, "--run-id", "run"]
    if step.startswith("summarize-"):
        prefix += ["--model", "qwen"]
    assert extract(prefix + [selector, model]) == 1
    assert extract(prefix + ["--schema", "prompts/graph_scene_v3.md"]) == 1
    assert extract(prefix + [selector, model, "--schema", "prompts/missing.md"]) == 1
    assert extract(prefix + [selector, model, "--schema", "prompts/graph_scene_v*.md"]) == 1
    assert extract(prefix + [selector, model, "--schema", "config.yaml"]) == 1
    assert extract(prefix + [selector, model, "--schema", "prompts/graph_scene_v3.md"]) == 0
    assert calls[-1][selector[2:]] == model
    assert calls[-1]["schema"].is_absolute()
    if step.startswith("extract-"):
        assert extract(prefix + [selector, model, "--source", model,
                                 "--schema", str(calls[-1]["schema"])]) == 1
    assert "gpus" not in calls[-1]
    with pytest.raises(SystemExit):
        extract(prefix + [selector, model, "--schema", "prompts/graph_scene_v3.md", "--gpus", "1"])


def test_dynamic_targets_and_custom_artifact_root(v5_context):
    config = v5_context.config
    assert len(select_arms(config)) == 5 and len(registry(config)) == 5
    config["protocol"]["arms"] = ["desc_qwen", "graph_gemini", "metadata"]
    config["artifacts_root"] = str(v5_context.root / "custom")
    (v5_context.root / "config.yaml").write_text(yaml.safe_dump(config))
    context = RunContext.load("new", root=v5_context.root)
    context.initialize()
    assert set(select_arms(config)) == {"desc_qwen", "graph_gemini", "metadata"}
    assert set(select_arms(config, ["graph_qwen"])) == {"graph_qwen"}
    with pytest.raises(ValueError):
        select_arms(config, ["GRAPH_V8_QWEN"])
    assert context.keyframes_dir == v5_context.root / "custom/preperation/resized_keyframes"
    assert context.source_assets_dir == v5_context.root / "custom/preperation/source_assets"
    assert context.cohort_dir == v5_context.root / "custom/preperation/cohort"
    assert context.run_root == v5_context.root / "custom/runs/new"
    assert context.prompt_path("prompts/graph_scene_v3.md").is_file()


@pytest.mark.parametrize("name", ["..", "a/b", "resized_keyframes", "source_assets", "a\\b", ""])
def test_run_id_cannot_escape_or_claim_the_shared_frame_directory(v5_context, name):
    with pytest.raises(ConfigError):
        RunContext.load(name, root=v5_context.root)


def test_missing_metadata_row_keeps_catalog_scope_and_records_failure(v5_context):
    from preparation.steps import prepare_cohort_step
    from pipeline_runtime import read_json, read_jsonl

    titles = v5_context.path("data", "titles_csv")
    titles.write_text("1,First\n2, \n3,Third\n")
    with pytest.raises(RuntimeError, match="unresolved assets"):
        prepare_cohort_step(v5_context)
    assert len(read_jsonl(v5_context.cohort_dir / "required_items.jsonl")) == 4
    assert read_json(v5_context.cohort_dir / "eligibility.json")["status"] == "blocked"
    assert read_jsonl(v5_context.cohort_dir / "preparation_failures.jsonl") == [
        {"item_id": "4", "reason": "missing_title"}
    ]
