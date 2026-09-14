from copy import deepcopy

import pytest
import yaml
from pipeline_fixtures import ROOT
from pipeline_fixtures import context as context  # noqa: PLC0414

from extraction import steps as extraction_steps
from pipeline_logging import log_step_start, step_settings
from pipeline_runtime import RunContext
from validation import steps as validation_steps


@pytest.mark.parametrize("schema", ["v3", "v4"])
@pytest.mark.parametrize(("step", "options", "setting"), [
    ("prepare-cohort", {"plan_only": True}, "validation.cohort.user_count"),
    ("prepare-input-data", {}, "extraction.visual_evidence.scene_duration"),
    ("extract-graph-scenes", {"model": "qwen"}, "extraction.graph.scene_max_new_tokens"),
    ("extract-graph-scenes", {"model": "gemini"}, "extraction.gemini.threads"),
    ("extract-description-scenes", {}, "extraction.description.scene_prompt"),
    ("summarize-graph", {"source": "qwen"}, "extraction.graph.summary_prompt"),
    ("summarize-graph", {"source": "gemini"}, "extraction.graph.summary_prompt"),
    ("summarize-description", {}, "extraction.description.summary_max_new_tokens"),
    ("embed-representations", {}, "validation.encoder.batch_size"),
    ("run-recommendation", {}, "validation.model.seeds"),
    ("run-diagnosis", {}, "validation.evaluation.bootstrap_samples"),
])
def test_each_step_logs_once_before_work(context, monkeypatch, capsys, schema, step, options, setting):
    context.config["schema_version"] = f"viewing-context-config/{schema}"

    class WorkStarted(Exception):
        pass

    def initialize(self):
        output = capsys.readouterr().out
        assert output.count("[STEP]") == 1
        assert f"[STEP] {step}\n" in output
        assert f"  {setting}:" in output
        assert "  force: false\n" in output
        assert f"  run_id: {self.run_id}\n" in output
        assert "  output_dir:" in output
        if step == "extract-graph-scenes" and options["model"] == "qwen":
            assert "  scene_concurrency: 8\n" in output
        raise WorkStarted

    monkeypatch.setattr(RunContext, "initialize", initialize)
    handlers = {**extraction_steps.STEP_HANDLERS, **validation_steps.STEP_HANDLERS}
    with pytest.raises(WorkStarted):
        handlers[step](context, **options)
    assert capsys.readouterr().out == ""


def test_gemini_log_uses_current_yaml_and_effective_token_limit(context, capsys):
    path = context.root / "config/pipeline.yaml"
    value = yaml.safe_load(path.read_text())
    value["models"]["gemini"].update(model_id="changed-model", max_output_tokens=4096)
    value["extraction"]["gemini"]["threads"] = 16
    value["extraction"]["graph"]["scene_max_new_tokens"] = 128
    path.write_text(yaml.safe_dump(value))
    loaded = RunContext.load("changed", root=context.root)
    before = deepcopy(loaded.config)
    log_step_start(loaded, "extract-graph-scenes", model="gemini", force=True)
    output = capsys.readouterr().out
    assert "  models.gemini.model_id: changed-model\n" in output
    assert "  models.gemini.max_output_tokens: 4096\n" in output
    assert "  extraction.gemini.threads: 16\n" in output
    assert "  force: true\n" in output
    assert "extraction.graph.scene_max_new_tokens" not in output
    assert "extraction.qwen" not in output
    assert "validation.model" not in output
    assert "  gpus:" not in output
    assert loaded.config == before
    assert not loaded.run_root.exists()  # Logging must not create an artifact.


@pytest.mark.parametrize("greedy", [True, False])
def test_gemini_graph_summary_logs_qwen_and_only_active_sampling(context, greedy):
    context.config["extraction"]["greedy_decoding"] = greedy
    settings = step_settings(context, "summarize-graph", source="gemini", gpus=2)
    assert settings["model"] == "qwen"
    assert settings["source"] == "gemini"
    assert settings["gpus"] == 2
    assert settings["output_dir"] == context.graph_summary_dir("gemini")
    assert settings["extraction.greedy_decoding"] is greedy
    assert ("extraction.summary_sampling.temperature" in settings) is not greedy
    assert ("extraction.summary_sampling.top_p" in settings) is not greedy
    assert not any(key.startswith("models.gemini") for key in settings)
    # Scene generation is always greedy and does not use the summary switch.
    scene = step_settings(context, "extract-description-scenes")
    assert "extraction.greedy_decoding" not in scene
    assert not any(key.startswith("extraction.summary_sampling") for key in scene)


def test_qwen_defaults_and_runtime_selection_are_reported(context):
    context.config["extraction"].pop("qwen", None)
    settings = step_settings(context, "extract-description-scenes")
    assert settings["gpus"] == 1
    assert settings["extraction.qwen.max_num_seqs"] == 32
    assert settings["extraction.qwen.max_model_len"] == "auto"
    assert settings["extraction.qwen.async_scheduling"] is None
    selected = step_settings(context, "run-recommendation", target=["DESC_QWEN", "METADATA"],
                             gpus=4, workers_per_gpu=2)
    assert selected["target"] == ["METADATA", "DESC_QWEN"]
    assert selected["selected_arms"] == ["SASRec_METADATA", "SASRec_DESC"]
    assert selected["gpus"] == 4 and selected["workers_per_gpu"] == 2
    assert step_settings(context, "run-recommendation")["gpus"] == "auto"
    assert "gpus" not in step_settings(context, "run-diagnosis")
    assert step_settings(context, "prepare-input-data", reuse_run_id="donor")["reuse_run_id"] == "donor"


def test_full_rolling_configuration_sections_are_reported(context):
    context.config.update(yaml.safe_load((ROOT / "config/pipeline.yaml").read_text()))
    cohort = step_settings(context, "prepare-cohort")
    assert cohort["data.pairs_csv"] == context.config["data"]["pairs_csv"]
    assert cohort["validation.cohort.timezone"] == "UTC"
    assert cohort["validation.cohort.exclude_final_day"] is True
    assert "extraction.qwen.max_num_seqs" not in cohort
    diagnostic = step_settings(context, "run-diagnosis")
    assert diagnostic["validation.evaluation.min_scene_coverage"] == 0.95
    assert diagnostic["validation.evaluation.max_arm_coverage_gap"] == 0.05
    assert diagnostic["validation.model.seeds"] == [42, 43, 44]
