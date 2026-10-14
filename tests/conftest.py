"""Shared current-v4 fixtures. Generation and encoding fakes never load remote/local models."""

from contextlib import contextmanager
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import yaml

from pipeline_runtime import RunContext
from validation.leave_one_out_data import DAY

ROOT = Path(__file__).resolve().parents[1]
GRAPH = {
    "medium": "live_action", "format": "narrative", "topics": [],
    "entities": [
        {"id": "person1", "name": "person", "attributes": ["red jacket"]},
        {"id": "person2", "name": "person", "attributes": ["blue shirt"]},
    ],
    "actions": [
        {"actor": "person1", "action": "looking at", "target": "person2", "tool": None, "location": None},
        {"actor": "person2", "action": "waving to", "target": "person1", "tool": None, "location": None},
    ],
}
PROSE = "A person in a red jacket looks at a person in a blue shirt, who waves back. The interaction appears to be a casual greeting."
GRAPH_TEXT = """[Context]
medium: live_action
format: narrative
topics: none
[Entities]
person1: person; red jacket
person2: person; blue shirt
[Actions]
person1 - looking at - person2; none; none
person2 - waving to - person1; none; none"""


@pytest.fixture
def current_context(tmp_path):
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    config["artifacts_root"] = "artifacts"
    config["data"].pop("titles_supplement_csv", None)
    for key in config["data"]:
        config["data"][key] = str(tmp_path / key)
    videos = Path(config["data"]["videos_dir"])
    videos.mkdir()
    for item in range(1, 5):
        (videos / f"{item}.mp4").write_bytes(f"video {item}".encode())
    Path(config["data"]["titles_csv"]).write_text(
        "1,First title\n2, \n3,Third title\n4,Fourth title\n"
    )
    origin = 1661990400000
    rows = [(u, i % 4 + 1, origin + i * DAY // 2) for u in range(1, 4) for i in range(24)]
    Path(config["data"]["pairs_csv"]).write_text(
        "user,item,timestamp\n" + "".join(f"{u},{i},{t}\n" for u, i, t in rows)
    )
    config["validation"]["cohort"].update(user_count=3, interaction_count=72, item_count=4)
    config["validation"]["model"].update(max_epochs=1, patience=1)
    config["validation"]["evaluation"]["bootstrap_samples"] = 100
    config["extraction"]["visual_evidence"]["image_resolution"] = [16, 8]
    for model in ("qwen3vl", "qwen3.5", "bge"):
        config["models"][model] = str(tmp_path / model)
        Path(config["models"][model]).mkdir()
        if model != "bge":
            import json
            model_type, architecture = {
                "qwen3vl": ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
                "qwen3.5": ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
            }[model]
            (Path(config["models"][model]) / "config.json").write_text(
                json.dumps({"model_type": model_type, "architectures": [architecture]})
            )
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    return RunContext.load("test_run", root=tmp_path)


@pytest.fixture
def ready_context(current_context, monkeypatch):
    from preparation.steps import prepare_cohort_step
    from preparation.input_data import prepare_input_data

    prepare_cohort_step(current_context)
    monkeypatch.setattr("preparation.video.media.probe_duration", lambda _: 10.0)

    def ffmpeg(args, **kwargs):
        assert args[0] == "ffmpeg"
        Image.new("RGB", (16, 8)).save(args[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("preparation.video.video_processor.subprocess.run", ffmpeg)
    prepare_input_data(current_context)
    return current_context


@pytest.fixture
def fake_models(monkeypatch):
    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            tasks = list(tasks)
            calls.append(tasks)
            for task in tasks:
                callback(task.task_id, GRAPH_TEXT if "[Entities]" in task.prompt else PROSE)
            return {}

        yield generate

    class Gemini:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, tasks, callback, **kwargs):
            tasks = list(tasks)
            calls.append(tasks)
            for task in tasks:
                callback(
                    SimpleNamespace(
                        task_id=task.task_id,
                        error=None,
                        response_diagnostics=None,
                        text=GRAPH_TEXT if "[Entities]" in task.prompt else PROSE,
                    )
                )

    class Encoder:
        def __init__(self, config):
            self.last_truncation = None

        def encode(self, texts):
            self.last_truncation = {"text_count": len(texts), "truncated_count": 0}
            return np.ones((len(texts), 1024), dtype=np.float32)

    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    monkeypatch.setattr("extraction.steps.GeminiWorkerPool", Gemini)
    monkeypatch.setattr("validation.features.BGETextEncoder", Encoder)
    return calls


@pytest.fixture
def generate_all():
    """Generate the six scene/summary sources shared by the nine current arms."""
    from arm_registry import generation_registry
    from extraction import steps

    def generate(context, summary_model="gemini"):
        for name, source in generation_registry(context.config).items():
            kind = source.representation
            getattr(steps, f"extract_{kind}_scenes")(
                context,
                arm=name,
                model=source.model,
                schema=f"prompts/scene_{kind}_v{8 if kind == 'graph' else 2}.md",
            )
            steps.summarize(
                context,
                arm=name,
                model=summary_model,
                schema=f"prompts/summary_{kind}_v{8 if kind == 'graph' else 5}.md",
            )

    return generate
