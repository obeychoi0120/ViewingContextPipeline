"""Shared v5 fixtures. Generation and encoding fakes never load remote/local models."""

from contextlib import contextmanager
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import yaml

from pipeline_runtime import RunContext
from validation.rolling_data import DAY

ROOT = Path(__file__).resolve().parents[1]
GRAPH = {
    "entities": [
        {"id": "person1", "name": "person", "attributes": ["red jacket"]},
        {"id": "person2", "name": "person", "attributes": ["blue shirt"]},
    ],
    "relations": [
        {"subject_id": "person1", "predicate": "looking at", "object_id": "person2"},
        {"subject_id": "person2", "predicate": "waving to", "object_id": "person1"},
    ],
    "context": ["possibly a casual greeting"],
}
PROSE = "A person in a red jacket looks at a person in a blue shirt, who waves back. The interaction appears to be a casual greeting."


@pytest.fixture
def v5_context(tmp_path):
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    config["artifacts_root"] = "artifacts"
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
    for model in ("qwen", "bge"):
        config["models"][model] = str(tmp_path / model)
        Path(config["models"][model]).mkdir()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    return RunContext.load("test_run", root=tmp_path)


@pytest.fixture
def ready_context(v5_context, monkeypatch):
    from validation.steps import prepare_cohort_step
    from extraction.preparation import prepare_input_data

    prepare_cohort_step(v5_context)
    monkeypatch.setattr("extraction.data_preparation.media.probe_duration", lambda _: 10.0)

    def ffmpeg(args, **kwargs):
        assert args[0] == "ffmpeg"
        Image.new("RGB", (16, 8)).save(args[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("extraction.data_preparation.video_processor.subprocess.run", ffmpeg)
    prepare_input_data(v5_context)
    return v5_context


@pytest.fixture
def fake_models(monkeypatch):
    import json

    calls = []

    @contextmanager
    def generator(**kwargs):
        def generate(tasks, callback):
            tasks = list(tasks)
            calls.append(tasks)
            for task in tasks:
                callback(task.task_id, json.dumps(GRAPH) if task.structured_output else PROSE)
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
                        text=json.dumps(GRAPH) if "JSON scene graph" in task.prompt else PROSE,
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
