"""CPU checks for the supported local backend's requests and recovery metadata."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
import sys

from PIL import Image
import pytest

from arm_registry import generation_registry
from artifact_io import read_json, write_json
from extraction.backends.qwen import QwenBackend
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.qwen_config import generation_settings
from extraction.scene_storage import ensure_generation_identity
from extraction.steps import prompt_provenance


def test_qwen_image_preparation_and_sampling(context, monkeypatch):
    model_path = context.path("models", "qwen3vl_2b")
    images = []
    for index in range(6):
        path = context.root / f"image{index}.png"
        Image.new("RGB", (640, 352), (index, 0, 0)).save(path)
        images.append(str(path))
    calls = []

    def template(messages, **kwargs):
        calls.append((messages, kwargs))
        assert [part["image"].getpixel((0, 0))[0] for part in messages[0]["content"][:-1]] == list(range(6))
        return "rendered"

    backend = QwenBackend(None, SimpleNamespace(apply_chat_template=template), str(model_path), None, [2], {})
    task = QwenGenerationTask(
        "task", tuple(images), "prompt", 384,
        **generation_settings("qwen3vl_2b", representation="graph"),
    )
    inputs, loaded = backend._prepare(task)
    try:
        assert inputs["prompt"] == "rendered" and len(inputs["multi_modal_data"]["image"]) == 6
        assert calls[0][1] == dict(tokenize=False, add_generation_prompt=True)
    finally:
        for image in loaded:
            image.close()
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(RequestOutputKind=SimpleNamespace(FINAL_ONLY="final")))
    params = backend.sampling_params(task)
    assert params["max_tokens"] == 384 and params["temperature"] == 0
    assert params["presence_penalty"] == 1.5 and params["repetition_penalty"] == 1.0
    assert params["extra_args"] == {"generated_token_repetition_penalty": 1.05}


def test_qwen_context_overflow_never_truncates(monkeypatch):
    calls = []

    async def render(inputs, **kwargs):
        return [{"prompt_token_ids": list(range(10))}]

    engine = SimpleNamespace(renderer=SimpleNamespace(render_cmpl_async=render), model_config=SimpleNamespace(max_model_len=20))
    backend = QwenBackend(engine, None, "unused", None, [], {})
    monkeypatch.setattr(backend, "_prepare", lambda task: ({"prompt": "full input"}, []))
    monkeypatch.setattr(backend, "sampling_params", lambda task: calls.append(task.max_new_tokens))
    monkeypatch.setitem(sys.modules, "vllm.renderers.params", SimpleNamespace(TokenizeParams=lambda **kw: kw))
    task = QwenGenerationTask("task", (), "full input", 384)
    with pytest.raises(RuntimeError, match="required=394.*max_model_len=20"):
        asyncio.run(backend.generate(task))
    assert calls == [384]


def test_qwen_generation_identity_preserves_initial_settings(context):
    arm = generation_registry(context.config)["graph_qwen3vl_2b"]
    initial = prompt_provenance(context, "prompts/scene_graph_v9_2b.md", arm)
    directory = context.scene_arm_dir(arm.name)
    ensure_generation_identity(directory, initial)
    before = read_json(directory / "generation_identity.json")
    changed = deepcopy(initial)
    changed["settings"]["generation"]["presence_penalty"] = 0
    ensure_generation_identity(directory, changed)
    assert read_json(directory / "generation_identity.json") == before
    write_json(directory / "old.jsonl", {})
    ensure_generation_identity(directory, changed, force=True)
    assert read_json(directory / "generation_identity.json") != before
    assert not (directory / "old.jsonl").exists()


def test_qwen_sizes_use_separate_sources_and_checkpoints(context):
    from extraction.qwen_config import validate_checkpoint

    small = generation_registry(context.config)["graph_qwen3vl_2b"]
    large = generation_registry(context.config)["graph_qwen3vl_8b"]
    assert context.scene_arm_dir(small.name) != context.scene_arm_dir(large.name)
    assert context.extraction_dir("graph", "qwen3vl_8b", "scenes") == context.scene_arm_dir(large.name)
    identities = [
        prompt_provenance(context, f"prompts/scene_graph_v9_{size}.md", arm)
        for size, arm in (("2b", small), ("8b", large))
    ]
    assert [item["scene_arm"] for item in identities] == ["graph_qwen3vl_2b", "graph_qwen3vl_8b"]
    assert identities[0]["model"]["path"] != identities[1]["model"]["path"]
    assert identities[1]["settings"]["generation"]["model"] == "qwen3vl_8b"
    config = read_json(context.path("models", "qwen3vl_2b") / "config.json")
    with pytest.raises(ValueError, match="states 8B"):
        validate_checkpoint("qwen3vl_8b", config, context.path("models", "qwen3vl_2b"))
    with pytest.raises(ValueError, match="states 8B"):
        context.config["models"]["qwen3vl_8b"] = context.config["models"]["qwen3vl_2b"]
        prompt_provenance(context, "prompts/scene_graph_v9_8b.md", large)
