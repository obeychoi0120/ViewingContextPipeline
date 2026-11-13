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


def test_qwen_requests_sampling_and_context_overflow(context, monkeypatch):
    # Images are prepared in order and sampling follows the graph settings.
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
    assert params["presence_penalty"] == 0.0 and params["repetition_penalty"] == 1.0
    assert params["extra_args"] == {"generated_token_repetition_penalty": 1.05}

    # A context overflow fails instead of truncating the input.
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


def test_qwen_identity_sizes_and_prompts(context):
    from pathlib import Path
    from extraction.qwen_config import validate_checkpoint

    # The generation identity keeps its initial settings unless forced.
    arm = generation_registry(context.config)["graph_qwen3vl_2b"]
    initial = prompt_provenance(context, "prompts/scene_graph_v9_core.md", arm)
    directory = context.scene_arm_dir(arm.name)
    ensure_generation_identity(directory, initial)
    before = read_json(directory / "generation_identity.json")
    changed = deepcopy(initial)
    changed["settings"]["generation"]["presence_penalty"] = 1.5
    ensure_generation_identity(directory, changed)
    assert read_json(directory / "generation_identity.json") == before
    write_json(directory / "old.jsonl", {})
    ensure_generation_identity(directory, changed, force=True)
    assert read_json(directory / "generation_identity.json") != before
    assert not (directory / "old.jsonl").exists()

    # Each extended prompt only appends an extension block to its core, verbatim.
    prompts = Path(__file__).resolve().parents[1] / "prompts"
    for version in ("v9", "v10"):
        core = (prompts / f"scene_graph_{version}_core.md").read_text(encoding="utf-8")
        extended = (prompts / f"scene_graph_{version}_extended.md").read_text(encoding="utf-8")
        assert extended.startswith(core), version
        assert extended[len(core):].lstrip().startswith("[Additional Instructions]"), version

    # Qwen sizes use separate sources and checkpoints.
    small = generation_registry(context.config)["graph_qwen3vl_2b"]
    large = generation_registry(context.config)["graph_qwen3vl_8b"]
    assert context.scene_arm_dir(small.name) != context.scene_arm_dir(large.name)
    assert context.extraction_dir("graph", "qwen3vl_8b", "scenes") == context.scene_arm_dir(large.name)
    identities = [
        prompt_provenance(context, f"prompts/scene_graph_v9_{name}.md", arm)
        for name, arm in (("core", small), ("extended", large))
    ]
    assert [item["scene_arm"] for item in identities] == ["graph_qwen3vl_2b", "graph_qwen3vl_8b"]
    assert identities[0]["model"]["path"] != identities[1]["model"]["path"]
    assert identities[1]["settings"]["generation"]["model"] == "qwen3vl_8b"
    config = read_json(context.path("models", "qwen3vl_2b") / "config.json")
    with pytest.raises(ValueError, match="expects 8B, but the checkpoint name states 2B"):
        validate_checkpoint("qwen3vl_8b", config, context.path("models", "qwen3vl_2b"))
    # A renamed checkpoint without a stated size is accepted.
    validate_checkpoint("qwen3vl_8b", config, context.path("models", "qwen3vl_2b").parent / "local-vl")
    with pytest.raises(ValueError, match="expects 8B"):
        context.config["models"]["qwen3vl_8b"] = context.config["models"]["qwen3vl_2b"]
        prompt_provenance(context, "prompts/scene_graph_v9_extended.md", large)


@pytest.mark.torch
def test_stop_before_repeated_line(monkeypatch):
    import numpy as np
    import torch
    from extraction.backends.qwen import QwenBackend
    from extraction.backends.qwen_repeat import (
        REPEAT_STOP_KEY, LineHashes, RepeatedLineStop, RepeatedLineStopV2, drop_repeated_last_line,
    )
    from extraction.cli import main

    NL, STOP, VOCAB = 9, 15, 16
    hashes = LineHashes([NL], "cpu")

    def repeated(*outputs):
        width = max(map(len, outputs))
        history = torch.tensor([o + [0] * (width - len(o)) for o in outputs])
        valid = torch.tensor([[i < len(o) for i in range(width)] for o in outputs])
        return hashes.repeated(history, valid, VOCAB).tolist()

    line = [1, 2, 3, NL]
    # Only a partial line identical to an earlier complete line counts; one-token lines may repeat.
    assert repeated(line + [1, 2, 3], line + [1, 2], line + [1, 2, 3, 4], [5, NL, 5], line, line + [4, 5, NL, 1, 2, 3]) == [
        True, False, False, False, False, True]
    assert repeated(line + [2, 1, 3]) == [False]

    # The stop token wins only where a line break would have been chosen.
    logits = torch.zeros(2, VOCAB)
    logits[:, NL] = 5.0
    logits[1, 4] = 7.0
    hashes.raise_stop(logits, torch.tensor([True, True]), torch.tensor([STOP, STOP]))
    assert logits.argmax(1).tolist() == [STOP, 4]
    untouched = torch.zeros(1, VOCAB)
    untouched[0, NL] = 5.0
    hashes.raise_stop(untouched, torch.tensor([False]), torch.tensor([STOP]))
    assert untouched.argmax(1).tolist() == [NL]

    # V1: live output lists; a request without the option is left alone.
    params = SimpleNamespace(extra_args={REPEAT_STOP_KEY: STOP})
    v1 = RepeatedLineStop(None, "cpu", False)
    v1.hashes = hashes
    outputs = [line + [1, 2, 3], line + [1, 2, 3]]
    v1.update_state(SimpleNamespace(removed=[], moved=[], added=[
        (0, params, [], outputs[0]), (1, SimpleNamespace(extra_args={}), [], outputs[1])]))
    logits = torch.zeros(2, VOCAB)
    logits[:, NL] = 1.0
    assert v1.apply(logits).argmax(1).tolist() == [STOP, NL]
    with pytest.raises(ValueError):
        RepeatedLineStop.validate_params(SimpleNamespace(extra_args={REPEAT_STOP_KEY: -1}))

    # V2: committed history on device plus this step's input token.
    prompt = [7, 7]
    tokens = torch.zeros(3, 20, dtype=torch.long)
    tokens[0, : len(prompt) + 6] = torch.tensor(prompt + line + [1, 2])
    states = SimpleNamespace(
        max_num_reqs=3, device="cpu",
        prompt_len=SimpleNamespace(gpu=torch.tensor([2, 2, 2]), np=np.array([2, 2, 2])),
        total_len=SimpleNamespace(gpu=torch.tensor([8, 8, 8])),
        all_token_ids=SimpleNamespace(gpu=tokens),
    )
    v2 = RepeatedLineStopV2(None, states)
    v2.hashes = hashes
    assert v2.add_request(0, params) and not v2.add_request(1, SimpleNamespace(extra_args=None))
    v2.apply_staged_writes()
    context = SimpleNamespace(
        expanded_idx_mapping=torch.tensor([0, 1]), idx_mapping_np=np.array([0, 1]),
        pos=torch.tensor([8, 8]), input_ids=torch.tensor([3, 3]),
        seq_lens_upper_bound_np=np.array([9, 9]),
    )
    tokens[1] = tokens[0]
    logits = torch.zeros(2, VOCAB)
    logits[:, NL] = 1.0
    assert v2.apply(logits, context).argmax(1).tolist() == [STOP, NL]

    # The backend sends the option only when enabled and drops the duplicate it left.
    assert drop_repeated_last_line("a\nb\na") == "a\nb"
    assert drop_repeated_last_line("a\nb\nc") == "a\nb\nc"
    assert "stop_repeated_lines" not in generation_settings("qwen3vl_2b", representation="graph")
    enabled = generation_settings("qwen3vl_2b", representation="graph", stop_repeated_lines=True)
    with pytest.raises(ValueError):
        generation_settings("qwen3vl_2b", representation="description", stop_repeated_lines=True)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(RequestOutputKind=SimpleNamespace(FINAL_ONLY="final")))
    backend = QwenBackend(None, None, "unused", None, [151645], {})
    sent = backend.sampling_params(QwenGenerationTask("t", (), "p", 8, **enabled))["extra_args"]
    assert sent[REPEAT_STOP_KEY] == 151645
    plain = generation_settings("qwen3vl_2b", representation="graph")
    assert REPEAT_STOP_KEY not in backend.sampling_params(QwenGenerationTask("t", (), "p", 8, **plain))["extra_args"]
    for model, step in (("gemini", "extract-graph-scenes"), ("qwen3vl_2b", "extract-description-scenes")):
        assert main([step, "--run-id", "r", "--dataset", "microlens_100k", "--schema", "x.md",
                     "--model", model, "--stop-repeated-lines"]) == 1
