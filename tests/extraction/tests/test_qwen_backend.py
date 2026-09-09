from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest
import torch

from extraction.backends.qwen import QwenBackend
from extraction.backends.qwen_penalty import GeneratedTokenPenalty, PENALTY_KEY
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.qwen_config import QWEN_DEFAULTS, qwen_settings


@pytest.fixture
def fake_vllm(monkeypatch):
    module = ModuleType("vllm")
    module.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    sampling = ModuleType("vllm.sampling_params")
    sampling.RequestOutputKind = SimpleNamespace(FINAL_ONLY="final")
    sampling.StructuredOutputsParams = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)
    renderer = ModuleType("vllm.renderers.params")
    renderer.TokenizeParams = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm.renderers.params", renderer)
    return module


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "formatted prompt"


class Engine:
    def __init__(self, length=8):
        self.model_config = SimpleNamespace(max_model_len=1024)
        self.renderer = self
        self.length = length
        self.calls = []

    async def render_cmpl_async(self, prompts, *, tok_params):
        assert tok_params.max_total_tokens is None
        assert tok_params.add_special_tokens is False
        self.prompts = prompts
        self.image_colors = [image.getpixel((0, 0))
                             for image in prompts[0].get("multi_modal_data", {}).get("image", [])]
        return [{"type": "multimodal", "prompt_token_ids": list(range(self.length))}]

    async def generate(self, prompt, params, request_id):
        self.calls.append((prompt, params, request_id))
        yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(
            text=" generated text ", token_ids=[1, 2], finish_reason="stop", stop_reason=7)])

    def shutdown(self, **_kwargs):
        self.closed = True


@pytest.fixture
def backend(fake_vllm):
    backend = QwenBackend(Engine(), Processor(), "model", ThreadPoolExecutor(2), [7, 9], {})
    yield backend
    backend.close()


def test_native_images_order_and_greedy_settings(backend, tmp_path):
    paths = []
    for color in ("red", "blue"):
        path = tmp_path / f"{color}.png"
        Image.new("RGB", (8, 8), color).save(path)
        paths.append(str(path))
    task = QwenGenerationTask("a:0", tuple(paths), "prompt", 32, repetition_penalty=1.05)
    output = asyncio.run(backend.generate(task))
    assert (output.text, output.prompt_tokens, output.output_tokens) == (" generated text ", 8, 2)
    assert (output.finish_reason, output.stop_reason) == ("stop", 7)
    content = backend.processor.messages[0]["content"]
    assert [item["type"] for item in content] == ["image", "image", "text"]
    assert content[-1]["text"] == "prompt"
    assert backend.engine.image_colors == [(255, 0, 0), (0, 0, 255)]
    _, params, request_id = backend.engine.calls[0]
    assert request_id == "a:0"
    assert (params.temperature, params.max_tokens, params.top_p, params.top_k) == (0, 32, 1, 0)
    assert params.repetition_penalty == 1
    assert params.extra_args[PENALTY_KEY] == 1.05
    assert params.stop_token_ids == [7, 9]
    assert params.output_kind == "final"


@pytest.mark.parametrize("kind", ["json", "grammar"])
def test_structured_output_and_generated_token_penalty_are_passed_together(backend, kind):
    from extraction.structured_output import GRAPH_JSON_SCHEMA, SUMMARY_GRAMMAR
    constraint = {kind: GRAPH_JSON_SCHEMA if kind == "json" else SUMMARY_GRAMMAR}
    task = QwenGenerationTask("structured", (), "prompt", 32,
                              repetition_penalty=1.15, structured_output=constraint)
    asyncio.run(backend.generate(task))
    params = backend.engine.calls[0][1]
    assert vars(params.structured_outputs) == constraint
    assert params.repetition_penalty == 1.0 and params.extra_args[PENALTY_KEY] == 1.15


def test_structured_output_compilation_error_never_generates_unconstrained(backend, monkeypatch):
    def fail(**kwargs):
        raise ValueError("grammar compilation failed")
    monkeypatch.setattr(sys.modules["vllm.sampling_params"], "StructuredOutputsParams", fail)
    with pytest.raises(RuntimeError, match="compilation"):
        asyncio.run(backend.generate(QwenGenerationTask("a", (), "prompt", 32,
                                                        structured_output={"grammar": "bad"})))
    assert not backend.engine.calls


@pytest.mark.parametrize("seed", [None, 43])
def test_text_only_sampling(backend, seed):
    task = QwenGenerationTask("summary", (), "summary prompt", 64, True, seed, 0.2, 0.8, 20)
    asyncio.run(backend.generate(task))
    params = backend.engine.calls[0][1]
    assert (params.temperature, params.top_p, params.top_k, params.seed) == (0.2, 0.8, 20, seed)
    assert "multi_modal_data" not in backend.engine.prompts[0]
    with pytest.raises(RuntimeError, match="requires temperature"):
        asyncio.run(backend.generate(replace(task, temperature=None)))


def test_expanded_image_context_is_checked_without_truncation(backend):
    backend.engine.length = 1000
    with pytest.raises(RuntimeError, match="a:0.*required=1032.*max_model_len=1024"):
        asyncio.run(backend.generate(QwenGenerationTask("a:0", (), "prompt", 32)))
    assert backend.engine.calls == []


def test_loader_parallelism_is_bounded_and_images_are_closed(backend, monkeypatch):
    import threading
    import time

    lock = threading.Lock()
    active, peak = 0, 0
    images = []

    def prepare(task):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        image = Image.new("RGB", (2, 2))
        images.append(image)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {"prompt": task.prompt}, [image]

    monkeypatch.setattr(backend, "_prepare", prepare)

    async def run():
        await asyncio.gather(*(backend.generate(QwenGenerationTask(str(i), (), "prompt", 32))
                               for i in range(8)))

    asyncio.run(run())
    assert peak == 2
    for image in images:
        with pytest.raises(ValueError, match="closed"):
            image.getpixel((0, 0))


def test_cancelled_preparation_releases_images_when_loader_finishes(backend, monkeypatch):
    import threading

    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    def prepare(task):
        started.set()
        assert release.wait(5)
        return {"prompt": task.prompt}, [SimpleNamespace(close=closed.set)]

    monkeypatch.setattr(backend, "_prepare", prepare)

    async def run():
        task = asyncio.create_task(backend.generate(QwenGenerationTask("a", (), "prompt", 32)))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        async def wait_closed():
            while not closed.is_set():
                await asyncio.sleep(0.001)
        await asyncio.wait_for(wait_closed(), timeout=5)

    asyncio.run(run())


def test_batch_penalty_matches_original_and_tracks_live_tokens():
    from benchmarks.qwen_transformers_reference import GeneratedTokenRepetitionPenalty

    penalty = GeneratedTokenPenalty(None, None, None)
    outputs = [2, 2, 3]
    params = SimpleNamespace(extra_args={PENALTY_KEY: 1.05})
    penalty.update_state(SimpleNamespace(removed=[], moved=[], added=[(0, params, [0, 1], outputs)]))
    logits = torch.tensor([[2.0, -2.0, 3.0, -3.0, 4.0]])
    expected = GeneratedTokenRepetitionPenalty(1.05, 2)(torch.tensor([[0, 1, 2, 2, 3]]), logits.clone())
    torch.testing.assert_close(penalty.apply(logits.clone()), expected)
    outputs.append(4)
    result = penalty.apply(logits.clone())
    assert result[0, 0] == logits[0, 0]  # Prompt tokens never penalized.
    assert result[0, 4] < logits[0, 4]
    assert not penalty.is_argmax_invariant()


def test_penalty_batch_removal_replacement_move_and_swap():
    penalty = GeneratedTokenPenalty(None, None, None)
    def params(n):
        return SimpleNamespace(extra_args={PENALTY_KEY: n})
    penalty.update_state(SimpleNamespace(removed=[], added=[], moved=[]))
    penalty.update_state(SimpleNamespace(removed=[], moved=[], added=[
        (0, params(2), [], [1]), (1, params(3), [], [2]), (2, params(1), [], [0]),
    ]))
    penalty.update_state(SimpleNamespace(removed=[], added=[], moved=[(0, 1, SimpleNamespace(name="SWAP"))]))
    result = penalty.apply(torch.ones(3, 4))
    assert result[0, 2] == pytest.approx(1 / 3)
    assert result[1, 1] == 0.5
    penalty.update_state(SimpleNamespace(removed=[1], added=[], moved=[(0, 2, SimpleNamespace(name="UNIDIRECTIONAL"))]))
    assert set(penalty.requests) == {2}
    penalty.update_state(SimpleNamespace(removed=[], moved=[], added=[(2, params(1), [], [])]))
    assert penalty.requests == {}


@pytest.mark.parametrize("value", [0, -1, float("nan"), True])
def test_invalid_penalty(value):
    with pytest.raises(ValueError):
        GeneratedTokenPenalty.validate_params(SimpleNamespace(extra_args={PENALTY_KEY: value}))


@pytest.mark.parametrize("settings", [
    {"max_num_seqs": 0}, {"renderer_num_workers": True}, {"max_model_len": False},
    {"gpu_memory_utilization": float("nan")}, {"async_scheduling": "auto"},
    {"enable_prefix_caching": 1}, {"unknown": 1},
    {"enable_chunked_prefill": False, "max_model_len": 20000, "max_num_batched_tokens": 100},
])
def test_invalid_runtime_settings(settings):
    with pytest.raises(ValueError):
        qwen_settings(settings)


def test_old_config_uses_same_defaults():
    assert qwen_settings() == QWEN_DEFAULTS
    assert qwen_settings({"max_num_seqs": 64})["max_num_seqs"] == 64


def test_engine_configuration_and_eos_are_explicit(fake_vllm, monkeypatch, tmp_path):
    import extraction.backends.qwen as module

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [11, 12], "temperature": 0.7}))
    arguments = []
    engine = Engine()
    engine.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=True))
    processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=7))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: processor)))
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", SimpleNamespace(AsyncEngineArgs=lambda **k: k))
    def create(args):
        arguments.append(args)
        return engine
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", SimpleNamespace(AsyncLLM=SimpleNamespace(from_engine_args=create)))
    monkeypatch.setattr(module, "version", lambda name: "0.28.0" if name == "vllm" else "test")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "fake CUDA")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=48 * 1024**3))
    backend = QwenBackend.from_pretrained(str(tmp_path), settings={"max_num_seqs": 8}, image_limit=6)
    try:
        args = arguments[0]
        assert args["max_model_len"] == -1
        assert args["max_num_seqs"] == 8
        assert args["dtype"] == "bfloat16" and args["kv_cache_dtype"] == "auto"
        assert args["model_impl"] == "vllm" and args["tensor_parallel_size"] == 1
        assert args["generation_config"] == "vllm"
        assert args["limit_mm_per_prompt"] == {"image": 6, "video": 0}
        assert args["structured_outputs_config"] == {"backend": "xgrammar"}
        assert backend.stop_token_ids == [11, 12]
        assert backend.runtime_info["settings"]["max_model_len"] == 1024
    finally:
        backend.close()
