import asyncio
from copy import deepcopy
from types import SimpleNamespace
import sys

from PIL import Image
import pytest

from artifact_io import read_json, read_jsonl, write_json
from arm_registry import generation_registry, registry, resolve_generation_arm
from extraction.backends.minicpm import MiniCPMBackend
from extraction.backends.qwen import QwenBackend
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.cli import main
from extraction.minicpm_config import minicpm_generation_settings, validate_minicpm_checkpoint
from extraction.minicpm_output import normalize_minicpm_output
from extraction.qwen_config import generation_settings
from extraction.scene_storage import ensure_generation_identity
from extraction.steps import extract_description_scenes, extract_graph_scenes, prompt_provenance, summarize
from helpers import FakeGenerator, NAMED_GRAPH_TEXT, install_doubles, ready_cohort
from pipeline_runtime import ConfigError, RunContext, _validate_config


@pytest.fixture
def minicpm_context(context):
    path = context.root / "models" / "minicpm"
    write_json(path / "config.json", {
        "model_type": "minicpmv4_7", "architectures": ["MiniCPMV4_7ForConditionalGeneration"],
        "dtype": "bfloat16",
    })
    (path / "chat_template.jinja").write_text("test template")
    context.config["models"]["minicpm"] = str(path)
    return context


def test_minicpm_optional_v6_config(context):
    current = deepcopy(context.config)
    _validate_config(current)
    old = deepcopy(current)
    del old["models"]["minicpm"]
    del old["extraction"]["graph"]["minicpm"]
    old["protocol"]["graph_extractors"].remove("minicpm")
    _validate_config(old)
    assert "graph_minicpm" not in generation_registry(old)
    assert registry(current) == registry(old)
    assert all(registry(current)[name] == arm for name, arm in registry(old).items())
    invalid = deepcopy(current)
    del invalid["models"]["minicpm"]
    with pytest.raises(ConfigError):
        _validate_config(invalid)
    invalid = deepcopy(current)
    del invalid["extraction"]["graph"]["minicpm"]
    with pytest.raises(ConfigError):
        _validate_config(invalid)
    invalid = deepcopy(current)
    invalid["protocol"]["graph_extractors"].remove("minicpm")
    with pytest.raises(ConfigError):
        _validate_config(invalid)
    invalid = deepcopy(current)
    invalid["models"]["minicpm"] = ""
    with pytest.raises(ConfigError):
        _validate_config(invalid)


@pytest.mark.parametrize("arm", [None, "graph_minicpm"])
def test_minicpm_cli_arm_default(minicpm_context, monkeypatch, arm):
    calls = []
    monkeypatch.setattr(RunContext, "load", lambda run_id, *, dataset: minicpm_context)
    from extraction.cli import STEP_HANDLERS
    monkeypatch.setitem(STEP_HANDLERS, "extract-graph-scenes", lambda context, **kw: calls.append(kw))
    args = ["extract-graph-scenes", "--run-id", "regression", "--dataset", "microlens_100k", "--schema", "prompts/scene_graph_v9.md",
            "--model", "minicpm", "--repetition-penalty", "1.1", "--presence-penalty", "0"]
    if arm:
        args += ["--arm", arm]
    assert main(args) == 0
    assert calls[0]["arm"] == "graph_minicpm"
    assert calls[0]["repetition_penalty"] == 1.1 and calls[0]["presence_penalty"] == 0


@pytest.mark.parametrize("step,model,arm", [
    ("extract-description-scenes", "minicpm", "graph_minicpm"),
    ("summarize", "minicpm", "graph_minicpm"),
    ("summarize", "gemini", "graph_minicpm"),
    ("extract-graph-scenes", "minicpm", "graph_qwen3vl"),
    ("extract-graph-scenes", "qwen3vl", None),
])
def test_minicpm_cli_rejects_unsupported(step, model, arm, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported invocation reached Run initialization")
    monkeypatch.setattr(RunContext, "load", unexpected)
    args = [step, "--run-id", "regression", "--dataset", "microlens_100k", "--schema", "prompts/scene_graph_v9.md", "--model", model]
    if arm:
        args += ["--arm", arm]
    assert main(args) == 1


def test_minicpm_public_api_rejects_other_stages(minicpm_context):
    with pytest.raises(ValueError, match="only Graph"):
        extract_description_scenes(minicpm_context, model="minicpm", schema="unused")
    with pytest.raises(ValueError):
        summarize(minicpm_context, model="minicpm", schema="unused", arm="graph_minicpm")
    with pytest.raises(ValueError, match="only Graph"):
        summarize(minicpm_context, model="gemini", schema="unused", arm="graph_minicpm")
    with pytest.raises(ValueError, match="requested extraction model"):
        resolve_generation_arm(minicpm_context.config, "graph_minicpm", "graph", "qwen3vl")
    for kind, phase in (("description", "scenes"), ("graph", "summaries")):
        with pytest.raises(ValueError):
            minicpm_context.extraction_dir(kind, "minicpm", phase)


def test_minicpm_checkpoint_and_generation_policy(minicpm_context):
    config = read_json(minicpm_context.path("models", "minicpm") / "config.json")
    validate_minicpm_checkpoint("minicpm", config)
    for delta in ({"model_type": "minicpmv4_6"}, {"architectures": ["Qwen3VLForConditionalGeneration"]},
                  {"dtype": "float16"}, {"quantization_config": {"bits": 4}}):
        with pytest.raises(ValueError):
            validate_minicpm_checkpoint("minicpm", {**config, **delta})
    settings = minicpm_generation_settings()
    expected = generation_settings("qwen3.5", representation="graph")
    assert settings == {**expected, "model": "minicpm"}
    assert settings["enable_thinking"] is False
    custom = minicpm_generation_settings(repetition_penalty=1.2, presence_penalty=0)
    assert custom["repetition_penalty"] == 1.2 and custom["presence_penalty"] == 0


@pytest.mark.parametrize("source,expected", [
    (r"one\ntwo\r\nthree\rfour", "one\ntwo\nthree\nfour"),
    (r'one\n"keep\nstring"\ntwo', 'one\n"keep\\nstring"\ntwo'),
    (r'{"text":"keep\nstring", "escaped":"a\\b"}', r'{"text":"keep\nstring", "escaped":"a\\b"}'),
    (r"one\n`keep\ncode`\ntwo", "one\n`keep\\ncode`\ntwo"),
    ("```json\n" + r'{"x":"a\nb"}' + "\n```" + r"\nafter", "```json\n" + r'{"x":"a\nb"}' + "\n```\nafter"),
    (r"one\n$x\ny$\n\(x\ny\)\nend", "one\n$x\\ny$\n\\(x\\ny\\)\nend"),
    (r"keep\\nbackslashes\nend", "keep\\\\nbackslashes\nend"),
    ("already\nnormal", "already\nnormal"),
])
def test_minicpm_newlines_preserve_literals(source, expected):
    assert normalize_minicpm_output(source) == expected


def test_minicpm_image_preparation_and_sampling(minicpm_context, monkeypatch):
    model_path = minicpm_context.path("models", "minicpm")
    images = []
    for index in range(6):
        path = minicpm_context.root / f"image{index}.png"
        Image.new("RGB", (640, 352), (index, 0, 0)).save(path)
        images.append(str(path))
    calls = []
    def template(messages, **kwargs):
        calls.append((messages, kwargs))
        assert [part["image"].getpixel((0, 0))[0] for part in messages[0]["content"][:-1]] == list(range(6))
        return "rendered"
    backend = MiniCPMBackend(None, SimpleNamespace(apply_chat_template=template), str(model_path), None, [2], {})
    task = QwenGenerationTask("task", tuple(images), "prompt", 384, **minicpm_generation_settings())
    inputs, loaded = backend._prepare(task)
    try:
        assert inputs["prompt"] == "rendered" and len(inputs["multi_modal_data"]["image"]) == 6
        assert calls[0][1] == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert MiniCPMBackend.engine_options() == {"mm_processor_kwargs": {"downsample_mode": "16x"}}
        assert QwenBackend.engine_options() == {}
    finally:
        for image in loaded:
            image.close()
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(RequestOutputKind=SimpleNamespace(FINAL_ONLY="final")))
    params = backend.sampling_params(task)
    assert params["max_tokens"] == 384 and params["temperature"] == 0
    assert params["presence_penalty"] == 1.5 and params["repetition_penalty"] == 1.0
    assert params["extra_args"] == {"generated_token_repetition_penalty": 1.05}


def test_minicpm_context_overflow_never_truncates(minicpm_context, monkeypatch):
    calls = []
    async def render(inputs, **kwargs):
        return [{"prompt_token_ids": list(range(10))}]
    engine = SimpleNamespace(renderer=SimpleNamespace(render_cmpl_async=render), model_config=SimpleNamespace(max_model_len=20))
    backend = MiniCPMBackend(engine, None, "unused", None, [], {})
    monkeypatch.setattr(backend, "_prepare", lambda task: ({"prompt": "full input"}, []))
    monkeypatch.setattr(backend, "sampling_params", lambda task: calls.append(task.max_new_tokens))
    monkeypatch.setitem(sys.modules, "vllm.renderers.params", SimpleNamespace(TokenizeParams=lambda **kw: kw))
    task = QwenGenerationTask("task", (), "full input", 384)
    with pytest.raises(RuntimeError, match="required=394.*max_model_len=20"):
        asyncio.run(backend.generate(task))
    assert calls == [384]


def test_minicpm_generation_identity(minicpm_context):
    context = minicpm_context
    arm = generation_registry(context.config)["graph_minicpm"]
    def provenance():
        return prompt_provenance(context, "prompts/scene_graph_v9.md", arm)
    initial = provenance()
    directory = context.scene_arm_dir(arm.name)
    ensure_generation_identity(directory, initial)
    ensure_generation_identity(directory, initial)
    initial_identity = read_json(directory / "generation_identity.json")
    assert initial["settings"]["processor"]["downsample_mode"] == "16x"
    for change in ("generation", "processor", "output_normalization"):
        changed = deepcopy(initial)
        changed["settings"][change] = "changed"
        ensure_generation_identity(directory, changed)
        assert read_json(directory / "generation_identity.json") == initial_identity
    (context.path("models", "minicpm") / "chat_template.jinja").write_text("changed template")
    ensure_generation_identity(directory, provenance())
    assert read_json(directory / "generation_identity.json") == initial_identity
    write_json(directory / "raw_responses" / "old.jsonl", {})
    ensure_generation_identity(directory, provenance(), force=True)
    assert read_json(directory / "generation_identity.json") != initial_identity
    assert not (directory / "raw_responses" / "old.jsonl").exists()


def test_minicpm_extract_store_resume_and_retry(minicpm_context, monkeypatch):
    from preparation.input_data import prepare_input_data
    context = minicpm_context
    ready_cohort(context)
    install_doubles(monkeypatch)
    prepare_input_data(context)
    raw = NAMED_GRAPH_TEXT.replace("\n", r"\n")
    factory = FakeGenerator(raw)
    monkeypatch.setattr("extraction.steps.qwen_generator", factory)
    options = dict(model="minicpm", schema="prompts/scene_graph_v9.md")
    extract_graph_scenes(context, **options)
    assert factory.tasks and all(task.model == "minicpm" for task in factory.tasks)
    assert len(factory.tasks[0].image_paths) == 6
    assert all(1 <= len(task.image_paths) <= 6 and task.max_new_tokens == 384 for task in factory.tasks)
    directory = context.scene_arm_dir("graph_minicpm")
    cid = factory.tasks[0].task_id.split(":")[0]
    output = directory / f"{cid}.jsonl"
    saved = read_jsonl(output)
    assert isinstance(saved[0]["scene_graph"], dict) and not saved[0]["warning"]
    assert read_jsonl(directory / "raw_responses" / f"{cid}.jsonl")[0]["raw_output"] == raw
    before, count = output.read_bytes(), len(factory.tasks)
    extract_graph_scenes(context, **options)
    assert len(factory.tasks) == count and output.read_bytes() == before
    # Warning and failed rows are retried; successful rows are retained.
    from extraction.graph_warnings import PARSE_FAILED
    from extraction.scene_storage import write_scene_records
    saved[0]["warning"] = [PARSE_FAILED]
    saved[1] = {"content_id": cid, "scene_idx": saved[1]["scene_idx"], "tokens": 1,
                "error": "interrupted", "raw_output": "partial"}
    write_scene_records(output, saved)
    extract_graph_scenes(context, **options)
    assert len(factory.tasks) == count + 2
    assert all(not row["warning"] for row in read_jsonl(output))
    count = len(factory.tasks)
    before = output.read_bytes()
    extract_graph_scenes(context, **options, presence_penalty=0)
    assert len(factory.tasks) == count and output.read_bytes() == before
    saved = read_jsonl(output)
    saved[0]["warning"] = [PARSE_FAILED]
    write_scene_records(output, saved)
    extract_graph_scenes(context, **options, presence_penalty=0)
    assert len(factory.tasks) == count + 1
    assert factory.tasks[-1].presence_penalty == 0
    assert read_json(directory / "generation_stats.json")["failed"] == 0
    truncated = FakeGenerator(raw, finish_reason="length")
    monkeypatch.setattr("extraction.steps.qwen_generator", truncated)
    extract_graph_scenes(context, **options, force=True)
    assert read_jsonl(output)[0]["raw_output"] == raw
    assert "truncated" in read_jsonl(output)[0]["error"]
