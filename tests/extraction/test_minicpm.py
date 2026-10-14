"""MiniCPM's independent settings, native rendering and generation identities."""

import asyncio
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from extraction.backends.minicpm import MiniCPMBackend, normalize_newlines
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.minicpm_config import minicpm_settings
from extraction import steps
from artifact_io import read_json, write_json
from pipeline_runtime import _validate_config, ConfigError


@pytest.mark.parametrize('value', [4, 16, None])
def test_settings_and_native_engine_options(current_context, value):
    section = {} if value is None else {'vision_downsample': value}
    current_context.config['extraction']['minicpm'] = section
    _validate_config(current_context.config)
    current_context.config['extraction']['qwen']['max_num_seqs'] = 1
    actual = minicpm_settings(section)
    assert actual['max_num_seqs'] == 32
    native = MiniCPMBackend.engine_options(actual)
    assert native['mm_processor_kwargs'] == {'downsample_mode': f'{value or 16}x', 'max_slice_nums': 9}
    assert 'vision_downsample' not in native and 'max_slice_nums' not in native
    assert 'enable_prefix_caching' not in native and 'async_scheduling' not in native


@pytest.mark.parametrize('value', ['4', '16', 4.0, 16.0, True, False, 0, 8, None, [], {}])
def test_invalid_downsample_rejected(current_context, value):
    current_context.config['extraction']['minicpm']['vision_downsample'] = value
    with pytest.raises(ConfigError, match='vision_downsample'):
        _validate_config(current_context.config)


def test_newline_normalization_preserves_json_escapes():
    text = r'[Context]\nmedium: live_action\n' + r'{"label": "line\nquote\" slash\\"}'
    assert normalize_newlines(text) == '[Context]\nmedium: live_action\n' + r'{"label": "line\nquote\" slash\\"}'
    fenced = r'```json\n{"label": "line\n"}\n```'
    assert normalize_newlines(fenced) == '```json\n' + r'{"label": "line\n"}' + '\n```'


@pytest.fixture
def native_backend(current_context, tmp_path, monkeypatch):
    import torch
    import transformers
    import extraction.backends.minicpm as mini
    import extraction.backends.vllm_backend as common

    model = current_context.path('models', 'minicpm')
    write_json(model / 'config.json', {'architectures': ['MiniCPMV4_7ForConditionalGeneration']})
    write_json(model / 'generation_config.json', {'eos_token_id': [248044, 248046]})
    calls = {}
    processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=1))

    def chat(messages, **kwargs):
        calls['chat'] = kwargs
        calls['images'] = [p['image'].size for p in messages[0]['content'][:-1]]
        return 'native template with six images'

    processor.apply_chat_template = chat
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', lambda *a, **k: processor)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'mock GPU')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(total_memory=100))
    monkeypatch.setattr(mini, 'version', lambda _: '5.18.0')
    monkeypatch.setattr(common, 'version', lambda _: 'test-build')
    monkeypatch.setattr(common, '_vllm_commit', lambda: 'c85616efc6f0eaa50ca1f1a09c32368aedffd3d8')

    async def render(inputs, **kwargs):
        calls['render'] = inputs
        return [{'prompt_token_ids': list(range(calls.get('prompt_length', 20)))}]

    async def generate(inputs, params, task_id):
        calls['sampling'] = params
        yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(
            text=r'[Context]\nmedium: live_action', token_ids=[1, 2],
            finish_reason='stop', stop_reason=248044)])

    engine = SimpleNamespace(renderer=SimpleNamespace(render_cmpl_async=render), generate=generate,
                             model_config=SimpleNamespace(max_model_len=8192),
                             vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=True)),
                             shutdown=lambda **k: None)
    def args(**kwargs):
        calls['engine'] = kwargs
        return kwargs

    modules = {
        'vllm': SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
        'vllm.engine.arg_utils': SimpleNamespace(AsyncEngineArgs=args),
        'vllm.v1.engine.async_llm': SimpleNamespace(AsyncLLM=SimpleNamespace(from_engine_args=lambda _: engine)),
        'vllm.model_executor.models.registry': SimpleNamespace(ModelRegistry=SimpleNamespace(
            get_supported_archs=lambda: ['MiniCPMV4_7ForConditionalGeneration'])),
        'vllm.renderers.params': SimpleNamespace(TokenizeParams=lambda **kwargs: kwargs),
        'vllm.sampling_params': SimpleNamespace(RequestOutputKind=SimpleNamespace(FINAL_ONLY='final')),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    paths = []
    for i in range(6):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (640, 352), (i, 0, 0)).save(path)
        paths.append(str(path))
    return model, calls, QwenGenerationTask('six-frame', tuple(paths), 'graph prompt', 384)


@pytest.mark.parametrize('downsample', [4, 16])
def test_native_backend_eos_thinking_and_context(native_backend, downsample):
    model, calls, task = native_backend
    backend = MiniCPMBackend.from_pretrained(str(model), settings={'vision_downsample': downsample})
    try:
        output = asyncio.run(backend.generate(task))
        assert calls['engine']['mm_processor_kwargs']['downsample_mode'] == f'{downsample}x'
        assert calls['chat']['enable_thinking'] is False
        assert calls['images'] == [(640, 352)] * 6
        assert calls['sampling']['stop_token_ids'] == [248044, 248046]
        assert output.text == '[Context]\nmedium: live_action'
        assert (output.prompt_tokens, output.output_tokens) == (20, 2)
        calls['prompt_length'] = 8000
        with pytest.raises(RuntimeError, match='required=8384.*exceeds max_model_len=8192'):
            asyncio.run(backend.generate(task))
        assert task.max_new_tokens == 384 and len(task.image_paths) == 6
    finally:
        backend.close()


def test_native_build_requirements(native_backend, monkeypatch):
    import extraction.backends.minicpm as mini
    model, _, _ = native_backend
    monkeypatch.setattr(mini, 'version', lambda _: '5.17.0')
    with pytest.raises(RuntimeError, match='transformers>=5.18.0'):
        MiniCPMBackend.from_pretrained(str(model))
    monkeypatch.setattr(mini, 'version', lambda _: '5.18.0')
    monkeypatch.setattr(sys.modules['vllm.model_executor.models.registry'].ModelRegistry,
                        'get_supported_archs', lambda: [])
    with pytest.raises(RuntimeError, match='PR #58674'):
        MiniCPMBackend.from_pretrained(str(model))


def test_generation_changes_block_resume_graph_and_summary(ready_context, fake_models):
    from arm_registry import registry
    from validation.graph_inputs import source_identity
    from validation.representation_inputs import documents_for_arm
    from extraction.minicpm_provenance import contract_identity

    context = ready_context
    context.config['extraction']['minicpm']['vision_downsample'] = 16
    arm = registry(context.config)['graph_minicpm']
    options = dict(model='minicpm', arm=arm.name, schema='prompts/scene_graph_v8.md')
    steps.extract_graph_scenes(context, **options)
    generation = context.scene_arm_dir(arm.name) / 'generation.json'
    old_identity = contract_identity(read_json(generation))
    old_graph_identity = source_identity(context, context.require_ready_cohort(), arm)
    calls = len(fake_models)
    steps.extract_graph_scenes(context, **options)
    assert len(fake_models) == calls  # completed scenes resume without engine startup
    steps.summarize(context, model='gemini', arm=arm.name, schema='prompts/summary_graph_v8.md')
    context.config['extraction']['minicpm']['vision_downsample'] = 4
    with pytest.raises(RuntimeError, match='--force'):
        steps.extract_graph_scenes(context, **options)
    with pytest.raises(RuntimeError, match='--force'):
        source_identity(context, context.require_ready_cohort(), arm)
    with pytest.raises(RuntimeError, match='--force'):
        steps.summarize(context, model='gemini', arm=arm.name, schema='prompts/summary_graph_v8.md')
    steps.extract_graph_scenes(context, **options, force=True)
    assert old_identity != contract_identity(read_json(generation))
    assert old_graph_identity != source_identity(context, context.require_ready_cohort(), arm)
    with pytest.raises(RuntimeError, match='Scene input changed'):
        documents_for_arm(context, context.require_ready_cohort(), arm)
    steps.summarize(context, model='gemini', arm=arm.name, schema='prompts/summary_graph_v8.md', force=True)
    assert documents_for_arm(context, context.require_ready_cohort(), arm)[0]['text']
    # Chat-template changes are also generation changes, including nested Jinja files.
    directory = context.path('models', 'minicpm') / 'chat_templates'
    directory.mkdir()
    (directory / 'default.jinja').write_text('new template')
    with pytest.raises(RuntimeError, match='--force'):
        steps.extract_graph_scenes(context, **options)


def test_force_interruption_cannot_reuse_old_mode(ready_context, fake_models, monkeypatch):
    from contextlib import contextmanager
    from extraction.scene_storage import read_scene_records

    context = ready_context
    context.config['extraction']['minicpm']['vision_downsample'] = 16
    options = dict(model='minicpm', arm='graph_minicpm', schema='prompts/scene_graph_v8.md')
    steps.extract_graph_scenes(context, **options)
    directory = context.scene_arm_dir('graph_minicpm')
    context.config['extraction']['minicpm']['vision_downsample'] = 4
    @contextmanager
    def interrupted(**kwargs):
        def generate(*args, **kwargs):
            raise KeyboardInterrupt
        yield generate
    with monkeypatch.context() as patch:
        patch.setattr(steps, 'qwen_generator', interrupted)
        with pytest.raises(KeyboardInterrupt):
            steps.extract_graph_scenes(context, **options, force=True)
    assert not list(directory.glob('*.jsonl'))
    steps.extract_graph_scenes(context, **options)
    assert len(list(directory.glob('*.jsonl'))) == 4
    assert read_scene_records(next(directory.glob('*.jsonl')))[0]['provenance']['settings']['minicpm']['vision_downsample'] == 4


@pytest.mark.parametrize('downsample', [4, 16])
def test_worker_dispatch_and_runtime_recording(monkeypatch, tmp_path, downsample):
    import queue
    from extraction.backends.qwen_workers import _worker_main
    from extraction.qwen_runtime import QwenRuntime
    from artifact_io import atomic_write_json

    calls = []
    backend = SimpleNamespace(runtime_info={'model_source': 'minicpm'}, close=lambda: calls.append('closed'))
    def load(model_path, *, settings, image_limit):
        calls.append((model_path, settings, image_limit))
        return backend
    monkeypatch.setattr(MiniCPMBackend, 'from_pretrained', load)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    monkeypatch.setenv('TOKENIZERS_PARALLELISM', 'false')
    tasks, events = queue.Queue(), queue.Queue()
    tasks.put(None)
    _worker_main(0, '0', 'local-mini', tasks, events, {'vision_downsample': downsample}, 6, 'minicpm')
    assert calls == [('local-mini', {'vision_downsample': downsample}, 6), 'closed']
    assert events.get_nowait()['kind'] == 'started'
    runtime = QwenRuntime(on_ready=lambda engines: atomic_write_json(tmp_path / 'runtime.json', engines, durable=True))
    runtime.engine_ready(events.get_nowait())
    assert read_json(tmp_path / 'runtime.json')['engines']['0']['model_source'] == 'minicpm'


def test_partial_extraction_resumes_remaining_contents(ready_context, fake_models, monkeypatch):
    from contextlib import contextmanager

    options = dict(model='minicpm', arm='graph_minicpm', schema='prompts/scene_graph_v8.md')
    original = steps.qwen_generator
    @contextmanager
    def interrupted(**kwargs):
        with original(**kwargs) as normal:
            def generate(tasks, callback):
                normal([next(iter(tasks))], callback)
                raise KeyboardInterrupt
            yield generate
    with monkeypatch.context() as patch:
        patch.setattr(steps, 'qwen_generator', interrupted)
        with pytest.raises(KeyboardInterrupt):
            steps.extract_graph_scenes(ready_context, **options)
    directory = ready_context.scene_arm_dir('graph_minicpm')
    completed = next(directory.glob('*.jsonl'))
    preserved = completed.read_bytes()
    steps.extract_graph_scenes(ready_context, **options)
    assert len(fake_models[-1]) == 3
    assert len(list(directory.glob('*.jsonl'))) == 4
    assert completed.read_bytes() == preserved


def test_qwen_wrapper_keeps_its_chat_and_processor_policy(native_backend):
    from extraction.backends.qwen import QwenBackend
    model, calls, task = native_backend
    backend = QwenBackend.from_pretrained(str(model))
    try:
        output = asyncio.run(backend.generate(task))
        assert 'enable_thinking' not in calls['chat']
        assert 'mm_processor_kwargs' not in calls['engine']
        assert calls['engine']['max_num_batched_tokens'] == 16384
        assert output.text == r'[Context]\nmedium: live_action'
        assert calls['sampling']['stop_token_ids'] == [248044, 248046]
    finally:
        backend.close()


def test_runtime_provenance_keeps_resumed_engine_versions(tmp_path):
    from extraction.minicpm_provenance import record_runtime, contract_identity
    from extraction.qwen_runtime import QwenRuntime

    contract = {'settings': {'vision_downsample': 16}}
    write_json(tmp_path / 'generation.json', contract)
    sessions = []
    for commit in ('old-build', 'new-build'):
        runtime = QwenRuntime(on_ready=lambda event: record_runtime(tmp_path, event))
        runtime.engine_ready({'worker_index': 0, 'versions': {'vllm': commit}, 'vllm_commit': commit})
        sessions.append(runtime.execution_id)
    document = read_json(tmp_path / 'runtime.json')
    assert document['generation_hash'] == contract_identity(contract)
    assert set(document['executions']) == set(sessions)
    assert document['executions'][sessions[0]]['0']['vllm_commit'] == 'old-build'
    assert document['executions'][sessions[1]]['0']['vllm_commit'] == 'new-build'
