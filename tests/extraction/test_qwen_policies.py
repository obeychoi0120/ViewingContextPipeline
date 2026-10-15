"""Model policy, rendered chat templates and actual vLLM request parameters."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from arm_registry import generated_arm, registry
from artifact_io import read_json, write_json
from extraction import steps
from extraction.backends.qwen import QwenBackend
from extraction.backends.qwen_workers import QwenGenerationTask
from extraction.qwen_config import generation_settings, validate_checkpoint
from pipeline_runtime import RunContext, ConfigError, _validate_config


@pytest.mark.parametrize('model', ['qwen3vl', 'qwen3.5'])
@pytest.mark.parametrize('summary', [False, True])
@pytest.mark.parametrize('representation', ['graph', 'description'])
@pytest.mark.parametrize('override', [None, 1.2])
def test_actual_sampling_params(model, summary, representation, override):
    pytest.importorskip('vllm')
    backend = object.__new__(QwenBackend)
    backend.stop_token_ids = [151645]
    task = QwenGenerationTask('a', (), 'prompt', 384, **generation_settings(
        model, representation=representation, summary=summary, repetition_penalty=override))
    params = backend.sampling_params(task)
    assert params.temperature == (1.0 if summary else 0.7)
    assert params.top_p == (1.0 if summary else 0.8)
    assert params.top_k == (40 if summary and model == 'qwen3vl' else 20)
    assert params.presence_penalty == (2.0 if summary else 1.5)
    assert params.min_p == 0.0 and params.repetition_penalty == 1.0
    assert params.max_tokens == 384 and params.stop_token_ids == [151645]
    expected_penalty = override if override is not None else (
        1.05 if representation == 'graph' and not summary else 1.0)
    assert params.extra_args['generated_token_repetition_penalty'] == expected_penalty
    recovery = replace(task, structured_output={'json': {'type': 'object'}})
    recovered = backend.sampling_params(recovery)
    assert recovered.temperature == params.temperature
    assert recovered.top_k == params.top_k and recovered.presence_penalty == params.presence_penalty
    assert recovered.repetition_penalty == 1.0 and recovered.extra_args == params.extra_args


def test_qwen35_template_disables_thinking(current_context):
    captured = {}

    def template(messages, **kwargs):
        captured.update(kwargs)
        return 'non-thinking prompt'

    backend = object.__new__(QwenBackend)
    backend.model_id = str(current_context.path('models', 'qwen3.5'))
    backend.processor = SimpleNamespace(apply_chat_template=template)
    inputs, images = backend._prepare(QwenGenerationTask('a', (), 'prompt', 512, **generation_settings('qwen3.5', representation='graph')))
    assert captured['enable_thinking'] is False
    assert inputs == {'prompt': 'non-thinking prompt'} and images == []
    with pytest.raises(ValueError, match='checkpoint does not match'):
        backend._prepare(QwenGenerationTask('a', (), 'prompt', 512, **generation_settings('qwen3vl', representation='graph')))


@pytest.mark.parametrize('model_type,architecture', [('qwen3_vl', 'Qwen3VLForConditionalGeneration'), ('qwen3_5', 'Qwen3_5ForConditionalGeneration')])
def test_checkpoint_identity_rejects_other_family(model_type, architecture):
    config = {'model_type': model_type, 'architectures': [architecture]}
    model = 'qwen3vl' if model_type == 'qwen3_vl' else 'qwen3.5'
    validate_checkpoint(model, config)
    with pytest.raises(ValueError, match='checkpoint does not match'):
        validate_checkpoint('qwen3.5' if model == 'qwen3vl' else 'qwen3vl', config)


def test_legacy_names_and_run_rejected(current_context):
    with pytest.raises(ValueError):
        generated_arm(current_context.config, 'graph', 'qwen')
    assert len(registry(current_context.config)) == 9
    old = current_context.run_root / 'extraction/scenes/graph_qwen'
    old.mkdir(parents=True)
    (old / 'original.jsonl').write_text('old artifact')
    with pytest.raises(ConfigError, match='legacy qwen Run'):
        RunContext.load(current_context.run_id, root=current_context.root)
    assert (old / 'original.jsonl').read_text() == 'old artifact'
    current_context.config['experiment_config_version'] = 'v5'
    with pytest.raises(ConfigError, match='v6'):
        _validate_config(current_context.config)


@pytest.mark.parametrize('model', ['qwen3vl', 'qwen3.5'])
@pytest.mark.parametrize('changed_setting', ['temperature', 'repetition_penalty'])
def test_policy_change_requires_force(ready_context, fake_models, model, changed_setting):
    options = dict(model=model, arm=f'graph_{model}', schema='prompts/scene_graph_v8.md')
    steps.extract_graph_scenes(ready_context, **options)
    directory = ready_context.scene_arm_dir(options['arm'])
    identity_path = directory / 'generation_identity.json'
    identity = read_json(identity_path)
    scene_file = next(directory.glob('*.jsonl'))
    before = scene_file.read_bytes()
    if changed_setting == 'repetition_penalty':
        identity['identity']['settings']['generation']['repetition_penalty'] = 1.0
        identity['identity']['settings']['repetition_penalty'] = [1.0]
    else:
        identity['identity']['settings']['generation']['temperature'] = 0.0
    write_json(identity_path, identity)
    count = len(fake_models)
    with pytest.raises(ValueError, match='policy mismatch'):
        steps.extract_graph_scenes(ready_context, **options)
    assert len(fake_models) == count
    assert scene_file.read_bytes() == before
    steps.extract_graph_scenes(ready_context, force=True, **options)
    assert len(fake_models) == count + 1
    assert all(task.repetition_penalty == 1.05 for task in fake_models[-1])
    restored = read_json(identity_path)['identity']['settings']
    assert restored['generation']['repetition_penalty'] == 1.05
    assert restored['repetition_penalty'] == [1.05]
    options['schema'] = 'prompts/summary_graph_v8.md'
    steps.summarize(ready_context, **options)
    document_path = next(ready_context.summary_arm_dir(options['arm']).glob('*.json'))
    document = read_json(document_path)
    document['provenance']['settings']['generation']['top_k'] = 1
    write_json(document_path, document)
    with pytest.raises(RuntimeError, match='policy mismatch'):
        steps.summarize(ready_context, **options)
    steps.summarize(ready_context, force=True, **options)
    assert read_json(document_path)['provenance']['settings']['generation']['top_k'] == (40 if model == 'qwen3vl' else 20)


def test_force_interruption_does_not_reuse_stale_policy(ready_context, fake_models, monkeypatch):
    options = dict(model='qwen3.5', arm='graph_qwen3.5', schema='prompts/scene_graph_v8.md')
    steps.extract_graph_scenes(ready_context, **options)
    directory = ready_context.scene_arm_dir(options['arm'])
    ready_context.config['extraction']['graph']['qwen3.5']['scene_max_new_tokens'] += 1

    @contextmanager
    def interrupted(**kwargs):
        def generate(tasks, callback):
            next(iter(tasks))
            raise KeyboardInterrupt
        yield generate

    monkeypatch.setattr(steps, 'qwen_generator', interrupted)
    with pytest.raises(KeyboardInterrupt):
        steps.extract_graph_scenes(ready_context, force=True, **options)
    assert not list(directory.glob('*.jsonl'))
