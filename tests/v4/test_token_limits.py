"""Stage limits follow the executing model, including cross-model summaries."""
import pytest

from extraction import steps
from pipeline_logging import step_settings
from pipeline_runtime import ConfigError, _validate_config, read_json


@pytest.mark.parametrize('representation', ['description', 'graph'])
@pytest.mark.parametrize('source', ['qwen', 'gemini'])
@pytest.mark.parametrize('summary_model', ['qwen', 'gemini'])
def test_distinct_model_limits_reach_requests_provenance_and_logs(
    ready_context, fake_models, representation, source, summary_model
):
    context = ready_context
    for index, kind in enumerate(['description', 'graph']):
        context.config['extraction'][kind] = {
            'qwen': {'scene_max_new_tokens': 301 + index, 'summary_max_new_tokens': 701 + index},
            'gemini': {'scene_max_new_tokens': 401 + index, 'summary_max_new_tokens': 901 + index},
        }
    _validate_config(context.config)
    arm = f"{'desc' if representation == 'description' else 'graph'}_{source}"
    scene_schema = f"prompts/scene_{representation}_v{'2' if representation == 'description' else '5'}.md"
    summary_schema = f"prompts/summary_{representation}_v{'5' if representation == 'description' else '6'}.md"
    getattr(steps, f'extract_{representation}_scenes')(
        context, arm=arm, model=source, schema=scene_schema,
    )
    scene_limit = context.config['extraction'][representation][source]['scene_max_new_tokens']
    assert all(task.max_new_tokens == scene_limit for task in fake_models[-1])
    assert step_settings(context, f'extract-{representation}-scenes', arm=arm,
                         model=source, schema=scene_schema)['max_new_tokens'] == scene_limit
    steps.summarize(context, arm=arm, model=summary_model, schema=summary_schema)
    summary_limit = context.config['extraction'][representation][summary_model]['summary_max_new_tokens']
    assert all(task.max_new_tokens == summary_limit for task in fake_models[-1])
    doc = read_json(next(context.summary_arm_dir(arm).glob('*.json')))
    assert doc['provenance']['settings']['max_new_tokens'] == summary_limit
    assert step_settings(context, f'summarize-{representation}', arm=arm, source=source,
                         model=summary_model, schema=summary_schema)['max_new_tokens'] == summary_limit


@pytest.mark.parametrize('invalid', [0, -1, True, '1024', None])
def test_invalid_model_limit_rejected(current_context, invalid):
    current_context.config['extraction']['graph']['gemini']['summary_max_new_tokens'] = invalid
    with pytest.raises(ConfigError, match='graph.gemini.summary_max_new_tokens'):
        _validate_config(current_context.config)


def test_missing_model_limits_rejected(current_context):
    del current_context.config['extraction']['description']['gemini']
    with pytest.raises(ConfigError, match='description must contain exactly qwen and gemini'):
        _validate_config(current_context.config)
