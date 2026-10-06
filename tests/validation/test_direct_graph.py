"""Direct graph input integrity, differentiability, invariance and execution scope."""
import copy
import json

import numpy as np
import pytest

from arm_registry import registry
from validation.graph_context import graph_context
from validation.graph_inputs import build_input, GraphStore, prepare, source_identity, verify
from validation.selection import prepare_validation_cohort


def scene():
    return {'medium': 'live_action', 'format': 'demonstration', 'topics': ['cooking'],
            'entities': [{'id': 'p', 'name': 'cook', 'attributes': ['red apron']},
                         {'id': 'c', 'name': 'carrot', 'attributes': []}],
            'actions': [{'actor': 'p', 'action': 'cutting', 'target': 'c', 'tool': None}]}


def write_scenes(context, cid, graphs):
    path = context.scene_arm_dir('graph_qwen') / f'{cid}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'content_id': cid, 'scene_idx': i, 'scene_graph': graph}) + '\n'
                            for i, graph in enumerate(graphs)))
    return path


@pytest.fixture
def graph_data(ready_context, monkeypatch):
    context = graph_context(ready_context, 'mean')
    cohort = prepare_validation_cohort(context)
    ids = [r['content_id'] for r in cohort['catalog']]
    alternate = scene()
    alternate['actions'][0]['action'] = 'washing'
    write_scenes(context, ids[0], [scene(), alternate])
    write_scenes(context, ids[1], ['parse failed'])
    actionless = scene()
    actionless['actions'] = []
    write_scenes(context, ids[2], [actionless])
    write_scenes(context, ids[3], [scene()])
    def encode(settings, texts):
        import hashlib
        return np.stack([np.random.default_rng(int.from_bytes(hashlib.sha256(t.encode()).digest()[:4]))
                         .normal(size=settings.embedding_dim) for t in texts]).astype(np.float32)
    monkeypatch.setattr('validation.graph_inputs.encode_texts', encode)
    prepare(context, target=['graph_qwen', 'graph_qwen_meta', 'meta'])
    return context, cohort


def test_input_defaults_isolation_actionless_and_missing(graph_data):
    context, cohort = graph_data
    store = GraphStore(context.representations_dir / 'graph_qwen_embeddings')
    assert store['video_offsets'].tolist() == [0, 2, 2, 3, 4]
    assert store['missing'][0].tolist() == [2, -1, -1, 0, 0, 1]
    for i in range(len(store['contexts'])):
        a, b = store['edge_offsets'][i:i+2]
        low, high = store['scene_offsets'][i:i+2]
        assert np.all(store['edges'][a:b, :2] >= low)
        assert np.all(store['edges'][a:b, :2] < high)
    assert np.all(store['titles'] == 0)
    stats = json.loads((context.representations_dir / 'graph_qwen_embeddings' / 'statistics.json').read_text())
    assert stats['counts']['raw'] == 1
    assert stats['counts']['actionless'] == 1
    assert stats['counts']['empty_video'] == 1
    assert len(store) == len(cohort['catalog'])
    assert prepare(context, target=['graph_qwen'])['reused_arms'] == ['graph_qwen']


def test_node_embedding_text_format_and_list_order(graph_data):
    context, cohort = graph_data
    graph = scene()
    graph['actions'][0]['action'] = 'cooking'
    graph['topics'] = ['food preparation', 'cooking']
    graph['entities'][0]['attributes'] = ['red apron', 'chef hat']
    cid = cohort['catalog'][0]['content_id']
    arm = registry(context.config)['graph_qwen']
    write_scenes(context, cid, [graph])
    texts, arrays, _ = build_input(context, cohort, arm)
    assert [texts[i] for i in arrays['nodes'][:3]] == [
        'Name: carrot\nAttributes:',
        'Name: cook\nAttributes: chef hat, red apron',
        'cooking',
    ]
    assert texts[arrays['contexts'][0]] == (
        'Format: demonstration\nMedium: live_action\nTopics: cooking, food preparation'
    )
    graph['topics'].reverse()
    graph['entities'][0]['attributes'].reverse()
    write_scenes(context, cid, [graph])
    reordered_texts, reordered_arrays, _ = build_input(context, cohort, arm)
    assert reordered_texts == texts
    for key in arrays:
        np.testing.assert_array_equal(reordered_arrays[key], arrays[key])


def test_json_input_version_cannot_reuse_labeled_input_cache(graph_data, monkeypatch):
    from validation import graph_inputs

    context, cohort = graph_data
    arm = registry(context.config)['graph_qwen']
    current_hash = source_identity(context, cohort, arm)
    with monkeypatch.context() as patch:
        patch.setattr(graph_inputs, 'VERSION', 'role-graph-inputs/v2')
        assert source_identity(context, cohort, arm) != current_hash
    directory = context.representations_dir / 'graph_qwen_embeddings'
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['version'] == 'role-graph-inputs/v3'
    manifest['version'] = 'role-graph-inputs/v2'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='stale'):
        verify(context, cohort, ['graph_qwen'])
    prepare(context, target=['graph_qwen'])
    verify(context, cohort, ['graph_qwen'])
    assert json.loads(manifest_path.read_text())['version'] == 'role-graph-inputs/v3'


@pytest.mark.parametrize('dimension,dtype', [(384, 'float32'), (1024, 'float64'), (None, None)])
def test_graph_cache_checks_actual_features_even_with_matching_provenance(graph_data, dimension, dtype):
    import shutil
    from validation.graph_inputs import bundle_valid
    from validation.shared_cache import SharedCache, checksum

    context, cohort = graph_data
    directory = context.representations_dir / 'graph_qwen_embeddings'
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    signature = manifest['input_hash']
    if dimension is None:
        (directory / 'features.npy').write_bytes(b'')
    else:
        np.save(directory / 'features.npy', np.zeros((manifest['feature_count'], dimension), dtype=dtype))
    manifest['checksums']['features.npy'] = checksum(directory / 'features.npy')
    manifest_path.write_text(json.dumps(manifest))
    assert not bundle_valid(directory, signature, 1024)
    with pytest.raises(ValueError, match='corrupt graph inputs'):
        verify(context, cohort, ['graph_qwen'])
    # Matching hashes are insufficient for shared-cache reuse as well.
    cache = SharedCache(context, 'graph_inputs', signature)
    shared_manifest = json.loads((cache.path / 'cache.json').read_text())
    for name in ('features.npy', 'manifest.json'):
        shutil.copy2(directory / name, cache.path / name)
        shared_manifest['files'][name] = checksum(cache.path / name)
    (cache.path / 'cache.json').write_text(json.dumps(shared_manifest))
    assert cache.valid()
    result = prepare(context, target=['graph_qwen'])
    assert result['generated_arms'] == ['graph_qwen']
    assert GraphStore(directory)['features'].shape == (manifest['feature_count'], 1024)


@pytest.mark.parametrize('corruption', ['duplicate', 'content_id', 'json'])
def test_input_integrity_is_fatal(graph_data, corruption):
    context, cohort = graph_data
    cid = cohort['catalog'][0]['content_id']
    p = context.scene_arm_dir('graph_qwen') / f'{cid}.jsonl'
    line = p.read_text().splitlines()[0]
    if corruption == 'duplicate':
        p.write_text(line + '\n' + line)
    elif corruption == 'content_id':
        p.write_text(line.replace(cid, 'wrong'))
    else:
        p.write_text('{broken')
    with pytest.raises(ValueError):
        build_input(context, cohort, registry(context.config)['graph_qwen'])


def test_invalid_warning_variable_size_and_cache_invalidation(graph_data):
    context, cohort = graph_data
    arm = registry(context.config)['graph_qwen']
    before = source_identity(context, cohort, arm)
    cid = cohort['catalog'][0]['content_id']
    large = scene()
    large['entities'] += [{'id': f'e{i}', 'name': 'plate', 'attributes': []} for i in range(30)]
    large['actions'] *= 8
    invalid = scene()
    invalid['actions'][0]['target'] = 'missing'
    p = write_scenes(context, cid, [large, invalid, scene()])
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    rows[2]['warning'] = 'bad output'
    p.write_text('\n'.join(map(json.dumps, rows)))
    _, arrays, stats = build_input(context, cohort, arm)
    assert arrays['scene_offsets'][1] == 40
    assert stats['counts']['invalid_structure'] == 1
    assert stats['counts']['warning'] == 1
    assert source_identity(context, cohort, arm) != before
    with pytest.raises(ValueError, match='stale'):
        verify(context, cohort, ['graph_qwen'])


@pytest.mark.parametrize('pool', ['mean'])
def test_model_gradients_invariance_and_checkpoint(graph_data, pool, tmp_path):
    torch = pytest.importorskip('torch')
    from validation.graph_model import new_graph_model, graph_batch
    from validation.steps import validation_config
    torch.set_num_threads(1)
    context, _ = graph_data
    context = graph_context(context, pool)
    config = validation_config(context)
    model = new_graph_model(context, config, 'graph_qwen_meta', torch.device('cpu'))
    batch = graph_batch(model.store, [0, 1, 2], 'cpu')
    encoder = model.graph_encoder
    output = encoder(batch)
    assert torch.isfinite(output).all() and torch.count_nonzero(output[1]) == 0
    perm = torch.randperm(len(batch['types']))
    inverse = torch.argsort(perm)
    reordered = dict(batch)
    for key in ('features', 'types', 'node_scenes'):
        reordered[key] = batch[key][perm]
    reordered['edges'] = batch['edges'].clone()
    reordered['edges'][:, :2] = inverse[batch['edges'][:, :2]]
    reordered['missing'] = batch['missing'].clone()
    reordered['missing'][:, 0] = inverse[batch['missing'][:, 0]]
    torch.testing.assert_close(encoder(reordered), output, atol=2e-6, rtol=1e-5)
    swapped = dict(batch)
    swapped['edges'] = batch['edges'].clone()
    for a, b in [(0, 1), (5, 6)]:
        swapped['edges'][batch['edges'][:, 2] == a, 2] = b
        swapped['edges'][batch['edges'][:, 2] == b, 2] = a
    assert not torch.allclose(encoder(swapped)[0], output[0])
    model.prepare_items(torch.tensor([1, 2, 3, 4]))
    users = model.user_vectors(torch.tensor([[0, 1, 3], [0, 3, 4]]))
    loss = torch.nn.functional.cross_entropy(users @ model.item_vectors(torch.tensor([4, 1])).T,
                                             torch.tensor([0, 1]))
    loss.backward()
    for module in [encoder.entity_projection, encoder.action_projection, encoder.context_projection,
                   encoder.role_fusion, encoder.scene_mlp, model.encoder, model.title_projection]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert not batch['features'].requires_grad
    model.eval()
    with torch.no_grad():
        expected = model.catalog_vectors()
        assert torch.count_nonzero(expected[1]) == 0
    torch.save(model.state_dict(), tmp_path / 'weights.pt')
    restored = new_graph_model(context, config, 'graph_qwen_meta', torch.device('cpu'))
    restored.load_state_dict(torch.load(tmp_path / 'weights.pt', weights_only=True))
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored.catalog_vectors(), expected)


def test_cli_rejects_invalid_modes_before_loading(monkeypatch):
    from validation.cli import main
    monkeypatch.setattr('validation.cli.RunContext.load', lambda _: pytest.fail('must reject before load'))
    with pytest.raises(SystemExit):
        main(['embed-representations', '--run-id', 'x', '--target', 'meta'])
    with pytest.raises(SystemExit):
        main(['run-recommendation', '--run-id', 'x', '--representation-mode', 'graph',
              '--scene-aggregation', 'attention', '--target', 'meta'])
    for step, mode, extra, arm in [
        ('run-recommendation', 'graph', [], 'meta'),
        ('run-diagnosis', 'graph', [], 'meta'),
        ('embed-representations', 'graph', ['--scene-aggregation', 'mean'], 'meta'),
        ('run-recommendation', 'text', ['--scene-aggregation', 'mean'], 'meta'),
        ('embed-representations', 'graph', [], 'desc_qwen_meta')]:
        assert main([step, '--run-id', 'x', '--representation-mode', mode, '--target', arm, *extra]) == 1


@pytest.mark.parametrize('pool', ['mean', 'gnn'])
def test_rolling_train_resume_diagnosis_and_hash(graph_data, pool, monkeypatch):
    torch = pytest.importorskip('torch')
    from validation.steps import validation_config, run_diagnosis
    from validation.rolling_data import EventTable
    from validation.rolling_recommendation import (run_combination, prepare_split, combination_dir,
                                                   combination_complete)
    from validation.representation_provenance import recommendation_identity
    from validation.selection import load_validation_cohort, training_signature
    from validation import recommendation_cache
    torch.set_num_threads(1)
    context, _ = graph_data
    context = graph_context(context, pool)
    config = validation_config(context)
    cohort = load_validation_cohort(context)
    table = EventTable(cohort['events'])
    split = cohort['plan']['splits'][0]
    identity = {'run_id': context.run_id, 'evaluation_date': split['evaluation_date'],
                'seed': config.model.seeds[0], 'arm': 'graph_qwen_meta',
                'deterministic': config.model.deterministic,
                'training_input_hash': training_signature(context, cohort, config),
                **recommendation_identity(context, 'graph_qwen_meta')}
    prepared = prepare_split(table, split)
    run_combination(context, config, table, split, identity, 'graph_qwen_meta', prepared, torch.device('cpu'))
    directory = combination_dir(context, split['evaluation_date'], identity['seed'], identity['arm'])
    assert combination_complete(directory, identity, len(prepared[0]['test']))
    assert recommendation_cache.valid_bundle(directory, identity, table, split)
    assert not combination_complete(directory, {**identity, 'scene_aggregation': 'wrong'}, len(prepared[0]['test']))
    assert np.load(directory / 'catalog_vectors.npy').shape == (4, 512)
    other = directory.parent / 'restore'
    assert recommendation_cache.restore(context, other, identity, table, split)
    assert recommendation_cache.valid_bundle(other, identity, table, split)
    (directory / 'complete.json').unlink()
    assert not combination_complete(directory, identity, len(prepared[0]['test']))
    assert recommendation_cache.restore(context, directory, identity, table, split)
    checkpoint = torch.load(directory / 'sasrec.pt', weights_only=True)
    from validation.graph_model import new_graph_model
    model = new_graph_model(context, config, 'graph_qwen_meta', 'cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        np.testing.assert_allclose(model.catalog_vectors().numpy(), np.load(directory / 'catalog_vectors.npy'),
                                   atol=1e-6)
    # Diagnose a complete one-day/one-seed miniature grid with existing validation logic.
    config.model.seeds = [identity['seed']]
    mini = copy.deepcopy(cohort)
    mini['plan']['splits'] = [split]
    mini['plan']['eligible_test_count'] = len(prepared[0]['test'])
    monkeypatch.setattr('validation.selection.load_validation_cohort', lambda *a, **k: mini)
    monkeypatch.setattr('validation.rolling_diagnosis.load_validation_cohort', lambda *a, **k: mini)
    monkeypatch.setattr('validation.steps.validation_config', lambda ctx: config)
    monkeypatch.setattr('validation.rolling_diagnosis.training_signature', lambda *a: identity['training_input_hash'])
    reuse = {'run_id': context.run_id, 'local': 0, 'shared': 0, 'generated': 1}
    (context.recommendations_dir / 'reuse.json').write_text(json.dumps(reuse))
    assert run_diagnosis(context, target=['graph_qwen_meta'], representation_mode='graph',
                         scene_aggregation=pool)['status'] == 'pass'
    diagnosis = json.loads(context.diagnosis_path.read_text())
    assert diagnosis['schema_version'] == 'rolling-diagnosis/v3'
    assert 'graph_inputs' not in diagnosis
    assert diagnosis['recommendations']['combination_count'] == 1
    assert diagnosis['recommendations']['reuse'] == reuse
    assert diagnosis['statistics']['status'] == 'computed'
    assert diagnosis['runtime_decision'] == {'status': 'pass', 'errors': []}
    assert diagnosis['paper_reference']['metadata_hr10'] == 0.046


def test_title_order_and_missing_inputs(graph_data):
    torch = pytest.importorskip('torch')
    from validation.graph_model import new_graph_model
    from validation.steps import validation_config
    context, cohort = graph_data
    # Fourth title remains, but Graph becomes unavailable.
    cid = cohort['catalog'][3]['content_id']
    write_scenes(context, cid, ['bad'])
    prepare(context, target=['graph_qwen', 'graph_qwen_meta'])
    for arm in ('graph_qwen', 'graph_qwen_meta'):
        model = new_graph_model(context, validation_config(context), arm, 'cpu').eval()
        # Trainable biases cannot introduce an Item for a completely missing video.
        with torch.no_grad():
            model.title_projection.bias.fill_(0.25)
            model.graph_encoder.scene_mlp.fc2.bias.fill_(0.75)
        from validation.graph_model import graph_batch
        batch = graph_batch(model.store, [0, 1, 2, 3], 'cpu')
        with torch.no_grad():
            values = model.catalog_vectors()
            title = model.title_projection(batch['titles'][3:4])
            title = title.masked_fill(~batch['title_available'][3:4, None].bool(), 0)
            fallback = model.graph_encoder.scene_mlp(torch.cat([title, torch.zeros(1, 384)], dim=-1))
        assert torch.count_nonzero(values[1]) == 0
        assert bool(torch.count_nonzero(values[3])) == (arm == 'graph_qwen_meta')
        if arm == 'graph_qwen_meta':
            torch.testing.assert_close(values[3:4], fallback)
        assert values.shape == (4, 512)
        assert (model.title_projection.in_features, model.title_projection.out_features) == (1024, 128)
        assert model.graph_encoder.output_dim == 512
        assert not hasattr(model, 'item_projection') and not hasattr(model, 'item_mlp')
        assert not any(isinstance(m, torch.nn.LayerNorm) for m in model.graph_encoder.modules())


def test_empty_and_single_scene_pooling(graph_data):
    torch = pytest.importorskip('torch')
    from validation.graph_model import SceneRoleEncoder, graph_batch
    import graph_v7_reference as reference
    context, _ = graph_data
    store = GraphStore(context.representations_dir / 'graph_qwen_embeddings')
    encoder = SceneRoleEncoder()
    expected = reference.SceneRoleEncoder()
    expected.load_state_dict(encoder.state_dict())
    with torch.no_grad():
        batch = graph_batch(store, [1, 2], 'cpu')
        a = encoder(batch)
        torch.testing.assert_close(a, expected(batch))
        assert torch.isfinite(a).all() and torch.count_nonzero(a[0]) == 0
        assert torch.count_nonzero(encoder(graph_batch(store, [1], 'cpu'))) == 0
    with pytest.raises(ValueError, match='mean or gnn'):
        SceneRoleEncoder(aggregation='invalid')


def test_four_gpu_feature_partition_and_order(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from validation import graph_inputs
    torch = pytest.importorskip('torch')
    calls = []
    class Encoder:
        def __init__(self, settings, device=None):
            self.device = device
            self.last_truncation = {}
        def encode(self, texts):
            calls.append((self.device, texts))
            return np.asarray([[int(t)] * 1024 for t in texts], dtype=np.float32)
    class Pool:
        def __init__(self, **kwargs):
            assert kwargs['max_workers'] == 4
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, fn, *args):
            result = fn(*args)
            return SimpleNamespace(result=lambda: result)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(graph_inputs, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr('validation.features.BGETextEncoder', Encoder)
    values = graph_inputs.encode_texts(SimpleNamespace(), [str(i) for i in range(11)])
    np.testing.assert_array_equal(values[:, 0], np.arange(11))
    assert [device for device, _ in calls] == ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    assert sum(len(texts) for _, texts in calls) == 11


def test_optional_explicit_role_states_and_scene_order(graph_data):
    torch = pytest.importorskip('torch')
    from validation.graph_model import SceneRoleEncoder, graph_batch
    context, cohort = graph_data
    arm = registry(context.config)['graph_qwen']
    cid = cohort['catalog'][0]['content_id']
    g = scene()
    g['actions'][0].update(receiver='unknown', location=None)
    write_scenes(context, cid, [g])
    _, arrays, _ = build_input(context, cohort, arm)
    assert arrays['missing'][0, -2:].tolist() == [0, 0]  # receiver is disabled
    invalid = scene()
    del invalid['actions'][0]['tool']
    write_scenes(context, cid, [invalid])
    _, _, stats = build_input(context, cohort, arm)
    assert stats['counts']['invalid_structure'] == 1
    store = GraphStore(context.representations_dir / 'graph_qwen_embeddings')
    batch = graph_batch(store, [0], 'cpu')
    # Swap the first two scene memberships and their Context features together.
    swapped = dict(batch)
    swapped['node_scenes'] = 1 - batch['node_scenes']
    swapped['contexts'] = batch['contexts'].flip(0)
    for pool in ('mean',):
        encoder = SceneRoleEncoder(aggregation=pool)
        torch.testing.assert_close(encoder(batch), encoder(swapped), atol=2e-6, rtol=1e-5)


def test_source_identity_title_model_and_catalog(graph_data):
    context, cohort = graph_data
    arm = registry(context.config)['graph_qwen_meta']
    original = source_identity(context, cohort, arm)
    changed = copy.deepcopy(cohort)
    changed['metadata_titles'][0]['title'] += ' modified'
    assert source_identity(context, changed, arm) != original
    changed = copy.deepcopy(cohort)
    changed['catalog'].reverse()
    changed['metadata_titles'].reverse()
    assert source_identity(context, changed, arm) != original
    changed['metadata_titles'].reverse()
    with pytest.raises(ValueError, match='order mismatch'):
        source_identity(context, changed, arm)
    context.config['validation']['encoder']['batch_size'] += 1
    assert source_identity(context, cohort, arm) != original


def test_graph_context_runs_in_spawned_workers(graph_data):
    import io
    from tqdm import tqdm
    from validation.steps import validation_config
    from validation.selection import load_validation_cohort, training_signature
    from validation.representation_provenance import recommendation_identity
    from validation.rolling_workers import run_parallel
    from validation.rolling_recommendation import combination_complete, combination_dir, phase_ids
    from validation.rolling_data import EventTable
    pytest.importorskip('torch')
    context, _ = graph_data
    cohort = load_validation_cohort(context)
    split = cohort['plan']['splits'][0]
    config = validation_config(context)
    jobs = []
    for seed in config.model.seeds[:2]:
        identity = {'run_id': context.run_id, 'evaluation_date': split['evaluation_date'],
                    'seed': seed, 'arm': 'graph_qwen',
                    'deterministic': config.model.deterministic,
                    'training_input_hash': training_signature(context, cohort, config),
                    **recommendation_identity(context, 'graph_qwen')}
        jobs.append((split, identity, 'graph_qwen'))
    with tqdm(total=2, file=io.StringIO()) as progress:
        assert run_parallel(context, jobs, ['cpu', 'cpu'], progress) == 2
    for split, identity, branch in jobs:
        directory = combination_dir(context, split['evaluation_date'], identity['seed'], branch)
        assert combination_complete(directory, identity, len(phase_ids(EventTable(cohort['events']), split, 'test')))


def test_deprecated_receiver_is_ignored_in_graph_inputs(graph_data):
    context, cohort = graph_data
    cid = cohort['catalog'][0]['content_id']
    arm = registry(context.config)['graph_qwen']
    write_scenes(context, cid, [scene()])
    texts, expected, stats = build_input(context, cohort, arm)
    for receiver in ['missing', 'c', ['p', 'c']]:
        graph = scene()
        graph['actions'][0]['receiver'] = receiver
        write_scenes(context, cid, [graph])
        actual_texts, actual, actual_stats = build_input(context, cohort, arm)
        assert actual_texts == texts and actual_stats == stats
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])


def test_v7_model_contract_reuses_inputs_and_rejects_old_checkpoint(graph_data):
    torch = pytest.importorskip('torch')
    from validation.graph_context import GRAPH_ARCHITECTURE, GRAPH_MODEL
    from validation.graph_model import new_graph_model
    from validation.representation_provenance import recommendation_identity
    from validation.recommendation_cache import cache_for
    from validation.steps import validation_config
    context, cohort = graph_data
    config = validation_config(context)
    arm = registry(context.config)['graph_qwen_meta']
    source_before = source_identity(context, cohort, arm)
    assert prepare(context, target=['graph_qwen_meta'])['reused_arms'] == ['graph_qwen_meta']
    assert source_identity(context, cohort, arm) == source_before
    current = recommendation_identity(context, arm.name)
    assert current['graph_architecture'] == GRAPH_ARCHITECTURE == 'sasrec-role-graph/v7'
    assert (GRAPH_MODEL['hidden_dim'], GRAPH_MODEL['scene_dim'], GRAPH_MODEL['title_dim']) == (128, 512, 128)
    previous = {**current, 'graph_architecture': 'sasrec-role-graph/v2',
                'graph_model': {'layers': 2, 'hidden_dim': 128, 'scene_dim': 512}}
    assert cache_for(context, current).key != cache_for(context, previous).key
    model = new_graph_model(context, config, arm.name, 'cpu')
    # Tiny legacy weight fixture exercises strict loading without historical models.
    old = {'graph_encoder.projection.weight': torch.zeros(128, 1024)}
    with pytest.raises(RuntimeError, match='state_dict'):
        model.load_state_dict(old)
    assert 'graph_encoder.title_projection.weight' in model.state_dict()
    assert 'item_projection.weight' not in model.state_dict()
    assert 'graph_encoder.layers.1.states' not in model.state_dict()


@pytest.mark.parametrize("pool", ["mean", "gnn"])
def test_title_only_meta_uses_common_v7_baseline(graph_data, pool):
    torch = pytest.importorskip('torch')
    from validation.model import seed_everything
    from validation.recommendation import _new_model
    from validation.graph_model import new_graph_model
    from validation.steps import validation_config
    context, _ = graph_data
    context = graph_context(context, pool)
    config = validation_config(context)
    store = GraphStore(context.representations_dir / 'meta_embeddings')
    seed_everything(42)
    actual = new_graph_model(context, config, 'meta', 'cpu')
    seed_everything(42)
    features = np.array(store['features'][store['titles']])
    expected = _new_model(config, item_count=len(store), branch='meta', device='cpu', features={
        'title_values': features, 'video_values': np.zeros_like(features),
        'title_available': store['titles'] != 0,
        'video_available': np.zeros(len(store), dtype=bool),
    })
    assert not hasattr(actual, 'graph_encoder')
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
    actual.eval()
    expected.eval()
    with torch.no_grad():
        torch.testing.assert_close(actual.catalog_vectors(), expected.catalog_vectors(), rtol=0, atol=0)
        history = torch.tensor([[1, 2, 0]])
        torch.testing.assert_close(actual.score_catalog(history), expected.score_catalog(history))


@pytest.mark.parametrize("pool", ["mean", "gnn"])
def test_entity_id_renaming_and_all_scene_orderings_preserve_outputs(graph_data, pool):
    torch = pytest.importorskip('torch')
    from validation.graph_model import new_graph_model
    from validation.steps import validation_config
    context, cohort = graph_data
    context = graph_context(context, pool)
    cid = cohort['catalog'][0]['content_id']
    first, second = scene(), scene()
    first['actions'].append({**first['actions'][0], 'action': 'washing'})
    second['topics'] = ['food preparation']
    write_scenes(context, cid, [first, second])
    prepare(context, target=['graph_qwen_meta'])
    original = new_graph_model(context, validation_config(context), 'graph_qwen_meta', 'cpu').eval()
    with torch.no_grad():
        expected = original.catalog_vectors().clone()
    renamed = []
    for graph in [second, first]:
        graph = copy.deepcopy(graph)
        graph['entities'].reverse()
        graph['actions'].reverse()
        mapping = {'p': 'renamed_a', 'c': 'renamed_z'}  # Reverse canonical Entity order too.
        for entity in graph['entities']:
            entity['id'] = mapping[entity['id']]
        for action in graph['actions']:
            for role in ('actor', 'target', 'tool', 'location'):
                if action.get(role) in mapping:
                    action[role] = mapping[action[role]]
        renamed.append(graph)
    write_scenes(context, cid, renamed)
    prepare(context, target=['graph_qwen_meta'])
    actual = new_graph_model(context, validation_config(context), 'graph_qwen_meta', 'cpu').eval()
    actual.load_state_dict(original.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(actual.catalog_vectors(), expected, atol=3e-6, rtol=1e-5)


def test_none_unknown_states_are_role_specific_and_receiver_is_ignored(graph_data):
    torch = pytest.importorskip('torch')
    from validation.graph_model import SceneRoleEncoder, graph_batch
    context, _ = graph_data
    store = GraphStore(context.representations_dir / 'graph_qwen_embeddings')
    batch = graph_batch(store, [0], 'cpu')
    # Isolate role-state effects from entity features and residual scene effects.
    batch['edges'] = batch['edges'][:0]
    batch['missing'][:, 1:] = 0
    encoder = SceneRoleEncoder()
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
        for slot in range(4):
            encoder.role_states[slot, 0].fill_(slot * 2 + 1)
            encoder.role_states[slot, 1].fill_(slot * 2 + 2)
    for slot, role in enumerate(encoder.ROLE_INDICES):
        with torch.no_grad():
            encoder.role_fusion.weight.zero_()
            encoder.role_fusion.weight[:, (slot + 1) * 128:(slot + 2) * 128] = torch.eye(128)
        none = encoder(batch)
        unknown_batch = {**batch, 'missing': batch['missing'].clone()}
        unknown_batch['missing'][:, role + 1] = 1
        unknown = encoder(unknown_batch)
        torch.testing.assert_close(none[:, 256:384], torch.full((1, 128), slot * 2 + 1.0))
        torch.testing.assert_close(unknown[:, 256:384], none[:, 256:384] + 1)
        receiver_batch = {**batch, 'missing': batch['missing'].clone()}
        receiver_batch['missing'][:, 4] = 1
        torch.testing.assert_close(encoder(receiver_batch), none, rtol=0, atol=0)
        encoder.zero_grad(set_to_none=True)
        unknown.sum().backward()
        assert encoder.role_states.grad[slot, 1].abs().sum() > 0
        assert not encoder.role_states.grad[slot, 0].any()


def test_scene_without_entities_or_actions_keeps_context(graph_data):
    torch = pytest.importorskip('torch')
    from validation.graph_model import new_graph_model
    from validation.steps import validation_config
    context, cohort = graph_data
    graph = scene()
    graph.update(entities=[], actions=[])
    write_scenes(context, cohort['catalog'][0]['content_id'], [graph])
    prepare(context, target=['graph_qwen'])
    model = new_graph_model(context, validation_config(context), 'graph_qwen', 'cpu')
    captured = []
    hook = model.graph_encoder.scene_mlp.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    model.prepare_items(torch.tensor([1]))
    values = model.item_vectors(torch.tensor([0, 1]))
    hook.remove()
    assert not values[0].any() and values[1].any()
    assert not captured[0][:, :384].any() and captured[0][:, 384:].any()
    values.sum().backward()
    assert model.graph_encoder.context_projection.weight.grad.abs().sum() > 0


def test_gnn_contract_reuses_features_but_separates_recommendation_cache(graph_data):
    from validation.representation_provenance import recommendation_identity
    from validation.recommendation_cache import cache_for
    from validation.graph_context import GRAPH_MODEL
    context, cohort = graph_data
    gnn = graph_context(context, 'gnn')
    arm = registry(context.config)['graph_qwen_meta']
    assert source_identity(context, cohort, arm) == source_identity(gnn, cohort, arm)
    assert prepare(gnn, target=[arm.name])['reused_arms'] == [arm.name]
    mean_identity = recommendation_identity(context, arm.name)
    gnn_identity = recommendation_identity(gnn, arm.name)
    assert mean_identity['embedding_hash'] == gnn_identity['embedding_hash']
    assert mean_identity['graph_model'] == GRAPH_MODEL
    assert mean_identity['graph_architecture'] == 'sasrec-role-graph/v7'
    assert gnn_identity['graph_architecture'] == 'sasrec-role-graph/v8'
    assert cache_for(context, mean_identity).key != cache_for(gnn, gnn_identity).key
    assert recommendation_identity(gnn, 'meta')['graph_model']['gcn_applied'] is False
    assert gnn.recommendations_dir.name == 'gnn'
    assert gnn.diagnosis_path.name == 'graph_gnn_diagnosis.json'
