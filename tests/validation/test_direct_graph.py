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
    context = graph_context(ready_context, 'attention')
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


@pytest.mark.parametrize('pool', ['mean', 'attention'])
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
    for module in [encoder.projection, encoder.layers[0].relations[0], model.encoder,
                   model.title_projection, model.item_norm,
                   *([encoder.attention] if pool == 'attention' else [])]:
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
    for step, mode, extra, arm in [
        ('run-recommendation', 'graph', [], 'meta'),
        ('run-diagnosis', 'graph', [], 'meta'),
        ('embed-representations', 'graph', ['--scene-aggregation', 'mean'], 'meta'),
        ('run-recommendation', 'text', ['--scene-aggregation', 'mean'], 'meta'),
        ('embed-representations', 'graph', [], 'desc_qwen_meta')]:
        assert main([step, '--run-id', 'x', '--representation-mode', mode, '--target', arm, *extra]) == 1


@pytest.mark.parametrize('pool', ['mean', 'attention'])
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
    assert run_diagnosis(context, target=['graph_qwen_meta'], representation_mode='graph',
                         scene_aggregation=pool)['status'] == 'pass'


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
        # Missing titles must stay zero even after trainable offsets change.
        with torch.no_grad():
            model.title_projection.bias.fill_(0.25)
            model.item_norm.bias.fill_(0.75)
        captured = []
        hook = model.item_norm.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        with torch.no_grad():
            values = model.catalog_vectors()
        hook.remove()
        title_ids = model.store['titles']
        with torch.no_grad():
            titles = torch.as_tensor(np.array(model.store['features'][title_ids]))
            expected = model.title_projection(titles)
            expected[title_ids == 0] = 0
        torch.testing.assert_close(captured[0][:, :128], expected)
        assert torch.count_nonzero(captured[0][3, 128:]) == 0
        assert torch.count_nonzero(values[1]) == 0
        assert bool(torch.count_nonzero(values[3])) == (arm == 'graph_qwen_meta')
        assert captured[0].shape == (4, 512)
        assert model.title_projection.in_features == 1024
        assert model.title_projection.out_features == 128
        assert len(model.graph_encoder.layers) == 1
        assert model.graph_encoder.output_dim == 384
        assert isinstance(model.graph_encoder.readout, torch.nn.Identity)
        assert not hasattr(model, 'item_projection')


def test_empty_and_single_scene_pooling(graph_data):
    torch = pytest.importorskip('torch')
    from validation.graph_model import RoleGraphEncoder, graph_batch
    context, _ = graph_data
    store = GraphStore(context.representations_dir / 'graph_qwen_embeddings')
    mean = RoleGraphEncoder(aggregation='mean')
    attention = RoleGraphEncoder(aggregation='attention')
    attention.load_state_dict(mean.state_dict(), strict=False)
    with torch.no_grad():
        batch = graph_batch(store, [1, 2], 'cpu')
        a, b = mean(batch), attention(batch)
        torch.testing.assert_close(a, b)
        assert torch.isfinite(a).all()
        assert torch.count_nonzero(a[0]) == 0
        assert torch.count_nonzero(mean(graph_batch(store, [1], 'cpu'))) == 0


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
    from validation.graph_model import RoleGraphEncoder, graph_batch
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
    for pool in ('mean', 'attention'):
        encoder = RoleGraphEncoder(aggregation=pool)
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


def test_v4_model_contract_reuses_inputs_and_rejects_old_checkpoint(graph_data):
    pytest.importorskip('torch')
    import graph_reference as v2
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
    assert current['graph_architecture'] == GRAPH_ARCHITECTURE == 'sasrec-role-graph/v6'
    assert (GRAPH_MODEL['layers'], GRAPH_MODEL['scene_dim'], GRAPH_MODEL['title_dim']) == (1, 384, 128)
    previous = {**current, 'graph_architecture': 'sasrec-role-graph/v2',
                'graph_model': {'layers': 2, 'hidden_dim': 128, 'scene_dim': 512, 'attention_dim': 128}}
    assert cache_for(context, current).key != cache_for(context, previous).key
    model = new_graph_model(context, config, arm.name, 'cpu')
    old = v2.GraphSASRec(store=model.store, aggregation='attention', item_count=len(model.store),
                        max_length=10, embedding_dim=512, num_blocks=2, num_heads=2, dropout=0.1,
                        arm='graph')
    with pytest.raises(RuntimeError, match='state_dict'):
        model.load_state_dict(old.state_dict())
    assert 'title_projection.weight' in model.state_dict()
    assert 'item_projection.weight' not in model.state_dict()
    assert 'graph_encoder.layers.1.states' not in model.state_dict()


def test_title_only_meta_uses_common_v4_baseline(graph_data):
    torch = pytest.importorskip('torch')
    from validation.model import SASRec, seed_everything
    from validation.graph_model import new_graph_model
    from validation.steps import validation_config
    context, _ = graph_data
    config = validation_config(context)
    store = GraphStore(context.representations_dir / 'meta_embeddings')
    seed_everything(42)
    actual = new_graph_model(context, config, 'meta', 'cpu')
    seed_everything(42)
    expected = SASRec(len(store), 10, 512, 2, 2, config.model.dropout, arm='metadata',
                      item_features=np.array(store['features'][store['titles']]))
    assert not hasattr(actual, 'graph_encoder')
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(actual.catalog_vectors(), expected.catalog_vectors(), rtol=0, atol=0)
