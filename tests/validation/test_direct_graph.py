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
            'actions': [{'actor': 'p', 'action': 'cutting', 'target': 'c', 'tool': None, 'location': 'unknown'}]}


def write_scenes(context, cid, graphs):
    path = context.scene_arm_dir('graph_qwen') / f'{cid}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'content_id': cid, 'scene_idx': i, 'tokens': None, 'warning': [], 'scene_graph': graph}) + '\n'
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
    rows[2]['warning'] = ['INVALID_REFERENCE']
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


@pytest.mark.parametrize('pool', ['mean', 'gnn', 'transformer', 'transformer_pos', 'transformer_local',
                                  'attn_pool', 'gnn_attn'])
def test_leave_one_out_train_resume_diagnosis_and_hash(graph_data, pool, monkeypatch):
    torch = pytest.importorskip('torch')
    from validation.steps import validation_config, run_diagnosis
    from validation.leave_one_out_data import EventTable
    from validation.leave_one_out_recommendation import (run_combination, prepare_split, combination_dir,
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
    identity = {'run_id': context.run_id, 'protocol': 'leave_one_out', 'split_hash': split['split_hash'],
                'seed': config.model.seeds[0], 'arm': 'graph_qwen_meta',
                'deterministic': config.model.deterministic,
                'training_input_hash': training_signature(context, cohort, config),
                **recommendation_identity(context, 'graph_qwen_meta')}
    prepared = prepare_split(table, split)
    run_combination(context, config, table, split, identity, 'graph_qwen_meta', prepared, torch.device('cpu'))
    directory = combination_dir(context, identity['seed'], identity['arm'])
    assert combination_complete(directory, identity, len(prepared[0]['test']))
    assert recommendation_cache.valid_bundle(directory, identity, table, split)
    old_evaluation = {**identity, 'evaluation_contract': {
        **identity['evaluation_contract'], 'checkpoint_metric': 'NDCG@10'}}
    assert not recommendation_cache.valid_bundle(directory, old_evaluation, table, split)
    assert recommendation_cache.cache_for(context, identity).key != recommendation_cache.cache_for(context, old_evaluation).key
    assert not recommendation_cache.restore(context, directory.parent / 'old-policy', old_evaluation, table, split)
    legacy = {**identity, 'graph_architecture':
              'sasrec-role-graph/v7' if pool == 'mean' else 'sasrec-role-graph/v8'}
    assert not combination_complete(directory, legacy, len(prepared[0]['test']))
    assert not recommendation_cache.valid_bundle(directory, legacy, table, split)
    assert not combination_complete(directory, {**identity, 'scene_aggregation': 'wrong'}, len(prepared[0]['test']))
    assert np.load(directory / 'catalog_vectors.npy').shape == (4, 512)
    execution = json.loads((directory / 'training.json').read_text())['execution']
    graph_training = execution['graph_training']
    assert ('scene_attention_summary' in graph_training) == (pool in ('attn_pool', 'gnn_attn'))
    assert ('scene_transformer_residual_scales' in graph_training) == pool.startswith('transformer')
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
    # Diagnose a complete one-seed miniature grid with the shared validation logic.
    config.model.seeds = [identity['seed']]
    mini = copy.deepcopy(cohort)
    mini['plan']['splits'] = [split]
    mini['plan']['eligible_test_count'] = len(prepared[0]['test'])
    monkeypatch.setattr('validation.selection.load_validation_cohort', lambda *a, **k: mini)
    monkeypatch.setattr('validation.leave_one_out_diagnosis.load_validation_cohort', lambda *a, **k: mini)
    monkeypatch.setattr('validation.steps.validation_config', lambda ctx: config)
    monkeypatch.setattr('validation.leave_one_out_diagnosis.training_signature', lambda *a: identity['training_input_hash'])
    reuse = {'run_id': context.run_id, 'local': 0, 'shared': 0, 'generated': 1}
    (context.recommendations_dir / 'reuse.json').write_text(json.dumps(reuse))
    assert run_diagnosis(context, target=['graph_qwen_meta'], representation_mode='graph',
                         scene_aggregation=pool)['status'] == 'pass'
    diagnosis = json.loads(context.diagnosis_path.read_text())
    assert diagnosis['schema_version'] == 'leave-one-out-diagnosis/v4'
    assert 'graph_inputs' not in diagnosis
    assert set(diagnosis['recommendations']) == {'means'}
    assert set(diagnosis['recommendations']['means']) == {'graph_qwen_meta'}
    assert set(diagnosis['statistics']) == {'status', 'comparisons', 'metric', 'comparison_policy', 'targets', 'comparison_count', 'correction'}
    assert 'cohort' not in diagnosis
    assert diagnosis['statistics']['status'] == 'computed'
    assert diagnosis['runtime_decision'] == {'status': 'pass', 'errors': []}
    assert 'paper_reference' not in diagnosis


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
            model.graph_encoder.scene_mlp.fc1.bias.fill_(0.75)
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


def test_graph_context_runs_in_spawned_workers(graph_data):
    import io
    from tqdm import tqdm
    from validation.steps import validation_config
    from validation.selection import load_validation_cohort, training_signature
    from validation.representation_provenance import recommendation_identity
    from validation.leave_one_out_workers import run_parallel
    from validation.leave_one_out_recommendation import combination_complete, combination_dir, phase_ids
    from validation.leave_one_out_data import EventTable
    pytest.importorskip('torch')
    context, _ = graph_data
    cohort = load_validation_cohort(context)
    split = cohort['plan']['splits'][0]
    config = validation_config(context)
    jobs = []
    for seed in config.model.seeds[:2]:
        identity = {'run_id': context.run_id, 'protocol': 'leave_one_out', 'split_hash': split['split_hash'],
                    'seed': seed, 'arm': 'graph_qwen',
                    'deterministic': config.model.deterministic,
                    'training_input_hash': training_signature(context, cohort, config),
                    **recommendation_identity(context, 'graph_qwen')}
        jobs.append((split, identity, 'graph_qwen'))
    with tqdm(total=2, file=io.StringIO()) as progress:
        assert run_parallel(context, jobs, ['cpu', 'cpu'], progress) == 2
    for split, identity, branch in jobs:
        directory = combination_dir(context, identity['seed'], branch)
        assert combination_complete(directory, identity, len(phase_ids(EventTable(cohort['events']), split, 'test')))


@pytest.mark.parametrize("pool", ["mean", "gnn", "transformer", "transformer_local", "attn_pool",
                                  "gnn_attn"])
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
    if pool.startswith('transformer'):  # Open the zero-initialized residual scales.
        with torch.no_grad():
            original.graph_encoder.scene_transformer.layers[0].residual_scale.fill_(0.5)
    if pool == 'attn_pool':  # Open the zero-initialized score output.
        with torch.no_grad():
            original.graph_encoder.attention_pool.score.weight.normal_()
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
    assert mean_identity['graph_model']['scene_mlp']['linear_layers'] == 1
    assert mean_identity['graph_architecture'] == 'sasrec-role-graph/v9'
    assert gnn_identity['graph_architecture'] == 'sasrec-role-graph/v10'
    assert cache_for(context, mean_identity).key != cache_for(gnn, gnn_identity).key
    assert recommendation_identity(gnn, 'meta')['graph_model']['gcn_applied'] is False
    assert gnn.recommendations_dir.name == 'gnn'
    assert gnn.diagnosis_path.name == 'graph_gnn_diagnosis.json'


def test_scene_order_changes_only_the_positional_transformer(graph_data):
    torch = pytest.importorskip('torch')
    from validation.graph_model import new_graph_model
    from validation.steps import validation_config
    context, cohort = graph_data
    cid = cohort['catalog'][0]['content_id']
    first, second = scene(), scene()
    second['topics'] = ['food preparation']
    vectors = {}
    for order in ((first, second), (second, first)):
        write_scenes(context, cid, list(order))
        for pool in ('transformer', 'transformer_pos'):
            pooled = graph_context(context, pool)
            prepare(pooled, target=['graph_qwen_meta'])
            torch.manual_seed(0)
            model = new_graph_model(pooled, validation_config(pooled), 'graph_qwen_meta', 'cpu')
            with torch.no_grad():  # Open the zero-initialized residual scales.
                model.graph_encoder.scene_transformer.layers[0].residual_scale.fill_(0.5)
                vectors.setdefault(pool, []).append(model.eval().catalog_vectors()[0].clone())
    torch.testing.assert_close(*vectors['transformer'], atol=3e-6, rtol=1e-5)
    assert not torch.allclose(*vectors['transformer_pos'], atol=1e-5)


def test_transformer_contracts_share_features_but_separate_results(graph_data):
    from validation.representation_provenance import recommendation_identity
    from validation.recommendation_cache import cache_for
    from validation.graph_context import (GNN_MODEL, GRAPH_MODEL, SCENE_AGGREGATIONS,
                                          TRANSFORMER_VARIANTS, validate_mode)
    from validation.cli import main
    context, cohort = graph_data
    arm = registry(context.config)['graph_qwen_meta']
    contexts = {pool: graph_context(context, pool) for pool in SCENE_AGGREGATIONS}
    identities = {pool: recommendation_identity(pooled, arm.name) for pool, pooled in contexts.items()}
    assert {i['embedding_hash'] for i in identities.values()} == {identities['mean']['embedding_hash']}
    assert len({cache_for(contexts[p], i).key for p, i in identities.items()}) == len(SCENE_AGGREGATIONS)
    assert identities['mean']['graph_model'] == GRAPH_MODEL
    assert identities['gnn']['graph_model'] == {**GNN_MODEL, 'gcn_applied': True}
    for pool, spec in TRANSFORMER_VARIANTS.items():
        pooled = contexts[pool]
        assert source_identity(context, cohort, arm) == source_identity(pooled, cohort, arm)
        identity = identities[pool]
        assert identity['graph_architecture'] == 'sasrec-role-graph/v11'
        model = identity['graph_model']
        assert model['aggregation'] == pool and model['transformer_applied'] is True
        assert {k: model['transformer'][k] for k in spec} == spec
        assert model['transformer']['ffn_dim'] == 2048 and model['transformer']['final_norm'] is False
        assert recommendation_identity(pooled, 'meta')['graph_model']['transformer_applied'] is False
        assert pooled.recommendations_dir.name == pool
        assert pooled.diagnosis_path.name == f'graph_{pool}_diagnosis.json'
        validate_mode('graph', pool, 'run-recommendation', [arm.name])
    with pytest.raises(ValueError):
        validate_mode('graph', 'attention', 'run-recommendation', [arm.name])
    with pytest.raises(SystemExit):
        main(['run-recommendation', '--run-id', context.run_id, '--target', arm.name,
              '--representation-mode', 'graph', '--scene-aggregation', 'attention'])


def test_attention_contracts_share_features_but_separate_results(graph_data):
    from validation.representation_provenance import recommendation_identity
    from validation.recommendation_cache import cache_for
    from validation.graph_context import (ATTENTION_AGGREGATIONS, GNN_MODEL,
                                          TRANSFORMER_VARIANTS, validate_mode)
    from validation.recommendation_contracts import DIAGNOSIS_ARCHITECTURE_VERSIONS
    context, cohort = graph_data
    arm = registry(context.config)['graph_qwen_meta']
    contexts = {pool: graph_context(context, pool)
                for pool in ('mean', 'gnn', *TRANSFORMER_VARIANTS, *ATTENTION_AGGREGATIONS)}
    identities = {pool: recommendation_identity(pooled, arm.name) for pool, pooled in contexts.items()}
    keys = {pool: cache_for(contexts[pool], identity).key for pool, identity in identities.items()}
    assert len(set(keys.values())) == len(keys)
    for pool in ATTENTION_AGGREGATIONS:
        pooled = contexts[pool]
        identity = identities[pool]
        assert source_identity(context, cohort, arm) == source_identity(pooled, cohort, arm)
        assert identity['embedding_hash'] == identities['mean']['embedding_hash']
        assert identity['graph_architecture'] == 'sasrec-role-graph/v12'
        assert identity['graph_architecture'] in DIAGNOSIS_ARCHITECTURE_VERSIONS
        assert identity['graph_model']['aggregation'] == pool
        assert pooled.recommendations_dir.name == pool
        assert pooled.diagnosis_path.name == f'graph_{pool}_diagnosis.json'
        validate_mode('graph', pool, 'run-recommendation', [arm.name])
    pool_model = identities['attn_pool']['graph_model']
    assert pool_model['attention_pool_applied'] is True and 'gcn' not in pool_model
    assert pool_model['attention_pool']['scene_exchange'] == 'none'
    assert pool_model['attention_pool']['score_init'] == 'zero_output_uniform_weights'
    meta = recommendation_identity(contexts['attn_pool'], 'meta')['graph_model']
    assert meta['attention_pool_applied'] is False
    gcn, reference = identities['gnn_attn']['graph_model']['gcn'], GNN_MODEL['gcn']
    # Everything except the neighbour-weight function matches gnn.
    shared = ('layers', 'hidden_dim', 'similarity_input', 'similarity_dim', 'similarity_eps',
              'residual', 'activation', 'bias', 'detach_similarity', 'connectivity')
    assert {k: gcn[k] for k in shared} == {k: reference[k] for k in shared}
    assert gcn['similarity'] == 'scaled_dot_product' and gcn['normalization'] == 'row_softmax'
    assert gcn['query_key_dim'] == 128 and gcn['query_key_input'] == 'unit_l2_times_sqrt_dim'
    assert pool_model['attention_pool']['score_input'] == 'unit_l2_times_sqrt_dim'
    assert identities['gnn_attn']['graph_model']['gcn_applied'] is True
    assert recommendation_identity(contexts['gnn_attn'], 'meta')['graph_model']['gcn_applied'] is False


def test_previous_graph_input_contract_is_not_reused(graph_data):
    from validation.graph_inputs import bundle_valid

    context, _ = graph_data
    directory = context.representations_dir / 'graph_qwen_embeddings'
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['version'] = 'role-graph-inputs/v3'
    path.write_text(json.dumps(manifest))
    assert not bundle_valid(directory, manifest['input_hash'], 1024)


def test_unsupported_scene_blocks_existing_graph_cache_without_mutation(graph_data):
    from extraction.errors import UnsupportedArtifactError

    context, cohort = graph_data
    cid = cohort['catalog'][0]['content_id']
    path = context.scene_arm_dir('graph_qwen') / f'{cid}.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['scene_graph']['actions'][0]['receiver'] = None
    path.write_text('\n'.join(map(json.dumps, rows)))
    cache = context.representations_dir / 'graph_qwen_embeddings' / 'manifest.json'
    original, manifest = path.read_bytes(), cache.read_bytes()
    with pytest.raises(UnsupportedArtifactError) as error:
        prepare(context, target=['graph_qwen'])
    assert str(path) in str(error.value)
    assert path.read_bytes() == original and cache.read_bytes() == manifest
