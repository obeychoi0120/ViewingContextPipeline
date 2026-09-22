"""Cross-run reuse of completed recommendation combinations."""
import json
import pickle
import shutil
import tempfile
from pathlib import Path

from extraction.recovery import fingerprint
from pipeline_runtime import read_json, write_json
from validation.model import torch
from validation.recommendation_contracts import ARCHITECTURE_VERSION, TRAINING_IMPLEMENTATION_VERSION
from validation.representation_provenance import read_state
from validation.shared_cache import SharedCache, checksum

FILES = ['sasrec.pt', 'training.json', 'per_event_metrics.jsonl', 'complete.json']


def cache_for(context, identity):
    key = fingerprint({'identity': {k: v for k, v in identity.items() if k != 'run_id'},
                       'implementation': TRAINING_IMPLEMENTATION_VERSION,
                       'architecture': ARCHITECTURE_VERSION})
    return SharedCache(context, 'recommendations', key)


def eligible(context, branch):
    return read_state(context, branch).get('shareable', False)


def valid_bundle(directory, identity, table, split):
    from validation.rolling_recommendation import combination_complete, phase_ids
    ids = phase_ids(table, split, 'test')
    if not combination_complete(directory, identity, len(ids)):
        return False
    try:
        training = read_json(directory / 'training.json')
        if training.get('split') != split or training.get('catalog_size') != len(table.items):
            return False
        expected = {int(i): table.rows[int(i)] for i in ids}
        for line in (directory / 'per_event_metrics.jsonl').read_text().splitlines():
            row = json.loads(line)
            source = expected.pop(row['event_id'])
            if any(row.get(k) != v for k, v in source.items()):
                return False
        checkpoint = torch.load(directory / 'sasrec.pt', map_location='cpu', weights_only=True)
        metadata = checkpoint['metadata']
        return (not expected and bool(checkpoint['state_dict'])
                and metadata.get('architecture_version') == ARCHITECTURE_VERSION
                and metadata.get('catalog_size') == len(table.items)
                and metadata.get('best_epoch') == training.get('best_epoch')
                and all(metadata.get(k) == v for k, v in identity.items()))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError, AttributeError):
        return False


def restore(context, directory, identity, table, split):
    try:
        return _restore(context, directory, identity, table, split)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError, AttributeError):
        return False


def _restore(context, directory, identity, table, split):
    cache = cache_for(context, identity)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        temporary = Path(temporary)
        manifest = cache.restore(temporary)
        if not manifest:
            return False
        old_identity = read_json(temporary / 'complete.json').get('identity', {})
        expected = {**identity, 'run_id': old_identity.get('run_id')}
        if not valid_bundle(temporary, expected, table, split):
            return False
        provenance = {'source_run_id': old_identity['run_id'], 'cache_key': cache.key}
        training = read_json(temporary / 'training.json')
        training.update(identity)
        training['reused_from'] = provenance
        write_json(temporary / 'training.json', training)
        complete = read_json(temporary / 'complete.json')
        complete.update(identity=identity, reused_from=provenance)
        write_json(temporary / 'complete.json', complete)
        metrics = temporary / 'per_event_metrics.jsonl'
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        with metrics.open('w') as stream:
            for row in rows:
                row.update(identity)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        checkpoint = torch.load(temporary / 'sasrec.pt', map_location='cpu', weights_only=True)
        checkpoint['metadata'].update(identity)
        checkpoint['metadata']['reused_from'] = provenance
        torch.save(checkpoint, temporary / 'sasrec.pt')
        complete['checksums'] = {name: checksum(temporary / name) for name in FILES[:-1]}
        write_json(temporary / 'complete.json', complete)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'complete.json').unlink(missing_ok=True)
        for name in FILES:
            shutil.copy2(temporary / name, directory / name)
    return True


def publish(context, directory, identity, table, split, *, force=False):
    if valid_bundle(directory, identity, table, split):
        cache_for(context, identity).publish(directory, FILES, origin={'run_id': context.run_id},
                                                   replace_corrupt=not force)
