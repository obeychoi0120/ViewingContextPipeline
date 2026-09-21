"""Explicit, resumable copies of legacy generation artifacts; originals stay untouched."""
from contextlib import contextmanager
import fcntl
import json

from arm_registry import registry, legacy_layout
from artifact_io import atomic_write_json, atomic_write_jsonl
from extraction.recovery import file_fingerprint
from extraction.scene_storage import read_scene_records, _payload
from extraction.summary_executor import check_summary_model, reuse_summary_document, summary_failure_rows, summary_model_from_document
from pipeline_runtime import read_json, read_jsonl

SCHEMA = "arm-layout-migration/v1"


@contextmanager
def migration_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.arm-layout.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def migrate_arm_layout(context, *, summary_model):
    if legacy_layout(context.config):
        raise ValueError("migrate-arm-layout requires the new nine-arm config contract")
    if summary_model not in {'qwen', 'gemini'}:
        raise ValueError('summary-model must be qwen or gemini')
    root = context.run_root / 'extraction'
    manifest_path = root / 'arm-layout-migration.json'
    with migration_lock(root):
        previous = read_json(manifest_path) if manifest_path.exists() else None
        if previous and previous.get('schema_version') != SCHEMA:
            raise ValueError('invalid arm migration manifest')
        if previous and previous['summary_model'] != summary_model:
            raise ValueError('migration already selected a different Summary model')
        writes, skipped, evidence = [], [], []

        def stage(source, target, value, jsonl=False):
            if target.exists():
                existing = read_jsonl(target) if jsonl else read_json(target)
                if existing != value:
                    raise ValueError(f'migration destination conflict: {target}')
            writes.append((target, value, jsonl))
            evidence.append({'source': str(source.relative_to(context.run_root)),
                             'source_hash': file_fingerprint(source),
                             'target': str(target.relative_to(context.run_root))})

        for arm in registry(context.config).values():
            if arm.model is None or arm.name != arm.scene_arm:
                continue
            old = root / arm.representation / arm.model / 'scenes'
            target = context.scene_arm_dir(arm.name)
            for path in sorted(old.glob('*.jsonl')):
                if path.name in {'failure.jsonl', 'failures.jsonl'}:
                    continue
                records = read_scene_records(path)
                stage(path, target / path.name, [_payload(path, row) for row in records], True)
            failures = {}
            for path in [old / 'failure.jsonl', old / 'failures.jsonl', *sorted((old / 'failures').glob('*.jsonl'))]:
                if path.is_file():
                    for row in read_jsonl(path):
                        cid = str(row.get('content_id', path.stem))
                        failures.setdefault(cid, {})[row['scene_idx']] = (path, {**row, 'content_id': cid})
            for cid, rows in failures.items():
                source = next(iter(rows.values()))[0]
                stage(source, target / 'failures' / f'{cid}.jsonl',
                      [r for _, r in sorted(rows.values(), key=lambda pair: pair[1]['scene_idx'])], True)

        for arm in registry(context.config).values():
            if arm.model is None or not arm.uses_title:
                continue
            old = root / arm.representation / arm.model / 'summaries' / summary_model
            target = context.summary_arm_dir(arm.name)
            check_summary_model(target, summary_model)
            failures = summary_failure_rows(old)
            migrated_failures = []
            for path in sorted(old.glob('*.json')):
                doc = reuse_summary_document(path, content_id=path.stem, arm=arm.scene_arm)
                prov = doc['provenance']
                if summary_model_from_document(doc) != summary_model:
                    raise ValueError(f'Summary model provenance mismatch: {path}')
                if (not isinstance(prov.get('english_title'), str)
                        or prov.get('uses_title') is False):
                    skipped.append({'source': str(path.relative_to(context.run_root)),
                                    'reason': 'title input is not verified by stored provenance'})
                    continue
                if prov.get('arm') != arm.scene_arm or prov.get('representation') != arm.representation:
                    raise ValueError(f'Summary source provenance mismatch: {path}')
                migration = {'schema_version': SCHEMA, 'source_arm': arm.scene_arm,
                             'target_arm': arm.name, 'source_path': str(path.relative_to(context.run_root)),
                             'source_hash': file_fingerprint(path)}
                copied = {**doc, 'arm': arm.name, 'migration': migration}
                failure = failures.get(path.stem)
                if failure and failure.get('summary_model', 'qwen') != summary_model:
                    raise ValueError(f'Summary failure model provenance mismatch: {path}')
                if doc['status'] in {'raw_fallback', 'failed'} or failure:
                    copied.update(status='failed', text='', word_count=0,
                                  violations=doc.get('violations') or [(failure or {}).get('error', 'legacy generation failed')])
                    migrated_failures.append({**(failure or {}), 'content_id': path.stem,
                                              'error': (failure or {}).get('error') or ', '.join(copied['violations']),
                                              'raw_output': '', 'summary_model': summary_model,
                                              'provenance': prov, 'migration': migration})
                stage(path, target / path.name, copied)
            for cid, row in failures.items():
                if not (old / f'{cid}.json').exists():
                    skipped.append({'source': str(old.relative_to(context.run_root)), 'content_id': cid,
                                    'reason': 'no Summary document with verified title provenance'})
            if migrated_failures or (target / 'failures.jsonl').exists():
                # The bundle's input document hashes also cover these normalized failures.
                fail_path = target / 'failures.jsonl'
                if fail_path.exists() and read_jsonl(fail_path) != migrated_failures:
                    raise ValueError(f'migration destination conflict: {fail_path}')
                writes.append((fail_path, migrated_failures, True))
        # Validate every destination before publishing anything. Partial copies are rerunnable.
        for target, value, jsonl in writes:
            if not target.exists():
                writer = atomic_write_jsonl if jsonl else atomic_write_json
                writer(target, value, durable=True)
        manifest = {'schema_version': SCHEMA, 'run_id': context.run_id,
                    'summary_model': summary_model, 'copied': evidence, 'skipped': skipped}
        atomic_write_json(manifest_path, manifest, durable=True)
    result = {'stage': 'migrate-arm-layout', 'file_count': len(writes), 'skipped': skipped}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result
