"""Explicitly normalize relocated Summary Arms with an original-file backup."""
import argparse
import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone

from arm_registry import registry, concat_layout
from artifact_io import atomic_write_json, atomic_write_jsonl
from extraction.arm_migration import normalize_summary_arm
from extraction.summary_executor import reuse_summary_document
from pipeline_runtime import RunContext
from validation.cache_identity import without_provenance_arm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    ctx = RunContext.load(args.run_id)
    if concat_layout(ctx.config):
        raise ValueError("Summary arm normalization is for historical v5/v6 layouts only")
    root = ctx.run_root / 'extraction' / 'summaries'
    backup = ctx.run_root.parents[1] / 'backups' / ctx.run_id / ('summary-arm-normalization-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    plans = []
    counts = Counter()

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    def transform(path, raw, arm):
        if path.suffix == '.jsonl':
            result = []
            for line in raw.decode().splitlines():
                row = json.loads(line)
                if 'provenance' in row:
                    row['provenance'] = without_provenance_arm(row['provenance'])
                if 'arm' in row:
                    row['arm'] = arm.name
                result.append(row)
            return result
        doc = json.loads(raw)
        return normalize_summary_arm(doc, arm, source_path=str(path.relative_to(ctx.run_root)), source_hash=digest(raw))

    for arm in registry(ctx.config).values():
        if arm.model is None:
            continue
        directory = root / arm.name
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or path.suffix not in {'.json', '.jsonl'}:
                continue
            raw = path.read_bytes()
            if path.suffix == '.json':
                doc = json.loads(raw)
                reuse_summary_document(path, content_id=path.stem, arm=doc.get('arm'))
                before = doc
            else:
                before = [json.loads(line) for line in raw.decode().splitlines()]
            result = transform(path, raw, arm)
            counts['checked_' + arm.name] += 1
            if result != before:
                plans.append((path, arm, digest(raw)))
    print('Preflight passed:', len(plans), 'files to normalize', flush=True)
    if not plans:
        return 0
    backup.mkdir(parents=True, exist_ok=False)
    with tarfile.open(backup / 'originals.tar.gz', 'w:gz', compresslevel=1) as archive:
        for path, arm, old_hash in plans:
            if digest(path.read_bytes()) != old_hash:
                raise RuntimeError(f'File changed during backup: {path}')
            archive.add(path, arcname=str(path.relative_to(ctx.run_root)), recursive=False)
    atomic_write_json(backup / 'manifest.json', {'run_id':ctx.run_id, 'state':'backed_up', 'file_count':len(plans), 'counts':dict(counts)}, durable=True)
    print('Backup complete:', backup, flush=True)
    with (backup / 'changes.jsonl').open('w') as journal:
        for i, (path, arm, old_hash) in enumerate(plans, 1):
            raw = path.read_bytes()
            if digest(raw) != old_hash:
                raise RuntimeError(f'File changed before normalization: {path}')
            result = transform(path, raw, arm)
            writer = atomic_write_jsonl if path.suffix == '.jsonl' else atomic_write_json
            writer(path, result, durable=True)
            journal.write(json.dumps({'path':str(path.relative_to(ctx.run_root)), 'before_sha256':old_hash, 'after_sha256':digest(path.read_bytes())}) + '\n')
            if i % 20000 == 0:
                journal.flush()
                print('Normalized', i, '/', len(plans), flush=True)
    atomic_write_json(backup / 'manifest.json', {'run_id':ctx.run_id, 'state':'complete', 'file_count':len(plans), 'counts':dict(counts)}, durable=True)
    print('Normalization complete:', backup, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
