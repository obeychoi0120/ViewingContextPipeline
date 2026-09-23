"""Reproduce structural counts and source-paired audit; manual notes are separate TSV."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/runs/v4_260922/extraction/scenes'
NOTES = {}
for line in (OUT / 'scene_gap_first100_review_notes.tsv').read_text().splitlines():
    video, scene, note = line.split('\t')
    key = int(video), int(scene)
    assert key not in NOTES
    NOTES[key] = note
stats = Counter()
predicates = Counter()
names = Counter()
records = []
source_files = []
missing = []
for video in range(1, 101):
    content_id = f'microlens_100k_{video:05}'
    arms = {}
    for arm in ('desc_qwen', 'graph_qwen'):
        path = BASE / arm / f'{content_id}.jsonl'
        data = path.read_bytes()
        source_files.append({'path': str(path.relative_to(ROOT)),
                             'sha256': hashlib.sha256(data).hexdigest()})
        arm_rows = {}
        for line_number, line in enumerate(data.decode().splitlines(), 1):
            row = json.loads(line)
            assert row['content_id'] == content_id
            assert row['scene_idx'] not in arm_rows
            arm_rows[row['scene_idx']] = (row, line_number)
        arms[arm] = arm_rows
        stats[arm + '_rows'] += len(arm_rows)
    for scene in sorted(set(arms['desc_qwen']) | set(arms['graph_qwen'])):
        d, dl = arms['desc_qwen'].get(scene, ({}, None))
        g, gl = arms['graph_qwen'].get(scene, ({}, None))
        graph = g.get('scene_graph')
        row = {'video': video, 'content_id': content_id, 'scene_idx': scene,
               'pair_status': 'matched' if d and g else 'missing_graph' if d else 'missing_desc',
               'graph_format': g.get('graph_format', 'structured') if g else 'missing',
               'description': d.get('description'), 'scene_graph': graph,
               'desc_path': str((BASE / 'desc_qwen' / f'{content_id}.jsonl').relative_to(ROOT)),
               'desc_line': dl,
               'graph_path': str((BASE / 'graph_qwen' / f'{content_id}.jsonl').relative_to(ROOT)),
               'graph_line': gl,
               'semantic_reviewed': (video, scene) in NOTES,
               'review_note_ko': NOTES.get((video, scene), '')}
        if d and g:
            stats['matched_pairs'] += 1
        else:
            missing.append({'video': video, 'scene_idx': scene, 'missing_arm': 'graph_qwen' if not g else 'desc_qwen'})
        if isinstance(graph, dict):
            stats['structured_scenes'] += 1
            entities, relations = graph['entities'], graph['relations']
            entity_by_id = {e['id']: e for e in entities}
            ids = set(entity_by_id)
            duplicate_signature = len({(e['name'], tuple(sorted(e.get('attributes', []))))
                                       for e in entities}) < len(entities)
            dangling = any(r['subject_id'] not in ids or r['object_id'] not in ids for r in relations)
            cycle = False
            if (len(entities) >= 3 and len(relations) == len(entities)
                    and len({r['predicate'] for r in relations}) == 1 and not dangling):
                edges = {r['subject_id']: r['object_id'] for r in relations}
                if len(edges) == len(entities) and len(set(edges.values())) == len(entities):
                    cursor = next(iter(ids))
                    seen = set()
                    while cursor not in seen:
                        seen.add(cursor)
                        cursor = edges[cursor]
                    cycle = len(seen) == len(entities)
            metrics = {'entity_count': len(entities), 'relation_count': len(relations),
                       'duplicate_signature': duplicate_signature, 'dangling_reference': dangling,
                       'uniform_full_cycle': cycle, 'over4entities': len(entities) > 4,
                       'over4relations': len(relations) > 4}
            row.update(metrics)
            for key, value in metrics.items():
                stats[key] += value
            stats['exact4entities'] += len(entities) == 4
            stats['exact4relations'] += len(relations) == 4
            names.update(e['name'] for e in entities)
            predicates.update(r['predicate'] for r in relations)
            for r in relations:
                stats['self_relations'] += r['subject_id'] == r['object_id']
                if r['predicate'] in ('cutting', 'stirring', 'eating', 'dribbling'):
                    key = r['predicate']
                    stats[key + '_relations'] += 1
                    stats[key + '_person_to_person'] += (
                        entity_by_id.get(r['subject_id'], {}).get('name') == 'person'
                        and entity_by_id.get(r['object_id'], {}).get('name') == 'person')
        elif isinstance(graph, str):
            stats['raw_text_scenes'] += 1
        records.append(row)
assert len(records) == 640 and stats['matched_pairs'] == 638
assert len(NOTES) == 125 and sum(r['semantic_reviewed'] for r in records) == 125
stats['semantic_reviewed_matched_pairs'] = sum(r['semantic_reviewed'] and r['pair_status'] == 'matched' for r in records)
stats['semantic_reviewed_missing_slots'] = sum(r['semantic_reviewed'] and r['pair_status'] != 'matched' for r in records)
meta = {
    'run_id': 'v4_260922', 'date': '2026-09-23',
    'selection': 'content_id 00001–00100, join on content_id and scene_idx',
    'semantic_sampling': 'All 100 first scenes (scene_idx=0), plus 23 targeted matched scenes and 2 missing-graph slots. Nonrandom qualitative sample; no video-level semantic prevalence estimates.',
    'structural_scope': 'All 638 Graph rows. Structural entity/relation counts only for 623 dict-form graphs; 15 raw text graphs retained but excluded from those denominators.',
    'reference_limit': 'Description is comparison reference, not video ground truth; frames not inspected.',
    'metric_definitions': {
        'duplicate_signature': 'At least two entities with exactly equal name and sorted attribute strings; does not prove duplicate real-world identity.',
        'dangling_reference': 'At least one relation endpoint ID absent from declared entities.',
        'uniform_full_cycle': 'At least three declared entities, same number of relations, one predicate, every entity has one incoming/outgoing edge and all form one directed cycle.',
        'over4entities': 'More than 4 entity entries; compare verified scene_graph_v4 prompt max.',
        'over4relations': 'More than 4 relation entries; duplicates counted as saved.',
        'person_to_person': 'Both declared endpoint names exactly person. Predicate-specific diagnostic, not automatically an error.'},
    'counts': dict(stats), 'predicate_counts': dict(predicates.most_common()),
    'entity_name_counts': dict(names.most_common()), 'missing': missing,
    'raw_text_slots': [{'video': r['video'], 'scene_idx': r['scene_idx']} for r in records if isinstance(r['scene_graph'], str)],
    'source_files': source_files}
(OUT / 'scene_gap_first100_stats.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
(OUT / 'scene_gap_first100_pairs.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
keys = list(dict.fromkeys(key for r in records for key in r))
with (OUT / 'scene_gap_first100_pairs.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for r in records:
        writer.writerow({**r, 'scene_graph': json.dumps(r['scene_graph'], ensure_ascii=False)})
print(json.dumps({'counts': stats, 'missing': missing}, ensure_ascii=False))
