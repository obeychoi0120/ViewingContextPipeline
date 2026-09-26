"""Read-only audit of existing outputs; no inference or artifact mutation."""
import collections as C
import re
import sys
import json
from pathlib import Path
import statistics as S

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from extraction.semantic_graph.parser import parse_or_repair_graph

OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/runs/v5_260923/extraction'
ARMS = ['desc_qwen', 'graph_qwen', 'graph_gemini']

def read(path):
    return json.loads(path.read_text())

def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []

def dist(values):
    if not values:
        return {'n': 0}
    v = sorted(values)
    return {'n': len(v), 'mean': round(S.mean(v), 2), 'median': S.median(v),
            'p90': v[int((len(v)-1)*.9)], 'p95': v[int((len(v)-1)*.95)], 'max': max(v)}

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def main():
    catalog = lines(ROOT/'artifacts/preparation/cohort/catalog.jsonl')
    ids = [r['content_id'] for r in sorted(catalog, key=lambda r: r['content_id'])[:500]]
    scenes, summaries, expected, failures = {}, {}, {}, {}
    evidence = {}
    for cid in ids:
        timestamps = ROOT/'artifacts/preparation/source_assets'/cid/'timestamp_fixed_30s.json'
        if timestamps.exists():
            data = [{"scene_idx": i, **r} for i,r in enumerate(read(timestamps))]
            evidence[cid] = data
            expected[cid] = {r['scene_idx'] for r in data}
        else:
            expected[cid] = set()
    for arm in ARMS:
        scenes[arm], summaries[arm], failures[arm] = {}, {}, {}
        for cid in ids:
            rows = lines(BASE/'scenes'/arm/f'{cid}.jsonl')
            if len(rows) != len({r['scene_idx'] for r in rows}):
                raise ValueError(f'duplicate scenes {arm} {cid}')
            scenes[arm][cid] = {r['scene_idx']: r for r in rows}
            path = BASE/'summaries'/arm/f'{cid}.json'
            summaries[arm][cid] = read(path) if path.exists() else None
            failures[arm][cid] = lines(BASE/'scenes'/arm/'failures'/f'{cid}.jsonl')
    stats = {'selection': {'ids': ids, 'count': len(ids), 'order': 'catalog content_id ascending'},
             'expected_scenes': sum(map(len, expected.values())), 'arms': {}}
    for arm in ARMS:
        rows = [r for cid in ids for r in scenes[arm][cid].values()]
        docs = [d for d in summaries[arm].values() if d]
        fail = [r for cid in ids for r in failures[arm][cid]]
        raw = [r for r in rows if isinstance(r.get('scene_graph'), str)]
        graphs = [r['scene_graph'] for r in rows if isinstance(r.get('scene_graph'), dict)]
        a = {'scene_files': sum(bool(scenes[arm][cid]) for cid in ids), 'saved_scenes': len(rows),
             'missing_scenes': [[cid, i] for cid in ids for i in sorted(expected[cid]-scenes[arm][cid].keys())],
             'extra_scenes': [[cid, i] for cid in ids for i in sorted(scenes[arm][cid].keys()-expected[cid])],
             'raw_scenes': len(raw), 'structured_scenes': len(graphs),
             'videos_with_raw': sum(any(isinstance(r.get('scene_graph'), str) for r in scenes[arm][cid].values()) for cid in ids),
             'warning_tags': dict(C.Counter(t for r in rows for t in r.get('warning', []))),
             'scene_output_tokens': dist([r['tokens'] for r in rows if type(r.get('tokens')) is int]),
             'failure_count': len(fail), 'failure_errors': dict(C.Counter(r['error'] for r in fail)),
             'summary_count': len(docs), 'summary_status': dict(C.Counter(d['status'] for d in docs)),
             'summary_words': dist([len(d.get('text','').split()) for d in docs if d['status']=='complete']),
             'summary_tokens': dist([d['tokens'] for d in docs if type(d.get('tokens')) is int]),
             'summary_models': dict(C.Counter(d.get('provenance',{}).get('model',{}).get('model_id','unknown') for d in docs)),
             'summary_prompt_hashes': dict(C.Counter(d.get('provenance',{}).get('prompt_hash','unknown') for d in docs)),
             'summary_source_runs': dict(C.Counter(str(d.get('provenance',{}).get('scene_path','')).split('/runs/')[-1].split('/')[0] for d in docs)),
             'summary_provenance_raw_total': sum(d.get('provenance',{}).get('raw_scene_count',0) for d in docs),
             'summary_over_350_words': sum(len(d.get('text','').split())>350 for d in docs)}
        if graphs:
            entities = [e for g in graphs for e in g.get('entities',[])]
            actions = [x for g in graphs for x in g.get('actions',[])]
            a.update(entity_counts=dist([len(g.get('entities',[])) for g in graphs]),
                     action_counts=dist([len(g.get('actions',[])) for g in graphs]),
                     empty_actions=sum(not g.get('actions') for g in graphs),
                     actions_total=len(actions), actions_with_tool=sum(x.get('tool') is not None for x in actions),
                     actions_null_actor=sum(x.get('actor') is None for x in actions),
                     actions_null_target=sum(x.get('target') is None for x in actions),
                     entity_names=C.Counter(e['name'].lower() for e in entities).most_common(30),
                     action_names=C.Counter(x['action'].lower() for x in actions).most_common(25),
                     topics=C.Counter(t.lower() for g in graphs for t in g.get('topics',[])).most_common(35),
                     media=dict(C.Counter(g.get('medium') for g in graphs)),
                     formats=dict(C.Counter(g.get('format') for g in graphs)),
                     self_actions=sum(x.get('actor') is not None and x.get('actor')==x.get('target') for x in actions),
                     exact_entity_duplicate_scenes=sum(len(g['entities'])!=len({(e['name'].lower(),tuple(e.get('attributes',[]))) for e in g['entities']}) for g in graphs))
        a['summary_noncomplete_ids'] = [cid for cid,d in summaries[arm].items() if not d or d['status'] != 'complete']
        a['cjk_output_scenes'] = sum(bool(re.search(r'[\u3400-\u9fff]', json.dumps(r.get('scene_graph',r.get('description','')),ensure_ascii=False))) for r in rows)
        if raw:
            a['raw_parser_replay'] = dict(C.Counter(parse_or_repair_graph(r['scene_graph']).error or 'parsed; downstream validation failed' for r in raw))
            action_lines = [[l.strip() for l in r['scene_graph'].split('[Actions]',1)[-1].splitlines() if ' - ' in l] for r in raw]
            a['raw_multiple_actions_same_line_candidates'] = sum(any(l.count(' - ')>=4 for l in ls) for ls in action_lines)
            a['raw_missing_tool_separator_candidates'] = sum(any(';' not in l for l in ls) for ls in action_lines)
        a['failure_max_new_tokens'] = dict(C.Counter(str(r.get('provenance',{}).get('settings',{}).get('max_new_tokens')) for r in fail))
        stats['arms'][arm] = a
    paired = [(cid,i) for cid in ids for i in sorted(scenes['graph_qwen'][cid].keys() & scenes['graph_gemini'][cid].keys())]
    structural = [(cid,i) for cid,i in paired if all(isinstance(scenes[a][cid][i]['scene_graph'],dict) for a in ARMS[1:])]
    stats['graph_pair'] = {'aligned':len(paired), 'both_structured':len(structural)}
    for key in ('medium','format'):
        stats['graph_pair'][key+'_agreement'] = sum(scenes['graph_qwen'][c][i]['scene_graph'].get(key)==scenes['graph_gemini'][c][i]['scene_graph'].get(key) for c,i in structural)
    stats['graph_pair']['paired_metrics'] = {}
    for arm in ARMS[1:]:
        gs = [scenes[arm][c][i]['scene_graph'] for c,i in structural]
        acts = [x for g in gs for x in g.get('actions',[])]
        stats['graph_pair']['paired_metrics'][arm] = {
            'scenes':len(gs),
            'mean_entities':round(S.mean(len(g['entities']) for g in gs),2),
            'mean_actions':round(S.mean(len(g['actions']) for g in gs),2),
            'empty_actions':sum(not g['actions'] for g in gs),
            'exact_entity_duplicate_scenes':sum(len(g['entities'])!=len({(e['name'].lower(),tuple(e.get('attributes',[]))) for e in g['entities']}) for g in gs),
            'actions_total':len(acts), 'actions_with_tool':sum(x.get('tool') is not None for x in acts),
            'exact_character_interaction_topic':sum('character interaction' in g.get('topics',[]) for g in gs),
            'black_hair_red_eyes_scenes':sum(any('black hair' in e.get('attributes',[]) and 'red eyes' in e.get('attributes',[]) for e in g['entities']) for g in gs),
            'prompt_limit_exceeded_scenes':sum(len(g['entities'])>6 or len(g['actions'])>4 or any(len(e.get('attributes',[]))>2 for e in g['entities']) for g in gs),
        }
    stats['graph_pair']['format_disagreements'] = [{'qwen':q,'gemini':g,'count':n} for (q,g),n in C.Counter((scenes['graph_qwen'][c][i]['scene_graph']['format'],scenes['graph_gemini'][c][i]['scene_graph']['format']) for c,i in structural if scenes['graph_qwen'][c][i]['scene_graph']['format']!=scenes['graph_gemini'][c][i]['scene_graph']['format']).most_common(15)]
    old=ROOT/'artifacts/runs/v4_260922/extraction'
    stats['desc_v4_byte_identical']={phase:sum((old/phase/'desc_qwen'/f'{cid}.{ext}').exists() and
                (old/phase/'desc_qwen'/f'{cid}.{ext}').read_bytes()==(BASE/phase/'desc_qwen'/f'{cid}.{ext}').read_bytes()
                for cid in ids) for phase,ext in [('scenes','jsonl'),('summaries','json')]}
    pairs=[]
    for cid in ids:
        for index in sorted(expected[cid] | set().union(*(set(scenes[a][cid]) for a in ARMS))):
            pairs.append({'content_id':cid,'scene_idx':index,
                          **{a:scenes[a][cid].get(index) for a in ARMS},
                          'evidence':next((r for r in evidence.get(cid,[]) if r['scene_idx']==index),None)})
    (OUT/'scene_pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
    (OUT/'summary_pairs.jsonl').write_text(''.join(json.dumps({'content_id':cid, **{a: {'text': (summaries[a][cid] or {}).get('text',''), 'status': (summaries[a][cid] or {}).get('status','missing')} for a in ARMS}},ensure_ascii=False)+'\n' for cid in ids))
    save('stats.json',stats)
    print(json.dumps({a:{k:v for k,v in d.items() if k not in {'missing_scenes','extra_scenes','entity_names','action_names','topics'}} for a,d in stats['arms'].items()},ensure_ascii=False,indent=2))
    print('expected',stats['expected_scenes'],'pair',stats['graph_pair'],'desc_v4',stats['desc_v4_byte_identical'])

if __name__=='__main__':main()
