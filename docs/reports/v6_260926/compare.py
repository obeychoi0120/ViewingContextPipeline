from pathlib import Path
import json,collections,statistics,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
base=ROOT/'artifacts/runs/v6_260926/extraction/scenes'
data={a:{(r['content_id'],r['scene_idx']):r for p in sorted((base/a).glob('microlens_*.jsonl')) for r in map(json.loads,p.read_text().splitlines())} for a in ['desc_gemini','graph_gemini']}
keys=sorted(data['desc_gemini'].keys()&data['graph_gemini'].keys())[:1000]
rows=[{'content_id':c,'scene_idx':i,**{a:data[a][c,i] for a in data}} for c,i in keys]
(OUT/'pairs1000.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
gs=[r['graph_gemini']['scene_graph'] for r in rows if isinstance(r['graph_gemini']['scene_graph'],dict)]
stats={'scope':'first 1000 common scenes sorted by content_id, scene_idx','scenes':len(rows),'videos':len({c for c,i in keys}),'last':keys[-1],'availability':{a:{'videos':len({c for c,i in d}),'scenes':len(d)} for a,d in data.items()},'structured':len(gs),'raw':len(rows)-len(gs),'warnings':dict(collections.Counter(t for r in rows for t in r['graph_gemini'].get('warning',[]))),'desc_words_mean':statistics.mean(len(r['desc_gemini']['description'].split()) for r in rows),'tokens_mean':{a:statistics.mean(r[a]['tokens'] for r in rows if isinstance(r[a].get('tokens'),int)) for a in data},'entities_mean':statistics.mean(len(g['entities']) for g in gs),'actions_mean':statistics.mean(len(g['actions']) for g in gs),'empty_actions':sum(not g['actions'] for g in gs),'formats':dict(collections.Counter(g['format'] for g in gs)),'topics':collections.Counter(t for g in gs for t in g['topics']).most_common(20),'sample_sha256':hashlib.sha256((OUT/'pairs1000.jsonl').read_bytes()).hexdigest()}
(OUT/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));print(json.dumps(stats,ensure_ascii=False,indent=2))
# Spread close reading across distinct videos, rather than selecting many scenes from one long video.
by={}
for r in rows:by.setdefault(r['content_id'],[]).append(r)
review=[v[j%len(v)] for j,v in enumerate(by.values()) if j%4==0]
(OUT/'review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review));print('review',len(review))
