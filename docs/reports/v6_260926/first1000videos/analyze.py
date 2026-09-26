"""Read-only comparison of existing scene artifacts, no inference."""
from pathlib import Path
import json,collections,statistics,hashlib
ROOT=Path(__file__).resolve().parents[4]; OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/runs/v6_260926/extraction'; ARMS=['desc_gemini','graph_gemini']
def readlines(p):
 return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
 catalog=readlines(ROOT/'artifacts/preparation/cohort/catalog.jsonl')
 ids=[r['content_id'] for r in sorted(catalog,key=lambda r:r['content_id'])[:1000]]
 data={a:{c:{r['scene_idx']:r for r in readlines(BASE/'scenes'/a/f'{c}.jsonl')} for c in ids} for a in ARMS}
 pairs=[];expected=set()
 for c in ids:
  p=ROOT/'artifacts/preparation/source_assets'/c/'timestamp_fixed_30s.json'
  ts=json.loads(p.read_text()) if p.exists() else []
  expected.update((c,i) for i in range(len(ts)))
  for i in sorted(set(range(len(ts)))|set(data[ARMS[0]][c])|set(data[ARMS[1]][c])):
   pairs.append({'content_id':c,'scene_idx':i,**{a:data[a][c].get(i) for a in ARMS},'evidence':ts[i] if i<len(ts) else None})
 common=[r for r in pairs if all(r[a] for a in ARMS)]
 gs=[r['graph_gemini']['scene_graph'] for r in common if isinstance(r['graph_gemini']['scene_graph'],dict)]
 stats={'scope':'first 1000 catalog videos by content_id; all expected scenes aligned','first':ids[0],'last':ids[-1], 'videos':len(ids),'expected_scenes':len(expected),'paired_scenes':len(common),'paired_videos':len({r['content_id'] for r in common}),'arms':{}}
 for a in ARMS:
  rows=[r for d in data[a].values() for r in d.values()];saved={(r['content_id'],r['scene_idx']) for r in rows}
  failures=[r for c in ids for r in readlines(BASE/'scenes'/a/'failures'/f'{c}.jsonl')]
  raw=[r for r in rows if isinstance(r.get('scene_graph'),str)]
  stats['arms'][a]={'saved_videos':sum(bool(v) for v in data[a].values()),'saved_scenes':len(rows),'missing_scenes':sorted(expected-saved),'raw':len(raw),'structured':sum(isinstance(r.get('scene_graph'),dict) for r in rows),'warnings':dict(collections.Counter(t for r in rows for t in r.get('warning',[]))),'failure_count':len(failures),'failure_errors':dict(collections.Counter(r.get('error') for r in failures)),'tokens_mean':statistics.mean(r['tokens'] for r in rows if type(r.get('tokens')) is int),'summary_files':sum((BASE/'summaries'/a/f'{c}.json').exists() for c in ids),'failure_prompt_hashes':dict(collections.Counter(r.get('provenance',{}).get('prompt_hash') for r in failures))}
 stats['graph_paired_structured']={'scenes':len(gs),'empty_actions':sum(not g['actions'] for g in gs),'entity_mean':statistics.mean(len(g['entities']) for g in gs),'actions_mean':statistics.mean(len(g['actions']) for g in gs),'formats':dict(collections.Counter(g['format'] for g in gs)),'topics':collections.Counter(t.lower() for g in gs for t in g['topics']).most_common(30)}
 stats['paired_desc_words_mean']=statistics.mean(len(r['desc_gemini']['description'].split()) for r in common)
 (OUT/'scene_pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
 stats['pairs_sha256']=hashlib.sha256((OUT/'scene_pairs.jsonl').read_bytes()).hexdigest()
 # 50 videos spread over the requested range; deterministic scene selection.
 review=[]
 for rank,c in enumerate(ids):
  if rank%20:continue
  available=[r for r in common if r['content_id']==c]
  if available:review.append(available[rank%len(available)])
 (OUT/'systematic50.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review))
 dump('stats.json',stats)
 print(json.dumps(stats,ensure_ascii=False,indent=2))
 print('systematic sample',len(review))
if __name__=='__main__':main()
