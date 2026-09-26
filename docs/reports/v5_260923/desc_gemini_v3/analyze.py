"""Audit saved descriptions only. No model calls or changes to run artifacts."""
from pathlib import Path
import collections,json,re,statistics
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/runs/v5_260923'
PATTERNS={
 'shot_narration':r'\b(close[- ]up|medium shot|wide shot|camera|frame|view shifts|cuts? to|sequence begins)\b',
 'sequencing':r'\b(initially|subsequently|finally|then|later|transitions?|sequence begins|scene (?:begins|ends))\b',
 'appearance':r'\b(hair|eyes|shirt|jacket|dress|robe|wearing|clad|outfit)\b',
 'text_mentions':r'\b(subtitles?|captions?|on-screen text|text overlay|text overlays|watermark|written|text reads|text states)\b',
 'audio_claims':r'\b(narrator|narration|voiceover|voice-over|soundtrack|music|singing|says|explains|discusses)\b',
 'interpretation':r'\b(suggesting|suggests|implying|implies|presumably|likely|appears to|seems to|possibly)\b',
 'viewer_pitch':r'\b(viewers? (?:who|interested|seeking|looking)|well-suited|appeals? to|ideal for|perfect for)\b',
 'instruction_focus':r'\b(tutorial|demonstrat\w+|compar\w+|technique|beginner|customiz\w+|upgrad\w+|review|explain\w+)\b',
}
def main():
 files=sorted((RUN/'extraction/scenes/desc_gemini').glob('microlens_*.jsonl'))[:1000]
 selected=[];total=0
 for rank,p in enumerate(files):
  rows=sorted([json.loads(l) for l in p.read_text().splitlines() if l.strip()],key=lambda r:r['scene_idx'])
  total+=len(rows)
  r=rows[rank%len(rows)]
  text=r['description']
  hits={k:len(re.findall(v,text,re.I)) for k,v in PATTERNS.items()}
  selected.append({**r,'word_count':len(text.split()),'signals':hits,'han_characters':len(re.findall(r'[\u3400-\u9fff]',text)),
                   'quoted_spans':re.findall(r'"([^"\n]+)"',text)})
 (OUT/'sample1000.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
 summary={'run':RUN.name,'available_video_files':len(list((RUN/'extraction/scenes/desc_gemini').glob('microlens_*.jsonl'))),
  'video_files':len(files),'scenes_in_first1000_videos':total,'sample_scenes':len(selected),
  'selection':'content_id ascending first 1000 videos; sorted scene index at zero-based video rank modulo scene count',
  'words':{'mean':round(statistics.mean(r['word_count'] for r in selected),2),'median':statistics.median(r['word_count'] for r in selected),'min':min(r['word_count'] for r in selected),'max':max(r['word_count'] for r in selected)},
  'words_over350':sum(r['word_count']>350 for r in selected),
  'signals_present':{k:sum(r['signals'][k]>0 for r in selected) for k in PATTERNS},
  'han_present':sum(r['han_characters']>0 for r in selected),'quoted_span_present':sum(bool(r['quoted_spans']) for r in selected),
  'signal_patterns':PATTERNS}
 (OUT/'stats.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
 # Reproducible systematic close-reading subset, supplemented by signal candidates.
 ids=list(range(0,1000,25))
 for key in ['text_mentions','audio_claims','interpretation','shot_narration']:
  ids.extend(sorted(range(1000),key=lambda i:selected[i]['signals'][key],reverse=True)[:5])
 ids.extend(i for i,r in enumerate(selected) if r['han_characters'])
 ids=list(dict.fromkeys(ids))
 (OUT/'review_candidates.jsonl').write_text(''.join(json.dumps(selected[i],ensure_ascii=False)+'\n' for i in ids))
 print('review candidates',len(ids))
if __name__=='__main__':main()
