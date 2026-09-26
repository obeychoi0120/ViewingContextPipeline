from pathlib import Path
import json,sys
from PIL import Image,ImageDraw
sys.path.insert(0,'src')
from visual_sampling import timestamp_stem
base=Path('docs/reports/v5_260923/desc_gemini_v3');(base/'evidence').mkdir(exist_ok=True)
r=list(map(json.loads,(base/'sample1000.jsonl').read_text().splitlines()))
for gi,ns in enumerate([[39,269,753],[96,651,850]]):
 sheet=Image.new('RGB',(1280,1200),'white');d=ImageDraw.Draw(sheet)
 for row,n in enumerate(ns):
  x=r[n-1];cid=x['content_id'];s=x['scene_idx']
  ts=json.loads((Path('artifacts/preparation/source_assets')/cid/'timestamp_fixed_30s.json').read_text())[s]['keyframe_timestamps']
  for j,t in enumerate(ts):
   p=Path('artifacts/preparation/resized_keyframes')/cid/(timestamp_stem(t)+'.png')
   im=Image.open(p).convert('RGB');im.thumbnail((426,176));xx=(j%3)*426;yy=row*400+(j//3)*200
   d.text((xx+3,yy+2),f'{cid} scene={s} t={t}',fill='black');sheet.paste(im,(xx,yy+20))
 sheet.save(base/'evidence'/f'contact_{gi+1}.jpg')
