from pathlib import Path
from PIL import Image,ImageDraw
import json,sys
sys.path.insert(0,'src')
from visual_sampling import timestamp_stem
out=Path('docs/reports/v6_260926/first1000videos/evidence');out.mkdir(exist_ok=True)
for gi,group in enumerate([[(110,0),(126,0),(41,5)],[(96,0),(131,0),(154,0)],[(121,1),(481,0),(621,0)],[(781,0),(941,4),(881,0)]]):
 sheet=Image.new('RGB',(1280,1200),'white');d=ImageDraw.Draw(sheet)
 for row,(n,s) in enumerate(group):
  cid=f'microlens_100k_{n:05d}';ts=json.loads((Path('artifacts/preparation/source_assets')/cid/'timestamp_fixed_30s.json').read_text())[s]['keyframe_timestamps']
  for j,t in enumerate(ts):
   p=Path('artifacts/preparation/resized_keyframes')/cid/(timestamp_stem(t)+'.png');im=Image.open(p).convert('RGB');im.thumbnail((426,176));xx=(j%3)*426;yy=row*400+(j//3)*200
   d.text((xx+3,yy+2),f'{cid} s={s} t={t}',fill='black');sheet.paste(im,(xx,yy+20))
 sheet.save(out/f'contact_{gi+1}.jpg')
