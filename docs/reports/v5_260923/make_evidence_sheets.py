import json,sys
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,'src')
from visual_sampling import timestamp_stem
out=Path('docs/reports/v5_260923/evidence');out.mkdir(exist_ok=True)
rows={(int(r['content_id'].split('_')[-1]),r['scene_idx']):r for r in map(json.loads,open('docs/reports/v5_260923/scene_pairs.jsonl'))}
groups=[[(8,1),(32,1),(48,0)],[(54,2),(84,2),(98,0)],[(81,0),(181,0),(201,0)],[(251,0),(341,0),(471,0)],[(12,0),(12,2),(96,0)]]
for gi,group in enumerate(groups):
 sheet=Image.new('RGB',(1280,3*400),'white');draw=ImageDraw.Draw(sheet)
 for ri,key in enumerate(group):
  r=rows[key];ts=r['evidence']['keyframe_timestamps']
  for j,t in enumerate(ts):
   p=Path('artifacts/preparation/resized_keyframes')/r['content_id']/(timestamp_stem(t)+'.png')
   im=Image.open(p).convert('RGB');im.thumbnail((426,176))
   x=j%3*426;y=ri*400+j//3*200
   draw.text((x+3,y+2),f'{key[0]:05d} scene={key[1]} t={t}',fill='black');sheet.paste(im,(x,y+20))
 sheet.save(out/f'contact_{gi+1}.jpg')
print('saved',len(groups),'sheets')
