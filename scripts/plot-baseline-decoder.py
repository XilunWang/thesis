"""Visualize same query at its best early stage and final stage; no inference."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pycocotools.coco import COCO
from pycocotools import mask as mu
out=ROOT/'reports/baseline-decoder-audit-20261031';rows=json.loads((out/'instances.json').read_text());gt=COCO(str(ROOT/'data/UIIS/annotations/val.json'));font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
selected=[124,126,280,463,125,11];canvas=Image.new('RGB',(1080,1200),'white')
for i,gid in enumerate(selected):
 r=next(x for x in rows if x['gt_id']==gid);a=gt.anns[gid];iid=a['image_id'];g=gt.annToMask(a).astype(bool);data=json.loads((out/f'image-{iid}.json').read_text());early=r['best_early_stage'];q=r['best_early_query']
 path=ROOT/'data/UIIS'/gt.imgs[iid]['file_name']
 if not path.exists():path=ROOT/'data/UIIS/val'/Path(gt.imgs[iid]['file_name']).name
 im=Image.open(path).convert('RGB');ys,xs=np.nonzero(g);side=max(64,int(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)*3));cx=(xs.min()+xs.max())//2;cy=(ys.min()+ys.max())//2;box=(max(0,cx-side//2),max(0,cy-side//2),min(im.width,cx+side//2),min(im.height,cy+side//2))
 for j,s in enumerate([None,early,9]):
  ar=np.array(im).copy()
  if s is None:ar[g]=(ar[g]*.4+np.array([30,255,70])*.6).astype('uint8');title=f'GT {gid} ({r["group"]})'
  else:
   pm=mu.decode(data['layers'][s]['rles'][q]).astype(bool);ar[pm]=(ar[pm]*.4+np.array([255,50,30])*.6).astype('uint8');ar[pm&g]=(ar[pm&g]*.4+np.array([255,235,0])*.6).astype('uint8');t=r['early_query_track'][s];title=f'S{s} Q{q}: IoU {t["iou"]:.3f}; class {t["correct_probability"]:.3f}'
  crop=Image.fromarray(ar).crop(box);ratio=min(356/crop.width,167/crop.height);crop=crop.resize((round(crop.width*ratio),round(crop.height*ratio)),Image.Resampling.NEAREST);canvas.paste(crop,(j*360+(356-crop.width)//2,i*200+29));ImageDraw.Draw(canvas).text((j*360+3,i*200+3),title,font=font,fill='black')
canvas.save(out/'same-query-cases.png')
# Complete stage-by-stage table as an image, with a fixed color key at IoU 0.5.
ordered=sorted(rows,key=lambda r:(r['group'],r['gt_id']));heat=Image.new('RGB',(1020,80+24*len(ordered)),'white');draw=ImageDraw.Draw(heat)
draw.text((6,4),'Best IoU among all 100 queries at each stage (GT-assisted; NOT recall/AP)',font=font,fill='black')
for s in range(10):draw.text((240+s*76,35),f'S{s}',font=font,fill='black')
for i,r in enumerate(ordered):
 y=70+i*24;draw.text((5,y),f'{r["gt_id"]} {"failure" if r["group"]=="failure" else "low-score"}',font=font,fill='black')
 for s in r['stages']:
  x=235+s['stage']*76;v=s['best_iou'];draw.rectangle((x,y,x+73,y+23),fill='#b6dfbd' if v>=.5 else '#f2d6c6');draw.text((x+6,y),f'{v:.3f}',font=font,fill='black')
heat.save(out/'all-stage-iou.png');print('DECODER_VISUALS_CREATED')
