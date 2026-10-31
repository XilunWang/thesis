"""Local mask shape diagnostics and exhaustive contact sheets, no inference."""
import json,sys,hashlib,math
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pycocotools.coco import COCO
from pycocotools import mask as mu
FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def overlay(im,mask,color):
 a=np.array(im).copy();a[mask]=(a[mask]*.45+np.array(color)*.55).astype('uint8');return Image.fromarray(a)
def fitted(im,size):
 im=im.copy();factor=min(size[0]/im.width,size[1]/im.height);im=im.resize((max(1,round(im.width*factor)),max(1,round(im.height*factor))),Image.Resampling.NEAREST);out=Image.new('RGB',size,'#242424');out.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return out

def main():
 src=ROOT/'reports/baseline-query-audit-20261031';out=ROOT/'reports/baseline-mask-shape-20261031';out.mkdir(exist_ok=True)
 paths=[src/'result.json',src/'instances.json',ROOT/'data/UIIS/annotations/val.json']+list(src.glob('image-*.json'));hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
 if (out/'summary.json').exists():assert json.loads((out/'summary.json').read_text())['input_hashes']==hashes, 'Inputs changed: preserve previous audit'
 gt=COCO(str(ROOT/'data/UIIS/annotations/val.json'));prior=json.loads((src/'instances.json').read_text());records=[];cards={'failure':[],'low_score_control':[]};full=[]
 for old in sorted(prior,key=lambda r:r['gt_id']):
  if old['area']>=1024:continue
  status=old['0.5']['status']
  if status not in ['no_geometry','selected_but_low_score']:continue
  group='failure' if status=='no_geometry' else 'low_score_control';a=gt.anns[old['gt_id']];iid=a['image_id'];g=gt.annToMask(a).astype(bool)
  data=json.loads((src/f'image-{iid}.json').read_text());queries=data['queries'];ious=mu.iou([q['segmentation'] for q in queries],[gt.annToRLE(a)],[0])[:,0];qi=int(ious.argmax());pm=mu.decode(queries[qi]['segmentation']).astype(bool)
  assert abs(float(ious[qi])-old['best_raw_iou'])<1e-12
  inter=int((pm&g).sum());ga=int(g.sum());pa=int(pm.sum());fn=ga-inter;fp=pa-inter;rec=inter/ga;prec=inter/pa if pa else 0
  others=np.zeros_like(g);neighbors=[]
  for b in gt.imgToAnns[iid]:
   if b['id']==a['id'] or b.get('iscrowd',0):continue
   bm=gt.annToMask(b).astype(bool);others|=bm;bi=int((pm&bm).sum());ba=int(bm.sum())
   if ba and bi/ba>=.5:neighbors.append(dict(gt_id=b['id'],category_id=b['category_id'],coverage=bi/ba,gt_overlap=int((g&bm).sum())))
  label='near_zero_coverage' if rec<.1 else 'underfill_dominant' if fn>fp else 'overfill_dominant' if fp>fn else 'equal_errors'
  row=dict(gt_id=a['id'],image_id=iid,category_id=a['category_id'],category=gt.cats[a['category_id']]['name'],area=a['area'],group=group,query=qi,iou=float(ious[qi]),recall=rec,precision=prec,gt_pixels=ga,pred_pixels=pa,fn=fn,fp=fp,error_type=label,other_gt_fp=int((pm&~g&others).sum()),multi_instance_coverage=rec>=.5 and bool(neighbors),neighbors_covered_half=neighbors)
  records.append(row)
  im=Image.open(ROOT/'data/UIIS'/gt.imgs[iid]['file_name']).convert('RGB') if (ROOT/'data/UIIS'/gt.imgs[iid]['file_name']).exists() else Image.open(ROOT/'data/UIIS/val'/Path(gt.imgs[iid]['file_name']).name).convert('RGB')
  ys,xs=np.nonzero(g);side=max(64,int(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)*3));cx=(xs.min()+xs.max())//2;cy=(ys.min()+ys.max())//2;box=(max(0,cx-side//2),max(0,cy-side//2),min(im.width,cx+side//2),min(im.height,cy+side//2))
  gm=overlay(im,g,(40,255,80));pred=overlay(im,pm,(255,60,50));pred=overlay(pred,g&pm,(255,230,20))
  card=Image.new('RGB',(480,260),'white');d=ImageDraw.Draw(card)
  d.text((6,3),f"GT{a['id']} {row['category']} Q{qi} IoU {ious[qi]:.4f}",font=FONT,fill='black')
  d.text((6,23),f"coverage {rec:.2f} purity {prec:.2f} FN {fn} FP {fp}",font=FONT,fill='black')
  for j,img in enumerate([im,gm,pred]):card.paste(fitted(img.crop(box),(158,175)),(j*160,48))
  d.text((6,227),f"{label}; multi-GT {row['multi_instance_coverage']}",font=FONT,fill='black');cards[group].append(card)
  whole=Image.new('RGB',(480,260),'white');whole.paste(fitted(pred,(480,230)),(0,25));ImageDraw.Draw(whole).text((4,3),f"GT{a['id']} Q{qi}: full prediction red; overlap yellow",font=FONT,fill='black');full.append(whole)
 def sheets(name,items):
  for start in range(0,len(items),9):
   page=Image.new('RGB',(1440,780),'#cccccc')
   for j,card in enumerate(items[start:start+9]):page.paste(card,((j%3)*480,(j//3)*260))
   page.save(out/f'{name}-{start//9+1:02}.jpg',quality=92)
 for name,items in cards.items():sheets(name,items)
 selected=[124,125,381,421,1437,11]
 combined=Image.new('RGB',(960,780),'white')
 for j,gid in enumerate(selected):
  row=next(r for r in records if r['gt_id']==gid);group=row['group'];index=[r['gt_id'] for r in records if r['group']==group].index(gid);combined.paste(cards[group][index],((j%2)*480,(j//2)*260))
 combined.save(out/'representative-cases.png')
 sheets('full-context',full)
 def summary(rs):return dict(count=len(rs),error_types=dict(Counter(r['error_type'] for r in rs)),multi_instance_coverage=sum(r['multi_instance_coverage'] for r in rs),by_class=dict(Counter(r['category'] for r in rs)),mean_iou=float(np.mean([r['iou'] for r in rs])))
 failures=[r for r in records if r['group']=='failure']
 details=dict(coverage_at_least_half=sum(r['recall']>=.5 for r in failures),coverage_at_least_08=sum(r['recall']>=.8 for r in failures),precision_below_01=sum(r['precision']<.1 for r in failures),iou_04_to_05=sum(.4<=r['iou']<.5 for r in failures),other_gt_majority_fp=sum(r['other_gt_fp']>r['fp']/2 for r in failures),multi_with_gt_overlap=sum(r['multi_instance_coverage'] and any(n['gt_overlap']>0 for n in r['neighbors_covered_half']) for r in failures))
 result=dict(failure_details=details,passed=True,groups={name:summary([r for r in records if r['group']==name]) for name in cards},input_hashes=hashes,source_sha256=digest(Path(__file__)),training=False,server_used=False)
 assert len(cards['failure'])==45 and len(cards['low_score_control'])==22
 assert all(digest(ROOT/p)==v for p,v in hashes.items())
 (out/'instances.json').write_text(json.dumps(records,indent=2));(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['groups'],indent=2))
if __name__=='__main__':main()
