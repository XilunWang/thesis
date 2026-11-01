"""Cached holdout rule replay, NOT recovered training labels."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT/'.local/bll-libs'),str(ROOT)]
import numpy as np
import cv2
from pycocotools import mask as mu
from scripts.instance_representative import make_layout
from scripts.position_grid4 import make_targets
OUT=ROOT/'reports/label-only-point-bias-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 scores=ROOT/'reports/label-only-audit-scores-20261101';visual=ROOT/'reports/label-only-visual-audit-20261101'
 request=json.loads((scores/'request.json').read_text());proof=json.loads((scores/'result.json').read_text());exp=json.loads((visual/'export.json').read_text())
 assert proof['passed'] and exp['passed']
 records=[];negatives=[];sources={}
 frozen=json.loads((ROOT/'reports/position-learning-plan-grid4-20261101/views.json').read_text())
 views={(v['image_id'],v['seed']):v for v in frozen['runs']}
 for c,r,e in zip(request['cases'],proof['runs'],exp['runs']):
  assert c['case_id']==r['case_id']==e['case_id']
  p=scores/r['file'];assert sha(p)==r['sha256'];a=np.load(p);valid=a['valid'].astype(bool)
  p=visual/e['file'];assert sha(p)==e['sha256'];im=cv2.imread(str(p));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float)
  cache=ROOT/'reports/position-learning-preflight-20261101'/f"holdout-{c['image_id']}-{c['seed']}.json";d=json.loads(cache.read_text());sources[cache.name]=sha(cache)
  assert d['input_sha256']==r['input_sha256'] and d['masks_sha256']==r['masks_sha256']
  masks=mu.decode(d['gt_rles']).transpose(2,0,1);h,w=d['padded_shape'];gh,gw=h//4,w//4
  t,v,old=make_targets(masks,d['annotation_ids'],d['original_areas'],d['valid_shape']);assert np.array_equal(v.numpy(),valid)
  assert {str(k):xy for k,xy in old.items()}==views[(c['image_id'],c['seed'])]['assignments']
  grad=np.hypot(cv2.Sobel(gray,cv2.CV_64F,1,0),cv2.Sobel(gray,cv2.CV_64F,0,1))
  def cells_mean(arr):
   pad=np.zeros((h,w));pad[:arr.shape[0],:arr.shape[1]]=arr
   count=np.zeros((h,w));count[:arr.shape[0],:arr.shape[1]]=1
   return pad.reshape(gh,4,gw,4).sum((1,3)).ravel()/np.maximum(1,count.reshape(gh,4,gw,4).sum((1,3)).ravel())
  bright=cells_mean(gray);edge=cells_mean(grad)
  layouts={s:make_layout(a[s],masks,d['annotation_ids'],d['original_areas'],valid) for s in ['initial','new100','new300','old300']}
  for s,l in layouts.items():
   assert c['cell'] not in l['representatives'].values()
   negatives.append(dict(case_id=c['case_id'],stage=s,positive=False,logit=float(a[s].ravel()[c['cell']]),direct_bce_direction='decrease'))
  for i,(gid,area) in enumerate(zip(d['annotation_ids'],d['original_areas'])):
   if not 0<area<=1024:continue
   m=masks[i];counts=m.reshape(gh,4,gw,4).sum((1,3)).ravel();support=np.flatnonzero(counts)
   chosen={'legacy':old[gid][1]*gw+old[gid][0],**{s:l['representatives'][gid] for s,l in layouts.items()}}
   stats={}
   for name,cell in chosen.items():
    x,y=cell%gw*4,cell//gw*4
    stats[name]=dict(cell=int(cell),foreground_pixels=int(counts[cell]),center_in_gt=bool(m[y+2,x+2]),brightness=float(bright[cell]),brightness_percentile_within_support=float(np.mean(bright[support]<=bright[cell])),gradient=float(edge[cell]),gradient_percentile_within_support=float(np.mean(edge[support]<=edge[cell])))
   records.append(dict(case_id=c['case_id'],gt_id=gid,support_cells=len(support),has_full_foreground_cell=bool((counts==16).any()),stages=stats))
 summaries={}
 for name in ['legacy','initial','new100','new300','old300']:
  items=[r['stages'][name] for r in records]
  summaries[name]=dict(n=len(items),center_outside=sum(not x['center_in_gt'] for x in items),partial_cells=sum(x['foreground_pixels']<16 for x in items),foreground_pixels_mean=float(np.mean([x['foreground_pixels'] for x in items])),brightness_percentile_mean=float(np.mean([x['brightness_percentile_within_support'] for x in items])),gradient_percentile_mean=float(np.mean([x['gradient_percentile_within_support'] for x in items])),different_from_legacy=sum(r['stages'][name]['cell']!=r['stages']['legacy']['cell'] for r in records))
 OUT.mkdir(exist_ok=True)
 result=dict(training_updates=0,model_forward_passes=0,scope='12 holdout views / 58 GT; rule replay only, not actual training label history',summaries=summaries,records=records,audited_candidates=negatives,cache_sha256=sources,source_result_sha256=sha(scores/'result.json'),script_sha256=sha(Path(__file__)))
 (OUT/'result.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
