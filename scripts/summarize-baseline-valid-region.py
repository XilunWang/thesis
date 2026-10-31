"""Paired geometry comparison for unchanged forward predictions."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools import mask as mu

def main():
 out=ROOT/'reports/baseline-valid-region-20261031';result=json.loads((out/'result.json').read_text());assert result['passed'];sel=json.loads((out/'selection.json').read_text());rows=[]
 for line in (out/'remote-hashes.txt').read_text(encoding='utf-8-sig').splitlines():
  a=line.split()
  if len(a)==2 and len(a[0])==64:assert hashlib.sha256((ROOT/a[1]).read_bytes()).hexdigest()==a[0]
 assert hashlib.sha256((ROOT/'scripts/audit-baseline-valid-region.py').read_bytes()).hexdigest()==result['source_sha256']
 for iid in sel['image_ids']:
  d=json.loads((out/f'image-{iid}.json').read_text());io=mu.iou(d['pred_rles'],d['gt_rles'],[0]*len(d['labels']));assert np.allclose(io,d['ious'],rtol=0,atol=1e-12)
  for j,area in enumerate(d['gt_pixels']):
   groups={}
   for name in ['baseline','valid_region']:
    runs=[]
    for run in d['runs']:
     x=run[name];q=x['assignment'].index(j+1);runs.append(dict(query=q,iou=float(io[q,j]),sampled_positive=x['sampled_gt_positive'][j],correct_probability=d['class_probabilities'][q][d['labels'][j]],costs=[c[q][j] for c in x['costs']]))
    groups[name]=runs
   rows.append(dict(image_id=iid,gt_index=j,area=area,label=d['labels'][j],padded=d['shape']!=d['valid_shape'],best_iou=float(io[:,j].max()),**groups))
 def summary(rs):
  ans={}
  for seed in range(3):
   b=[r['baseline'][seed]['iou'] for r in rs];v=[r['valid_region'][seed]['iou'] for r in rs];delta=np.array(v)-b
   ans[str(seed)]=dict(gt=len(rs),baseline_qualified=sum(x>=.5 for x in b),valid_qualified=sum(x>=.5 for x in v),gained=sum(x<.5<=y for x,y in zip(b,v)),lost=sum(y<.5<=x for x,y in zip(b,v)),baseline_mean_iou=float(np.mean(b)),valid_mean_iou=float(np.mean(v)),improved_ge01=int((delta>=.1).sum()),worsened_ge01=int((delta<=-.1).sum()),any_decrease=int((delta < -1e-12).sum()),worst_delta=float(delta.min()),baseline_zero_points=sum(r['baseline'][seed]['sampled_positive']==0 for r in rs),valid_zero_points=sum(r['valid_region'][seed]['sampled_positive']==0 for r in rs))
  ans['query_changed_across_seeds']={name:sum(len({x['query'] for x in r[name]})>1 for r in rs) for name in ['baseline','valid_region']}
  return ans
 result=dict(passed=True,groups={name:summary([r for r in rows if lo<=r['area']<hi]) for name,lo,hi in [('small',0,1024),('non_small',1024,float('inf'))]},images=len(sel['image_ids']),no_padding_images=sorted({r['image_id'] for r in rows if not r['padded']}),download_hashes_and_rle_ious_verified=True)
 (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'instances.json').write_text(json.dumps(rows,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
