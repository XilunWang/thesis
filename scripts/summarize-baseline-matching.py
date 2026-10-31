"""Summarize frozen full-mask geometry versus actual matching assignments."""
import json,sys,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools import mask as mu

def main():
 out=ROOT/'reports/baseline-matching-audit-20261031';result=json.loads((out/'result.json').read_text());assert result['passed'];sel=json.loads((out/'selection.json').read_text())
 for line in (out/'remote-hashes.txt').read_text(encoding='utf-8-sig').splitlines():
  parts=line.split()
  if len(parts)==2 and len(parts[0])==64:assert hashlib.sha256((ROOT/parts[1]).read_bytes()).hexdigest()==parts[0]
 assert hashlib.sha256((ROOT/'scripts/audit-baseline-matching.py').read_bytes()).hexdigest()==result['source_sha256']
 rows=[]
 for iid in sel['image_ids']:
  d=json.loads((out/f'image-{iid}.json').read_text());io=mu.iou(d['pred_rles'],d['gt_rles'],[0]*len(d['labels']));assert np.allclose(io,d['ious'],rtol=0,atol=1e-12)
  for j,(label,area) in enumerate(zip(d['labels'],d['gt_pixels'])):
   best=int(io[:,j].argmax());probs=np.array(d['class_probabilities']);runs=[]
   for run in d['runs']:
    assignment=run['assignment'];assigned=assignment.index(j+1);status='self' if assignment[best]==j+1 else 'background' if assignment[best]==0 else 'other_gt'
    cost=np.array(run['costs']);delta=cost[:,best,j]-cost[:,assigned,j]
    good=np.flatnonzero(io[:,j]>=.5).tolist();goodbg=[q for q in good if assignment[q]==0];goodother=[q for q in good if assignment[q]>0 and assignment[q]!=j+1]
    runs.append(dict(seed=run['seed'],best_query_status=status,assigned_query=assigned,assigned_iou=float(io[assigned,j]),assigned_correct_probability=float(probs[assigned,label]),best_query_costs=cost[:,best,j].tolist(),assigned_query_costs=cost[:,assigned,j].tolist(),best_minus_assigned_costs=delta.tolist(),best_minus_assigned_total=float(delta.sum()),good_background_queries=goodbg,good_other_gt_queries=goodother,sampled_positive=run['sampled_gt_positive'][j],sampled_atleast_half=run['sampled_gt_atleast_half'][j],sampled_mass=run['sampled_gt_mass'][j]))
   rows.append(dict(image_id=iid,gt_index=j,label=label,area=area,best_query=best,best_iou=float(io[best,j]),best_correct_probability=float(probs[best,label]),best_background_probability=float(probs[best,7]),runs=runs,assigned_query_changed=len({r['assigned_query'] for r in runs})>1,cost_names=d['runs'][0]['cost_names']))
 def summarize(rs):
  primary=[r['runs'][0] for r in rs];gaps=[r for r in rs if r['best_iou']>=.5>r['runs'][0]['assigned_iou']]
  return dict(gt=len(rs),best_geometry05=sum(r['best_iou']>=.5 for r in rs),assigned_geometry05=sum(r['assigned_iou']>=.5 for r in primary),best_query_status=dict(Counter(r['best_query_status'] for r in primary)),good_best_background=sum(r['best_iou']>=.5 and r['runs'][0]['best_query_status']=='background' for r in rs),good_background_exists=sum(bool(r['good_background_queries']) for r in primary),geometry_gap_count=len(gaps),geometry_gap_ge01=sum(r['best_iou']-r['runs'][0]['assigned_iou']>=.1 for r in gaps),geometry_gap_best_background=sum(r['runs'][0]['best_query_status']=='background' for r in gaps),zero_positive_sample_primary=sum(r['sampled_positive']==0 for r in primary),zero_atleast_half_sample_primary=sum(r['sampled_atleast_half']==0 for r in primary),assignment_changed_over3=sum(r['assigned_query_changed'] for r in rs),per_seed=[dict(assigned_geometry05=sum(r['runs'][s]['assigned_iou']>=.5 for r in rs),geometry_gap=sum(r['best_iou']>=.5>r['runs'][s]['assigned_iou'] for r in rs),zero_positive_samples=sum(r['runs'][s]['sampled_positive']==0 for r in rs)) for s in range(3)])
 small=[r for r in rows if r['area']<1024]
 details=dict(small_zero_foreground_any_seed=sum(any(x['sampled_positive']==0 for x in r['runs']) for r in small),good_best_background_with_good_assigned=sum(r['best_iou']>=.5 and r['runs'][0]['best_query_status']=='background' and r['runs'][0]['assigned_iou']>=.5 for r in small))
 cache_path=ROOT/'reports/uiis-pixel-bll-data-20261031/images.jsonl';cache={r['id']:r for r in [json.loads(l) for l in cache_path.read_text().splitlines()]}
 details['valid_shapes']={str(iid):cache[iid]['valid_shape'] for iid in sel['image_ids']}
 details['existing_augmentation_cache_sha256']=hashlib.sha256(cache_path.read_bytes()).hexdigest()
 summary=dict(details=details,passed=True,images=len(sel['image_ids']),matching_runs=3*len(sel['image_ids']),groups={name:summarize([r for r in rows if lo<=r['area']<hi]) for name,lo,hi in [('all',0,float('inf')),('small',0,1024),('non_small',1024,float('inf'))]},source_hash_verified=True,all_exported_mask_ious_verified=True,download_hashes_verified=True)
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'instances.json').write_text(json.dumps(rows,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
