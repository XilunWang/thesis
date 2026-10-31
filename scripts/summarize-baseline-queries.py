"""Summarize fixed query audit; GT-assisted maxima are not attainable recall."""
import json,hashlib,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu
def main():
 out=ROOT/'reports/baseline-query-audit-20261031'
 raw=json.loads((out/'result.json').read_text());assert raw['passed']
 gt=COCO(str(ROOT/'data/UIIS/annotations/val.json'))
 for line in (out/'remote-hashes.txt').read_text(encoding='utf-8-sig').splitlines():
  parts=line.split()
  if len(parts)==2 and len(parts[0])==64:
   assert hashlib.sha256((ROOT/parts[1]).read_bytes()).hexdigest()==parts[0]
 images={c['image_id']:json.loads((out/f"image-{c['image_id']}.json").read_text()) for c in raw['checks']}
 old={r['gt_id']:r for r in json.loads((ROOT/'reports/baseline-priorities-20261031/instances.json').read_text())}
 rows=[]
 for r in raw['records']:
  prev=old[r['gt_id']];qs=r['queries'];assert len(qs)==100
  exported=images[r['image_id']]['queries']
  overlap=mu.iou([q['segmentation'] for q in exported],[gt.annToRLE(gt.anns[r['gt_id']])],[0])[:,0]
  assert np.allclose(overlap,[q['iou'] for q in qs],rtol=0,atol=1e-12)
  bestsame=max((q['iou'] for q in qs if q['correct_class_selected']),default=0)
  bestany=max((q['iou'] for q in qs if q['any_class_selected']),default=0)
  assert abs(bestsame-prev['best_same_iou'])<1e-10 and abs(bestany-prev['best_any_iou'])<1e-10
  rr={k:v for k,v in r.items() if k!='queries'};rr['matched']=prev['matched'];rr['saved_best_any_iou']=bestany
  rr['best_raw_iou']=max(q['iou'] for q in qs)
  for t in [.5,.75]:
   good=[q for q in qs if q['iou']>=t];selected=[q for q in good if q['correct_class_selected']]
   high=[q for q in selected if q['correct_class_final_score']>=.5]
   status='no_geometry' if not good else 'correct_class_not_topk' if not selected else 'selected_but_low_score' if not high else 'eligible_above_threshold'
   rr[str(t)]=dict(status=status,geometric_queries=len(good),selected_correct_queries=len(selected),high_score_queries=len(high),
    best_geometry_query=max(qs,key=lambda q:q['iou']),best_correct_score_good_query=max(good,key=lambda q:q['correct_class_final_score']) if good else None)
  rows.append(rr)
 def summarize(rs):
  return dict(gt=len(rs),previous_matched=sum(r['matched'] for r in rs),previous_missed=sum(not r['matched'] for r in rs),
   all_status={str(t):dict(Counter(r[str(t)]['status'] for r in rs)) for t in [.5,.75]},
   missed_status={str(t):dict(Counter(r[str(t)]['status'] for r in rs if not r['matched'])) for t in [.5,.75]},
   previously_no_saved_geometry=sum(r['saved_best_any_iou']<.5 for r in rs),
   recovered_geometry_before_topk=sum(r['saved_best_any_iou']<.5<=r['best_raw_iou'] for r in rs))
 result=dict(passed=True,images=len(raw['checks']),reproduced_predictions=100*len(raw['checks']),max_score_error=max(c['score_max_abs_error'] for c in raw['checks']),
  unique_selected_queries_range=[min(c['unique_selected_queries'] for c in raw['checks']),max(c['unique_selected_queries'] for c in raw['checks'])],
  groups={name:summarize([r for r in rows if lo<=r['area']<hi]) for name,lo,hi in [('tiny',0,256),('other_small',256,1024),('all_small',0,1024),('medium',1024,9216),('large',9216,float('inf'))]})
 low=[r for r in rows if r['area']<1024 and r['0.5']['status']=='selected_but_low_score']
 candidates=[r['0.5']['best_correct_score_good_query'] for r in low]
 result['low_score_small']=dict(count=len(low),best_candidate_class_below05=sum(q['correct_class_probability']<.5 for q in candidates),best_candidate_top_background=sum(q['top_class']==7 for q in candidates),mask_quality_range=[min(q['mask_quality'] for q in candidates),max(q['mask_quality'] for q in candidates)])
 result['download_hashes_and_all_exported_mask_ious_verified']=True
 (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'instances.json').write_text(json.dumps(rows))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()

