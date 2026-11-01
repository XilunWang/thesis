"""Same-input per-GT E0 detection versus saved position coverage."""
import collections,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT)]
import numpy as np
from pycocotools import mask as mu
OUT=ROOT/'reports/fixed12-e0-complementarity-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def detection_match(ious,labels,scores,gtlabels,gtids,scoremin=.5,ioumin=.5):
 matched={}
 for pred in sorted(range(len(scores)),key=lambda i:(-scores[i],i)):
  if scores[pred]<scoremin:continue
  candidates=[j for j in range(len(gtids)) if j not in matched and gtlabels[j]==labels[pred] and ious[pred,j]>=ioumin]
  if candidates:matched[min(candidates,key=lambda j:(-ious[pred,j],gtids[j]))]=pred
 return matched

def position_match(selected,support):
 chosen=set(selected);owner={}
 def visit(g,seen):
  for c in sorted(support[g]&chosen):
   if c in seen:continue
   seen.add(c)
   if c not in owner or visit(owner[c],seen):owner[c]=g;return True
  return False
 for g in sorted(support):visit(g,set())
 return {g:c for c,g in owner.items()}

def main():
 # Class, score, IoU and one-to-one/tie semantics at exact thresholds.
 matched=detection_match(np.array([[.5,.5],[.9,.1],[.1,.9]]),[0,1,0],[.5,.9,.49],[0,0],[20,10]);assert matched=={1:0}
 export_dir=OUT/'attempt-2';proof=load(export_dir/'result.json');planpath=ROOT/'docs/fixed12-e0-complementarity-plan-20261101.json';plan=load(planpath);assert proof['passed'] and sha(planpath)==proof['plan_sha256']
 views={(v['image_id'],v['seed']):v for v in load(ROOT/'reports/position-learning-plan-grid4-20261101/views.json')['runs']}
 folders={'geometric':'position-grid4-execution-20261101','adaptive':'label-only-pilot-execution-20261101','coverage':'coverage-priority-pilot-execution-20261101'};evals={}
 for name,folder in folders.items():
  path=ROOT/f'reports/{folder}/eval-300.json';assert sha(path)==plan['source_sha256'][path.relative_to(ROOT).as_posix()]
  evals[name]={(r['image_id'],r['augmentation_seed']):r for r in load(path)['records'] if r['split']=='holdout' and r['source']=='head' and r['k']==100}
 records=[];perview=[]
 for exported,c in zip(proof['runs'],plan['cases']):
  assert exported['case_id']==c['case_id'];path=export_dir/exported['file'];assert sha(path)==exported['sha256'];d=load(path);key=(d['image_id'],d['seed']);v=views[key]
  assert d['input_sha256']==v['input_sha256'] and d['masks_sha256']==v['masks_sha256']
  cache=load(ROOT/f"reports/position-learning-preflight-20261101/holdout-{key[0]}-{key[1]}.json")
  assert d['annotation_ids']==cache['annotation_ids'] and d['labels']==cache['labels'] and d['original_areas']==cache['original_areas']
  masks=mu.decode(cache['gt_rles']).transpose(2,0,1);vh,vw=d['valid_shape'];assert np.array_equal(mu.decode(d['gt_rles']).transpose(2,0,1),masks[:,:vh,:vw])
  iou=mu.iou(d['query_rles'],d['gt_rles'],[0]*len(d['labels']));top=np.asarray(d['selected_query_class_flat_indices']);qs=top//7;labels=top%7;scores=np.asarray(d['selected_scores'])
  probs=np.asarray(d['class_probabilities']);quality=np.asarray(d['mask_quality']);assert np.allclose(scores,probs[qs,labels]*quality[qs],rtol=1e-6,atol=1e-8)
  matched=detection_match(iou[qs],labels,scores,d['labels'],d['annotation_ids'],plan['matching']['score_min'],plan['matching']['mask_iou_min'])
  h,w=cache['padded_shape'];support={int(g):set(map(int,np.flatnonzero(m.reshape(h//4,4,w//4,4).any(axis=(1,3))))) for g,a,m in zip(d['annotation_ids'],d['original_areas'],masks) if 0<a<=1024}
  position={name:position_match(rows[key]['selected_cells'],support) for name,rows in evals.items()}
  for name in folders:assert len(position[name])==evals[name][key]['occupied']
  this=[]
  for j,(gid,area) in enumerate(zip(d['annotation_ids'],d['original_areas'])):
   if not 0<area<=1024:continue
   good=iou[:,j]>=.5;eligible=np.flatnonzero((labels==d['labels'][j])&good[qs]);high=eligible[scores[eligible]>=.5]
   status='detected' if j in matched else 'no_qualified_geometry' if not good.any() else 'correct_class_not_selected' if not len(eligible) else 'low_score' if not len(high) else 'one_to_one_competition'
   q=int(iou[:,j].argmax());r=dict(case_id=c['case_id'],image_id=key[0],seed=key[1],gt_id=gid,category_id=d['labels'][j]+1,original_area=area,baseline_detected=j in matched,baseline_status=status,matched_query=int(qs[matched[j]]) if j in matched else None,matched_score=float(scores[matched[j]]) if j in matched else None,best_query=q,best_iou=float(iou[q,j]),best_correct_selected_good_score=float(scores[eligible].max()) if len(eligible) else None,position_hit={name:gid in pos for name,pos in position.items()},position_cell={name:pos.get(gid) for name,pos in position.items()})
   records.append(r);this.append(r)
  perview.append(dict(case_id=c['case_id'],image_id=key[0],small=len(this),baseline_detected=sum(r['baseline_detected'] for r in this),baseline_missed=sum(not r['baseline_detected'] for r in this),position_covers_missed={name:sum(not r['baseline_detected'] and r['position_hit'][name] for r in this) for name in folders}))
 assert len(records)==58 and len(perview)==12
 groups={}
 for group,pred in [('all',lambda r:True),('detected',lambda r:r['baseline_detected']),('missed',lambda r:not r['baseline_detected'])]:
  subset=[r for r in records if pred(r)];groups[group]=dict(n=len(subset),hits={name:sum(r['position_hit'][name] for r in subset) for name in folders})
 reasons={status:dict(n=sum(r['baseline_status']==status for r in records),hits={name:sum(r['baseline_status']==status and r['position_hit'][name] for r in records) for name in folders}) for status in sorted({r['baseline_status'] for r in records})}
 transitions={}
 for name in ['adaptive','coverage']:
  transitions[name]=dict(new_missed_vs_geometric=[r['gt_id'] for r in records if not r['baseline_detected'] and r['position_hit'][name] and not r['position_hit']['geometric']],lost_missed_vs_geometric=[r['gt_id'] for r in records if not r['baseline_detected'] and not r['position_hit'][name] and r['position_hit']['geometric']])
 result=dict(passed=True,training_updates=0,groups=groups,reasons=reasons,transitions=transitions,per_view=perview,records=records,plan_sha256=sha(planpath),export_result_sha256=sha(export_dir/'result.json'),script_sha256=sha(Path(__file__)),interpretation='Location coverage only; no modified segmentation output, no rescued TP or AP claim. Purposive sample and fixed threshold, not population estimate.')
 (OUT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['groups','reasons','transitions','per_view']},indent=2))
if __name__=='__main__':main()
