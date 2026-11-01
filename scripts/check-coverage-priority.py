"""Frozen offline geometry plus paired actual-score checks. No model."""
import json,hashlib,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT/'.local/bll-libs'),str(ROOT)]
import numpy as np
from pycocotools import mask as mu
from scripts.coverage_priority import representatives
from scripts.instance_representative import make_layout
OUT=ROOT/'reports/coverage-priority-check-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def measure(reps,masks,ids,areas,valid):
 gh,gw=valid.shape;rows=[]
 for gid,a,m in zip(ids,areas,masks):
  if not 0<a<=1024:continue
  gid=int(gid);c=reps[gid];counts=m.reshape(gh,4,gw,4).sum((1,3)).ravel();assert valid.ravel()[c] and counts[c]>0
  rows.append(dict(gt_id=gid,cell=int(c),foreground_pixels=int(counts[c]),center_in_gt=bool(m[c//gw*4+2,c%gw*4+2]),augmented_area=int(m.sum()),tiny=bool(m.sum()<=16),max_coverage=int(counts.max()),below_available_max=bool(counts[c]<counts.max()),has_full_cell=bool((counts==16).any())))
 assert len(rows)==len(reps)==len(set(reps.values()))
 return rows
def summary(rows):
 return dict(instances=len(rows),partial=sum(r['foreground_pixels']<16 for r in rows),center_outside=sum(not r['center_in_gt'] for r in rows),tiny=sum(r['tiny'] for r in rows),no_full_cell=sum(not r['has_full_cell'] for r in rows),below_available_max=sum(r['below_available_max'] for r in rows),foreground_pixels_sum=sum(r['foreground_pixels'] for r in rows))
def main():
 planpath=ROOT/'docs/coverage-priority-check-plan-20261101.json';plan=json.loads(planpath.read_text());vp=ROOT/'reports/position-learning-plan-grid4-20261101/views.json';assert sha(vp)==plan['views_sha256'];views=json.loads(vp.read_text())
 neighbors={(x['image_id'],x['seed']):x['pairs'] for x in views['frozen_neighbor_observations']}
 scores=ROOT/'reports/label-only-audit-scores-20261101';proof=json.loads((scores/'result.json').read_text());score_runs={(r['image_id'],r['seed']):r for r in proof['runs']}
 records=[];actual=[];cache_hash={};pairs=0
 for v in views['runs']:
  key=(v['image_id'],v['seed']);p=ROOT/'reports/position-learning-preflight-20261101'/f"{v['split']}-{key[0]}-{key[1]}.json";d=json.loads(p.read_text());cache_hash[p.name]=sha(p)
  assert d['input_sha256']==v['input_sha256'] and d['masks_sha256']==v['masks_sha256']
  h,w=d['padded_shape'];masks=mu.decode(d['gt_rles']).transpose(2,0,1) if d['gt_rles'] else np.zeros((0,h,w),np.uint8)
  valid=np.zeros((h//4,w//4),bool);vh,vw=d['valid_shape'][:2];valid[:(vh+3)//4,:(vw+3)//4]=True
  for sign in [-1,1]:
   z=sign*np.arange(valid.size,dtype=float).reshape(valid.shape)
   reps=representatives(z,masks,d['annotation_ids'],d['original_areas'],valid)
   rows=measure(reps,masks,d['annotation_ids'],d['original_areas'],valid)
   assert set(map(int,v['assignments']))==set(reps)
   for a,b in neighbors.get(key,[]):assert reps[a]!=reps[b];pairs+=1
   records.append(dict(image_id=key[0],seed=key[1],split=v['split'],synthetic_sign=sign,summary=summary(rows),rows=rows))
  if key in score_runs:
   r=score_runs[key];p=scores/r['file'];assert sha(p)==r['sha256'];arr=np.load(p);assert np.array_equal(valid,arr['valid'])
   for stage in ['initial','new100','new300','old300']:
    z=arr[stage];old=make_layout(z,masks,d['annotation_ids'],d['original_areas'],valid)['representatives'];new=representatives(z,masks,d['annotation_ids'],d['original_areas'],valid)
    before=measure(old,masks,d['annotation_ids'],d['original_areas'],valid);after=measure(new,masks,d['annotation_ids'],d['original_areas'],valid)
    actual.append(dict(case_id=r['case_id'],stage=stage,before=before,after=after,changed=sum(old[g]!=new[g] for g in old)))
  if len(records)%128==0:print('VIEWS',len(records)//2,flush=True)
 # Minimal collision test: two IDs can share the best cell, but both must survive.
 m=np.zeros((2,4,8),np.uint8);m[0,:,:4]=1;m[0,0,4]=1;m[1,0,:4]=1
 reps=representatives(np.zeros((1,2)),m,[1,2],[17,4],np.ones((1,2),bool));assert reps=={1:1,2:0}
 # Truly impossible support must raise, never silently delete an instance.
 impossible=np.zeros((2,4,4),np.uint8);impossible[:,0,0]=1
 try:representatives(np.zeros((1,1)),impossible,[1,2],[1,1],np.ones((1,1),bool))
 except ValueError:pass
 else:raise AssertionError('Impossible collision was silently accepted')
 paired={}
 for stage in ['initial','new100','new300','old300']:
  subset=[r for r in actual if r['stage']==stage];before=summary([x for r in subset for x in r['before']]);after=summary([x for r in subset for x in r['after']]);paired[stage]=dict(before=before,after=after,changed=sum(r['changed'] for r in subset))
  assert after['instances']==before['instances'] and after['partial']<=before['partial'] and after['center_outside']<=before['center_outside']
 total={str(sign):summary([x for r in records if r['synthetic_sign']==sign for x in r['rows']]) for sign in [-1,1]}
 assert len(records)==768 and len(actual)==48
 OUT.mkdir(exist_ok=True);result=dict(passed=True,training_updates=0,model_forwards=0,views=384,geometry_checks=768,actual_score_checks=48,pair_checks=pairs,geometry_totals=total,paired_actual=paired,records=records,actual=actual,collision_fallback_passed=True,impossible_collision_rejected=True,plan_sha256=sha(planpath),cache_sha256=cache_hash,code_sha256={f:sha(ROOT/'scripts'/f) for f in ['coverage_priority.py','check-coverage-priority.py']})
 (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['passed','views','pair_checks','geometry_totals','paired_actual']},indent=2))
if __name__=='__main__':main()
