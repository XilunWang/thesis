"""Frozen single-cell paired corruption checks; arrays only, no model."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
import cv2
from pycocotools import mask as mu
OUT=ROOT/'reports/conditional-neighbor-rule-20261101'
def construct(a,b,occupied,valid):
 original=dict(candidate=a.copy(),control=a.copy(),active=False,reason='insufficient_exclusive_support')
 if min(int((a&~b).sum()),int((b&~a).sum()))<2:return original
 donor=b&~a&valid;other=~occupied&valid
 distance=np.rint(cv2.distanceTransform((~a).astype('uint8'),cv2.DIST_C,3)).astype(int)
 for radius in sorted(set(distance[donor].tolist())):
  ni=np.flatnonzero(donor&(distance==radius));ci=np.flatnonzero(other&(distance==radius))
  if not len(ni) or not len(ci):continue
  n,c=int(ni[0]),int(ci[0]);candidate=a.copy();control=a.copy();candidate.flat[n]=True;control.flat[c]=True
  assert np.all(candidate[a]) and np.all(control[a])
  assert int((candidate&~a).sum())==int((control&~a).sum())==1
  assert not np.array_equal(candidate,b) and not np.array_equal(control,b)
  return dict(candidate=candidate,control=control,active=True,reason='paired',neighbor_cell=list(map(int,np.unravel_index(n,a.shape))),control_cell=list(map(int,np.unravel_index(c,a.shape))),radius=int(radius))
 original['reason']='no_equal_distance_valid_control';return original

def main():
 assert not (OUT/'result.json').exists()
 audit=json.loads((ROOT/'reports/coral-support-audit-20261101/result.json').read_text());pairs=[p for p in audit['pairs'] if not p['overlap'] and p['distance']<=32];assert len(pairs)==3
 refs={r['id']:r for r in map(json.loads,(ROOT/'reports/uiis-pixel-bll-data-20261031/images.jsonl').read_text().splitlines())};rows=[]
 for pair in pairs:
  iid=pair['image_id'];p=ROOT/f'reports/coral-augmented-export-20261101/image-{iid}.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==audit['source_sha256'][str(p.relative_to(ROOT))]
  d=json.loads(p.read_text());masks=mu.decode(d['gt_rles']).transpose(2,0,1).astype(bool);i,j=pair['indices'];h,w=masks.shape[-2:];vh,vw=refs[iid]['valid_shape']
  for s in [32,16,8]:
   supports=[m.reshape(h//s,s,w//s,s).any((1,3)) for m in masks];occupied=np.any(supports,axis=0)
   valid=np.zeros((h//s,w//s),dtype=bool);valid[:vh//s,:vw//s]=True
   ab=construct(supports[i],supports[j],occupied,valid);ba=construct(supports[j],supports[i],occupied,valid)
   collision=any(np.array_equal(ab[k],ba[k]) for k in ['candidate','control'])
   if collision:
    ab=dict(candidate=supports[i],control=supports[i],active=False,reason='pair_collision');ba=dict(candidate=supports[j],control=supports[j],active=False,reason='pair_collision')
   for target,neighbor,x in [(i,j,ab),(j,i,ba)]:
    a=supports[target]
    if x['active']:
     for k in ['candidate','control']:assert np.all(valid[x[k]&~a]) and int((x[k]&~a).sum())==1
     assert occupied[tuple(x['neighbor_cell'])] and not occupied[tuple(x['control_cell'])]
    else:assert np.array_equal(x['candidate'],a) and np.array_equal(x['control'],a)
    rows.append(dict(image_id=iid,target_index=target,neighbor_index=neighbor,stride=s,**{k:v for k,v in x.items() if k not in ['candidate','control']},support_cells=int(a.sum()),output_sha256={k:hashlib.sha256(x[k].tobytes()).hexdigest() for k in ['candidate','control']}))
 summary={s:dict(directional_cases=sum(r['stride']==s for r in rows),active=sum(r['stride']==s and r['active'] for r in rows),fallback=sum(r['stride']==s and not r['active'] for r in rows)) for s in [32,16,8]}
 result=dict(passed=True,model_built=False,training_updates=0,summary=summary,rows=rows,selected_images=[12,309,326,337,381,414],pair_images=sorted({r['image_id'] for r in rows}),unique_targets=sorted({(r['image_id'],r['target_index']) for r in rows}),scope='Pairwise offline geometry only; shared target appears with different neighbors; no full-batch query construction.')
 (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
