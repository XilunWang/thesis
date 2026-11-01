"""Geometry-only audit of frozen augmented GT masks; no torch/model/training."""
import hashlib,json,sys
from itertools import combinations
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
import cv2
from pycocotools import mask as mu
SRC=ROOT/'reports/baseline-matching-audit-20261031';OUT=ROOT/'reports/neighbor-support-audit-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def support(mask,s):
 h,w=mask.shape;assert h%s==0 and w%s==0
 # For aligned integer reductions, area interpolation > 1e-8 is equivalent
 # to any occupied pixel in the block, for these binary masks and strides.
 return mask.reshape(h//s,s,w//s,s).any(axis=(1,3))
def main():
 assert not (OUT/'result.json').exists()
 saved={}
 for line in (SRC/'remote-hashes.txt').read_text().splitlines():
  if line.strip():h,p=line.split(maxsplit=1);saved[p]=h
 ids=json.loads((SRC/'selection.json').read_text())['image_ids'];images=[];pairs=[];digests={};total=small=0
 for iid in ids:
  p=SRC/f'image-{iid}.json';digest=sha(p);assert digest==saved[str(p.relative_to(ROOT)).replace('\\','/')];digests[str(p.relative_to(ROOT))]=digest
  rec=json.loads(p.read_text());masks=mu.decode(rec['gt_rles']).transpose(2,0,1).astype(bool);areas=masks.sum((1,2));assert areas.tolist()==rec['gt_pixels'];assert list(masks.shape[1:])==rec['shape']
  total+=len(masks);small+=int(sum(areas<1024));scaled={s:[support(m,s) for m in masks] for s in [32,16,8]}
  # Independent OpenCV area interpolation verifies the integer block rule.
  for s,ss in scaled.items():
   for m,b in zip(masks,ss):assert np.array_equal(b,cv2.resize(m.astype('float32'),(b.shape[1],b.shape[0]),interpolation=cv2.INTER_AREA)>1e-8)
  images.append(dict(image_id=iid,instances=len(masks),small=int(sum(areas<1024)),zero_support={s:sum(not b.any() for b in ss) for s,ss in scaled.items()}))
  for i,j in combinations(range(len(masks)),2):
   if rec['labels'][i]!=rec['labels'][j]:continue
   overlap=int(np.count_nonzero(masks[i]&masks[j]));distance=0. if overlap else float(cv2.distanceTransform((~masks[i]).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[masks[j]].min())
   row=dict(image_id=iid,indices=[i,j],label=rec['labels'][i],areas=[int(areas[i]),int(areas[j])],has_small=bool(min(areas[i],areas[j])<1024),pixel_overlap=overlap,min_pixel_center_distance=distance,levels={})
   for s,ss in scaled.items():
    a,b=ss[i],ss[j];ex_a=int((a&~b).sum());ex_b=int((b&~a).sum());common=int((a&b).sum())
    # Bidirectional strict partial-neighbor witness: keep own full support,
    # add first row-major exclusive neighbor cell, leave >=1 unadded.
    feasible=ex_a>=2 and ex_b>=2
    witness=None
    if feasible:
     ai=a.copy();bj=b.copy();ai.flat[np.flatnonzero(b&~a)[0]]=True;bj.flat[np.flatnonzero(a&~b)[0]]=True
     assert not np.array_equal(ai,bj) and np.all(ai[a]) and np.all(bj[b]);witness=dict(added_cells_each=1,distinct=True)
    row['levels'][s]=dict(cells=[int(a.sum()),int(b.sum())],common=common,exclusive=[ex_a,ex_b],identical=bool(np.array_equal(a,b)),one_contained=bool(ex_a==0 or ex_b==0),strict_partial_feasible=feasible,witness=witness)
   pairs.append(row)
 groups={}
 for limit in [1,8,16,32]:
  rr=[p for p in pairs if p['min_pixel_center_distance']<=limit and p['has_small'] and p['pixel_overlap']==0]
  groups[limit]=dict(pairs=len(rr),images=sorted({p['image_id'] for p in rr}),levels={s:dict(identical=sum(p['levels'][s]['identical'] for p in rr),one_contained=sum(p['levels'][s]['one_contained'] for p in rr),strict_partial_feasible=sum(p['levels'][s]['strict_partial_feasible'] for p in rr)) for s in [32,16,8]},all_levels_feasible=sum(all(p['levels'][s]['strict_partial_feasible'] for s in [32,16,8]) for p in rr))
 assert total==70 and small==32
 result=dict(passed=True,images=images,total=total,small=small,same_class_pairs=len(pairs),groups=groups,pairs=pairs,input_sha256=digests,scope='Frozen 12 train images, one historical augmentation each. Geometry-only necessary checks, not model separability or AP.',rules='Same class; at least one area<1024; nonoverlapping GT; minimum foreground pixel-center distance<=32px primary, 1/8/16 sensitivity. Fixed integer area support at strides32/16/8. Strict partial witness adds one exclusive cell and leaves at least one exclusive cell on each side.')
 (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(total=total,small=small,same_class_pairs=len(pairs),groups=groups),indent=2))
if __name__=='__main__':main()
