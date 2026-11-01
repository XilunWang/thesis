import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.local/analysis-libs'));import numpy as np;import cv2
p=Path('reports/bll-real-gradient-20261030-small-rank0/trace.npz');rows=[]
with np.load(p,allow_pickle=False) as d:
 masks=d['original_mask_targets'].astype(bool);labels=d['matched_labels'];gi=d['gi'];areas=masks.sum((1,2));done=set()
 for i in range(len(masks)):
  if labels[i]!=1 or not 0<areas[i]<1024:continue
  dist=cv2.distanceTransform((~masks[i]).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
  for j in range(len(masks)):
   if j==i or labels[j]!=1:continue
   key=tuple(sorted([i,j]))
   if key in done:continue
   done.add(key);gap=float(dist[masks[j]].min());overlap=int((masks[i]&masks[j]).sum())
   if overlap or gap>32:continue
   levels={}
   for s in [32,16,8]:
    a=masks[i].reshape(1024//s,s,1024//s,s).any((1,3));b=masks[j].reshape(1024//s,s,1024//s,s).any((1,3));ex=[int((a&~b).sum()),int((b&~a).sum())];levels[s]=dict(cells=[int(a.sum()),int(b.sum())],exclusive=ex,identical=bool(np.array_equal(a,b)),strict_partial_feasible=min(ex)>=2)
   rows.append(dict(trace_indices=[i,j],gt_indices=[int(gi[i]),int(gi[j])],areas=[int(areas[i]),int(areas[j])],distance=gap,levels=levels))
print(json.dumps(rows,indent=2));Path('reports/coral-neighbor-coverage-20261101/additional-cache-pairs.json').write_text(json.dumps(dict(source=str(p),pairs=rows,scope='Extra historical cache, not part of deterministic six-image selection'),indent=2))
