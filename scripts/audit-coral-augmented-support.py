"""Audit the six preselected coral training augmentations, without a model."""
import json,hashlib,sys
from itertools import combinations
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
import cv2
from pycocotools import mask as mu
OUT=ROOT/'reports/coral-support-audit-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (OUT/'result.json').exists()
 exported=ROOT/'reports/coral-augmented-export-20261101';proof=json.loads((exported/'result.json').read_text());assert proof['passed'] and not proof['model_built'];assert proof['source_sha256']==sha(ROOT/'scripts/export-coral-augmented-masks.py')
 for row in proof['images']:assert sha(exported/f"image-{row['image_id']}.json")==row['sha256']
 old=ROOT/'reports/baseline-matching-audit-20261031';hashes={line.split(maxsplit=1)[1]:line.split()[0] for line in (old/'remote-hashes.txt').read_text().splitlines() if line.strip()}
 p12=old/'image-12.json';assert sha(p12)==hashes['reports/baseline-matching-audit-20261031/image-12.json']
 rows=[];images=[];source_hashes={}
 for iid in [12,309,326,337,381,414]:
  p=p12 if iid==12 else exported/f'image-{iid}.json';source_hashes[str(p.relative_to(ROOT))]=sha(p);d=json.loads(p.read_text());masks=mu.decode(d['gt_rles']).transpose(2,0,1).astype(bool);areas=masks.sum((1,2));assert areas.tolist()==d['gt_pixels'];ids=[i for i,c in enumerate(d['labels']) if c==1];scales={}
  for s in [32,16,8]:
   h,w=masks.shape[-2:];ss=[m.reshape(h//s,s,w//s,s).any((1,3)) for m in masks]
   for i in ids:assert np.array_equal(ss[i],cv2.resize(masks[i].astype('float32'),(w//s,h//s),interpolation=cv2.INTER_AREA)>1e-8)
   scales[s]=ss
  images.append(dict(image_id=iid,coral=len(ids),small=int(sum(areas[i]<1024 for i in ids)),zero_support={s:sum(not ss[i].any() for i in ids) for s,ss in scales.items()}))
  for i,j in combinations(ids,2):
   if min(areas[i],areas[j])>=1024:continue
   overlap=int((masks[i]&masks[j]).sum());gap=float(cv2.distanceTransform((~masks[i]).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[masks[j]].min())
   levels={}
   for s,ss in scales.items():
    a,b=ss[i],ss[j];ea=int((a&~b).sum());eb=int((b&~a).sum());feasible=min(ea,eb)>=2
    if feasible:
     aa=a.copy();bb=b.copy();aa.flat[np.flatnonzero(b&~a)[0]]=True;bb.flat[np.flatnonzero(a&~b)[0]]=True;assert not np.array_equal(aa,bb)
    levels[s]=dict(cells=[int(a.sum()),int(b.sum())],exclusive=[ea,eb],identical=bool(np.array_equal(a,b)),one_contained=min(ea,eb)==0,strict_partial_feasible=feasible)
   rows.append(dict(image_id=iid,indices=[i,j],areas=[int(areas[i]),int(areas[j])],overlap=overlap,distance=gap,levels=levels))
 summaries={}
 for limit in [1,8,16,32]:
  rr=[r for r in rows if not r['overlap'] and r['distance']<=limit];summaries[limit]=dict(pairs=len(rr),images=sorted({r['image_id'] for r in rr}),levels={s:{k:sum(r['levels'][s][k] for r in rr) for k in ['identical','one_contained','strict_partial_feasible']} for s in [32,16,8]})
 result=dict(passed=True,images=images,pairs=rows,summary=summaries,source_sha256=source_hashes,training_updates=0,scope='Six frozen real train augmentations. Areas and distances after actual augmentation; geometric feasibility only.')
 (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(images=images,summary=summaries),indent=2))
if __name__=='__main__':main()
