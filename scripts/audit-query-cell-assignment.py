"""GT-only maximum matching on occupied stride8 cells. Not proposal inference."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools import mask as mu
OUT=ROOT/'reports/query-cell-assignment-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def assign(candidates,slots=1):
 owner={}
 def visit(gid,seen):
  for cell in candidates[gid]:
   for slot in range(slots):
    node=(cell,slot)
    if node in seen:continue
    seen.add(node)
    if node not in owner or visit(owner[node],seen):owner[node]=gid;return True
  return False
 for gid in sorted(candidates):visit(gid,set())
 matches={gid:node for node,gid in owner.items()};assert len(matches)==len(owner)
 # Alternating reachability: if unmatched targets remain, validate a Hall deficit.
 left=set(candidates)-set(matches);right=set();queue=list(left)
 while queue:
  gid=queue.pop()
  for cell in candidates[gid]:
   for slot in range(slots):
    node=(cell,slot)
    if node in right:continue
    right.add(node);assert node in owner,'augmenting path remains'
    other=owner[node]
    if other not in left:left.add(other);queue.append(other)
 assert len(left)-len(right)==len(candidates)-len(matches)
 return matches,dict(gt_ids=sorted(left),available_slots=len(right),deficit=len(left)-len(right))
def main():
 OUT.mkdir(exist_ok=False)
 assert len(assign({1:[0,1],2:[0]})[0])==2
 assert assign({1:[0],2:[0]})[1]['deficit']==1
 assert len(assign({1:[0],2:[0]},2)[0])==2
 src=ROOT/'reports/identity-augmentation-check-20261101';proof=json.loads((src/'result.json').read_text());assert proof['passed']
 runs=[];hashes={str((src/'result.json').relative_to(ROOT)):sha(src/'result.json')}
 for run in proof['runs']:
  p=src/f"image-{run['image_id']}-seed-{run['seed']}.json";hashes[str(p.relative_to(ROOT))]=sha(p);d=json.loads(p.read_text())
  if d['pipeline_filtered']:runs.append(dict(image_id=d['image_id'],seed=d['seed'],filtered=True));continue
  masks=mu.decode(d['gt_rles']).transpose(2,0,1);h,w=masks.shape[-2:];assert h%8==w%8==0
  candidates={};initial={};small_masks={}
  for gid,area,mask in zip(d['annotation_ids'],d['original_areas'],masks):
   if not 0<area<=1024:continue
   yy,xx=np.nonzero(mask);j=int(np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2));cx,cy=int(xx[j])//8,int(yy[j])//8;initial[gid]=cy*(w//8)+cx
   counts=mask.reshape(h//8,8,w//8,8).sum((1,3));cells=np.flatnonzero(counts)
   candidates[gid]=sorted(map(int,cells),key=lambda c:((c//(w//8)-cy)**2+(c%(w//8)-cx)**2,-int(counts.flat[c]),c));small_masks[gid]=mask
  matched,witness=assign(candidates);two,witness2=assign(candidates,2)
  assert assign(dict(reversed(list(candidates.items()))))[0]==matched
  locations=[]
  for gid,(cell,slot) in matched.items():
   assert cell in candidates[gid] and slot==0
   cy,cx=divmod(cell,w//8);ys,xs=np.nonzero(small_masks[gid][cy*8:(cy+1)*8,cx*8:(cx+1)*8]);assert len(xs)
   j=int(np.argmin((xs-3.5)**2+(ys-3.5)**2));x,y=int(cx*8+xs[j]),int(cy*8+ys[j]);assert small_masks[gid][y,x]
   locations.append(dict(annotation_id=gid,cell=[cx,cy],point=[x,y],moved=cell!=initial[gid]))
  runs.append(dict(image_id=d['image_id'],seed=d['seed'],filtered=False,original_small=len(candidates),representative_unique_cells=len(set(initial.values())),one_slot_matched=len(matched),one_slot_unmatched=sorted(set(candidates)-set(matched)),one_slot_witness=witness,two_slot_matched=len(two),two_slot_witness=witness2,moved_instances=sum(r['moved'] for r in locations),assignments=locations,budget20_deficit_lower_bound=max(0,len(candidates)-20)))
 rr=[r for r in runs if not r['filtered']]
 summary={k:sum(r[k] for r in rr) for k in ['original_small','representative_unique_cells','one_slot_matched','two_slot_matched','moved_instances','budget20_deficit_lower_bound']}
 summary.update(one_slot_deficit=sum(r['one_slot_witness']['deficit'] for r in rr),two_slot_deficit=sum(r['two_slot_witness']['deficit'] for r in rr),max_small_per_augmentation=max(r['original_small'] for r in rr))
 result=dict(passed=True,summary=summary,runs=runs,source_sha256=hashes,training_updates=0,scope='GT-assisted geometry only, 6 fixed train images / 12 seeds; every occupied cell eligible, no learned proposal scores, no NMS, no AP; not residual-target budget.')
 (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
 for r in rr:
  if r['one_slot_witness']['deficit']:print(r['image_id'],r['seed'],r['one_slot_witness'])
if __name__=='__main__':main()
