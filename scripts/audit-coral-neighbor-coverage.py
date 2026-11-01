"""Train-only coral coverage and cache inventory; no model or augmentation."""
import json,hashlib,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
import cv2
from pycocotools.coco import COCO
OUT=ROOT/'reports/coral-neighbor-coverage-20261101'
def main():
 assert not (OUT/'result.json').exists()
 ann=ROOT/'data/UIIS/annotations/train.json';gt=COCO(str(ann));assert len(gt.imgs)==3937
 cid=next(k for k,v in gt.cats.items() if v['name']=='reefs');cc=[a for a in gt.anns.values() if a['category_id']==cid and not a.get('iscrowd',0)]
 small=[a for a in cc if 0<a['area']<1024];groups=defaultdict(list)
 for a in cc:groups[a['image_id']].append(a)
 pairs=[]
 for iid in sorted(groups):
  aa=groups[iid]
  if len(aa)<2 or not any(0<a['area']<1024 for a in aa):continue
  masks={a['id']:gt.annToMask(a).astype(bool) for a in aa};done=set()
  for a in aa:
   if not 0<a['area']<1024:continue
   am=masks[a['id']];dist=cv2.distanceTransform((~am).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
   for b in aa:
    key=tuple(sorted([a['id'],b['id']]))
    if a['id']==b['id'] or key in done:continue
    done.add(key);bm=masks[b['id']]
    if not am.any() or not bm.any():continue
    overlap=int((am&bm).sum());gap=float(dist[bm].min())
    if gap<=32:pairs.append(dict(image_id=iid,gt_ids=list(key),distance=gap,overlap_pixels=overlap))
 clean=[p for p in pairs if p['overlap_pixels']==0];source_ids={p['image_id'] for p in clean}
 cachepath=ROOT/'reports/uiis-pixel-bll-data-20261031/images.jsonl';cache=[json.loads(l) for l in cachepath.read_text().splitlines()];assert len(cache)==3937
 aug=[]
 for row in cache:
  areas=[a for c,a in zip(row.get('labels',[]),row.get('gt_pixels',[])) if c==cid-1]
  if len(areas)>=2 and any(0<a<1024 for a in areas):aug.append(dict(image_id=row['id'],seed=row['seed'],coral_instances=len(areas),small=sum(0<a<1024 for a in areas),source_near_pair=row['id'] in source_ids,valid_shape=row['valid_shape']))
 inventory=[]
 for p in sorted((ROOT/'reports').glob('bll-real-gradient*/trace.npz')):
  with np.load(p,allow_pickle=False) as d:
   labels=d['matched_labels'];areas=d['original_mask_targets'].sum((1,2));n=int((labels==cid-1).sum());ns=int(((labels==cid-1)&(areas>0)&(areas<1024)).sum())
   inventory.append(dict(path=str(p.relative_to(ROOT)),coral=n,small_coral=ns))
 selection=sorted([r for r in aug if r['source_near_pair']],key=lambda r:r['image_id'])[:6]
 summary=dict(images=3937,coral_category_id=cid,coral_instances=len(cc),coral_images=len(groups),source_small_coral=len(small),source_small_images=len({a['image_id'] for a in small}),source_near_nonoverlap_pairs=len(clean),source_near_images=len(source_ids),source_near_small_gt=len({gid for p in clean for gid in p['gt_ids'] if 0<gt.anns[gid]['area']<1024}),overlapping_near_pairs=sum(p['overlap_pixels']>0 for p in pairs),cached_augmentation_candidate_images=len(aug),cached_candidate_intersect_source_near=sum(a['source_near_pair'] for a in aug),distance_sensitivity={t:dict(pairs=sum(p['distance']<=t for p in clean),images=len({p['image_id'] for p in clean if p['distance']<=t})) for t in [1,8,16,32]})
 result=dict(summary=summary,source_pairs=pairs,augmentation_metadata_candidates=aug,trace_inventory=inventory,selection=selection,input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ann,cachepath]},limitations='Original-annotation area and original pixel distances are not augmented sizes/distances. Cache metadata supplies areas/classes but no spatial masks. Selection is deterministic first6, not representative. No validation/test data used.')
 (OUT/'result.json').write_text(json.dumps(result,indent=2));(OUT/'selection.json').write_text(json.dumps(dict(images=selection,rule='First6 ascending train image_id with original nonoverlapping same-class near pair<=32 and cached actual augmentation containing >=2 reefs and >=1 small reef; fixed historical seeds; export masks only, no model',training_authorized=False),indent=2));print(json.dumps(dict(summary=summary,selection=selection,trace_inventory=inventory),indent=2))
if __name__=='__main__':main()
