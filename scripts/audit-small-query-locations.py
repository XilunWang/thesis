"""Training-only geometry and budget audit. No feature model, inference or training."""
import hashlib,json,sys
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu
OUT=ROOT/'reports/small-query-location-audit-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def point(mask):
 y,x=np.nonzero(mask)
 if not len(x):return None
 # Select a real foreground pixel closest to foreground centroid; deterministic row-major tie.
 j=int(np.argmin((x-x.mean())**2+(y-y.mean())**2));return [int(x[j]),int(y[j])]
def analyze(rows):
 groups=defaultdict(list)
 for r in rows:
  if r['point'] is not None:groups[r['image_id']].append(r)
 result={}
 for stride in [8,16]:
  collisions=[];neighbor_pairs=0;unique_total=0;cell_class_total=0
  for iid,rs in groups.items():
   cells=defaultdict(list)
   for r in rs:cells[r['point'][0]//stride,r['point'][1]//stride].append(r)
   unique_total+=len(cells);cell_class_total+=len({(xy,r['category_id']) for xy,rr in cells.items() for r in rr})
   for xy,rr in cells.items():
    if len(rr)>1:collisions.append(dict(image_id=iid,cell=xy,ids=[r['id'] for r in rr],categories=[r['category_id'] for r in rr]))
   for a,b in combinations(rs,2):
    dx=abs(a['point'][0]//stride-b['point'][0]//stride);dy=abs(a['point'][1]//stride-b['point'][1]//stride)
    if a['category_id']==b['category_id'] and max(dx,dy)==1:neighbor_pairs+=1
  result[str(stride)]=dict(collision_cells=len(collisions),collision_images=len({c['image_id'] for c in collisions}),instances_in_collision_cells=sum(len(c['ids']) for c in collisions),one_query_per_cell_excess=sum(len(c['ids'])-1 for c in collisions),unique_spatial_cells=unique_total,unique_cell_class_pairs=cell_class_total,adjacent_same_class_pairs=neighbor_pairs,collision_cases=collisions)
 return result
def budgets(counts):
 vals=list(counts.values())
 return {'maximum':max(vals,default=0),'positive_images':sum(v>0 for v in vals),'budgets':{str(k):{'images_exceeding':sum(v>k for v in vals),'excess_instances':sum(max(0,v-k) for v in vals)} for k in [8,16,20,32]}}
def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists();ann=ROOT/'data/UIIS/annotations/train.json'
 assert sha(ann)=='9cc586e559f54533a938e34a89017d47784c3646bc43ab2cd0d88e140ef5b49d'
 gt=COCO(str(ann));assert len(gt.imgs)==3937
 rows=[];counts={iid:0 for iid in gt.imgs};classes=Counter();hashes={str(ann.relative_to(ROOT)):sha(ann)}
 for a in gt.anns.values():
  if a.get('iscrowd',0) or not 0<a['area']<=1024:continue
  mask=gt.annToMask(a);pt=point(mask);counts[a['image_id']]+=1;classes[gt.cats[a['category_id']]['name']]+=1
  rows.append(dict(id=a['id'],image_id=a['image_id'],category_id=a['category_id'],area=a['area'],point=pt))
 cachepath=ROOT/'reports/uiis-pixel-bll-data-20261031/images.jsonl';hashes[str(cachepath.relative_to(ROOT))]=sha(cachepath)
 cache=[json.loads(l) for l in cachepath.read_text().splitlines()];assert len(cache)==3937
 filtered=[r['id'] for r in cache if 'gt_pixels' not in r]
 assert all(r.get('pipeline_filtered') and r.get('processing_ok') for r in cache if 'gt_pixels' not in r)
 augcounts={r['id']:sum(0<x<=1024 for x in r['gt_pixels']) for r in cache if 'gt_pixels' in r}
 src=ROOT/'reports/baseline-matching-audit-20261031';selection=json.loads((src/'selection.json').read_text())['image_ids']
 paths=[src/f'image-{iid}.json' for iid in selection]+[ROOT/f'reports/coral-augmented-export-20261101/image-{iid}.json' for iid in [309,326,337,381,414]]
 augmented=[];missing_source_ids=[]
 for path in paths:
  hashes[str(path.relative_to(ROOT))]=sha(path);d=json.loads(path.read_text());masks=mu.decode(d['gt_rles']).transpose(2,0,1)
  assert masks.sum((1,2)).tolist()==d['gt_pixels']
  assert d['image_id'] in gt.imgs
  if 'gt_ids' not in d:missing_source_ids.append(d['image_id'])
  for j,(mask,area,label) in enumerate(zip(masks,d['gt_pixels'],d['labels'])):
   if 0<area<=1024:augmented.append(dict(id=f"{d['image_id']}:{j}",image_id=d['image_id'],category_id=label+1,area=area,point=point(mask)))
 result=dict(passed=True,training_updates=0,original={'images':3937,'small_gt':len(rows),'small_gt_by_class':classes,'empty_raster_masks':sum(r['point'] is None for r in rows),'budgets':budgets(counts),'geometry':analyze(rows)},cached_augmentation={'images':3937,'images_with_area_records':len(augcounts),'pipeline_filtered_ids':filtered,'small_definition':'augmented raster area<=1024; NOT original-small identity','small_instances':sum(augcounts.values()),'budgets':budgets(augcounts)},augmented_mask_subset={'images':len(paths),'small_instances':len(augmented),'missing_original_gt_ids_images':missing_source_ids,'geometry':analyze(augmented)},source_sha256=hashes,limitations=['Raw coordinates are not augmented feature grids.','Subset is historical selected17 images, not representative.','3937 augmentation cache lacks spatial masks and stable original GT IDs.','One chosen foreground point may collide; another point or multiple slots could resolve some collisions.','Budget counts are all-small GT, not residual misses; no learned proposal recall or AP measured.'])
 (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'points.json').write_text(json.dumps(dict(original=rows,augmented=augmented),ensure_ascii=False),encoding='utf-8')
 for section in ['original','cached_augmentation','augmented_mask_subset']:
  x=result[section].copy()
  if 'geometry' in x:x['geometry']={s:{k:v for k,v in r.items() if k!='collision_cases'} for s,r in x['geometry'].items()}
  print(section,json.dumps(x,ensure_ascii=False))
if __name__=='__main__':main()
