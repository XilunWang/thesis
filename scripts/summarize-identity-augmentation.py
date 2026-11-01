"""Verify exported identity masks locally and summarize original-small positions."""
import importlib.util,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
from pycocotools import mask as mu
spec=importlib.util.spec_from_file_location('locations',ROOT/'scripts/audit-small-query-locations.py');locations=importlib.util.module_from_spec(spec);spec.loader.exec_module(locations)
def main():
 out=ROOT/'reports/identity-augmentation-check-20261101';proof=json.loads((out/'result.json').read_text());assert proof['passed'] and proof['training_updates']==0
 raw={a['id']:a for a in json.loads((ROOT/'data/UIIS/annotations/train.json').read_text())['annotations']};points=[];runs=[];removed=Counter()
 for run in proof['runs']:
  d=json.loads((out/f"image-{run['image_id']}-seed-{run['seed']}.json").read_text());first=d['stages'][0]
  source={i for i,ig in zip(first['ids'],first['ignore']) if not ig and 0<raw[i]['area']<=1024};ids=d.get('annotation_ids',[])
  record=dict(image_id=d['image_id'],seed=d['seed'],filtered=d['pipeline_filtered'],source_small=len(source),retained_small=0,removed_small=len(source-set(ids)),small_now_over1024=d.get('original_small_now_large',0),large_now_under1024=d.get('original_large_now_small',0))
  prev=set()
  for stage in d['stages']:
   curr={i for i,ig in zip(stage.get('ids',[]),stage.get('ignore',[])) if not ig and 0<raw[i]['area']<=1024}
   if stage['stage']!='LoadAnnotations':removed[stage['stage']]+=len(prev-curr)
   prev=curr
  if ids:
   masks=mu.decode(d['gt_rles']).transpose(2,0,1);assert masks.sum((1,2)).tolist()==d['augmented_areas'] and len(set(ids))==len(ids)
   for gid,area,mask in zip(ids,d['original_areas'],masks):
    assert area==raw[gid]['area'] and raw[gid]['image_id']==d['image_id']
    if 0<area<=1024:
     record['retained_small']+=1;pt=locations.point(mask);assert pt is not None
     points.append(dict(id=gid,image_id=f"{d['image_id']}:{d['seed']}",category_id=raw[gid]['category_id'],point=pt))
  runs.append(record)
 result=dict(passed=True,images=len({r['image_id'] for r in runs}),augmentations=len(runs),successful_items=sum(not x['filtered'] for x in runs),runs=runs,original_small_occurrences=sum(x['source_small'] for x in runs),retained_small_occurrences=sum(x['retained_small'] for x in runs),removed_small_occurrences=sum(x['removed_small'] for x in runs),small_now_over1024=sum(x['small_now_over1024'] for x in runs),large_now_under1024=sum(x['large_now_under1024'] for x in runs),removed_by_stage=removed,geometry=locations.analyze(points))
 assert sum(removed.values())==result['removed_small_occurrences']
 (out/'local-verification.json').write_text(json.dumps(result,indent=2));print('IDENTITY_LOCAL_VERIFICATION_PASSED',len(runs))
if __name__=='__main__':main()
