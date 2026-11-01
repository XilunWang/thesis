"""Read-only cache identity audit; never join image IDs across official splits."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT)]
import numpy as np
from pycocotools import mask as mu
OUT=ROOT/'reports/position-baseline-alignment-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def main():
 train=read('data/UIIS/annotations/train.json');val=read('data/UIIS/annotations/val.json');tr={r['id']:r for r in train['images']};va={r['id']:r for r in val['images']}
 viewpath=ROOT/'reports/position-learning-plan-grid4-20261101/views.json';views=json.loads(viewpath.read_text())['runs'];ids={r['image_id'] for r in views};holdout={r['image_id'] for r in views if r['split']=='holdout'}
 manifest=read('reports/position-learning-plan-grid4-20261101/manifest.json');assert sha(ROOT/'data/UIIS/annotations/train.json')==manifest['source_sha256'] and sha(viewpath)==manifest['views_sha256']
 query=read('reports/baseline-query-audit-20261031/result.json');matching=read('reports/baseline-matching-audit-20261031/result.json')
 for p,h in query['input_hashes'].items():
  if (ROOT/p).is_file():assert sha(ROOT/p)==h,p
 for p,h in matching['input_hashes'].items():
  if (ROOT/p).is_file():assert sha(ROOT/p)==h,p
 qids=set(read('reports/baseline-query-audit-20261031/selection.json')['image_ids']);selection=read('reports/baseline-matching-audit-20261031/selection.json');mids=set(selection['image_ids'])
 false_id_matches=[dict(image_id=i,train_file=tr[i]['file_name'],val_file=va[i]['file_name']) for i in sorted(ids&qids)]
 photo_hashes={};missing={};available={}
 for split,images in [('position_train',[tr[i] for i in sorted(ids)]),('official_val',list(va.values()))]:
  hashes={};absent=[]
  for item in images:
   p=ROOT/'data/UIIS'/('train' if split=='position_train' else 'val')/item['file_name']
   if not p.is_file():absent.append(item['id']);continue
   hashes.setdefault(sha(p),[]).append(item['id'])
  photo_hashes[split]=hashes;missing[split]=absent;available[split]=len(images)-len(absent)
 exact_photo_matches=[dict(train_ids=photo_hashes['position_train'][h],val_ids=photo_hashes['official_val'][h],sha256=h) for h in sorted(photo_hashes['position_train'].keys()&photo_hashes['official_val'].keys())]
 evals={k:read(f'reports/{folder}/eval-300.json') for k,folder in [('geometric','position-grid4-execution-20261101'),('adaptive','label-only-pilot-execution-20261101'),('coverage','coverage-priority-pilot-execution-20261101')]}
 overlaps=[];evidence={}
 for iid in sorted(ids&mids):
  seed=selection['augmentation_seed_base']+iid;v=next((r for r in views if r['image_id']==iid and r['seed']==seed),None)
  if v is None:overlaps.append(dict(image_id=iid,same_seed=False));continue
  basepath=ROOT/f'reports/baseline-matching-audit-20261031/image-{iid}.json';cachepath=ROOT/f"reports/position-learning-preflight-20261101/{v['split']}-{iid}-{seed}.json";d=json.loads(basepath.read_text());c=json.loads(cachepath.read_text())
  bm=mu.decode(d['gt_rles']).transpose(2,0,1);cm=mu.decode(c['gt_rles']).transpose(2,0,1);exact=np.array_equal(bm,cm);assert exact and d['labels']==c['labels'];assert c['input_sha256']==v['input_sha256'] and c['masks_sha256']==v['masks_sha256']
  iou=mu.iou(d['pred_rles'],d['gt_rles'],[0]*len(d['labels']));assert np.allclose(iou,d['ious'],rtol=0,atol=1e-12)
  h,w=c['padded_shape'];support={int(g):set(map(int,np.flatnonzero(m.reshape(h//4,4,w//4,4).any(axis=(1,3))))) for g,a,m in zip(c['annotation_ids'],c['original_areas'],cm) if 0<a<=1024}
  targets=[]
  for j,(gid,area) in enumerate(zip(c['annotation_ids'],c['original_areas'])):
   if not 0<area<=1024:continue
   q=int(iou[:,j].argmax());label=c['labels'][j];prob=float(d['class_probabilities'][q][label]);quality=float(d['mask_quality'][q]);hits={}
   for name,ev in evals.items():
    row=next(r for r in ev['records'] if r['image_id']==iid and r['augmentation_seed']==seed and r['source']=='head' and r['k']==100)
    selected=set(row['selected_cells']);owner={}
    def visit(g,seen):
     for cell in sorted(support[g]&selected):
      if cell in seen:continue
      seen.add(cell)
      if cell not in owner or visit(owner[cell],seen):owner[cell]=g;return True
     return False
    for g in sorted(support):visit(g,set())
    assert len(owner)==row['occupied'];hits[name]=gid in owner.values()
   targets.append(dict(gt_id=gid,original_area=area,augmented_pixels=int(cm[j].sum()),label_zero_based=label,best_query=q,best_iou=float(iou[q,j]),correct_class_probability=prob,mask_quality=quality,probability_times_quality=prob*quality,position_hits=hits))
  overlaps.append(dict(image_id=iid,file=tr[iid]['file_name'],split=v['split'],seed=seed,same_seed=True,GT_masks_exact=True,baseline_input_tensor_hash_available=False,baseline_mode=matching['model_forward_mode'],position_mode='eval, frozen E0',formal_detection_pair=False,targets=targets))
  evidence[str(basepath.relative_to(ROOT))]=sha(basepath);evidence[str(cachepath.relative_to(ROOT))]=sha(cachepath)
 coralids={309,326,337,381,414};coral=[]
 for iid in sorted(ids&coralids):
  p=ROOT/f'reports/coral-augmented-export-20261101/image-{iid}.json';d=json.loads(p.read_text());v=next(r for r in views if r['image_id']==iid and r['seed']==d['seed']);c=read(f"reports/position-learning-preflight-20261101/{v['split']}-{iid}-{v['seed']}.json")
  coral.append(dict(image_id=iid,seed=v['seed'],GT_masks_exact=bool(np.array_equal(mu.decode(d['gt_rles']),mu.decode(c['gt_rles']))),has_baseline_predictions='pred_rles' in d));evidence[str(p.relative_to(ROOT))]=sha(p)
 instances=read('reports/baseline-query-audit-20261031/instances.json');miss=[r for r in instances if r['area']<=1024 and not r['matched']]
 dup=[dict(image_id=i,file=t['file_name'],in_position_sample=i in ids) for i,t in tr.items() if t['file_name']=='XL_266.jpg']
 result=dict(audit_completed=True,training_updates=0,model_forwards=0,position_images=len(ids),position_holdout_images=len(holdout),position_views=len(views),baseline_query_images=len(qids),baseline_small_misses=len(miss),false_integer_ID_overlaps=false_id_matches,photo_hash_available=available,photo_hash_missing=missing,exact_photo_matches=exact_photo_matches,known_official_duplicate=dup,train_prediction_image_overlap=sorted(ids&mids),train_prediction_holdout_image_overlap=sorted(holdout&mids),train_geometric_alignment=overlaps,GT_only_overlap=coral,strict_missed_target_position_pairs=0,complementarity_conclusion='Not estimable from available paired evidence; zero evaluable missed-target pairs is not zero rescued targets.',source_sha256={**evidence,'views':sha(viewpath),'baseline_query_result':sha(ROOT/'reports/baseline-query-audit-20261031/result.json'),'baseline_matching_result':sha(ROOT/'reports/baseline-matching-audit-20261031/result.json')},script_sha256=sha(Path(__file__)))
 OUT.mkdir(exist_ok=True);(OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','photo_hash_missing','script_sha256']},indent=2))
if __name__=='__main__':main()
