"""Frozen E0 only, exact saved augmentation. No parameter updates."""
import fcntl,hashlib,importlib.util,inspect,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pycocotools import mask as mu
OUT=ROOT/'reports/fixed12-e0-complementarity-20261101/attempt-2'
PLAN=ROOT/'docs/fixed12-e0-complementarity-plan-20261101.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encode(masks):
 result=[]
 for m in masks:
  r=mu.encode(np.asfortranarray(m.astype(np.uint8)));r['counts']=r['counts'].decode('ascii');result.append(r)
 return result
def main(result):
 plan=json.loads(PLAN.read_text());assert len(plan['cases'])==12
 spec=importlib.util.spec_from_file_location('base',ROOT/'scripts/run-position-grid4-pilot.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 pilot=base.Pilot();pilot.head.requires_grad_(False).eval()
 def forbidden(*a,**kw):raise AssertionError('No updates allowed')
 pilot.optimizer.step=forbidden
 ck=ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth';assert sha(ck)==plan['checkpoint_sha256']
 fusion=pilot.model.panoptic_fusion_head;assert fusion.num_classes==fusion.num_things_classes==7
 (OUT/'postprocess-source.txt').write_text(inspect.getsource(type(fusion).predict)+'\n'+inspect.getsource(type(fusion).instance_postprocess))
 result.update(plan_sha256=sha(PLAN),checkpoint_sha256=sha(ck),runs=[])
 with torch.no_grad():
  for c in plan['cases']:
   row=next(r for r in pilot.views if r['split']=='holdout' and (r['image_id'],r['seed'])==(c['image_id'],c['seed']))
   data,_,valid=pilot.data(row);sample=data['data_samples'][0];vh,vw=row['valid_shape'][:2]
   assert list(sample.metainfo['img_shape'][:2])==[vh,vw]
   cls,padded=pilot.model.panoptic_head.predict(pilot.model.extract_feat(data['inputs']),data['data_samples'])
   actual=fusion.predict(cls,padded,data['data_samples'],rescale=False)[0]['ins_results']
   logits=padded[0,:,:vh,:vw];probs=cls[0].softmax(-1);assert probs.shape==(100,8)
   binary=logits>0;quality=(logits.sigmoid()*binary).flatten(1).sum(1)/(binary.flatten(1).sum(1)+1e-6)
   cp,top=probs[:,:7].flatten().topk(100,sorted=False);queries=top//7;labels=top%7
   old_scores=cp*quality[queries]
   selected_logits=logits[queries];is_thing=labels<fusion.num_things_classes
   selected_logits=selected_logits[is_thing];selected_binary=(selected_logits>0).float()
   selected_quality=(selected_logits.sigmoid()*selected_binary).flatten(1).sum(1)/(selected_binary.flatten(1).sum(1)+1e-6)
   scores=cp[is_thing]*selected_quality
   old_error=float((old_scores-actual.scores).abs().max())
   assert torch.equal(actual.labels,labels) and torch.equal(actual.masks,binary[queries]) and torch.equal(actual.scores,scores)
   gt=sample.gt_instances;gr=encode(gt.masks.masks[:,:vh,:vw]);pr=encode(binary.cpu().numpy())
   payload=dict(case_id=c['case_id'],image_id=c['image_id'],seed=c['seed'],input_sha256=row['input_sha256'],masks_sha256=row['masks_sha256'],valid_shape=[vh,vw],padded_shape=row['padded_shape'],annotation_ids=gt.annotation_ids.tolist(),original_areas=gt.original_areas.tolist(),labels=gt.labels.tolist(),gt_rles=gr,query_rles=pr,class_probabilities=probs.cpu().tolist(),mask_quality=quality.cpu().tolist(),selected_query_class_flat_indices=top.cpu().tolist(),selected_scores=scores.cpu().tolist(),postprocess_exact=True,all_query_formula_max_abs_difference=old_error)
   file=f"case-{c['case_id']:02d}.json";(OUT/file).write_text(json.dumps(payload));result['runs'].append(dict(case_id=c['case_id'],file=file,sha256=sha(OUT/file),input_exact=True,masks_exact=True,postprocess_exact=True,all_query_formula_max_abs_difference=old_error))
   print('EXPORTED',c['case_id'],c['image_id'],flush=True)
 pilot.protected();assert all(p.grad is None for p in pilot.head.parameters());assert sha(ck)==plan['checkpoint_sha256']
 result.update(passed=True,original_model_unchanged=True,position_head_forward_calls=0,script_sha256=sha(Path(__file__)))
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'result.json').exists();result=dict(passed=False,training_updates=0)
 try:
  with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(result)
 except BaseException:result['error']=traceback.format_exc();raise
 finally:(OUT/'result.json').write_text(json.dumps(result,indent=2))
