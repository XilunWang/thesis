"""Export one fixed real training forward for independent local BLL inspection. No optimizer."""
import copy,fcntl,hashlib,json,random,runpy,sys,types
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 from mmengine.dataset import pseudo_collate
 from mmdet.registry import MODELS,DATASETS
 from mmdet.utils import register_all_modules
 import scripts.bll_mask2former_head as adapter
 from scripts.query_bll import build_bll_targets
 from scripts.bll_core import boundary
 out=R/'reports/bll-one-image-trace';out.mkdir(parents=True,exist_ok=False)
 with (R/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  register_all_modules();torch.set_num_threads(2)
  random.seed(20261021);np.random.seed(20261021);torch.manual_seed(20261021)
  cfg=runpy.run_path(str(R/'scripts/prepare-uiis-bll.py'))['build_config']()
  ck=R/'work_dirs/uiis_e1_bll_r50_36e/epoch_36.pth';digest=sha(ck)
  assert digest=='6d89f04575ec2a28b441c3f681e38627cf2273638de3ee6394bedb03279e8176'
  dataset=DATASETS.build(cfg.train_dataloader.dataset);dataset.full_init()
  index=min(range(len(dataset)),key=lambda i:dataset.get_data_info(i)['img_id']);info=dataset.get_data_info(index)
  item=dataset.prepare_data(index);assert item is not None and item['data_samples'].img_id==info['img_id']
  raw_input=item['inputs'].numpy().copy()
  model=MODELS.build(cfg.model).cuda();state=torch.load(ck,map_location='cpu')['state_dict'];model.load_state_dict(state,strict=True);model.train()
  arrays={};records=[];head=model.panoptic_head;original_targets=head._get_targets_single;original_loss=adapter.query_boundary_loss
  def capture_targets(self,*args,**kwargs):
   result=original_targets(*args,**kwargs)
   records.append(tuple(x.detach().cpu().numpy().copy() for x in [result[0],result[2],result[4],result[6].pos_assigned_gt_inds]))
   return result
  head._get_targets_single=types.MethodType(capture_targets,head)
  def tap(previous,final,gts,matches,shapes,**kw):
   value,diag=original_loss(previous,final,gts,matches,shapes,**kw)
   assert len(gts)==1
   qi,gi=matches[0];target,valid=build_bll_targets(gts[0][gi],tuple(final.shape[-2:]),shapes[0]);pb=(previous[0,qi].sigmoid()>=.5)&valid
   region=(boundary(pb,kw['boundary_width'])|boundary(target,kw['boundary_width']))&valid
   for k,v in dict(previous=previous,final=final,gt=gts[0],qi=qi,gi=gi,target=target,valid=valid,region=region).items():arrays[k]=v.detach().cpu().numpy().copy()
   arrays['valid_shape']=np.array(shapes[0]);arrays['production_loss']=np.array(float(value))
   arrays['weight']=np.array(kw['loss_weight']);arrays['width']=np.array(kw['boundary_width'])
   return value,diag
  adapter.query_boundary_loss=tap
  try:
   with torch.no_grad():
    data=model.data_preprocessor(pseudo_collate([item]),training=True)
    losses=model(**data,mode='loss')
  finally:adapter.query_boundary_loss=original_loss;head._get_targets_single=original_targets
  assert len(records)==head.num_transformer_decoder_layers+1
  labels,targets,qi,gi=records[-1]
  assert np.array_equal(qi,arrays['qi']) and np.array_equal(gi,arrays['gi'])
  assert np.array_equal(targets,arrays['gt'][gi])
  assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
  assert all(p.grad is None for p in model.parameters())
  assert sha(ck)==digest
  arrays.update(input_bgr=raw_input,matched_labels=labels[qi],original_mask_targets=targets)
  np.savez_compressed(out/'trace.npz',**arrays)
  meta=dict(passed=True,image_id=info['img_id'],filename=Path(info['img_path']).name,seed=20261021,checkpoint_sha256=digest,training=False,optimizer_created=False,model_state_unchanged=True,actual_train_pipeline=True,layer_assignments_observed=len(records),matched=len(qi),losses={k:float(v) for k,v in losses.items()},trace_sha256=sha(out/'trace.npz'),source_sha256={f:sha(R/f) for f in ['scripts/query_bll.py','scripts/bll_core.py','scripts/bll_mask2former_head.py','scripts/export-bll-one-image.py']})
  (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
