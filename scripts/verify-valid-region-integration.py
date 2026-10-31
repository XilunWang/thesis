"""Integration acceptance: same-forward parity, every-stage sampling, finite backward; no step."""
import copy,fcntl,hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmcv.ops import point_sample
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS,MODELS
from mmdet.utils import register_all_modules
from mmdet.models.dense_heads import Mask2FormerHead
from scripts.valid_region_head import ValidRegionMask2FormerHead

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def main():
 out=ROOT/'reports/valid-region-integration-20261031';out.mkdir(exist_ok=True);assert not (out/'result.json').exists()
 with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);register_all_modules();torch.set_num_threads(2)
  cfg=Config.fromfile(str(ROOT/'work_dirs/uiis_baseline_r50_36e/config-resume.py'));ck=ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth';ann=ROOT/'data/UIIS/annotations/train.json';hashes={str(p.relative_to(ROOT)):sha(p) for p in [ck,ann]}
  assert sha(ck)=='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
  dc=copy.deepcopy(cfg.train_dataloader.dataset);ds=DATASETS.build(dc);ds.full_init();idx={ds.get_data_info(i)['img_id']:i for i in range(len(ds))}
  cfg.model.panoptic_head.type='ValidRegionMask2FormerHead';model=MODELS.build(cfg.model).cuda();head=model.panoptic_head;assert head.match_valid_region is False
  state=torch.load(ck,map_location='cpu')['state_dict'];model.load_state_dict(state,strict=True)
  for name in ['forward','predict','loss_by_feat','_loss_by_feat_single']:
   assert getattr(ValidRegionMask2FormerHead,name) is getattr(Mask2FormerHead,name)
  batches=[('mixed_sizes',[345,620],[]),('no_padding',[8,12],[]),('mixed_empty',[345,620],[1]),('all_empty',[345,620],[0,1])];results=[]
  for name,ids,empty in batches:
   model.train();model.zero_grad(set_to_none=True);assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
   items=[]
   for iid in ids:seed(20261012+iid);items.append(ds.prepare_data(idx[iid]))
   data=model.data_preprocessor(pseudo_collate(items),training=True);samples=data['data_samples']
   for i in empty:samples[i].gt_instances=samples[i].gt_instances[:0]
   targets=head.preprocess_gt([s.gt_instances for s in samples],[None]*2);metas=[s.metainfo for s in samples]
   allcls,allmasks=head(model.extract_feat(data['inputs']),samples);assert len(allcls)==10
   original_assign=head.assigner.assign;original_target=head._get_targets_single
   def run(mode):
    calls=[];head.match_valid_region=(mode=='on');implementation=Mask2FormerHead._get_targets_single if mode=='parent' else ValidRegionMask2FormerHead._get_targets_single
    def observed(cls,masks,gt,meta):
     rng=torch.cuda.get_rng_state();coords=torch.rand((1,head.num_points,2),device=cls.device);torch.cuda.set_rng_state(rng)
     h,w=gt.masks.shape[-2:];vh,vw=meta['img_shape'][:2]
     if mode=='on' and len(gt.labels):coords=coords*coords.new_tensor([vw/w,vh/h])
     with torch.no_grad():
      expected_pred=point_sample(masks.unsqueeze(1),coords.repeat(len(cls),1,1)).squeeze(1)
      expected_gt=point_sample(gt.masks.unsqueeze(1).float(),coords.repeat(len(gt.labels),1,1)).squeeze(1)
     holder={}
     def checked(pred_instances,gt_instances,img_meta=None,**kwargs):
      assert torch.equal(pred_instances.masks,expected_pred) and torch.equal(gt_instances.masks,expected_gt)
      assigned=original_assign(pred_instances=pred_instances,gt_instances=gt_instances,img_meta=img_meta,**kwargs);holder['assignment']=assigned.gt_inds
      return assigned
     head.assigner.assign=checked
     try:ret=implementation(head,cls,masks,gt,meta)
     finally:head.assigner.assign=original_assign
     a=holder['assignment'];pos=ret[4];assert len(pos)==len(gt.labels)
     expected_labels=gt.labels.new_full((head.num_queries,),head.num_classes);expected_labels[pos]=gt.labels[a[pos]-1]
     assert torch.equal(ret[0],expected_labels) and torch.equal(ret[2],gt.masks[a[pos]-1])
     calls.append(dict(valid_shape=[vh,vw],gt_count=len(gt.labels),positive_count=len(pos)))
     return ret
    head._get_targets_single=observed;seed(20261031)
    try:losses=head.loss_by_feat(allcls,allmasks,targets,metas)
    finally:head._get_targets_single=original_target;head.assigner.assign=original_assign
    assert len(calls)==20 and all(torch.isfinite(v).all() for v in losses.values())
    return losses,torch.cuda.get_rng_state(),calls
   with torch.no_grad():
    parent,prng,pcalls=run('parent');off,orng,ocalls=run('off')
   assert parent.keys()==off.keys() and all(torch.equal(parent[k],off[k]) for k in parent) and torch.equal(prng,orng)
   on,onrng,calls=run('on');assert on.keys()==parent.keys();assert torch.equal(prng,onrng)
   if name=='no_padding':assert all(torch.equal(parent[k],on[k]) for k in parent)
   if name=='all_empty':assert all(float(v)==0 for k,v in on.items() if 'mask' in k or 'dice' in k)
   sum(on.values()).backward();grad=[(k,p.grad) for k,p in model.named_parameters() if p.grad is not None];assert grad and all(torch.isfinite(g).all() for k,g in grad)
   assert any(g.abs().max()>0 for k,g in grad)
   assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items())
   row=dict(batch=name,image_ids=ids,empty_indices=empty,loss_terms=len(on),parent_off_exact=True,rng_exact=True,all_stage_sampling_exact=True,calls=calls,losses={k:float(v) for k,v in on.items()},finite_gradient_tensors=len(grad),nonzero_gradient_tensors=sum(bool(g.abs().max()>0) for k,g in grad))
   results.append(row);print(name,'ALL_STAGE_LOSS_BACKWARD_PASSED',flush=True)
   del allcls,allmasks,on,off,parent,grad,data,targets,samples,items
  model.zero_grad(set_to_none=True);model.eval();seed(20261012+345);item=ds.prepare_data(idx[345]);batch=pseudo_collate([item])
  with torch.no_grad():
   head.match_valid_region=False;off=model.test_step(copy.deepcopy(batch))[0].pred_instances
   head.match_valid_region=True;on=model.test_step(copy.deepcopy(batch))[0].pred_instances
   assert torch.equal(off.masks,on.masks) and torch.equal(off.scores,on.scores) and torch.equal(off.labels,on.labels)
  # Invalid valid shape must fail instead of sampling outside the canvas.
  from mmengine.structures import InstanceData
  fake=InstanceData(labels=torch.tensor([0],device='cuda'),masks=torch.ones((1,4,4),device='cuda'))
  try:head._get_targets_single(torch.zeros((100,8),device='cuda'),torch.zeros((100,2,2),device='cuda'),fake,dict(img_shape=(5,4)))
  except ValueError:invalid_rejected=True
  else:raise AssertionError('invalid shape accepted')
  assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items()) and all(sha(ROOT/p)==v for p,v in hashes.items())
  result=dict(passed=True,training_updates=0,optimizer_created=False,model_state_unchanged=True,default_off=True,inference_exact=True,invalid_shape_rejected=invalid_rejected,batches=results,input_hashes=hashes,source_sha256=sha(Path(__file__)),head_sha256=sha(ROOT/'scripts/valid_region_head.py'))
  (out/'result.json').write_text(json.dumps(result,indent=2));print('VALID_REGION_INTEGRATION_PASSED',flush=True)
if __name__=='__main__':main()
