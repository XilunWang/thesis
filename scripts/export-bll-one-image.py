"""Fixed real training image: supervision and paired loss gradients. No optimizer."""
import argparse,copy,fcntl,hashlib,json,random,runpy,sys,types
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def gradient_comparison(a,b):
 a=a.detach().double().reshape(-1);b=b.detach().double().reshape(-1)
 assert torch.isfinite(a).all() and torch.isfinite(b).all()
 na=float(a.norm());nb=float(b.norm());active=(a!=0)&(b!=0)
 return dict(original_norm=na,bll_norm=nb,bll_to_original_norm=nb/na if na else None,
  cosine=float(torch.dot(a,b)/(a.norm()*b.norm())) if na and nb else None,
  jointly_nonzero=int(active.sum()),opposite_sign=int(((a*b<0)&active).sum()))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--image-id',type=int,default=0);args=parser.parse_args()
 from mmengine.dataset import pseudo_collate
 from mmdet.registry import MODELS,DATASETS
 from mmdet.utils import register_all_modules
 import scripts.bll_mask2former_head as adapter
 from scripts.query_bll import build_bll_targets
 from scripts.bll_core import boundary
 suffix='' if args.image_id==0 else '-id'+str(args.image_id)
 out=R/('reports/bll-real-gradient-20261030'+suffix);out.mkdir(parents=True,exist_ok=False)
 with (R/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  register_all_modules();torch.set_num_threads(2)
  random.seed(20261021);np.random.seed(20261021);torch.manual_seed(20261021)
  cfg=runpy.run_path(str(R/'scripts/prepare-uiis-bll.py'))['build_config']()
  ck=R/'work_dirs/uiis_e1_bll_r50_36e/epoch_36.pth';digest=sha(ck)
  assert digest=='6d89f04575ec2a28b441c3f681e38627cf2273638de3ee6394bedb03279e8176'
  dataset=DATASETS.build(cfg.train_dataloader.dataset);dataset.full_init()
  index=next(i for i in range(len(dataset)) if dataset.get_data_info(i)['img_id']==args.image_id);info=dataset.get_data_info(index)
  item=dataset.prepare_data(index);assert item is not None and item['data_samples'].img_id==info['img_id']
  raw_input=item['inputs'].numpy().copy()
  model=MODELS.build(cfg.model).cuda();state=torch.load(ck,map_location='cpu')['state_dict'];model.load_state_dict(state,strict=True);model.train()
  arrays={};records=[];live={};head=model.panoptic_head;original_targets=head._get_targets_single;original_loss=adapter.query_boundary_loss
  def capture_targets(self,*args,**kwargs):
   result=original_targets(*args,**kwargs)
   records.append(tuple(x.detach().cpu().numpy().copy() for x in [result[0],result[2],result[4],result[6].pos_assigned_gt_inds]))
   return result
  head._get_targets_single=types.MethodType(capture_targets,head)
  def tap(previous,final,gts,matches,shapes,**kw):
   live['final']=final
   value,diag=original_loss(previous,final,gts,matches,shapes,**kw)
   assert len(gts)==1
   qi,gi=matches[0];target,valid=build_bll_targets(gts[0][gi],tuple(final.shape[-2:]),shapes[0]);pb=(previous[0,qi].sigmoid()>=.5)&valid
   region=(boundary(pb,kw['boundary_width'])|boundary(target,kw['boundary_width']))&valid
   for k,v in dict(previous=previous,final=final,gt=gts[0],qi=qi,gi=gi,target=target,valid=valid,region=region).items():arrays[k]=v.detach().cpu().numpy().copy()
   arrays['valid_shape']=np.array(shapes[0]);arrays['production_loss']=np.array(float(value.detach()))
   arrays['weight']=np.array(kw['loss_weight']);arrays['width']=np.array(kw['boundary_width'])
   return value,diag
  adapter.query_boundary_loss=tap
  try:
   data=model.data_preprocessor(pseudo_collate([item]),training=True)
   losses=model(**data,mode='loss')
   # Compare the actual weighted final-layer losses from this SAME forward/matching.
   # This is not the complete original objective (classification/auxiliary excluded).
   original=losses['loss_mask']+losses['loss_dice']
   parameter=head.mask_embed[-1].weight
   g_original=torch.autograd.grad(original,(live['final'],parameter),retain_graph=True)
   g_bll=torch.autograd.grad(losses['loss_bll'],(live['final'],parameter))
   comparisons=dict(final_logits=gradient_comparison(g_original[0],g_bll[0]),
                    mask_embed_final_weight=gradient_comparison(g_original[1],g_bll[1]))
   arrays['gradient_original']=g_original[0].detach().cpu().numpy()
   arrays['gradient_bll']=g_bll[0].detach().cpu().numpy()
  finally:adapter.query_boundary_loss=original_loss;head._get_targets_single=original_targets
  assert len(records)==head.num_transformer_decoder_layers+1
  labels,targets,qi,gi=records[-1]
  assert np.array_equal(qi,arrays['qi']) and np.array_equal(gi,arrays['gi'])
  assert np.array_equal(targets,arrays['gt'][gi])
  assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
  assert all(p.grad is None for p in model.parameters())
  assert sha(ck)==digest
  z=arrays['final'][0,arrays['qi']].astype(np.float64);target=arrays['target'];region=arrays['region']
  oracle=(np.logaddexp(0,z)-z*target)[region].sum()/max(int(region.sum()),1)*float(arrays['weight'])
  assert np.isclose(oracle,float(arrays['production_loss']),rtol=2e-6,atol=1e-7)
  gt=arrays['gt'][arrays['gi']];n,hp,wp=gt.shape;h,w=target.shape[-2:]
  coverage=gt.reshape(n,h,hp//h,w,wp//w).mean(axis=(2,4));arrays['coverage']=coverage
  vh,vw=arrays['valid_shape'];full=np.zeros((h,w),bool);full[:vh//(hp//h),:vw//(wp//w)]=True
  minority=region&full&(coverage>0)&(coverage<.5)
  ga=arrays['gradient_original'][0,arrays['qi']];gb=arrays['gradient_bll'][0,arrays['qi']]
  comparisons['minority_full_cells']=dict(count=int(minority.sum()),
    original_pushes_down_bll_up=int((minority&(ga>0)&(gb<0)).sum()),
    original_nonzero=int((minority&(ga!=0)).sum()))
  arrays.update(input_bgr=raw_input,matched_labels=labels[qi],original_mask_targets=targets)
  np.savez_compressed(out/'trace.npz',**arrays)
  meta=dict(passed=True,image_id=info['img_id'],filename=Path(info['img_path']).name,seed=20261021,checkpoint_sha256=digest,training=False,model_training_mode=True,optimizer_created=False,model_state_unchanged=True,actual_train_pipeline=True,comparisons=comparisons,independent_bce=float(oracle),scope='One fixed training augmentation at E1 epoch36; original final mask BCE+Dice only, not total training objective or causal proof',layer_assignments_observed=len(records),matched=len(qi),losses={k:float(v.detach()) for k,v in losses.items()},trace_sha256=sha(out/'trace.npz'),source_sha256={f:sha(R/f) for f in ['scripts/query_bll.py','scripts/bll_core.py','scripts/bll_mask2former_head.py','scripts/export-bll-one-image.py']})
  (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
