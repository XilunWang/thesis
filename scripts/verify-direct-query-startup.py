"""Independent-process startup/RNG and diagnostic optimizer restore checks; no steps."""
import argparse,copy,gc,json,sys,hashlib,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
from scripts.ap_screen_runtime import fingerprint,rng_state,restore_rng,ScreenFiniteOptimWrapper,sha
import scripts.valid_region_head
import scripts.direct_query_feature_head
OUT=ROOT/'reports/direct-query-startup-20261101'

def payload(batch):
 return dict(inputs=batch['inputs'],samples=[dict(meta=s.metainfo,labels=s.gt_instances.labels,boxes=s.gt_instances.bboxes,masks=s.gt_instances.masks.to_ndarray()) for s in batch['data_samples']])

def block_step(*args,**kwargs):raise RuntimeError('OPTIMIZER_STEP_FORBIDDEN_IN_PREFLIGHT')

def setup(mode):
 path=ROOT/('reports/valid-region-ap-screen-plan-20261031/control.py' if mode=='control' else 'reports/direct-query-screen-preparation-20261101/candidate.py')
 cfg=Config.fromfile(str(path));cfg.work_dir=str(OUT/(mode+'-runner'));cfg.optim_wrapper.type='ScreenFiniteOptimWrapper'
 if mode=='restore':cfg.load_from=None
 r=Runner.from_cfg(cfg);stages={'after_constructor':fingerprint(rng_state())}
 # Mirror installed Runner.train initialization without entering its train loop.
 r._train_loop=r.build_train_loop(r._train_loop)
 r.optim_wrapper=r.build_optim_wrapper(r.optim_wrapper);r.scale_lr(r.optim_wrapper,r.auto_scale_lr)
 if r.param_schedulers is not None:r.param_schedulers=r.build_param_scheduler(r.param_schedulers)
 r.call_hook('before_run');r._init_model_weights();stages['after_init_weights']=fingerprint(rng_state())
 r.load_or_resume();r.optim_wrapper.initialize_count_status(r.model,r.iter,r.train_loop.max_iters)
 r._maybe_compile('train_step');r.optim_wrapper.optimizer.step=block_step
 stages['after_load']=fingerprint(rng_state())
 return r,stages

def snapshot(r):return dict(model=fingerprint(r.model.state_dict()),optimizer=fingerprint(r.optim_wrapper.state_dict()),rng=fingerprint(rng_state()),epoch=r.epoch,iteration=r.iter)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['control','raw','candidate','restore']);a=parser.parse_args();mode=a.mode
 OUT.mkdir(exist_ok=True);assert not (OUT/(mode+'.json')).exists()
 register_all_modules();torch.set_num_threads(2)
 r,stages=setup(mode);opt=r.optim_wrapper.optimizer
 if mode=='restore':
  p=OUT/'diagnostic-optimizer-fixture.pth';saved=torch.load(p,map_location='cpu');assert saved['diagnostic_only'] and saved['actual_parameter_updates']==0
  r.resume(str(p));restore_rng(saved['diagnostic_rng']);expected=json.loads((OUT/'fixture-expected.json').read_text())
  assert snapshot(r)==expected
  result=dict(passed=True,optimizer_states=len(r.optim_wrapper.optimizer.state),fresh_process=True,diagnostic_only=True,actual_parameter_updates=0,fixture_sha256=sha(p))
  (OUT/'restore.json').write_text(json.dumps(result,indent=2));print('RESTORE_PASSED',flush=True);return
 initial=torch.load(ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth',map_location='cpu')['state_dict']
 state=r.model.state_dict();base={k:v for k,v in state.items() if not k.startswith('panoptic_head.direct_c')}
 assert fingerprint(base)==fingerprint(initial);assert not opt.state and r.iter==r.epoch==0
 extras={k:v for k,v in state.items() if k not in initial}
 if mode=='control':assert not extras
 else:assert set(extras)=={'panoptic_head.direct_c3.weight','panoptic_head.direct_c4.weight'} and all(torch.count_nonzero(v)==0 for v in extras.values())
 before=fingerprint(state);named={id(p):name for name,p in r.model.named_parameters()};groups=[]
 for g in opt.param_groups:
  for p in g['params']:
   groups.append(dict(name=named[id(p)],lr=g['lr'],weight_decay=g.get('weight_decay',0),requires_grad=p.requires_grad,numel=p.numel()))
 for g in groups:
  if g['name'].startswith('panoptic_head.direct_c'):assert g['lr']==1e-5 and g['weight_decay']==.05
 sampler=r.train_dataloader.sampler
 sampler_order=list(iter(sampler));worker=dict(num_workers=r.train_dataloader.num_workers,worker_init=repr(r.train_dataloader.worker_init_fn),sampler_type=type(sampler).__name__,sampler_seed=getattr(sampler,'seed',None))
 r.call_hook('before_train');r.call_hook('before_train_epoch');r.model.train()
 iterator=iter(r.train_dataloader);batches=[]
 with torch.no_grad():
  for i in range(3):
   batch=next(iterator);raw_hash=fingerprint(payload(batch));ids=[int(s.img_id) for s in batch['data_samples']]
   data=r.model.data_preprocessor(batch,training=True)
   before_rng=fingerprint(rng_state());loss=r.model.loss(data['inputs'],data['data_samples']);assert all(torch.isfinite(v).all() for v in loss.values())
   batches.append(dict(index=i,image_ids=ids,augmented_payload_sha256=raw_hash,preprocessed_inputs_sha256=fingerprint(data['inputs']),before_loss_rng=before_rng,after_loss_rng=fingerprint(rng_state()),loss_sha256=fingerprint(loss),loss_values={k:float(v) for k,v in loss.items()}))
   print(mode,'BATCH',i,ids,flush=True);del data,loss,batch
 if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
 del iterator
 assert fingerprint(r.model.state_dict())==before and not opt.state and r.iter==r.epoch==0
 assert all(p.grad is None for p in r.model.parameters())
 result=dict(passed=True,mode=mode,model_base_exact=True,optimizer_initially_empty=True,parameter_updates=0,gradients_created=False,stages=stages,groups=groups,worker=worker,sampler_order_sha256=fingerprint(sampler_order),batches=batches,head_sha256=sha(ROOT/'scripts/direct_query_feature_head.py'))
 if mode=='candidate':
  # Explicit artificial moments, never used to compute a parameter update.
  for group in opt.param_groups:
   for p in group['params']:
    if p.requires_grad:opt.state[p]=dict(step=torch.tensor(1.),exp_avg=torch.full_like(p,.01),exp_avg_sq=torch.full_like(p,.02))
  expected=snapshot(r);(OUT/'fixture-expected.json').write_text(json.dumps(expected,indent=2))
  r.save_checkpoint(str(OUT),'diagnostic-optimizer-fixture.pth',by_epoch=False,meta=dict(epoch=0,iter=0))
  p=OUT/'diagnostic-optimizer-fixture.pth';saved=torch.load(p,map_location='cpu');saved.update(diagnostic_only=True,actual_parameter_updates=0,diagnostic_rng=rng_state());torch.save(saved,p)
  assert snapshot(r)==expected;result['fixture_states']=len(opt.state)
 (OUT/(mode+'.json')).write_text(json.dumps(result,indent=2));print(mode,'STARTUP_CHECKED_NO_UPDATES',flush=True)
if __name__=='__main__':
 with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
