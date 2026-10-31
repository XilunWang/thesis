"""Shared AP-screen contracts and finite/budget guards; no execution on import."""
import hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.optim import OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'reports/valid-region-ap-screen-plan-20261031'
PREFLIGHT=ROOT/'reports/valid-region-ap-screen-entry-20261031'
ARMS=('control','candidate')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def contract(arm):
 assert arm in ARMS
 return dict(arm=arm,match_valid_region=arm=='candidate',config_sha256=sha(PLAN/(arm+'.py')),head_sha256=sha(ROOT/'scripts/valid_region_head.py'),epochs=3,iterations=5907)
def check_config(cfg,arm):
 m=json.loads((PLAN/'preparation.json').read_text())
 assert sha(PLAN/(arm+'.py'))==m['configs'][arm] and sha(ROOT/'scripts/valid_region_head.py')==m['head_sha256']
 assert cfg.model.panoptic_head.match_valid_region is (arm=='candidate')
 assert cfg.train_cfg.max_epochs==3 and cfg.param_scheduler==[] and not cfg.resume
 assert cfg.optim_wrapper.optimizer.lr==1e-5 and cfg.train_dataloader.batch_size==2
 assert cfg.val_cfg is None and cfg.randomness.seed==20261012
 assert sha(Path(cfg.load_from))==m['checkpoint_sha256']
 assert sha(ROOT/'data/UIIS/annotations/train.json')==m['train_sha256'] and sha(ROOT/'data/UIIS/annotations/val.json')==m['test_sha256']
def validate_checkpoint(saved,expected,final=False,allow_fixture=False):
 assert saved['screen_contract']==expected,'checkpoint/config/switch mismatch'
 assert not saved.get('diagnostic_only',False) or allow_fixture,'diagnostic checkpoint cannot train/evaluate'
 e,i=saved['meta']['epoch'],saved['meta']['iter'];assert 0<=e<=3 and 0<=i<=5907
 if final:
  assert (e,i)==(3,5907),'only complete epoch3 can be evaluated'
  assert not saved.get('partial_stop',False)
  assert len(saved['optimizer']['state'])==345
  assert all(int(v['step'])==5907 for v in saved['optimizer']['state'].values())
 return True
def fingerprint(obj):
 h=hashlib.sha256()
 def visit(x):
  if torch.is_tensor(x):
   y=x.detach().cpu().contiguous();h.update(str((y.dtype,tuple(y.shape))).encode());h.update(y.numpy().tobytes())
  elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str):h.update(repr(k).encode());visit(x[k])
  elif isinstance(x,(list,tuple)):
   h.update(type(x).__name__.encode())
   for y in x:visit(y)
  else:h.update(repr(x).encode())
 visit(obj);return h.hexdigest()
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
@OPTIM_WRAPPERS.register_module()
class ScreenFiniteOptimWrapper(OptimWrapper):
 def backward(self,loss,**kwargs):
  if not torch.isfinite(loss).all():raise FloatingPointError('nonfinite loss before backward')
  return super().backward(loss,**kwargs)
 def step(self,**kwargs):
  params=[p for group in self.optimizer.param_groups for p in group['params']]
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('nonfinite gradient before step')
  super().step(**kwargs)
  if any(not torch.isfinite(p).all() for p in params):raise FloatingPointError('nonfinite parameter after step')
class APScreenGuard(Hook):
 priority='VERY_HIGH'
 def __init__(self,expected,deadline):self.expected=expected;self.deadline=deadline
 def before_train_iter(self,runner,batch_idx,data_batch=None):
  if runner.epoch>=3 or runner.iter>=5907:raise RuntimeError('training budget exhausted')
  if time.time()>=self.deadline:
   runner.save_checkpoint(runner.work_dir,'budget_stop.pth',meta=dict(epoch=runner.epoch,iter=runner.iter),by_epoch=False)
   raise TimeoutError('four-hour budget reached; stopped before next update')
 def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
  if outputs is not None:
   for value in outputs.values():
    if torch.is_tensor(value) and not torch.isfinite(value).all():raise FloatingPointError('nonfinite logged loss')
 def before_save_checkpoint(self,runner,checkpoint):
  assert runner.model.panoptic_head.match_valid_region is self.expected['match_valid_region']
  checkpoint['screen_contract']=self.expected;checkpoint['screen_rng']=rng_state()
  checkpoint['partial_stop']=time.time()>=self.deadline
