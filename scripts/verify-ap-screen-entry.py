"""No-update entry acceptance: real Runner loading/serialization, control-flow tests, E0 replay."""
import argparse,copy,fcntl,gc,importlib.util,json,logging,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.loops import EpochBasedTrainLoop
from mmdet.utils import register_all_modules
from scripts.ap_screen_runtime import *
import scripts.valid_region_head
spec=importlib.util.spec_from_file_location('screen_entry',ROOT/'scripts/run-valid-region-ap-screen.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
OUT=PREFLIGHT;WORK=ROOT/'work_dirs/valid_region_entry_probe_retry1'

def initialized(arm,folder):
 runner,cfg=entry.create_runner(arm,folder)
 runner._train_loop=runner.build_train_loop(runner._train_loop)
 runner.optim_wrapper=runner.build_optim_wrapper(runner.optim_wrapper);runner.param_schedulers=[]
 runner.load_or_resume()
 return runner,cfg

def snapshot(r):return dict(model=fingerprint(r.model.state_dict()),optimizer=fingerprint(r.optim_wrapper.state_dict()),scheduler=fingerprint(r.param_schedulers),epoch=r.epoch,iteration=r.iter,rng=fingerprint(rng_state()),switch=r.model.panoptic_head.match_valid_region)

def guards():
 expected=contract('control');guard=APScreenGuard(expected,time.time()+600)
 class FakeModel:
  panoptic_head=type('Head',(),{'match_valid_region':False})()
  def train(self):pass
  def train_step(self,data,optim_wrapper=None):return {'loss':torch.tensor(0.)}
 class Fake:
  model=FakeModel();optim_wrapper=None;val_loop=None;work_dir=str(WORK/'fake');logger=logging.getLogger('fake')
  def __init__(self):self.loop=None;self.saved=[];self.iter_calls=0
  @property
  def epoch(self):return self.loop.epoch
  @property
  def iter(self):return self.loop.iter
  def call_hook(self,name,**kwargs):
   if name=='before_train_iter':guard.before_train_iter(self,**kwargs);self.iter_calls+=1
   elif name=='after_train_iter':guard.after_train_iter(self,**kwargs)
  def save_checkpoint(self,*args,**kwargs):self.saved.append(kwargs)
 fake=Fake();fake.loop=EpochBasedTrainLoop(fake,torch.utils.data.DataLoader(list(range(1969)),batch_size=1),max_epochs=3,val_interval=4)
 fake.loop.run();assert fake.epoch==3 and fake.iter==5907 and fake.iter_calls==5907
 try:guard.before_train_iter(fake,0)
 except RuntimeError:pass
 else:raise AssertionError('fourth epoch accepted')
 fake.loop._epoch=1;fake.loop._iter=1969;guard.deadline=time.time()-1
 try:guard.before_train_iter(fake,0)
 except TimeoutError:pass
 else:raise AssertionError('deadline ignored')
 assert len(fake.saved)==1 and fake.iter_calls==5907
 param=torch.nn.Parameter(torch.tensor([1.]));optim=torch.optim.AdamW([param],lr=1e-5);wrapper=ScreenFiniteOptimWrapper(optim)
 try:wrapper.backward(torch.tensor(float('nan'),requires_grad=True))
 except FloatingPointError:pass
 else:raise AssertionError('NaN loss accepted')
 param.grad=torch.tensor([float('inf')])
 try:wrapper.step()
 except FloatingPointError:pass
 else:raise AssertionError('Inf gradient accepted')
 assert float(param)==1 and not optim.state
 # End-to-end orchestration abort: an injected OOM must prevent evaluation and arm B.
 called=[];oldtrain,oldeval=entry.train_arm,entry.evaluate_config
 def oom(arm,deadline):called.append(arm);raise torch.cuda.OutOfMemoryError('injected OOM; no real allocation failure')
 def forbidden(*a,**k):raise AssertionError('evaluation ran after OOM')
 entry.train_arm=oom;entry.evaluate_config=forbidden;failure=OUT/'injected-oom';failure.mkdir()
 try:
  try:entry.execute_sequence(failure,time.time()+60)
  except torch.cuda.OutOfMemoryError:pass
  else:raise AssertionError('OOM swallowed')
 finally:entry.train_arm,entry.evaluate_config=oldtrain,oldeval
 assert called==['control'] and not json.loads((failure/'process-result.json').read_text())['completed']
 # Final-evaluation metadata and diagnostic rejection; tensors are not fabricated model weights.
 minimal=dict(screen_contract=expected,meta=dict(epoch=3,iter=5907),optimizer=dict(state={k:dict(step=5907) for k in range(345)}))
 assert validate_checkpoint(minimal,expected,final=True)
 for change in ['epoch','iter','switch','diagnostic','partial']:
  x=copy.deepcopy(minimal)
  if change=='epoch':x['meta']['epoch']=2
  if change=='iter':x['meta']['iter']=5906
  if change=='switch':x['screen_contract']['match_valid_region']=True
  if change=='diagnostic':x['diagnostic_only']=True
  if change=='partial':x['partial_stop']=True
  try:validate_checkpoint(x,expected,final=True)
  except AssertionError:pass
  else:raise AssertionError('invalid final accepted: '+change)
 return dict(native_epoch_loop_simulated_steps=5907,real_optimizer_steps=0,fourth_epoch_rejected=True,deadline_saved_and_stopped=True,nonfinite_pre_step_rejected=True,oom_aborts_pair=True,final_metadata_negative_checks=5)

def child(arm):
 target=WORK/(arm+'.pth');saved=torch.load(target,map_location='cpu');validate_checkpoint(saved,contract(arm),allow_fixture=True)
 cfg=Config.fromfile(str(PLAN/(arm+'.py')));cfg.work_dir=str(WORK/(arm+'-restore'));cfg.optim_wrapper.type='ScreenFiniteOptimWrapper';cfg.load_from=None
 runner=Runner.from_cfg(cfg);runner.resume(str(target));restore_rng(saved['screen_rng']);expected=json.loads((WORK/(arm+'-expected.json')).read_text())
 assert snapshot(runner)==expected
 assert len(runner.optim_wrapper.optimizer.state)==345 and runner.epoch==1 and runner.iter==1969
 # Fixture is explicitly unusable for actual training/evaluation.
 try:validate_checkpoint(saved,contract(arm))
 except AssertionError:pass
 else:raise AssertionError('fixture accepted')
 (OUT/(arm+'-restore.json')).write_text(json.dumps(dict(passed=True,exact_snapshot=expected,optimizer_states=345,training_updates=0,diagnostic_fixture_rejected_for_execution=True),indent=2));print(arm,'FRESH_PROCESS_RESTORE_PASSED',flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--child',choices=ARMS);args=parser.parse_args();register_all_modules();torch.set_num_threads(2)
 if args.child:child(args.child);return
 OUT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=False);assert not (OUT/'acceptance.json').exists()
 report=dict(passed=False,training_started=False,optimizer_updates=0)
 try:
  with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   configs={a:Config.fromfile(str(PLAN/(a+'.py'))) for a in ARMS}
   a,b=[copy.deepcopy(configs[k].to_dict()) for k in ARMS]
   for c in (a,b):c['model']['panoptic_head']['match_valid_region']=None;c['work_dir']=None;c['test_evaluator']['outfile_prefix']=None
   assert a==b
   import shutil
   assert shutil.disk_usage(ROOT).free>10*1024**3
   duplicate=sha(ROOT/'data/UIIS/train/XL_266.jpg')==sha(ROOT/'data/UIIS/val/XL_679.jpg');assert duplicate
   report['guards']=guards();report['arms']={}
   code=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run-valid-region-ap-screen.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   assert code.returncode!=0 and 'explicit --execute required' in code.stdout
   (OUT/'no-execute.log').write_text(code.stdout)
   for arm in ARMS:
    assert not Path(configs[arm].work_dir).exists()
    runner,cfg=initialized(arm,WORK/(arm+'-fresh'));source=torch.load(cfg.load_from,map_location='cpu')
    assert runner.epoch==runner.iter==0 and not runner.optim_wrapper.optimizer.state
    assert len(runner.train_dataloader.dataset)==3937 and len(runner.train_dataloader)==1969
    assert fingerprint(runner.model.state_dict())==fingerprint(source['state_dict'])
    assert runner.model.panoptic_head.match_valid_region is (arm=='candidate')
    lrs=[g['lr'] for g in runner.optim_wrapper.optimizer.param_groups];assert set(lrs)=={cfg.optim_wrapper.optimizer.lr,cfg.optim_wrapper.optimizer.lr*cfg.optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult}
    # Existing trained E0 moments are serialization fixtures only, never initial optimizer state.
    runner.optim_wrapper.load_state_dict(source['optimizer'])
    for group,lr in zip(runner.optim_wrapper.optimizer.param_groups,lrs):group['lr']=lr
    assert len(runner.optim_wrapper.optimizer.state)==345
    runner._train_loop._epoch=1;runner._train_loop._iter=1969
    runner.register_hook(APScreenGuard(contract(arm),time.time()+3600));before=snapshot(runner)
    runner.save_checkpoint(str(WORK),arm+'.pth',meta=dict(epoch=1,iter=1969),by_epoch=False)
    fixture=WORK/(arm+'.pth');saved=torch.load(fixture,map_location='cpu');saved['diagnostic_only']=True;torch.save(saved,fixture)
    assert snapshot(runner)==before
    (WORK/(arm+'-expected.json')).write_text(json.dumps(before,indent=2))
    del saved,source,runner;gc.collect();torch.cuda.empty_cache()
    with (OUT/(arm+'-child.log')).open('w') as f:subprocess.run([sys.executable,'-B',__file__,'--child',arm],stdout=f,stderr=subprocess.STDOUT,check=True)
    assert json.loads((OUT/(arm+'-restore.json')).read_text())['passed']
    report['arms'][arm]=dict(config_loaded=True,fresh_epoch_iter=[0,0],fresh_optimizer_empty=True,lr_groups=sorted(set(lrs)),checkpoint_exact=True,nonempty_optimizer_restore=True)
    print(arm,'ENTRY_AND_RESTORE_PASSED',flush=True)
   # Exercise actual evaluation path on unchanged E0, clearly marked baseline replay.
   probe_cfg=copy.deepcopy(configs['candidate']);checkpoint=Path(probe_cfg.load_from)
   evaldir=OUT/'baseline-evaluation-replay';report['evaluation']=entry.evaluate_config(probe_cfg,checkpoint,evaldir,time.time()+1800)
   original=json.loads((ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json').read_text());replayed=json.loads((evaldir/'predictions.segm.json').read_text())
   assert replayed==original,'baseline predictions changed in new evaluation entry'
   raw=entry.compare_predictions({'E0':evaldir/'predictions.segm.json'});assert abs(raw['coco']['E0']['stats'][0]-32.6222177)<1e-5
   (OUT/'baseline-raw-ap.json').write_text(json.dumps(raw,indent=2))
   report.update(passed=True,duplicate_preserved=duplicate,baseline_predictions_exact=True,no_training_work_dirs_created=all(not Path(c.work_dir).exists() for c in configs.values()))
   paths=['scripts/ap_screen_runtime.py','scripts/run-valid-region-ap-screen.py','scripts/valid_region_head.py','reports/valid-region-ap-screen-plan-20261031/control.py','reports/valid-region-ap-screen-plan-20261031/candidate.py','reports/valid-region-ap-screen-plan-20261031/preparation.json']
   report['sources']={p:sha(ROOT/p) for p in paths}
 except BaseException:
  report['error']=traceback.format_exc();raise
 finally:(OUT/'acceptance.json').write_text(json.dumps(report,indent=2))
 print('AP_SCREEN_ENTRY_ACCEPTED_NO_TRAINING',flush=True)
if __name__=='__main__':main()
