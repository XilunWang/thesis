"""Explicitly gated candidate-only 3-epoch screen; no training on import."""
import argparse,copy,gc,hashlib,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.direct_query_execution import BUDGET_SECONDS,check_next_update,check_output_paths,sequence,validate_final
PLAN=ROOT/'reports/direct-query-screen-preparation-20261101'
SEAL=ROOT/'reports/direct-query-execution-20261101/acceptance.json'
RUN=ROOT/'reports/direct-query-ap-screen-run'
EVAL=ROOT/'reports/direct-query-ap-screen-candidate'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
 if not args.execute:parser.error('explicit --execute required; no training started')
 # Dependencies are imported only after the explicit execution gate.
 import fcntl,math
 import torch
 from mmengine.config import Config
 from mmengine.hooks import Hook
 from mmengine.runner import Runner
 from mmdet.utils import register_all_modules
 from scripts.ap_screen_runtime import ScreenFiniteOptimWrapper,rng_state,fingerprint
 import scripts.direct_query_feature_head
 seal=json.loads(SEAL.read_text());assert seal['passed']
 for name,digest in seal['sources'].items():assert sha(ROOT/name)==digest,'source changed after execution checks: '+name
 prep=json.loads((PLAN/'preparation.json').read_text());startup=json.loads((ROOT/'reports/direct-query-startup-20261101/acceptance.json').read_text())
 assert startup['passed'] and prep['runtime_preflight_passed'] and prep['historical_A_usable_for_exploratory_screen']
 cfg=Config.fromfile(str(PLAN/'candidate.py'))
 assert sha(PLAN/'candidate.py')==prep['candidate_config_sha256'] and sha(ROOT/'scripts/direct_query_feature_head.py')==prep['head_sha256']==startup['head_sha256']
 assert cfg.model.panoptic_head.type=='DirectQueryFeatureHead' and cfg.model.panoptic_head.direct_query_features
 assert not cfg.resume and cfg.train_cfg.max_epochs==3 and cfg.train_cfg.val_interval==4 and cfg.val_cfg is None
 assert cfg.param_scheduler==[] and cfg.randomness.seed==20261012 and cfg.train_dataloader.batch_size==2
 assert cfg.optim_wrapper.optimizer.lr==1e-5 and cfg.optim_wrapper.type=='OptimWrapper'
 assert cfg.work_dir==str(ROOT/'work_dirs/direct_query_ap_screen_candidate')
 assert cfg.test_evaluator.outfile_prefix==str(EVAL/'predictions')
 expected=dict(version=1,head_sha256=sha(ROOT/'scripts/direct_query_feature_head.py'),config_sha256=sha(PLAN/'candidate.py'),epochs=3,iterations=5907,direct_query_features=True)
 original_manifest=json.loads((ROOT/'reports/valid-region-ap-screen-plan-20261031/preparation.json').read_text())
 references={Path(cfg.load_from):prep['initial_weights_sha256'],ROOT/'data/UIIS/annotations/train.json':original_manifest['train_sha256'],ROOT/'data/UIIS/annotations/val.json':original_manifest['test_sha256'],ROOT/'reports/valid-region-ap-screen-control/predictions.segm.json':prep['old_A_predictions_sha256'],ROOT/'work_dirs/valid_region_ap_screen_control/epoch_3.pth':prep['old_A_checkpoint_sha256_from_verified_record']}
 original_prediction=ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json'
 references[original_prediction]=json.loads((ROOT/'reports/feature-readout-export-20261101/result.json').read_text())['input_hashes']['reports/uiis-baseline-36e-evaluation/predictions.segm.json']
 assert all(sha(p)==digest for p,digest in references.items())
 spec=importlib.util.spec_from_file_location('old_screen_evaluation_helpers',ROOT/'scripts/run-valid-region-ap-screen.py');helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
 class Guard(Hook):
  priority='VERY_HIGH'
  def before_train(self,runner):
   assert runner.epoch==runner.iter==0 and not runner.optim_wrapper.optimizer.state
   assert len(runner.train_dataloader.dataset)==3937 and len(runner.train_dataloader)==1969
   initial=torch.load(cfg.load_from,map_location='cpu')['state_dict'];state=runner.model.state_dict()
   extra={k:v for k,v in state.items() if k not in initial}
   assert set(extra)=={'panoptic_head.direct_c3.weight','panoptic_head.direct_c4.weight'} and all(torch.count_nonzero(v)==0 for v in extra.values())
   assert fingerprint({k:state[k] for k in initial})==fingerprint(initial)
   names={id(p):k for k,p in runner.model.named_parameters()};actual=[]
   for group in runner.optim_wrapper.optimizer.param_groups:
    for p in group['params']:actual.append(dict(name=names[id(p)],lr=group['lr'],weight_decay=group.get('weight_decay',0),requires_grad=p.requires_grad,numel=p.numel()))
   checked=json.loads((ROOT/'reports/direct-query-startup-20261101/candidate.json').read_text())['groups'];assert actual==checked
   (RUN/'actual-start.json').write_text(json.dumps(dict(initial_state_exact=True,optimizer_empty=True,groups=actual),indent=2))
  def before_train_iter(self,runner,batch_idx,data_batch=None):
   try:check_next_update(runner.epoch,runner.iter,time.time(),deadline)
   except TimeoutError:
    runner.save_checkpoint(runner.work_dir,'budget_stop.pth',meta=dict(epoch=runner.epoch,iter=runner.iter),by_epoch=False);raise
  def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
   for v in (outputs or {}).values():
    if torch.is_tensor(v) and not torch.isfinite(v).all():raise FloatingPointError('nonfinite logged output')
    if isinstance(v,(float,int)) and not math.isfinite(v):raise FloatingPointError('nonfinite logged output')
  def before_save_checkpoint(self,runner,checkpoint):
   assert runner.model.panoptic_head.direct_query_features
   checkpoint['direct_query_contract']=expected;checkpoint['direct_query_rng']=rng_state();checkpoint['partial_stop']=time.time()>=deadline
 def train():
  c=copy.deepcopy(cfg);c.optim_wrapper.type='ScreenFiniteOptimWrapper';runner=Runner.from_cfg(c);runner.register_hook(Guard())
  runner.train();assert (runner.epoch,runner.iter)==(3,5907)
  p=Path(cfg.work_dir)/'epoch_3.pth';saved=torch.load(p,map_location='cpu');validate_final(saved,expected)
  assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
  del saved,runner;gc.collect();torch.cuda.empty_cache();return p
 def evaluate(p):
  saved=torch.load(p,map_location='cpu');validate_final(saved,expected);del saved
  return helpers.evaluate_config(cfg,p,EVAL,deadline)
 def compare():
  if time.time()>=deadline:raise TimeoutError('no time left for comparison')
  r=helpers.compare_predictions({'E0':original_prediction,'control':ROOT/'reports/valid-region-ap-screen-control/predictions.segm.json','candidate':EVAL/'predictions.segm.json'})
  old=json.loads((ROOT/'reports/valid-region-ap-screen-run/comparison.json').read_text())
  for k in ['E0','control']:assert all(abs(x-y)<1e-8 for x,y in zip(r['coco'][k]['stats'],old['coco'][k]['stats']))
  (RUN/'comparison.json').write_text(json.dumps(r,indent=2));return r
 def persist(r):
  r.update(started_at=started,deadline=deadline,elapsed_seconds=time.time()-started,reused_control=True)
  (RUN/'process-result.json').write_text(json.dumps(r,indent=2))
 register_all_modules();torch.set_num_threads(2)
 with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  check_output_paths([Path(cfg.work_dir),EVAL,RUN])
  RUN.mkdir();started=time.time();deadline=started+BUDGET_SECONDS
  try:sequence(train,evaluate,compare,persist)
  finally:assert all(sha(p)==digest for p,digest in references.items()),'reference artifact modified'
if __name__=='__main__':main()
