"""Explicitly gated 3+3 epoch AP screen. Importing or omitting --execute never trains."""
import argparse,copy,fcntl,gc,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
from scripts.ap_screen_runtime import *
import scripts.valid_region_head

def create_runner(arm,work_dir=None):
 cfg=Config.fromfile(str(PLAN/(arm+'.py')));check_config(cfg,arm)
 if work_dir is not None:cfg.work_dir=str(work_dir)
 cfg.optim_wrapper.type='ScreenFiniteOptimWrapper'
 runner=Runner.from_cfg(cfg);return runner,cfg

def evaluate_config(cfg,checkpoint,output,deadline):
 if time.time()>=deadline:raise TimeoutError('no time left for evaluation')
 output=Path(output);output.mkdir(exist_ok=False,parents=True)
 c=copy.deepcopy(cfg);c.load_from=str(checkpoint);c.resume=False;c.work_dir=str(output/'runner');c.test_evaluator.outfile_prefix=str(output/'predictions')
 class Count(Hook):
  def __init__(self):self.ids=[]
  def before_test_iter(self,runner,batch_idx,data_batch=None):
   if time.time()>=deadline:raise TimeoutError('evaluation deadline exceeded')
  def after_test_iter(self,runner,batch_idx,data_batch=None,outputs=None):self.ids.extend(int(s.img_id) for s in outputs)
 runner=Runner.from_cfg(c);counter=Count();runner.register_hook(counter);metrics=runner.test()
 expected={i['id'] for i in json.loads((ROOT/'data/UIIS/annotations/val.json').read_text())['images']}
 assert len(counter.ids)==691 and set(counter.ids)==expected
 result=dict(completed=True,evaluated_images=691,checkpoint_sha256=sha(checkpoint),metrics={k:float(v) for k,v in metrics.items()})
 (output/'metrics.json').write_text(json.dumps(result,indent=2));del runner;gc.collect();torch.cuda.empty_cache();return result

def train_arm(arm,deadline):
 cfg=Config.fromfile(str(PLAN/(arm+'.py')));work=Path(cfg.work_dir)
 if work.exists():raise FileExistsError('refuse existing work directory: '+str(work))
 runner,cfg=create_runner(arm);runner.register_hook(APScreenGuard(contract(arm),deadline))
 runner.train()
 assert runner.epoch==3 and runner.iter==5907
 path=work/'epoch_3.pth';saved=torch.load(path,map_location='cpu');validate_checkpoint(saved,contract(arm),final=True)
 del saved,runner;gc.collect();torch.cuda.empty_cache()
 return cfg,path

def compare_predictions(paths):
 import numpy as np
 from pycocotools.coco import COCO
 from pycocotools.cocoeval import COCOeval
 gt=COCO(str(ROOT/'data/UIIS/annotations/val.json'));table={}
 for label,path in paths.items():
  preds=json.loads(Path(path).read_text());clean=[{k:v for k,v in r.items() if k!='bbox'} for r in preds]
  ev=COCOeval(gt,gt.loadRes(clean),'segm');ev.params.maxDets=[100,300,1000];ev.evaluate();ev.accumulate();ev.summarize()
  stats=(ev.stats*100).tolist();classap={}
  for k,cid in enumerate(ev.params.catIds):
   x=ev.eval['precision'][:,:,k,0,-1];x=x[x>-1];classap[gt.cats[cid]['name']]=float(x.mean()*100) if x.size else None
  table[label]=dict(stats=stats,class_ap=classap)
 if set(table)=={'E0','control','candidate'}:
  e,a,b=[table[k]['stats'] for k in ['E0','control','candidate']]
  gates=dict(overall=b[0]>=max(e[0],a[0])+.5,small=b[3]>=a[3]+.5 and b[3]>=e[3],large=b[5]>=max(e[5],a[5])-.3)
 else:gates={}
 return dict(coco=table,gates=gates,worth_replication=bool(gates) and all(gates.values()),note='single-seed exploratory AP screen; no automatic next training')

def execute_sequence(out,deadline):
 started=time.time();result=dict(completed=False,started_at=started,deadline=deadline,stages=[])
 try:
  trained=[]
  for arm in ARMS:
   cfg,path=train_arm(arm,deadline);result['stages'].append(dict(arm=arm,training_complete=True));trained.append((arm,cfg,path))
   (out/'progress.json').write_text(json.dumps(result,indent=2))
  for index,(arm,cfg,path) in enumerate(trained):
   metrics=evaluate_config(cfg,path,ROOT/f'reports/valid-region-ap-screen-{arm}',deadline);result['stages'][index]['evaluation']=metrics
   (out/'progress.json').write_text(json.dumps(result,indent=2))
  comparison=compare_predictions({'E0':ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json',**{a:ROOT/f'reports/valid-region-ap-screen-{a}/predictions.segm.json' for a in ARMS}})
  (out/'comparison.json').write_text(json.dumps(comparison,indent=2));result['completed']=True
 except BaseException:
  result['error']=traceback.format_exc();raise
 finally:
  result['elapsed_seconds']=time.time()-started;(out/'process-result.json').write_text(json.dumps(result,indent=2))
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
 if not args.execute:parser.error('explicit --execute required; no training started')
 seal=json.loads((PREFLIGHT/'acceptance.json').read_text());assert seal['passed']
 for path,digest in seal['sources'].items():assert sha(ROOT/path)==digest,'changed after acceptance: '+path
 register_all_modules();torch.set_num_threads(2)
 with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Refuse stale or partially completed work before creating a run record.
  for arm in ARMS:
   cfg=Config.fromfile(str(PLAN/(arm+'.py')));check_config(cfg,arm)
   assert not Path(cfg.work_dir).exists() and not (ROOT/f'reports/valid-region-ap-screen-{arm}').exists()
  out=ROOT/'reports/valid-region-ap-screen-run';out.mkdir(exist_ok=False)
  references={str(p):sha(p) for p in [ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json',ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth']}
  try:execute_sequence(out,time.time()+4*3600)
  finally:assert all(sha(p)==h for p,h in references.items())
if __name__=='__main__':main()
