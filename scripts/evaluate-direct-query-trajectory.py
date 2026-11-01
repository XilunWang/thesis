"""Retrospective epoch1/2 evaluation only; never train or replace final-screen results."""
import argparse,fcntl,gc,importlib.util,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/direct-query-trajectory-20261101'

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
 if not args.evaluate:parser.error('explicit --evaluate required; no model executed')
 import torch
 from mmengine.config import Config
 from mmdet.utils import register_all_modules
 from scripts.ap_screen_runtime import sha,contract,validate_checkpoint
 import scripts.direct_query_feature_head
 spec=importlib.util.spec_from_file_location('saved_screen_helpers',ROOT/'scripts/run-valid-region-ap-screen.py')
 helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
 seal=json.loads((ROOT/'reports/direct-query-execution-20261101/acceptance.json').read_text())
 for name,digest in seal['sources'].items():assert sha(ROOT/name)==digest,name
 configs={'control':ROOT/'reports/valid-region-ap-screen-plan-20261031/control.py','candidate':ROOT/'reports/direct-query-screen-preparation-20261101/candidate.py'}
 works={'control':ROOT/'work_dirs/valid_region_ap_screen_control','candidate':ROOT/'work_dirs/direct_query_ap_screen_candidate'}
 paths={(arm,e):works[arm]/f'epoch_{e}.pth' for arm in configs for e in [1,2]}
 for path in paths.values():assert path.is_file(),str(path)
 assert not OUT.exists(),'refuse overwriting retrospective outputs'
 register_all_modules();torch.set_num_threads(2)
 with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  OUT.mkdir();started=time.time();deadline=started+3600
  result=dict(completed=False,training_updates=0,retrospective_only=True,started_at=started,evaluations={})
  def save():
   result['elapsed_seconds']=time.time()-started
   (OUT/'process-result.json').write_text(json.dumps(result,indent=2))
  try:
   preflight={}
   for (arm,epoch),path in paths.items():
    saved=torch.load(path,map_location='cpu')
    assert not saved.get('diagnostic_only') and not saved.get('partial_stop')
    assert (saved['meta']['epoch'],saved['meta']['iter'])==(epoch,1969*epoch)
    states=saved['optimizer']['state'];assert len(states)==(345 if arm=='control' else 347)
    assert all(int(s['step'])==1969*epoch for s in states.values())
    if arm=='control':validate_checkpoint(saved,contract('control'),final=False)
    else:
     expected=dict(version=1,head_sha256=sha(ROOT/'scripts/direct_query_feature_head.py'),config_sha256=sha(configs[arm]),epochs=3,iterations=5907,direct_query_features=True)
     assert saved['direct_query_contract']==expected
    assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
    preflight[f'{arm}_{epoch}']=dict(checkpoint_sha256=sha(path),epoch=epoch,iteration=1969*epoch)
    del saved;gc.collect()
   result['preflight']=preflight;save()
   predictions={}
   for epoch in [1,2]:
    for arm in ['control','candidate']:
     key=f'{arm}_{epoch}';path=paths[arm,epoch];cfg=Config.fromfile(str(configs[arm]))
     assert cfg.train_cfg.max_epochs==3 and not cfg.resume
     assert cfg.model.panoptic_head.type==('ValidRegionMask2FormerHead' if arm=='control' else 'DirectQueryFeatureHead')
     if arm=='control':assert not cfg.model.panoptic_head.match_valid_region
     else:assert cfg.model.panoptic_head.direct_query_features
     # evaluate_config invokes Runner.test only, never train or optimizer updates.
     result['evaluations'][key]=helpers.evaluate_config(cfg,path,OUT/key,deadline);save()
     predictions[key]=OUT/key/'predictions.segm.json'
   comparison=helpers.compare_predictions(predictions)
   final=json.loads((ROOT/'reports/direct-query-ap-screen-run/comparison.json').read_text())
   comparison['coco'].update({f'{arm}_3':final['coco'][arm] for arm in configs})
   comparison['E0']=final['coco']['E0']
   comparison['note']='Retrospective trajectory, no best-epoch substitution, original epoch3 screen gates unchanged.'
   (OUT/'trajectory.json').write_text(json.dumps(comparison,indent=2))
   for (arm,epoch),path in paths.items():assert sha(path)==preflight[f'{arm}_{epoch}']['checkpoint_sha256']
   result['completed']=True
  except BaseException:
   result['error']=traceback.format_exc();raise
  finally:save()
if __name__=='__main__':main()
