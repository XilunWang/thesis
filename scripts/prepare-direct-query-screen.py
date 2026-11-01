"""Prepare a direct-query screening draft and audit old A evidence; never train."""
import ast,copy,hashlib,json,pprint,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def diff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  out=[]
  for k in sorted(a.keys()|b.keys()):
   key=f'{path}.{k}' if path else k
   if k not in a or k not in b:out.append(key)
   else:out+=diff(a[k],b[k],key)
  return out
 return [] if a==b else [path]
def main():
 old=ROOT/'reports/valid-region-ap-screen-plan-20261031';pre=json.loads((old/'preparation.json').read_text());seal=json.loads((ROOT/'reports/valid-region-ap-screen-entry-20261031/acceptance.json').read_text())
 assert sha(old/'control.py')==pre['configs']['control']==seal['sources']['reports/valid-region-ap-screen-plan-20261031/control.py']
 # Reuse the existing literal-only config validator; no framework or model is imported.
 validator=runpy.run_path(str(ROOT/'scripts/prepare-valid-region-ap-screen.py'))['check_literal']
 for n in ast.parse((old/'control.py').read_text()).body:
  if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):continue
  assert isinstance(n,ast.Assign);validator(n.value)
 cfg={k:v for k,v in runpy.run_path(str(old/'control.py')).items() if not k.startswith('__')}
 assert cfg['model']['panoptic_head']['match_valid_region'] is False
 assert cfg['model']['panoptic_head']['num_things_classes']==7
 assert sha(ROOT/'checkpoints/e0-backup/epoch_36.pth')==pre['checkpoint_sha256']
 assert sha(ROOT/'data/UIIS/annotations/train.json')==pre['train_sha256']
 assert sha(ROOT/'data/UIIS/annotations/val.json')==pre['test_sha256']
 verification=json.loads((ROOT/'reports/valid-region-ap-screen-run/local-verification.json').read_text())
 pred=ROOT/'reports/valid-region-ap-screen-control/predictions.segm.json';assert sha(pred)==verification['checks']['control']['predictions_sha256']
 rows=json.loads(pred.read_text());assert len(rows)==69100 and len({r['image_id'] for r in rows})==691
 process=json.loads((ROOT/'reports/valid-region-ap-screen-run/process-result.json').read_text());assert process['completed']
 metrics=json.loads((ROOT/'reports/valid-region-ap-screen-control/metrics.json').read_text());assert metrics['completed'] and metrics['checkpoint_sha256']==verification['checks']['control']['checkpoint_sha256']
 acceptance=json.loads((ROOT/'reports/direct-query-integration-20261101/result.json').read_text());assert acceptance['passed']
 head=ROOT/'scripts/direct_query_feature_head.py'
 startup_path=ROOT/'reports/direct-query-startup-20261101/acceptance.json'
 startup=json.loads(startup_path.read_text()) if startup_path.exists() else {}
 runtime_ok=bool(startup.get('passed') and startup.get('head_sha256')==sha(head))
 assert sha(head)==acceptance['source_hashes']['scripts/direct_query_feature_head.py'] or runtime_ok
 new=copy.deepcopy(cfg);remote='/root/autodl-tmp/underwater-segmentation'
 new['custom_imports']={'imports':['scripts.direct_query_feature_head'],'allow_failed_imports':False}
 new['model']['panoptic_head'].pop('match_valid_region');new['model']['panoptic_head'].update(type='DirectQueryFeatureHead',direct_query_features=True)
 new['work_dir']=remote+'/work_dirs/direct_query_ap_screen_candidate'
 new['test_evaluator']['outfile_prefix']=remote+'/reports/direct-query-ap-screen-candidate/predictions'
 differences=diff(cfg,new);allowed=['custom_imports.imports','model.panoptic_head.direct_query_features','model.panoptic_head.match_valid_region','model.panoptic_head.type','test_evaluator.outfile_prefix','work_dir'];assert differences==allowed
 assert new['load_from']==cfg['load_from'] and not new['resume'] and new['optim_wrapper']==cfg['optim_wrapper']
 assert new['train_cfg']['max_epochs']==3 and new['param_scheduler']==[] and new['randomness']['seed']==20261012
 groups=[]
 for name,count in [('panoptic_head.direct_c3.weight',131072),('panoptic_head.direct_c4.weight',262144)]:
  keys=new['optim_wrapper']['paramwise_cfg']['custom_keys'];assert not any(k in name for k in keys)
  groups.append(dict(name=name,parameters=count,expected_lr=new['optim_wrapper']['optimizer']['lr'],expected_weight_decay=new['optim_wrapper']['optimizer']['weight_decay'],actual_optimizer_checked=runtime_ok))
 out=ROOT/'reports/direct-query-screen-preparation-20261101';out.mkdir(exist_ok=True)
 text='"""DRAFT: runtime RNG/optimizer/resume gates are pending. No training authorization."""\n'+''.join(k+' = '+pprint.pformat(v,sort_dicts=False,width=100)+'\n' for k,v in new.items())
 compile(text,'candidate.py','exec');(out/'candidate.py').write_text(text,encoding='utf8')
 comparison=json.loads((ROOT/'reports/valid-region-ap-screen-run/comparison.json').read_text());a=comparison['coco']['control']['stats'];e=comparison['coco']['E0']['stats']
 result=dict(local_preparation_passed=True,training_started=False,server_checked=runtime_ok,old_A_protocol_evidence_valid=True,strict_paired_reuse_approved=False,runtime_preflight_passed=runtime_ok,actual_optimizer_checked=runtime_ok,data_rng_replay_checked=runtime_ok,nonempty_optimizer_restore_checked=runtime_ok,initial_weights_sha256=pre['checkpoint_sha256'],old_A_config_sha256=sha(old/'control.py'),old_A_predictions_sha256=sha(pred),old_A_checkpoint_sha256_from_verified_record=metrics['checkpoint_sha256'],candidate_config_sha256=sha(out/'candidate.py'),head_sha256=sha(head),config_differences=differences,expected_new_parameter_groups=groups,old_A_metrics=dict(AP=a[0],APs=a[3],APl=a[5]),gates=dict(AP=max(a[0],e[0])+.5,APs=max(a[3]+.5,e[3]),APl=max(a[5],e[5])-.3),pending=['Reconstruct sealed A startup and candidate startup without optimizer steps; compare sampler/worker seeds, augmented batch fingerprints and CPU/CUDA RNG before loss sampling.','Resolve RNG differences without changing baseline A history; do not merely reseed candidate and claim an exact historical pair.','Verify actual optimizer groups, empty initial state and synthetic nonempty-state save/restore without optimizer steps; distinguish fixture from real interrupted-training recovery.','Freeze independent finite-loss/gradient guards, budget, checkpoint contract and final-only evaluation entry before any training authorization.'])
 if runtime_ok:
  result['historical_A_usable_for_exploratory_screen']=True
  result['historical_step_by_step_equivalence_proven']=False
  result['startup_acceptance_sha256']=sha(startup_path)
  result['pending']=['Independent finite/budget/checkpoint contracts and execution/evaluation orchestration are not prepared; no training authorization.','Diagnostic nonempty optimizer restore does not prove recovery after real updates.']
 (out/'preparation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
