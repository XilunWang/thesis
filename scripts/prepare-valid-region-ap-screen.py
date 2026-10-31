"""Prepare and validate paired AP-screen configs; never build a model or train."""
import ast,copy,hashlib,json,pprint,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E0_SHA='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def check_literal(node):
 if isinstance(node,ast.Constant):return
 if isinstance(node,(ast.List,ast.Tuple)):
  for x in node.elts:check_literal(x)
  return
 if isinstance(node,ast.Dict):
  for x in node.keys+node.values:check_literal(x)
  return
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args:
  for k in node.keywords:
   assert k.arg is not None;check_literal(k.value)
  return
 if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):check_literal(node.operand);return
 raise ValueError('Unexpected executable config expression: '+ast.dump(node))
def main():
 base=ROOT/'reports/uiis-baseline-36e-evaluation/evaluation-config.py'
 for n in ast.parse(base.read_text()).body:
  assert isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) for t in n.targets)
  check_literal(n.value)
 cfg={k:v for k,v in runpy.run_path(str(base)).items() if not k.startswith('__')}
 assert cfg['train_cfg']['max_epochs']==36 and cfg['model']['panoptic_head']['type']=='Mask2FormerHead'
 ck=ROOT/'checkpoints/e0-backup/epoch_36.pth';assert sha(ck)==E0_SHA
 train=ROOT/'data/UIIS/annotations/train.json';test=ROOT/'data/UIIS/annotations/val.json'
 assert len(json.loads(train.read_text())['images'])==3937 and len(json.loads(test.read_text())['images'])==691
 remote='/root/autodl-tmp/underwater-segmentation'
 cfg['custom_imports']=dict(imports=['scripts.valid_region_head'],allow_failed_imports=False)
 cfg['model']['panoptic_head'].update(type='ValidRegionMask2FormerHead',match_valid_region=False)
 cfg['optim_wrapper']['optimizer']['lr']=1e-5
 cfg['param_scheduler']=[]
 cfg['train_cfg']=dict(type='EpochBasedTrainLoop',max_epochs=3,val_interval=4)
 cfg['val_cfg']=None;cfg['val_dataloader']=None;cfg['val_evaluator']=None
 cfg['load_from']=remote+'/work_dirs/uiis_baseline_r50_36e/epoch_36.pth';cfg['resume']=False
 cfg['randomness']=dict(seed=20261012)
 cfg['default_hooks']['checkpoint']=dict(type='CheckpointHook',by_epoch=True,interval=1,max_keep_ckpts=3,save_last=True,save_optimizer=True,save_param_scheduler=True)
 out=ROOT/'reports/valid-region-ap-screen-plan-20261031';out.mkdir(exist_ok=True)
 arms={}
 for name,enabled in [('control',False),('candidate',True)]:
  c=copy.deepcopy(cfg);c['model']['panoptic_head']['match_valid_region']=enabled
  c['work_dir']=remote+'/work_dirs/valid_region_ap_screen_'+name
  c['test_evaluator']['outfile_prefix']=remote+'/reports/valid-region-ap-screen-'+name+'/predictions'
  arms[name]=c
  text='"""Frozen 3-epoch AP screen; preparation is not training authorization."""\n'+''.join(k+' = '+pprint.pformat(v,sort_dicts=False,width=100)+'\n' for k,v in c.items())
  (out/(name+'.py')).write_text(text,encoding='utf-8');compile(text,name+'.py','exec')
 a,b=copy.deepcopy(arms['control']),copy.deepcopy(arms['candidate'])
 for c in (a,b):
  c['model']['panoptic_head']['match_valid_region']=None;c['work_dir']=None;c['test_evaluator']['outfile_prefix']=None
 assert a==b
 manifest=dict(prepared=True,training_started=False,server_checked=False,framework_config_load_checked=False,checkpoint_restore_checked=False,arms=['control','candidate'],epochs_per_arm=3,iterations_per_arm=5907,total_updates_if_authorized=11814,seed=20261012,learning_rate=1e-5,checkpoint_sha256=E0_SHA,train_sha256=sha(train),test_sha256=sha(test),reference_config_sha256=sha(base),head_sha256=sha(ROOT/'scripts/valid_region_head.py'),pair_difference_only=['model.panoptic_head.match_valid_region','work_dir','test_evaluator.outfile_prefix'],configs={name:sha(out/(name+'.py')) for name in arms})
 (out/'preparation.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
