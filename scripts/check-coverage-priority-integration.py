"""Fixed GPU two-view no-update check; separate process checkpoint restore."""
import argparse,fcntl,hashlib,importlib.util,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.coverage_priority_objective import objective,settings,settings_hash
from scripts.representative_label_only import objective as previous
from scripts.position_pilot_runtime import save_checkpoint,load_checkpoint
OUT=ROOT/'reports/coverage-priority-integration-20261101'
KEYS=[(2971,21263983),(3151,20264163)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main(restore):
 spec=importlib.util.spec_from_file_location('base',ROOT/'scripts/run-position-grid4-pilot.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 pilot=base.Pilot()
 def forbidden(*a,**k):raise AssertionError('No optimizer updates permitted')
 pilot.optimizer.step=forbidden
 codes={n:sha(ROOT/'scripts'/n) for n in ['coverage_priority.py','coverage_priority_objective.py','check-coverage-priority-integration.py']}
 options=dict(manifest_hash=settings_hash(True),code_hashes=codes,view_hash=pilot.options['view_hash'],kind='coverage_priority_preflight')
 if restore:
  state=load_checkpoint(OUT/'step0.pth',pilot.head,pilot.optimizer,**options)
  assert state['step']==0 and state['cursor']==0 and not pilot.optimizer.state
 else:
  request=json.loads((ROOT/'reports/label-only-audit-scores-20261101/request.json').read_text());p=ROOT/'work_dirs/position_label_only_pilot/step300.pth'
  assert sha(p)==request['stages']['new300']['sha256']['step300.pth']
  pilot.head.load_state_dict(torch.load(p,map_location='cpu')['head'],strict=True)
 pilot.head.eval();before={k:v.detach().clone() for k,v in pilot.head.state_dict().items()}
 pieces=[pilot.data(next(r for r in pilot.views if (r['image_id'],r['seed'])==key)) for key in KEYS]
 with torch.no_grad():features=pilot.model.extract_feat(torch.cat([d['inputs'] for d,t,v in pieces]))
 target=torch.stack([t for d,t,v in pieces]).cuda();valid=torch.stack([v for d,t,v in pieces]).cuda()
 instances=[]
 for data,t,v in pieces:
  gt=data['data_samples'][0].gt_instances;instances.append(dict(masks=gt.masks.masks,ids=gt.annotation_ids.tolist(),areas=gt.original_areas.tolist()))
 z=pilot.head(features[1],features[2]);snapshot=z.detach().clone()
 old,oldinfo=previous(z,target,valid,instances,enabled=True)
 off,offinfo=objective(z,target,valid,instances)
 assert torch.equal(old,off) and torch.equal(oldinfo['labels'],offinfo['labels'])
 gold=torch.autograd.grad(old,z,retain_graph=True)[0];goff=torch.autograd.grad(off,z,retain_graph=True)[0];assert torch.equal(gold,goff)
 on,info=objective(z,target,valid,instances,coverage_priority=True)
 g=torch.autograd.grad(on,z,retain_graph=True)[0];labels=info['labels']
 assert torch.isfinite(on) and torch.isfinite(g).all()
 assert (g[(labels==1)&valid]<0).all() and (g[(labels==0)&valid]>0).all() and not g[~valid].any()
 unchanged=(labels==oldinfo['labels'])&valid;assert torch.equal(g[unchanged],gold[unchanged])
 pair_count=0
 for i,key in enumerate(KEYS):
  reps=info['layouts'][i]['representatives'];assert len(reps)==int(target[i].sum())==len(set(reps.values()))
  for a,b in pilot.neighbors.get(key,[]):assert reps[a]!=reps[b];pair_count+=1
 pilot.optimizer.zero_grad(set_to_none=True);on.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pilot.head.parameters())
 norm=float(torch.sqrt(sum((p.grad**2).sum() for p in pilot.head.parameters())))
 assert norm>0;pilot.protected();assert all(torch.equal(v,before[k]) for k,v in pilot.head.state_dict().items())
 with torch.no_grad():assert torch.equal(pilot.head(features[1],features[2]),snapshot)
 observed=dict(logits_sha256=array_sha(snapshot.cpu().numpy()),labels_sha256=array_sha(labels.cpu().numpy()),old_loss=float(old.detach()),new_loss=float(on.detach()),positive_counts=target.sum((1,2)).tolist(),neighbor_pairs=pair_count,gradient_norm=norm)
 if restore:
  expected=json.loads((OUT/'result.json').read_text())['observed']
  for key in ['logits_sha256','labels_sha256','old_loss','new_loss','positive_counts','neighbor_pairs']:assert observed[key]==expected[key],key
  (OUT/'fresh-restore.json').write_text(json.dumps(dict(passed=True,training_updates=0,observed=observed,optimizer_empty=True,settings=settings(True)),indent=2));print('FRESH_RESTORE_PASSED');return
 pilot.head.zero_grad(set_to_none=True)
 save_checkpoint(OUT/'step0.pth',pilot.head,pilot.optimizer,**options,step=0,cursor=0,elapsed_seconds=0)
 try:load_checkpoint(OUT/'step0.pth',pilot.head,pilot.optimizer,**{**options,'manifest_hash':settings_hash(False)})
 except ValueError:pass
 else:raise AssertionError('Wrong switch checkpoint was accepted')
 load_checkpoint(OUT/'step0.pth',pilot.head,pilot.optimizer,**options)
 assert not pilot.optimizer.state and all(torch.equal(v,before[k]) for k,v in pilot.head.state_dict().items())
 pilot.protected()
 (OUT/'result.json').write_text(json.dumps(dict(passed=True,training_updates=0,settings=settings(True),observed=observed,default_off_exact=True,unchanged_label_gradients_exact=True,signs_correct=True,parameters_unchanged=True,same_process_restore=True,wrong_switch_rejected=True,checkpoint_sha256=sha(OUT/'step0.pth'),code_sha256=codes,limitations='Two fixed views, final saved head as diagnostic input. Empty optimizer state; not resume of trained Adam, no training launch or AP claim.'),indent=2))
 print('GPU_NO_UPDATE_PASSED',json.dumps(observed))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--restore-check',action='store_true');args=parser.parse_args();OUT.mkdir(exist_ok=True)
 assert not (OUT/('fresh-restore.json' if args.restore_check else 'result.json')).exists()
 with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:main(args.restore_check)
  except BaseException:(OUT/('restore-failure.txt' if args.restore_check else 'failure.txt')).write_text(traceback.format_exc());raise
