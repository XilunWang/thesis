"""Local artifact, score, label-log and final metric verification."""
import hashlib,json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT/'.local/bll-libs'),str(ROOT)]
import numpy as np
from scripts.position_grid4 import select
from scripts.coverage_pilot_metrics import compare
OUT=ROOT/'reports/coverage-priority-pilot-execution-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def array_sha(a):return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
 artifacts=load(OUT/'artifact-sha256.json')
 for name,h in artifacts.items():assert sha(OUT/name)==h,name
 proc=load(OUT/'process-result.json');result=load(OUT/'result.json');assert proc['exit_code']==0 and result['completed'] and result['step']==300 and result['original_preserved']
 steps=[json.loads(s) for s in (OUT/'steps.jsonl').read_text().splitlines()];traces=[json.loads(s) for s in (OUT/'training-labels.jsonl').read_text().splitlines()]
 assert [r['step'] for r in steps]==list(range(1,301));assert len(traces)==300
 assert all(math.isfinite(r[k]) for r in steps for k in ['loss','gradient_norm','elapsed_seconds']);assert all(a['elapsed_seconds']<=b['elapsed_seconds'] for a,b in zip(steps,steps[1:]))
 views=load(ROOT/'reports/position-learning-plan-grid4-20261101/views.json');viewmap={(r['image_id'],r['seed']):r for r in views['runs']};learn=[v for v in views['runs'] if v['split']=='learn'];plan=load(ROOT/'docs/coverage-priority-pilot-plan-20261101.json')
 for i,t in enumerate(traces):
  assert t['step_before']==i and t['attempted_update']==i+1 and len(t['views'])==2
  for j,v in enumerate(t['views']):
   visit=i*2+j;cycle,off=divmod(visit,len(learn));expected=learn[int(np.random.default_rng(plan['settings']['order_seed']+cycle).permutation(len(learn))[off])]
   assert (v['image_id'],v['seed'])==(expected['image_id'],expected['seed']) and v['split']=='learn'
   assert v['input_sha256']==expected['input_sha256'] and v['masks_sha256']==expected['masks_sha256']
   reps=v['representatives'];assert {str(r['gt_id']) for r in reps}==set(expected['assignments'])
   assert len({r['cell'] for r in reps})==len(reps) and all(1<=r['foreground_pixels']<=16 for r in reps)
   h,w=expected['padded_shape'];lab=np.zeros((h//4,w//4),np.float32)
   for r in reps:lab.flat[r['cell']]=1
   assert array_sha(lab)==v['label_sha256']
 evaluations={s:load(OUT/f'eval-{s}.json') for s in [0,100,300]}
 assert evaluations[0]['records']==load(ROOT/'reports/position-grid4-execution-20261101/eval-0.json')['records']
 for stage,ev in evaluations.items():
  index=load(OUT/f'full-scores-{stage}/index.json');assert len(index['runs'])==128 and index['checkpoint_sha256']==sha(OUT/f'step{stage}.pth')
  selected={(r['image_id'],r['augmentation_seed']):r['selected_cells'] for r in ev['records'] if r['split']=='holdout' and r['source']=='head' and r['k']==200}
  for r in index['runs']:
   v=viewmap[(r['image_id'],r['seed'])];assert v['input_sha256']==r['input_sha256'] and v['masks_sha256']==r['masks_sha256']
   p=OUT/f'full-scores-{stage}'/r['file'];assert sha(p)==r['sha256'];a=np.load(p);z=a['logits'];assert z.dtype==np.float32 and np.isfinite(z).all()
   keep=np.zeros_like(a['valid']);h,w=v['valid_shape'][:2];keep[:(h+3)//4,:(w+3)//4]=True;assert np.array_equal(keep,a['valid'])
   cells,_=select(z,v['valid_shape'],200);assert cells.tolist()==selected[(r['image_id'],r['seed'])]
 recomputed=compare(ROOT,evaluations[300],load(ROOT/'reports/label-only-pilot-execution-20261101/eval-300.json'),load(ROOT/'reports/position-grid4-execution-20261101/eval-300.json'),evaluations[0],plan,views)
 for k,v in recomputed.items():assert result[k]==v,k
 import torch
 initial=torch.load(OUT/'step0.pth',map_location='cpu',weights_only=False);old=torch.load(ROOT/'reports/position-grid4-execution-20261101/step0.pth',map_location='cpu',weights_only=False)
 assert all(torch.equal(v,old['head'][k]) for k,v in initial['head'].items()) and not initial['optimizer']['state']
 final=torch.load(OUT/'step300.pth',map_location='cpu',weights_only=False);latest=torch.load(OUT/'latest.pth',map_location='cpu',weights_only=False)
 assert final['step']==latest['step']==300 and final['cursor']==600 and final['kind']=='coverage_priority_pilot' and final['manifest_hash']==sha(ROOT/'docs/coverage-priority-pilot-plan-20261101.json')
 assert all(torch.equal(v,latest['head'][k]) for k,v in final['head'].items());assert final['optimizer']['state']
 proof=dict(passed=True,artifacts_verified=len(artifacts),training_steps=300,actual_label_visits=600,full_score_views_verified=384,initial_weights_exact=True,initial_adam_empty=True,initial_records_exact=True,final_metrics_recomputed=True,latest_final_equal=True,recorded_gt_pixel_counts_not_independently_redecoded=True,script_sha256=sha(Path(__file__)))
 (OUT/'local-verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
if __name__=='__main__':main()
