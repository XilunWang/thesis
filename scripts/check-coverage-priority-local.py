"""CPU boundary cases: empty, one pixel, adjacent conflict, padding."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT/'.local/bll-libs'),str(ROOT)]
import numpy as np
import torch
from scripts.coverage_priority_objective import objective,settings_hash
from scripts.representative_label_only import objective as previous
from scripts.position_grid4 import make_targets

def main():
 torch.set_num_threads(2);runs=[]
 for name in ['empty','one_pixel','adjacent_fallback']:
  n=0 if name=='empty' else 1 if name=='one_pixel' else 2
  m=np.zeros((n,8,12),np.uint8)
  if name=='one_pixel':m[0,0,0]=1
  if name=='adjacent_fallback':m[0,:4,:4]=1;m[0,0,4]=1;m[1,0,1]=1
  ids=list(range(1,n+1));areas=m.sum((1,2)).tolist()
  t,v,_=make_targets(m,ids,areas,(4,8));items=[dict(masks=m,ids=ids,areas=areas)]
  for dtype in [torch.float32,torch.float64]:
   z=torch.linspace(-2,2,6,dtype=dtype).reshape(1,2,3).requires_grad_();target=t[None].to(dtype);valid=v[None]
   old,details=previous(z,target,valid,items,enabled=True);off,offdetails=objective(z,target,valid,items)
   assert torch.equal(old,off) and torch.equal(details['labels'],offdetails['labels'])
   oldg=torch.autograd.grad(old,z,retain_graph=True)[0];assert torch.equal(oldg,torch.autograd.grad(off,z,retain_graph=True)[0])
   on,info=objective(z,target,valid,items,coverage_priority=True);g=torch.autograd.grad(on,z)[0];lab=info['labels'];assert int(lab.sum())==n
   assert torch.isfinite(on) and torch.isfinite(g).all() and not g[~valid].any()
   assert (g[(lab==1)&valid]<0).all() and (g[(lab==0)&valid]>0).all()
   same=(lab==details['labels'])&valid;assert torch.equal(g[same],oldg[same])
   if name=='empty':assert torch.equal(on,old) and torch.equal(on,torch.nn.functional.binary_cross_entropy_with_logits(z[valid],torch.zeros_like(z[valid]),reduction='none').mean())
   reps=info['layouts'][0]['representatives'];assert len(reps)==len(set(reps.values()))==n
   runs.append(dict(case=name,dtype=str(dtype),positive_count=n,passed=True))
 assert settings_hash(False)!=settings_hash(True)
 out=ROOT/'reports/coverage-priority-integration-20261101';out.mkdir(exist_ok=True);(out/'local.json').write_text(json.dumps(dict(passed=True,training_updates=0,runs=runs),indent=2));print('LOCAL_PASSED',len(runs))
if __name__=='__main__':main()
