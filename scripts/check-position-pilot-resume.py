"""CPU roundtrip in a new process; real step0 and explicitly synthetic moment state."""
import json
import random
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.isolated_position_head import IsolatedPositionHead
from scripts.position_pilot_runtime import digest, save_checkpoint, load_checkpoint, view_at
OUT = ROOT/'reports/position-pilot-resume-20261101'
PLAN = ROOT/'reports/position-learning-plan-20261101/manifest.json'


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def make():
    plan=json.loads(PLAN.read_text()); h=IsolatedPositionHead()
    o=torch.optim.AdamW(h.parameters(),lr=plan['optimizer']['lr'],weight_decay=plan['optimizer']['weight_decay'],
                       betas=tuple(plan['optimizer']['betas']),eps=plan['optimizer']['eps'])
    return h,o


def draws():
    return [random.random(),float(np.random.random()),torch.rand(3).tolist()]


def forbidden(*args,**kwargs):
    raise AssertionError('Optimizer step forbidden during acceptance')


def restore(kind):
    fixture=torch.load(OUT/f'{kind}-expected.pth',map_location='cpu')
    h,o=make(); opts=fixture['options']
    data=load_checkpoint(OUT/f'{kind}.pth',h,o,**opts)
    assert draws()==fixture['draws']
    with torch.no_grad():
        assert torch.equal(h(*fixture['features']),fixture['prediction'])
    assert equal(h.state_dict(),fixture['head']) and equal(o.state_dict(),fixture['optimizer'])
    assert view_at(fixture['views'],data['cursor'],20261102)==fixture['next_view']
    assert {id(p) for g in o.param_groups for p in g['params']}=={id(p) for p in h.parameters()}
    for key in ['manifest_hash','view_hash','kind','code_hashes']:
        bad=dict(opts);bad[key]='wrong'
        try:load_checkpoint(OUT/f'{kind}.pth',h,o,**bad)
        except ValueError:pass
        else:raise AssertionError('Accepted incompatible checkpoint')
    print(kind,'CROSS_PROCESS_EXACT',flush=True)


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2)
    random.seed(20261101);np.random.seed(20261101);torch.manual_seed(20261101)
    head,opt=make();initial={k:v.clone() for k,v in head.state_dict().items()}
    features=[torch.randn(1,512,8,8),torch.randn(1,1024,4,4)]
    with torch.no_grad():pred=head(*features)
    views=[(i,s) for i in range(5) for s in range(2)]
    codes={n:digest(ROOT/'scripts'/n) for n in ['isolated_position_head.py','position_pilot_runtime.py']}
    for kind,step in [('pilot',0),('synthetic_optimizer_fixture',7)]:
        if step:
            # Fabricated nonzero Adam moments exercise serialization, not a training result.
            for p in head.parameters():
                opt.state[p]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(p,0.125),exp_avg_sq=torch.full_like(p,0.25))
        options=dict(manifest_hash=digest(PLAN),code_hashes=codes,view_hash=digest(PLAN),kind=kind)
        save_checkpoint(OUT/f'{kind}.pth',head,opt,**options,step=step,cursor=step*2,elapsed_seconds=0)
        expected=dict(options=options,draws=draws(),features=features,prediction=pred,
                      head=head.state_dict(),optimizer=opt.state_dict(),views=views,next_view=view_at(views,step*2,20261102))
        torch.save(expected,OUT/f'{kind}-expected.pth')
        subprocess.run([sys.executable,'-B',str(Path(__file__)), '--restore',kind],check=True)
    assert equal(initial,head.state_dict())
    result=dict(passed=True,training_updates=0,head_parameters_unchanged=True,
                real_step0_restored=True,synthetic_nonzero_moments_restored=True,
                cross_process_rng_and_cursor_exact=True,incompatible_checkpoints_rejected=True,
                limitation='CPU serialization only; no actual trained Adam moments, GPU optimizer update or full training interruption tested',
                source_sha256={n:digest(ROOT/'scripts'/n) for n in ['isolated_position_head.py','position_pilot_runtime.py','check-position-pilot-resume.py']})
    (OUT/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    torch.optim.AdamW.step=forbidden
    if len(sys.argv)>1:
        assert sys.argv[1]=='--restore';restore(sys.argv[2])
    else:main()
