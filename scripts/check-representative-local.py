"""Actual torch/autograd objective against independent cached numpy reference."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT/'.local/bll-libs'),str(ROOT)]
import numpy as np
import torch
from pycocotools import mask as mu
from scripts.instance_representative import objective,make_layout,terms_for_layout,WEIGHTS
from scripts.isolated_position_head import position_loss


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('reference',ROOT/'scripts/check-instance-representative-cache.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    reference=json.loads((ROOT/'reports/instance-representative-cache-20261101/result.json').read_text())
    frozen=json.loads((ROOT/'reports/position-learning-plan-grid4-20261101/views.json').read_text())
    views={(v['image_id'],v['seed']):v for v in frozen['runs']}
    runs=[]
    for row in reference['scored']+reference['geometry_stress']:
        v=views[(row['image_id'],row['seed'])]
        path=ROOT/'reports/position-learning-preflight-20261101'/f"{v['split']}-{row['image_id']}-{row['seed']}.json"
        assert sha(path)==reference['source_sha256'][str(path.relative_to(ROOT))]
        d=json.loads(path.read_text());masks=mu.decode(d['gt_rles']).transpose(2,0,1)
        gh,gw=np.array(d['padded_shape'])//4
        valid=np.zeros((gh,gw),dtype=bool);vh,vw=d['valid_shape'][:2];valid[:(vh+3)//4,:(vw+3)//4]=True
        if 'stage' in row:
            arrays=np.load(ROOT/'reports/position-grid4-coral-scores-20261101'/f"image-{row['image_id']}-seed-{row['seed']}.npz")
            scores=arrays[f"logits_{row['stage']}"]
        else:
            scores=np.arange(gh*gw,dtype=float).reshape(gh,gw)/(gh*gw)
            if row['synthetic_priority']=='row_major':scores=-scores
        support={int(g):set(map(int,np.flatnonzero(m.reshape(gh,4,gw,4).any(axis=(1,3)))))
                 for g,a,m in zip(d['annotation_ids'],d['original_areas'],masks) if 0<a<=1024}
        reps=ref.allocate(support,scores.ravel());groups=ref.duplicate_groups(support,reps,int(gw))
        assert {str(g):c for g,c in reps.items()}==row['representatives']
        item=dict(masks=masks,ids=d['annotation_ids'],areas=d['original_areas'])
        maxerror=0.
        for dtype in [torch.float64,torch.float32]:
            z=torch.tensor(scores,dtype=dtype).unsqueeze(0).requires_grad_()
            keep=torch.from_numpy(valid).unsqueeze(0)
            target=torch.zeros_like(z)
            for x,y in v['assignments'].values():target[0,y,x]=1
            a,_=objective(z,target,keep,enabled=False)
            b=position_loss(z,target,keep)
            assert torch.equal(a,b)
            assert torch.equal(torch.autograd.grad(a,z)[0],torch.autograd.grad(b,z)[0])
            value,details=objective(z,target,keep,[item],enabled=True)
            actual_layout=details['layouts'][0]
            assert actual_layout['representatives']==reps and actual_layout['groups']==groups
            values,grads=ref.objective_parts(z.detach().numpy().ravel().astype(float),valid.ravel(),reps,groups)
            expected=sum(WEIGHTS[k]*g for k,g in grads.items()).reshape(gh,gw)
            grad=torch.autograd.grad(value,z)[0][0].numpy()
            error=float(np.max(np.abs(grad-expected)));maxerror=max(maxerror,error)
            assert np.allclose(grad,expected,rtol=1e-5,atol=1e-7)
            assert np.isclose(float(value.detach()),sum(WEIGHTS[k]*v for k,v in values.items()),rtol=1e-5,atol=1e-7)
            assert not grad[~valid].any()
            assert all(grad.ravel()[c]<0 for c in reps.values())
            assert all(grad.ravel()[c]>0 for cells in groups.values() for c in cells)
        runs.append(dict(image_id=row['image_id'],seed=row['seed'],stage=row.get('stage'),
                         synthetic_priority=row.get('synthetic_priority'),targets=len(reps),max_gradient_abs_error=maxerror))
    # Empty image, one-cell target, impossible overlap and invalid padding.
    masks=np.zeros((0,8,8),dtype=np.uint8)
    z=torch.tensor([[[2.,-2.],[float('nan'),float('nan')]]],requires_grad=True)
    valid=torch.tensor([[[True,True],[False,False]]])
    value,_=objective(z,torch.zeros_like(z),valid,[dict(masks=masks,ids=[],areas=[])],enabled=True)
    value.backward();assert torch.isfinite(value) and torch.isfinite(z.grad).all() and not z.grad[~valid].any()
    m=np.zeros((1,8,8),dtype=np.uint8);m[0,0,0]=1
    layout=make_layout(np.zeros((2,2)),m,[1],[1],valid[0].numpy())
    assert layout['representatives']=={1:0} and layout['groups']=={1:[]}
    rejected=False
    try:make_layout(np.zeros((2,2)),np.concatenate([m,m]),[1,2],[1,1],valid[0].numpy())
    except ValueError:rejected=True
    assert rejected
    out=ROOT/'reports/representative-integration-20261101';out.mkdir(exist_ok=True)
    result=dict(passed=True,training_updates=0,weights=WEIGHTS,cached_checks=len(runs),
        dtype_checks=len(runs)*2,default_off_loss_and_gradient_exact=True,
        representative_gradients_negative=True,duplicate_gradients_positive=True,
        no_positive_negative_conflicts=True,padding_gradients_zero=True,
        empty_and_single_cell_passed=True,insufficient_capacity_rejected=True,
        max_gradient_abs_error=max(r['max_gradient_abs_error'] for r in runs),runs=runs,
        source_reference_sha256=sha(ROOT/'reports/instance-representative-cache-20261101/result.json'),
        code_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'scripts/instance_representative.py']})
    (out/'local-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


if __name__=='__main__':main()
