"""CPU helper acceptance; no model or optimizer."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.pixel_bll import pixel_boundary_loss


def main():
    torch.set_num_threads(2)
    out=R/'reports/bll-pixel-helper-20261031';out.mkdir(exist_ok=False)
    tests=json.loads((R/'reports/bll-candidate-audit-20261030/single-pixel-tests.json').read_text())
    for row in tests:
        vh,vw=row['valid'];y,x=row['pixel'];hp=(vh+3)//4*4;wp=(vw+3)//4*4
        gt=torch.zeros(1,hp,wp);gt[0,y,x]=1
        z=torch.zeros(1,1,hp//4,wp//4,requires_grad=True);prev=torch.full_like(z,-4,requires_grad=True)
        loss,diag=pixel_boundary_loss(prev,z,[gt],[(torch.tensor([0]),torch.tensor([0]))],[(vh,vw)],loss_weight=1)
        assert torch.isfinite(loss) and diag['selected_pixels']>0
        loss.backward();assert torch.isfinite(z.grad).all() and prev.grad is None
    checks=[]
    folders=['bll-real-gradient-20261030'+('' if i==0 else f'-id{i}') for i in range(4)]
    folders += [f'bll-real-gradient-20261030-small-rank{i}' for i in range(3)]
    for folder in folders:
        a=np.load(R/'reports'/folder/'trace.npz')
        z=torch.tensor(a['final'],requires_grad=True);prev=torch.from_numpy(a['previous']);gt=torch.from_numpy(a['gt'])
        qi=torch.from_numpy(a['qi']);gi=torch.from_numpy(a['gi']);shape=tuple(map(int,a['valid_shape']))
        loss,diag=pixel_boundary_loss(prev,z,[gt],[(qi,gi)],[shape],loss_weight=1)
        gp,=torch.autograd.grad(loss,z)
        ref=torch.tensor(a['final'],requires_grad=True);hp,wp=gt.shape[-2:];h,w=ref.shape[-2:]
        sel=torch.from_numpy(a['region']).repeat_interleave(hp//h,1).repeat_interleave(wp//w,2)
        valid=torch.zeros(hp,wp,dtype=torch.bool);valid[:shape[0],:shape[1]]=True;sel &= valid
        up=F.interpolate(ref[0,qi,None],(hp,wp),mode='bilinear',align_corners=False)[:,0]
        expected=F.binary_cross_entropy_with_logits(up,gt[gi].float(),reduction='none')[sel].mean()
        gr,=torch.autograd.grad(expected,ref)
        torch.testing.assert_close(loss,expected,rtol=2e-6,atol=1e-7)
        torch.testing.assert_close(gp,gr,rtol=2e-5,atol=1e-8)
        checks.append(dict(folder=folder,loss=float(loss.detach()),selected=diag['selected_pixels']))
        print('TRACE_OK',folder,flush=True)
    # Two differently sized valid regions; global denominator, not image mean.
    gt=torch.zeros(1,8,8);gt[0,2,2]=1
    pair=(torch.tensor([0]),torch.tensor([0]))
    z=torch.zeros(2,1,2,2,requires_grad=True);z.data[1]=2
    prev=torch.full_like(z,-4)
    combined,diag=pixel_boundary_loss(prev,z,[gt,gt],[pair,pair],[(8,8),(5,5)],loss_weight=1)
    singles=[];counts=[]
    for i,shape in enumerate([(8,8),(5,5)]):
        v,d=pixel_boundary_loss(prev[i:i+1],z[i:i+1],[gt],[pair],[shape],loss_weight=1)
        singles.append(v);counts.append(d['selected_pixels'])
    torch.testing.assert_close(combined,sum(v*n for v,n in zip(singles,counts))/sum(counts))
    empty=torch.empty(0,dtype=torch.long)
    loss,_=pixel_boundary_loss(prev[:1],z[:1],[gt],[(empty,empty)],[(8,8)],loss_weight=1)
    assert float(loss)==0 and torch.equal(torch.autograd.grad(loss,z,retain_graph=True)[0],torch.zeros_like(z))
    disabled,_=pixel_boundary_loss(prev,z,[],[],[],loss_weight=0);assert float(disabled)==0
    rejected=0
    for kwargs in [dict(loss_weight=-1),dict(loss_weight=float('nan'))]:
        try:pixel_boundary_loss(prev[:1],z[:1],[gt],[pair],[(8,8)],**kwargs)
        except ValueError:rejected+=1
    bad=gt.clone();bad[0,7,7]=1
    try:pixel_boundary_loss(prev[:1],z[:1],[bad],[pair],[(5,5)],loss_weight=1)
    except ValueError:rejected+=1
    duplicate=(torch.tensor([0,0]),torch.tensor([0,0]))
    try:pixel_boundary_loss(prev[:1],z[:1],[gt],[duplicate],[(8,8)],loss_weight=1)
    except ValueError:rejected+=1
    assert rejected==4
    result=dict(passed=True,single_pixel_cases=len(tests),saved_trace_checks=checks,
        batch_normalization=True,empty_and_disabled=True,negative_controls=rejected,no_model_or_optimizer=True)
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('HELPER_PASS')


if __name__=='__main__':main()
