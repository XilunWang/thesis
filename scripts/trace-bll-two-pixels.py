"""Exact bilinear coupling attribution for two frozen FN pixels. No updates."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R=Path(__file__).resolve().parents[1]


def weights(y,x,h,w,hp,wp):
    sy=(y+.5)*h/hp-.5;sx=(x+.5)*w/wp-.5
    iy=math.floor(sy);ix=math.floor(sx);dy=sy-iy;dx=sx-ix
    result={}
    for yy,wy in [(iy,1-dy),(iy+1,dy)]:
        for xx,wx in [(ix,1-dx),(ix+1,dx)]:
            key=(min(max(yy,0),h-1),min(max(xx,0),w-1))
            result[key]=result.get(key,0)+wy*wx
    return result


def main():
    torch.set_num_threads(2)
    out=R/'reports/bll-two-pixels-20261031';out.mkdir(exist_ok=False)
    folder=R/'reports/bll-real-gradient-20261030-small-rank0'
    meta=json.loads((folder/'metadata.json').read_text())
    assert hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest()==meta['trace_sha256']
    a=np.load(folder/'trace.npz');j=int(np.where(a['qi']==38)[0][0]);gi=int(a['gi'][j])
    gt=torch.from_numpy(a['gt'][gi]).bool();hp,wp=gt.shape
    z=torch.tensor(a['final'][0,38],dtype=torch.float64,requires_grad=True);h,w=z.shape
    up=F.interpolate(z[None,None],(hp,wp),mode='bilinear',align_corners=False)[0,0]
    valid=torch.zeros_like(gt);vh,vw=map(int,a['valid_shape']);valid[:vh,:vw]=True
    selected=torch.from_numpy(a['region'][j]).repeat_interleave(hp//h,0).repeat_interleave(wp//w,1)&valid
    loss=F.binary_cross_entropy_with_logits(up,gt.double(),reduction='none')[selected].mean()
    grad,=torch.autograd.grad(loss,z)
    direction=-F.interpolate(grad[None,None],(hp,wp),mode='bilinear',align_corners=False)[0,0]
    bad=torch.nonzero(selected&gt&(up.detach()<0)&(direction<0)).tolist();assert len(bad)==2
    probs=up.detach().sigmoid().numpy();denom=int(selected.sum());rows=[]
    for y,x in bad:
        coeff=weights(y,x,h,w,hp,wp)
        manual_logit=sum(v*float(z.detach()[yy,xx]) for (yy,xx),v in coeff.items())
        assert abs(manual_logit-float(up.detach()[y,x]))<1e-10
        # Independent sparse interpolation overlap K_pq = sum_j A_pj A_qj.
        coupling=np.zeros((hp,wp),dtype=np.float64)
        for yy in range(max(0,y-12),min(hp,y+13)):
            for xx in range(max(0,x-12),min(wp,x+13)):
                other=weights(yy,xx,h,w,hp,wp)
                coupling[yy,xx]=sum(v*other.get(k,0) for k,v in coeff.items())
        chosen=selected.numpy()&(coupling>0);fg=chosen&gt.numpy();bg=chosen&~gt.numpy()
        contributions=coupling*(probs-gt.numpy())*selected.numpy()/denom
        fg_up=float(-contributions[fg].sum());bg_down=float(contributions[bg].sum())
        np.testing.assert_allclose(contributions.sum(),-float(direction[y,x]),rtol=1e-10,atol=1e-12)
        ys,xs=np.where(chosen)
        top=sorted(zip(*np.where(bg)),key=lambda p:contributions[p],reverse=True)[:5]
        row=dict(pixel_yx=[y,x],gt=bool(gt[y,x]),selected=bool(selected[y,x]),probability=float(probs[y,x]),
            nodes=[dict(grid_yx=list(k),weight=v,logit=float(z.detach()[k])) for k,v in coeff.items()],
            contributing_bbox_yxyx=[int(ys.min()),int(xs.min()),int(ys.max()),int(xs.max())],
            shared_selected_fg=int(fg.sum()),shared_selected_bg=int(bg.sum()),
            fg_raise=fg_up,bg_lower=bg_down,net_direction=fg_up-bg_down,
            fg_multiplier_required_for_positive_direction=bg_down/fg_up,
            current_balanced_fg_multiplier=int((selected&~gt).sum())/int((selected&gt).sum()),
            without_shared_background_direction=fg_up,
            independent_highres_pixel_direction=(1-float(probs[y,x]))/denom,
            top_background=[dict(pixel_yx=[int(yy),int(xx)],probability=float(probs[yy,xx]),
                                 coupling=float(coupling[yy,xx]),contribution=float(contributions[yy,xx])) for yy,xx in top])
        rows.append(row)
    result=dict(passed=True,image_id=7,query=38,gt_index=gi,trace_sha256=meta['trace_sha256'],
        selected_fg=int((selected&gt).sum()),selected_bg=int((selected&~gt).sum()),rows=rows,
        scope='Exact local derivative in frozen native-logit space. Background omission and independent highres coordinates are algebraic counterfactuals, not deployable fixes or updated predictions.')
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
