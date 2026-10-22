"""Foreground preservation, valid-edge handling and real BLL analytic gradients."""
import hashlib
import json
from pathlib import Path
import torch
from query_bll import build_bll_targets, query_boundary_loss, TARGET_RULE


def run(device):
    count=0
    # Exhaust every source-pixel position, including partial bottom/right blocks,
    # an image smaller than the stride and unequal height/width strides.
    for hp,wp,h,w,vh,vw in [(8,12,2,3,7,9),(8,12,2,3,2,2),(6,12,3,3,5,11)]:
        masks=torch.zeros(vh*vw,hp,wp,device=device)
        for y in range(vh):
            for x in range(vw): masks[y*vw+x,y,x]=1
        snapshot=masks.clone()
        target,valid=build_bll_targets(masks,(h,w),(vh,vw))
        expected=torch.zeros_like(target)
        for y in range(vh):
            for x in range(vw): expected[y*vw+x,y//(hp//h),x//(wp//w)]=1
        assert torch.equal(target,expected) and torch.equal(masks,snapshot)
        assert (target.flatten(1).sum(1)==1).all()
        # Background previous predictions: each isolated occupied cell should
        # receive a foreground BCE gradient, including partial edge cells.
        previous=torch.full((1,len(masks),h,w),-10.,device=device,requires_grad=True)
        final=torch.zeros_like(previous,requires_grad=True)
        idx=torch.arange(len(masks),device=device)
        loss,diag=query_boundary_loss(previous,final,[masks],[(idx,idx)],[(vh,vw)],loss_weight=1)
        loss.backward()
        foreground=target.bool()
        assert (final.grad[0][foreground]<0).all()
        torch.testing.assert_close(final.grad[0][foreground],torch.full_like(final.grad[0][foreground],-.5/diag['selected_pixels']))
        assert not final.grad[0][:,~valid].any() and previous.grad is None
        assert torch.isfinite(loss) and torch.isfinite(final.grad).all()
        empty,keep=build_bll_targets(masks[:0],(h,w),(vh,vw))
        assert empty.shape==(0,h,w) and torch.equal(keep,valid)
        count+=len(masks)
    bad=torch.zeros(1,8,8,device=device);bad[0,7,7]=1
    try: build_bll_targets(bad,(2,2),(7,7))
    except ValueError: pass
    else: raise AssertionError('Nonzero padding accepted')
    return dict(device=device,passed=True,exhaustive_single_pixel_cases=count)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2)
    result=dict(passed=False,target_rule=TARGET_RULE,torch=torch.__version__)
    try:
        result['results']=[run('cpu')]+([run('cuda:0')] if args.cuda else [])
        result['passed']=True
    finally:
        root=Path(__file__).resolve().parents[1]
        result['sources']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/query_bll.py','scripts/bll_core.py','scripts/validate-bll-targets.py']}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
