"""All 384 cached labels plus local CPU forward/backward; zero updates."""
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.local/analysis-libs'),str(ROOT/'.local/bll-libs')]
import numpy as np
import torch
from pycocotools import mask as mu
from scripts.position_grid4 import Grid4PositionHead, make_targets, position_loss, select, REPRESENTATION
SRC=ROOT/'reports/position-learning-preflight-20261101'
OUT=ROOT/'reports/position-grid4-cache-20261101'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arrsha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()


def main(result):
    torch.set_num_threads(2);torch.manual_seed(20261101)
    proof=json.loads((SRC/'result.json').read_text());assert len(proof['runs'])==384
    capacity=json.loads((ROOT/'reports/position-grid-comparison-20261101/result.json').read_text())
    spec=importlib.util.spec_from_file_location('comparison',ROOT/'scripts/compare-position-grid-capacity.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    rows=[];total=0;neighbors=[];distinct_pairs=set();hashes={}
    for i,r in enumerate(proof['runs']):
        p=SRC/f"{r['split']}-{r['image_id']}-{r['seed']}.json";d=json.loads(p.read_text())
        hashes[p.name]=sha(p);assert hashes[p.name]==capacity['source_hashes'][p.name]
        h,w=d['padded_shape'];vh,vw=d['valid_shape'][:2]
        masks=mu.decode(d['gt_rles']).transpose(2,0,1) if d['gt_rles'] else np.zeros((0,h,w),dtype=np.uint8)
        mask_hash=arrsha(masks)
        target,valid,assignment=make_targets(masks,d['annotation_ids'],d['original_areas'],d['valid_shape'])
        assert arrsha(masks)==mask_hash==d['masks_sha256']
        matched,witness=reference.module.assign(reference.candidates_for(masks,d['annotation_ids'],d['original_areas'],4))
        assert witness['deficit']==0
        assert assignment=={gid:[cell%(w//4),cell//(w//4)] for gid,(cell,slot) in matched.items()}
        assert target.shape==(h//4,w//4) and int(valid.sum())==((vh+3)//4)*((vw+3)//4)
        assert len(assignment)==sum(0<a<=1024 for a in d['original_areas'])==int(target.sum())
        total+=len(assignment)
        # All valid cells and physical coordinates are checked, not only GT-positive ones.
        flat,xy=select(np.zeros(target.shape,dtype=np.float32),d['valid_shape'],target.numel())
        assert np.array_equal(flat,np.flatnonzero(valid.numpy()))
        cx=flat%(w//4);cy=flat//(w//4)
        expected=np.stack(((cx*4+np.minimum(cx*4+4,vw))/(2*vw),(cy*4+np.minimum(cy*4+4,vh))/(2*vh)),axis=1)
        assert np.array_equal(xy,expected) and np.all((xy>=0)&(xy<1))
        logits=torch.zeros_like(target)[None].requires_grad_()
        loss=position_loss(logits,target[None],valid[None]);loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and not logits.grad[~valid[None]].any()
        corrupted=logits.detach().clone();corrupted[~valid[None]]=float('nan')
        assert torch.equal(loss.detach(),position_loss(corrupted,target[None],valid[None]))
        # Reference oracle only verifies the selector does not discard adjacent positives.
        picked,_=select(target.numpy(),d['valid_shape'],100)
        assert int(target.flatten()[picked].sum())==min(100,len(assignment))
        row=dict(split=r['split'],image_id=r['image_id'],seed=r['seed'],positives=len(assignment),
                 input_sha256=d['input_sha256'],masks_sha256=mask_hash,valid_shape=d['valid_shape'],padded_shape=d['padded_shape'],
                 assignments=assignment,target_sha256=arrsha(target.numpy()),valid_sha256=arrsha(valid.numpy()))
        rows.append(row)
        # Freeze the OLD 8-pixel neighborhood observations, not a newly smaller neighborhood.
        if r['split']=='holdout':
            old={int(k):v for k,v in d['assignments'].items()};keys=sorted(old);pairs=[]
            for j,a in enumerate(keys):
                for b in keys[j+1:]:
                    if max(abs(old[a][q]-old[b][q]) for q in [0,1])<=1:
                        pairs.append([a,b]);distinct_pairs.add((a,b))
            if pairs:neighbors.append(dict(image_id=r['image_id'],seed=r['seed'],pairs=pairs))
        if (i+1)%64==0:print('CACHE_CHECKED',i+1,flush=True)
    assert total==2350 and len(distinct_pairs)==46 and len({r['image_id'] for r in neighbors})==8
    # Verify the real module's four learned output channels and exact spatial ordering.
    head=Grid4PositionHead();assert sum(p.numel() for p in head.parameters())==344964
    with torch.no_grad():
        head.predictor[-1].weight.zero_();head.predictor[-1].bias.copy_(torch.tensor([1.,2.,3.,4.]))
        output=head(torch.zeros(1,512,4,4),torch.zeros(1,1024,2,2))
    assert torch.equal(output[0],torch.tensor([[1.,2.],[3.,4.]]).repeat(4,4))
    # Standalone valid edges, adjacency, invalid scores and remaining capacity guard.
    grid=np.full((3,3),np.nan);grid[:2,:2]=0
    assert select(grid,[5,5],100)[0].tolist()==[0,1,3,4]
    assert select(np.array([[9.,8.],[1.,0.]]),[8,8],2)[0].tolist()==[0,1]
    grid[0,0]=np.nan
    try:select(grid,[5,5])
    except ValueError:pass
    else:raise AssertionError('Nonfinite valid score accepted')
    try:make_targets(np.ones((2,4,4),dtype=np.uint8),[1,2],[1,1],[4,4])
    except ValueError:pass
    else:raise AssertionError('Unrepresentable targets silently accepted')
    head=Grid4PositionHead();initial={k:v.clone() for k,v in head.state_dict().items()}
    c3=torch.randn(1,512,8,8,requires_grad=True);c4=torch.randn(1,1024,4,4,requires_grad=True)
    logits=head(c3,c4);targets=torch.zeros_like(logits);targets[0,4,4]=targets[0,4,5]=1
    loss=position_loss(logits,targets,torch.ones_like(targets,dtype=torch.bool));loss.backward()
    assert c3.grad is None and c4.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in head.parameters())
    assert all(torch.equal(initial[k],v) for k,v in head.state_dict().items())
    head.zero_grad(set_to_none=True)
    empty=head(c3,c4);loss=position_loss(empty,torch.zeros_like(empty),torch.ones_like(empty,dtype=torch.bool));loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in head.parameters())
    for val in [-1000.,1000.]:
        x=torch.full((1,2,2),val,requires_grad=True);t=torch.tensor([[[1.,0.],[0.,0.]]])
        l=position_loss(x,t,torch.ones_like(t,dtype=torch.bool));l.backward();assert torch.isfinite(l) and torch.isfinite(x.grad).all()
    result.update(passed=True,representation=REPRESENTATION,views=384,targets=total,parameters=344964,
                  runs=rows,frozen_neighbor_observations=neighbors,distinct_neighbor_pairs=len(distinct_pairs),
                  channel_mapping_exact=True,all_padding_gradients_zero=True,coordinates_exact=True,
                  head_gradients_finite=True,detached_inputs=True,parameters_unchanged=True,
                  source_hashes=hashes,code_sha256={n:sha(ROOT/'scripts'/n) for n in ['position_grid4.py','isolated_position_head.py','identity_data_pipeline.py','verify-position-grid4-cache.py']})


if __name__=='__main__':
    OUT.mkdir(exist_ok=False);result=dict(passed=False,training_updates=0)
    try:main(result)
    except BaseException:result['error']=traceback.format_exc();raise
    finally:(OUT/'result.json').write_text(json.dumps(result,indent=2))
