"""Local single-factor proposal: adaptive labels with unchanged legacy BCE."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT/'.local/bll-libs'),str(ROOT)]
import numpy as np
import torch
from pycocotools import mask as mu
from scripts.instance_representative import make_layout
from scripts.isolated_position_head import position_loss


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    source=ROOT/'reports/instance-representative-cache-20261101/result.json'
    reference=json.loads(source.read_text())
    views_path=ROOT/'reports/position-learning-plan-grid4-20261101/views.json'
    assert sha(views_path)==reference['source_sha256'][str(views_path.relative_to(ROOT))]
    views={(v['image_id'],v['seed']):v for v in json.loads(views_path.read_text())['runs']}
    score_source=ROOT/'reports/position-grid4-coral-scores-20261101'
    score_proof=json.loads((score_source/'result.json').read_text())
    score_hash={r['file']:r['sha256'] for r in score_proof['runs']}
    runs=[]
    for row in reference['scored']+reference['geometry_stress']:
        v=views[(row['image_id'],row['seed'])]
        path=ROOT/'reports/position-learning-preflight-20261101'/f"{v['split']}-{row['image_id']}-{row['seed']}.json"
        assert sha(path)==reference['source_sha256'][str(path.relative_to(ROOT))]
        d=json.loads(path.read_text());masks=mu.decode(d['gt_rles']).transpose(2,0,1)
        gh,gw=np.array(d['padded_shape'])//4
        valid=np.zeros((gh,gw),dtype=bool);vh,vw=d['valid_shape'][:2];valid[:(vh+3)//4,:(vw+3)//4]=True
        if 'stage' in row:
            name=f"image-{row['image_id']}-seed-{row['seed']}.npz"
            assert sha(score_source/name)==score_hash[name]
            arrays=np.load(score_source/name,allow_pickle=False)
            scores=arrays[f"logits_{row['stage']}"]
        else:
            scores=np.arange(gh*gw,dtype=float).reshape(gh,gw)/(gh*gw)
            if row['synthetic_priority']=='row_major':scores=-scores
        layout=make_layout(scores,masks,d['annotation_ids'],d['original_areas'],valid)
        reps=layout['representatives']
        assert {str(g):c for g,c in reps.items()}==row['representatives']
        old=torch.zeros((1,gh,gw));new=torch.zeros_like(old)
        for x,y in v['assignments'].values():old[0,y,x]=1
        for c in reps.values():new.reshape(-1)[c]=1
        keep=torch.from_numpy(valid).unsqueeze(0)
        assert int(old.sum())==int(new.sum())==len(reps)
        unchanged=(old==new)&keep
        for dtype in [torch.float32,torch.float64]:
            z=torch.tensor(scores,dtype=dtype).unsqueeze(0).requires_grad_()
            oldloss=position_loss(z,old.to(dtype),keep)
            newloss=position_loss(z,new.to(dtype),keep)
            oldgrad=torch.autograd.grad(oldloss,z)[0]
            newgrad=torch.autograd.grad(newloss,z)[0]
            assert torch.equal(oldgrad[unchanged],newgrad[unchanged])
            assert torch.isfinite(newloss) and torch.isfinite(newgrad).all()
            assert (newgrad[(new==1)&keep]<0).all()
            assert (newgrad[(new==0)&keep]>0).all()
            assert not newgrad[~keep].any()
            # Partition includes every valid negative exactly once, no ignored foreground.
            negatives=set(layout['outside'])|{c for cells in layout['groups'].values() for c in cells}
            expected=set(map(int,torch.nonzero(((new==0)&keep).reshape(-1)).reshape(-1).tolist()))
            assert negatives==expected and not negatives&set(reps.values())
        for pair in row['pairs']:
            a,b=pair['ids'];assert reps[a]!=reps[b]
            assert float(newgrad.reshape(-1)[reps[a]])<0 and float(newgrad.reshape(-1)[reps[b]])<0
        runs.append(dict(image_id=row['image_id'],seed=row['seed'],stage=row.get('stage'),
            synthetic_priority=row.get('synthetic_priority'),targets=len(reps),
            positive_cells_replaced=int(((new==1)&(old==0)).sum()),
            unchanged_cells=int(unchanged.sum()),negative_cells=len(negatives),pairs=len(row['pairs'])))
    # Empty-image weighting remains the original full negative mean, not 0.25.
    z=torch.tensor([[[1.,-1.]]],requires_grad=True);keep=torch.ones_like(z,dtype=torch.bool)
    empty=torch.zeros_like(z)
    value=position_loss(z,empty,keep)
    expected=torch.nn.functional.softplus(z).mean()
    assert torch.allclose(value,expected,atol=1e-7,rtol=1e-6)
    result=dict(passed=True,training_updates=0,cached_checks=len(runs),dtype_checks=len(runs)*2,
        scope='Label-only prototype audit, not a trained model or GPU parameter-direction check',
        representative_count_preserved=True,all_nonrepresentative_valid_cells_negative=True,
        unchanged_label_logit_gradients_exact=True,empty_image_legacy_weight=True,
        positive_logit_signs_correct=True,padding_gradients_zero=True,
        pair_checks=sum(r['pairs'] for r in runs),runs=runs,
        source_sha256={str(source.relative_to(ROOT)):sha(source),str(views_path.relative_to(ROOT)):sha(views_path)},
        code_sha256={n:sha(ROOT/'scripts'/n) for n in ['check-representative-label-only.py','instance_representative.py','isolated_position_head.py']})
    out=ROOT/'reports/representative-label-only-20261101';out.mkdir(exist_ok=True)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','source_sha256','code_sha256']},indent=2))


if __name__=='__main__':main()
