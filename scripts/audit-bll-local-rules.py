"""Two fixed, symmetric local error rules on all seven saved training traces."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R=Path(__file__).resolve().parents[1]
RULES=('ordinary','error_double','error_only')


def main():
    torch.set_num_threads(2)
    out=R/'reports/bll-local-rules-20261031';out.mkdir(exist_ok=False)
    folders=['bll-real-gradient-20261030'+('' if i==0 else f'-id{i}') for i in range(4)]
    folders += [f'bll-real-gradient-20261030-small-rank{i}' for i in range(3)]
    rows=[];sources=[]
    for folder in folders:
        path=R/'reports'/folder;meta=json.loads((path/'metadata.json').read_text())
        assert hashlib.sha256((path/'trace.npz').read_bytes()).hexdigest()==meta['trace_sha256']
        sources.append(dict(folder=folder,sha256=meta['trace_sha256']))
        a=np.load(path/'trace.npz');vh,vw=map(int,a['valid_shape'])
        for j,gi in enumerate(a['gi']):
            gt=torch.from_numpy(a['gt'][gi]).bool();hp,wp=gt.shape
            z=torch.tensor(a['final'][0,a['qi'][j]],requires_grad=True);h,w=z.shape
            up=F.interpolate(z[None,None],(hp,wp),mode='bilinear',align_corners=False)[0,0]
            valid=torch.zeros_like(gt);valid[:vh,:vw]=True
            sel=torch.from_numpy(a['region'][j]).repeat_interleave(hp//h,0).repeat_interleave(wp//w,1)&valid
            pos=sel&gt;neg=sel&~gt;pred=up.detach()>=0;wrong=pred!=gt
            groups=dict(fn=pos&~pred,fp=neg&pred,tp=pos&pred,tn=neg&~pred)
            lossmap=F.binary_cross_entropy_with_logits(up,gt.float(),reduction='none')
            row=dict(image_id=meta['image_id'],query=int(a['qi'][j]),area=int(gt.sum()),
                     size='small' if int(gt.sum())<1024 else 'non_small',rules={})
            for name in RULES:
                weight=sel.float()
                if name=='error_double':weight=weight*(1+wrong.float())
                if name=='error_only':weight=weight*wrong
                denom=weight.sum().clamp(min=1)
                loss=(lossmap*weight).sum()/denom
                g,=torch.autograd.grad(loss,z,retain_graph=True)
                expected=(up.detach().sigmoid()-gt.float())*weight/denom
                oracle,=torch.autograd.grad(up,z,grad_outputs=expected,retain_graph=True)
                torch.testing.assert_close(g,oracle,rtol=2e-5,atol=1e-8)
                assert torch.isfinite(g).all()
                norm=g.norm();direction=-F.interpolate((g/norm.clamp(min=1e-30))[None,None],(hp,wp),mode='bilinear',align_corners=False)[0,0]
                stats={}
                for label,mask in groups.items():
                    d=direction[mask]*(1 if label in ('fn','tp') else -1)
                    stats[label]=dict(count=int(mask.sum()),correct_direction=int((d>1e-10).sum()),
                        opposite_direction=int((d< -1e-10).sum()),sum_desired_direction=float(d.sum()))
                row['rules'][name]=dict(groups=stats,selected_positive=int(((weight>0)&gt).sum()),
                    selected_negative=int(((weight>0)&~gt).sum()),gradient_norm=float(norm),
                    known_two_directions=[float(direction[y,x]) for y,x in [(622,1007),(623,1007)]] if meta['image_id']==7 and int(a['qi'][j])==38 else None)
            rows.append(row)
        print('DONE',meta['image_id'],flush=True)
    totals={}
    for size in ['small','non_small']:
        rr=[r for r in rows if r['size']==size];totals[size]={}
        for rule in RULES:
            agg={}
            for label in ['fn','fp','tp','tn']:
                parts=[r['rules'][rule]['groups'][label] for r in rr]
                agg[label]={k:sum(x[k] for x in parts) for k in parts[0]}
            totals[size][rule]=dict(instances=len(rr),groups=agg,
                instances_without_selected_positive=sum(r['rules'][rule]['selected_positive']==0 for r in rr))
    # Reproduce earlier fixed small-instance comparator before interpreting candidates.
    base=totals['small']['ordinary']['groups']
    assert (base['fn']['count'],base['fn']['opposite_direction'],base['fp']['count'],base['fp']['opposite_direction'])==(179,2,1644,6)
    result=dict(passed=True,protocol='Frozen logit0 error masks inside OLD selection; detached weights. Ordinary, error weight2/correct weight1, and error-only. No weight search. Unit gradient per instance. All 7 saved images, not fresh holdout.',rows=rows,totals=totals,sources=sources)
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
