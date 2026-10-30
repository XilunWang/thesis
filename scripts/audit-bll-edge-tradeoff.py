"""Frozen small-instance logits: repair vs suppression directions, no update."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    out = R / 'reports/bll-edge-tradeoff-20261030'
    out.mkdir(exist_ok=False)
    rows = []
    for rank in range(3):
        folder = R / f'reports/bll-real-gradient-20261030-small-rank{rank}'
        meta = json.loads((folder / 'metadata.json').read_text())
        assert hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest() == meta['trace_sha256']
        a = np.load(folder/'trace.npz')
        vh, vw = map(int, a['valid_shape'])
        for j, gi in enumerate(a['gi']):
            gt = torch.from_numpy(a['gt'][gi]).bool()
            area = int(gt.sum())
            if not 0 < area < 1024: continue
            hp, wp = gt.shape
            z = torch.tensor(a['final'][0, a['qi'][j]], requires_grad=True)
            h, w = z.shape
            up = F.interpolate(z[None,None], (hp,wp), mode='bilinear', align_corners=False)[0,0]
            selected = torch.from_numpy(a['region'][j]).repeat_interleave(hp//h,0).repeat_interleave(wp//w,1)
            valid = torch.zeros_like(gt);valid[:vh,:vw]=True
            selected &= valid
            pos=selected&gt;neg=selected&~gt
            npix=int(pos.sum());nb=int(neg.sum());assert npix and nb
            lossmap=F.binary_cross_entropy_with_logits(up,gt.float(),reduction='none')
            lp=lossmap[pos].sum();ln=lossmap[neg].sum()
            gp,=torch.autograd.grad(lp,z,retain_graph=True)
            gn,=torch.autograd.grad(ln,z,retain_graph=True)
            ordinary=(gp+gn)/(npix+nb)
            balanced=.5*gp/npix+.5*gn/nb
            expected,=torch.autograd.grad(.5*lossmap[pos].mean()+.5*lossmap[neg].mean(),z)
            torch.testing.assert_close(balanced,expected,rtol=2e-5,atol=1e-8)
            predicted=up.detach()>=0
            groups={'missed_foreground':pos&~predicted,'correct_foreground':pos&predicted,
                    'excess_background':neg&predicted,'correct_background':neg&~predicted}
            row=dict(image_id=meta['image_id'],query=int(a['qi'][j]),area=area,selected_fg=npix,selected_bg=nb,
                relative_fg_weight_balanced=nb/npix,counts={k:int(v.sum()) for k,v in groups.items()},variants={})
            for name,g in [('ordinary',ordinary),('balanced',balanced)]:
                assert torch.isfinite(g).all()
                # Unit-native-logit-norm infinitesimal descent direction; NO logits changed.
                norm=g.norm();direction=-F.interpolate((g/norm)[None,None],(hp,wp),mode='bilinear',align_corners=False)[0,0]
                metrics={}
                for k,mask in groups.items():
                    d=direction[mask]
                    toward_fg=k.endswith('foreground')
                    desired=d if toward_fg else -d
                    metrics[k]=dict(pixels=len(d),helped=int((desired>1e-10).sum()),
                        harmed=int((desired< -1e-10).sum()),
                        mean_desired_direction=float(desired.mean()) if len(d) else None)
                row['variants'][name]=dict(native_gradient_norm=float(norm),groups=metrics)
            rows.append(row)
    assert len(rows)==4
    totals={}
    for name in ['ordinary','balanced']:
        totals[name]={}
        for group in rows[0]['counts']:
            rr=[r['variants'][name]['groups'][group] for r in rows]
            n=sum(x['pixels'] for x in rr)
            totals[name][group]=dict(pixels=n,helped=sum(x['helped'] for x in rr),harmed=sum(x['harmed'] for x in rr),
                mean_desired_direction=sum((x['mean_desired_direction'] or 0)*x['pixels'] for x in rr)/n if n else None)
    result=dict(passed=True,rows=rows,totals=totals,
        protocol='Threshold logit0 frozen; same saved inputs/old selection. Ordinary pixel mean vs per-instance 0.5 foreground mean + 0.5 background mean. Normalize native gradient to unit norm for direction comparison. No optimization step or predicted new masks.',
        limits='Only four small instances; fixed E1 outputs, not a candidate-trained model. Selected pixels only; not AP/recall or full network parameter effects.')
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(totals=totals,instances=[dict(image_id=r['image_id'],query=r['query'],counts=r['counts'],weight=r['relative_fg_weight_balanced']) for r in rows]),indent=2))


if __name__=='__main__':main()
