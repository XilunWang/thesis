"""Fixed saved logits: candidate supervision only. No model/optimizer/update."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def real_labels():
    from pycocotools.coco import COCO
    sys.path.insert(0,str(ROOT))
    from scripts.query_bll import build_bll_targets
    from scripts.bll_core import boundary
    torch.set_num_threads(2)
    annotation=ROOT/'data/UIIS/annotations/train.json'
    coco=COCO(str(annotation));rows=[]
    for iid in sorted(coco.imgs)[:128]:
        for ann in coco.imgToAnns[iid]:
            if ann.get('iscrowd',0):continue
            mask=coco.annToMask(ann);vh,vw=mask.shape;hp=(vh+3)//4*4;wp=(vw+3)//4*4
            gt=torch.zeros(1,hp,wp);gt[0,:vh,:vw]=torch.from_numpy(mask)
            occ,valid=build_bll_targets(gt,(hp//4,wp//4),(vh,vw))
            region=boundary(occ)&valid
            valid_pixels=torch.zeros(hp,wp);valid_pixels[:vh,:vw]=1
            counts=valid_pixels.reshape(hp//4,4,wp//4,4).sum((1,3))
            coverage=gt.reshape(1,hp//4,4,wp//4,4).sum((2,4))/counts.clamp(min=1)
            selected=region.repeat_interleave(4,1).repeat_interleave(4,2)
            soft_retained=bool(((coverage>0)&region).any())
            pixel_retained=bool(((gt>0)&selected).any())
            assert soft_retained and pixel_retained
            rows.append(dict(image_id=iid,gt_id=ann['id'],small=ann['area']<1024,
                coverage_positive_selected=soft_retained,original_pixel_positive_selected=pixel_retained,
                would_vanish_if_coverage_thresholded=not bool((coverage>=.5).any())))
    result=dict(passed=True,images=128,instances=len(rows),small=sum(r['small'] for r in rows),
        positive_supervision_lost=0,coverage_threshold_half_lost=sum(r['would_vanish_if_coverage_thresholded'] for r in rows),
        coverage_threshold_half_lost_small=sum(r['small'] and r['would_vanish_if_coverage_thresholded'] for r in rows),
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),rows=rows,
        scope='Native GT only, first128 training images; no augmentations or predictions; GT-only region, not actual prediction union.')
    path=ROOT/'reports/bll-candidate-audit-20261030/real-labels.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


def candidate(z, gt, region, valid_shape, kind):
    n, h, w = z.shape
    hp, wp = gt.shape[-2:]; vh, vw = map(int, valid_shape)
    sh, sw = hp // h, wp // w
    valid = torch.zeros((hp, wp)); valid[:vh, :vw] = 1
    counts = valid.reshape(h, sh, w, sw).sum((1, 3))
    positive = gt.reshape(n, h, sh, w, sw).sum((2, 4))
    coverage = positive / counts.clamp(min=1)
    if kind == 'coverage':
        loss = F.binary_cross_entropy_with_logits(z, coverage, reduction='none')[region].mean()
        retained = ((coverage > 0) & region).flatten(1).any(1)
    elif kind == 'original_pixels':
        up = F.interpolate(z[:, None], size=(hp, wp), mode='bilinear', align_corners=False)[:, 0]
        selected = region.repeat_interleave(sh, 1).repeat_interleave(sw, 2) & valid.bool()
        loss = F.binary_cross_entropy_with_logits(up, gt, reduction='none')[selected].mean()
        retained = ((gt > 0) & selected).flatten(1).any(1)
    else:
        raise ValueError(kind)
    grad, = torch.autograd.grad(loss, z)
    assert torch.isfinite(loss) and torch.isfinite(grad).all()
    return float(loss.detach()), grad.numpy(), retained.numpy(), coverage.numpy()


def compare(ga, gb, minority, old):
    overlap = (ga != 0) & (gb != 0)
    conflict = ga * gb < 0
    na, nb = np.linalg.norm(ga), np.linalg.norm(gb)
    return dict(cosine=float(np.sum(ga * gb)/(na * nb)), norm_ratio=float(nb/na),
        opposite_all=int(conflict.sum()), overlap=int(overlap.sum()),
        minority_original_down_candidate_up=int((minority & (ga > 0) & (gb < 0)).sum()),
        new_opposite_vs_old=int((conflict & ~(ga * old < 0)).sum()))


def main():
    torch.set_num_threads(2)
    out = ROOT / 'reports/bll-candidate-audit-20261030'
    out.mkdir(exist_ok=False)
    source_paths = ['scripts/query_bll.py','scripts/bll_core.py','scripts/bll_mask2former_head.py']
    source_hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
    rows = []
    for iid in range(4):
        folder = ROOT / ('reports/bll-real-gradient-20261030' + ('' if iid == 0 else f'-id{iid}'))
        meta = json.loads((folder/'metadata.json').read_text())
        assert source_hashes == {p: meta['source_sha256'][p] for p in source_paths}
        assert hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest() == meta['trace_sha256']
        a = np.load(folder/'trace.npz'); qi = a['qi']
        gt = torch.tensor(a['gt'][a['gi']], dtype=torch.float32)
        region = torch.tensor(a['region'], dtype=torch.bool)
        ga = a['gradient_original'][0,qi].astype(float); old = a['gradient_bll'][0,qi].astype(float)
        n,h,w = ga.shape; hp,wp = gt.shape[-2:]; vh,vw = a['valid_shape']
        full = np.zeros((h,w),bool); full[:vh//(hp//h),:vw//(wp//w)] = True
        minority = a['region'] & full & (a['coverage']>0) & (a['coverage']<.5)
        row = dict(image_id=iid, instances=n, areas=gt.sum((1,2)).tolist(), trace_sha256=meta['trace_sha256'],
                   occupancy=compare(ga,old,minority,old))
        for kind in ['coverage','original_pixels']:
            z = torch.tensor(a['final'][0,qi],dtype=torch.float32,requires_grad=True)
            value,g,retained,cov = candidate(z,gt,region,a['valid_shape'],kind)
            assert retained.all()
            row[kind] = dict(loss=value,retained=int(retained.sum()),**compare(ga,g,minority,old))
            if kind == 'coverage':
                expected = (torch.sigmoid(z).detach().numpy()-cov)*a['region']/int(region.sum())
                np.testing.assert_allclose(g,expected,rtol=1e-5,atol=1e-8)
        rows.append(row)

    # Every source single-pixel location, including partial-valid cells and <stride images.
    # Use the production target/boundary solely to keep region fixed for this audit.
    import sys
    sys.path.insert(0,str(ROOT))
    from scripts.query_bll import build_bll_targets
    from scripts.bll_core import boundary
    tests=[]
    for vh,vw in [(4,4),(5,7),(1,1),(3,2),(8,8)]:
        hp=(vh+3)//4*4;wp=(vw+3)//4*4;h=hp//4;w=wp//4
        for y in range(vh):
            for x in range(vw):
                gt=torch.zeros(1,hp,wp);gt[0,y,x]=1
                occ,valid=build_bll_targets(gt,(h,w),(vh,vw))
                region=boundary(occ)&valid
                record=dict(valid=[vh,vw],pixel=[y,x])
                for kind in ['coverage','original_pixels']:
                    z=torch.zeros(1,h,w,requires_grad=True)
                    loss,g,retained,cov=candidate(z,gt,region,(vh,vw),kind)
                    assert retained.all()
                    if kind=='coverage':assert np.all(g[:,~valid.numpy()]==0)
                    record[kind]=dict(retained=True,gradient_sum=float(g.sum()),max_coverage=float(cov.max()))
                tests.append(record)
    # Demonstrates why nonzero soft labels do not guarantee binary-mask retention.
    interior=[t for t in tests if t['valid']==[4,4]]
    assert all(t['coverage']['max_coverage']==1/16 for t in interior)
    assert all(t['coverage']['gradient_sum']>0 for t in interior)
    result=dict(passed=True,rows=rows,synthetic_single_pixel_cases=len(tests),
        single_pixel_4x4=dict(soft_target=.0625,gradient_at_probability_half=.4375,
            thresholded_at_half_would_disappear=True),source_hashes=source_hashes,
        scope='Same saved E1 logits/matches/old region; target and loss-resolution probes only. No model forward, parameter gradient, training or AP evaluation.')
    assert source_hashes == {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'single-pixel-tests.json').write_text(json.dumps(tests),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['--real-labels-only']:real_labels()
    elif not sys.argv[1:]:main()
    else:raise SystemExit('Use no arguments or --real-labels-only')
