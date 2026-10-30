"""Decompose original-GT pixel BCE on saved real small-instance logits; no updates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    out = ROOT / 'reports/bll-small-pixel-audit-20261030'
    out.mkdir(exist_ok=False)
    rows = []; inputs = []
    for rank in range(3):
        folder = ROOT / f'reports/bll-real-gradient-20261030-small-rank{rank}'
        meta = json.loads((folder / 'metadata.json').read_text())
        assert meta['passed'] and meta['model_state_unchanged'] and not meta['optimizer_created']
        assert hashlib.sha256((folder / 'trace.npz').read_bytes()).hexdigest() == meta['trace_sha256']
        a = np.load(folder / 'trace.npz')
        gt = torch.from_numpy(a['gt'][a['gi']]).float()
        region = torch.from_numpy(a['region']).bool()
        n, hp, wp = gt.shape; h, w = region.shape[-2:]; vh, vw = map(int, a['valid_shape'])
        valid = torch.zeros(hp, wp, dtype=torch.bool); valid[:vh, :vw] = True
        selected = region.repeat_interleave(hp // h, 1).repeat_interleave(wp // w, 2) & valid
        denominator = int(selected.sum()); assert denominator > 0
        z = torch.tensor(a['final'][0, a['qi']], requires_grad=True)
        up = F.interpolate(z[:, None], size=(hp, wp), mode='bilinear', align_corners=False)[:, 0]
        pos = selected & (gt > 0); neg = selected & (gt == 0)
        loss_map = F.binary_cross_entropy_with_logits(up, gt, reduction='none')
        lp = loss_map[pos].sum() / denominator; ln = loss_map[neg].sum() / denominator
        gp, = torch.autograd.grad(lp, z, retain_graph=True)
        gn, = torch.autograd.grad(ln, z, retain_graph=True)
        total, = torch.autograd.grad(loss_map[selected].mean(), z, retain_graph=True)
        torch.testing.assert_close(gp + gn, total, rtol=2e-5, atol=1e-8)
        assert (gp <= 1e-10).all() and (gn >= -1e-10).all()
        assert torch.isfinite(total).all()
        # Independent BCE scalar formula; decomposition keeps SAME full selected-pixel denominator.
        independent = (F.softplus(up) - up * gt)[selected].double().mean()
        assert abs(float(independent.detach()) - float((lp + ln).detach())) < 2e-6
        assert all(pos.flatten(1).any(1))
        for j in range(n):
            area = int(gt[j].sum())
            if not 0 < area < 1024:
                continue
            support = gp[j] < 0
            p = -gp[j]; b = gn[j]; net = total[j]
            # Derivative of mean foreground logit wrt native logits, without taking a step.
            mean_fg = up[j][gt[j].bool()].mean()
            sensitivity, = torch.autograd.grad(mean_fg, z, retain_graph=True)
            s = sensitivity[j]
            positive_drive = float((s * p).sum())
            background_drive = float((s * b).sum())
            row = dict(image_id=meta['image_id'],filename=meta['filename'],rank=rank,
                query=int(a['qi'][j]),gt_index=int(a['gi'][j]),area=area,
                selected_fg=int(pos[j].sum()),selected_bg=int(neg[j].sum()),
                foreground_logit_support=int(support.sum()),
                background_overcomes_fg_cells=int((support & (net > 0)).sum()),
                fg_abs_gradient_sum=float(p.sum()),background_on_fg_support_sum=float(b[support].sum()),
                cancelled_fg_fraction=float(torch.minimum(p,b).sum()/p.sum()),
                positive_drive_on_gt_mean=positive_drive,background_drive_on_gt_mean=background_drive,
                background_to_positive_drive=background_drive/positive_drive if positive_drive else None,
                direct_logit_descent_would_lower_gt_mean=background_drive>positive_drive,
                selected_foreground_retained=bool(pos[j].any()),
                mean_gt_probability=float(up[j][gt[j].bool()].sigmoid().mean().detach()))
            rows.append(row)
        inputs.append(dict(image_id=meta['image_id'],seed=meta['seed'],trace_sha256=meta['trace_sha256'],
                           matched=n,global_selected_pixel_denominator=denominator,
                           foreground_loss=float(lp.detach()),background_loss=float(ln.detach()),
                           decomposition_verified=True))
    assert rows and len({r['image_id'] for r in rows}) == 3
    result = dict(passed=True,images=3,small_instances=len(rows),
        positive_supervision_lost=sum(not r['selected_foreground_retained'] for r in rows),
        instances_with_local_bg_dominance=sum(r['background_overcomes_fg_cells']>0 for r in rows),
        instances_with_gt_mean_downward_pressure=sum(r['direct_logit_descent_would_lower_gt_mean'] for r in rows),
        rows=rows,inputs=inputs,
        scope='Saved E1 outputs; original-pixel candidate only. Fixed original selection. Logit gradients, not network-parameter updates or prediction disappearance proof.')
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
