"""Local production-helper audit. Synthetic logits only; no model or optimizer."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.query_bll import build_bll_targets, query_boundary_loss
from scripts.bll_core import boundary


def independent_boundary(mask, width=3):
    padded = np.pad(mask.astype(np.float64), ((0, 0), (width, width), (width, width)))
    sums = np.zeros_like(mask, dtype=np.float64)
    for y in range(2 * width + 1):
        for x in range(2 * width + 1):
            sums += padded[:, y:y + mask.shape[1], x:x + mask.shape[2]]
    size = (2 * width + 1) ** 2
    return np.abs(size * mask - sums) > .15 * size


def check(gt, previous, logits, valid_shape=None):
    """Check production targets, region, loss and every final-logit gradient."""
    n, hp, wp = gt.shape
    h, w = logits.shape[-2:]
    valid_shape = valid_shape or (hp, wp)
    target, valid = build_bll_targets(gt, (h, w), valid_shape)
    arr = gt.numpy().reshape(n, h, hp // h, w, wp // w)
    expected_target = arr.max(axis=(2, 4))
    np.testing.assert_array_equal(target.numpy(), expected_target)
    previous = previous.detach().clone().requires_grad_(True)
    final = logits.detach().clone().requires_grad_(True)
    indices = torch.arange(n)
    loss, diag = query_boundary_loss(previous[None], final[None], [gt],
                                     [(indices, indices)], [valid_shape], loss_weight=1.)
    pb = (previous.detach().numpy() >= 0) & valid.numpy()
    region = (independent_boundary(pb) | independent_boundary(expected_target)) & valid.numpy()
    np.testing.assert_array_equal(boundary(torch.from_numpy(pb)).numpy(), independent_boundary(pb))
    z = final.detach().numpy().astype(np.float64)
    pixels = np.logaddexp(0., z) - z * expected_target
    expected_loss = pixels[region].sum() / max(int(region.sum()), 1)
    gradient = (1 / (1 + np.exp(-z)) - expected_target) * region / max(int(region.sum()), 1)
    loss.backward()
    np.testing.assert_allclose(float(loss.detach()), expected_loss, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(final.grad.numpy(), gradient, rtol=3e-6, atol=1e-8)
    assert previous.grad is None
    assert diag['selected_pixels'] == int(region.sum())
    return dict(loss=float(loss.detach()), selected=int(region.sum()),
                per_instance=region.sum(axis=(1, 2)).tolist(),
                gradient_abs_sum_per_instance=np.abs(final.grad.numpy()).sum(axis=(1, 2)).tolist()), region


def main():
    torch.set_num_threads(2)
    out = ROOT / 'reports/bll-objective-audit-20261029'
    out.mkdir(parents=True, exist_ok=False)
    sources = ['scripts/query_bll.py', 'scripts/bll_core.py', 'scripts/bll_mask2former_head.py']
    before = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    result = dict(scope='Synthetic logits, production helpers, independent NumPy oracle; no model/optimizer/training', source_sha256=before)

    # Sub-grid-positioned rectangle: compare occupancy-favouring and coverage-favouring logits.
    gt = torch.zeros(1, 128, 128)
    gt[:, 33:94, 35:92] = 1
    target, _ = build_bll_targets(gt, (32, 32), (128, 128))
    coverage = gt.reshape(1, 32, 4, 32, 4).mean(dim=(2, 4))
    prev = (target * 2 - 1) * 4
    occ_prob = target * .9 + .05
    cov_prob = coverage.clamp(.05, .95)
    occ, region = check(gt, prev, torch.logit(occ_prob))
    cov, _ = check(gt, prev, torch.logit(cov_prob))
    assert occ['loss'] < cov['loss']
    result['mixed_edge_objective'] = dict(occupancy_favouring=occ, coverage_favouring=cov,
        selected_mixed_cells=int(((coverage > 0) & (coverage < 1)).numpy()[region].sum()),
        warning='Coverage is a diagnostic reference, not the exact original point-sampling loss or a proposed fix.')

    # Selection is a neighbourhood-disagreement test, not a constant-width dilation.
    shapes = []
    for side in [1, 2, 3, 8, 20]:
        grid = torch.zeros(1, 64, 64)
        grid[:, 20:20 + side, 20:20 + side] = 1
        hi = grid.repeat_interleave(4, -2).repeat_interleave(4, -1)
        stats, region = check(hi, torch.full_like(grid, -4), torch.zeros_like(grid))
        shapes.append(dict(side_grid_cells=side, foreground_cells=side * side,
                           selected_foreground=int((region & grid.numpy().astype(bool)).sum()),
                           selected_background=int((region & ~grid.numpy().astype(bool)).sum()), **stats))
    assert shapes[0]['selected_background'] == 0 and shapes[1]['selected_background'] == 0
    result['gt_only_selection_by_size'] = shapes

    # A detached wrong previous mask adds background supervision, not foreground labels.
    base, base_region = check(gt, torch.full_like(target, -4), torch.zeros_like(target))
    wrong = torch.full_like(target, -4)
    wrong[:, 1:6, 1:6] = 4
    extra, extra_region = check(gt, wrong, torch.zeros_like(target))
    added = extra_region & ~base_region
    assert added.sum() > 0 and target.numpy()[added].sum() == 0
    result['wrong_previous_region'] = dict(gt_only=base, with_wrong_previous=extra,
        added_background_cells=int(added.sum()),
        gt_region_gradient_scale_ratio=base['selected'] / extra['selected'],
        interpretation='Added cells push foreground probability DOWN, while global averaging dilutes each original selected-cell gradient.')

    # Global pixel mean gives large/perimeter-rich selections greater aggregate weight.
    grid = torch.zeros(2, 64, 64)
    grid[0, 20, 20] = 1
    grid[1, 20:40, 20:40] = 1
    hi = grid.repeat_interleave(4, -2).repeat_interleave(4, -1)
    stats, _ = check(hi, torch.full_like(grid, -4), torch.zeros_like(grid))
    result['normalization_equal_logits'] = stats
    assert stats['per_instance'][1] > stats['per_instance'][0]

    # Padding and query/GT pairing: permutation must preserve the scalar loss.
    padded = torch.zeros(2, 32, 32)
    padded[0, 18:21, 24:27] = 1
    padded[1, 4:16, 4:16] = 1
    z = torch.linspace(-3, 3, 128).reshape(2, 8, 8)
    padded_stats, _ = check(padded, torch.full_like(z, -4), z, (21, 27))
    final = z.flip(0)[None]
    loss, _ = query_boundary_loss(torch.full_like(final, -4), final, [padded],
        [(torch.tensor([1, 0]), torch.tensor([0, 1]))], [(21, 27)], loss_weight=1.)
    assert abs(float(loss) - padded_stats['loss']) < 1e-6
    result['padding_and_permuted_pairing'] = dict(passed=True, **padded_stats)
    assert before == {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    result.update(passed=True, production_sources_unchanged=True, previous_selection_detached=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
