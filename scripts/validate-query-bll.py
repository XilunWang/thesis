"""Synthetic tests of query/GT alignment and padding, not a full-head test."""
import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import traceback
from pathlib import Path

import numpy as np
import torch
from query_bll import query_boundary_loss

ROOT = Path(__file__).resolve().parents[1]


def run():
    torch.set_num_threads(1)
    torch.manual_seed(20261021)
    spec = importlib.util.spec_from_file_location('bll_test_oracle', ROOT/'scripts/validate-bll.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    gt = [torch.zeros(2, 48, 64), torch.zeros(1, 48, 64)]
    gt[0][0, 8:32, 8:28] = 1
    gt[0][1, 20:44, 36:60] = 1
    gt[1][0, 8:32, 12:44] = 1
    valid = [(48, 64), (37, 53)]
    matches = [(torch.tensor([3, 1]), torch.tensor([1, 0])),
               (torch.tensor([2]), torch.tensor([0]))]
    previous = torch.full((2, 4, 12, 16), -3.)
    previous[:, 0] = 3  # unrelated query must not affect selected instances
    for i, (qi, gi) in enumerate(matches):
        for query, target in zip(qi.tolist(), gi.tolist()):
            previous[i, query] = torch.roll(gt[i][target, ::4, ::4], 1, 1)*6-3
    previous.requires_grad_()
    final = torch.randn_like(previous).requires_grad_()
    snapshots = [x.detach().clone() for x in [previous, final, *gt]]
    loss, info = query_boundary_loss(previous, final, gt, matches, valid, loss_weight=1)
    # Independent stride slicing + count-neighbor oracle + stable NumPy BCE.
    selected = np.zeros(tuple(final.shape), dtype=bool)
    targets = np.zeros(tuple(final.shape), dtype=np.float32)
    for i, (qi, gi) in enumerate(matches):
        gh, gw = valid[i][0]//4, valid[i][1]//4
        keep = np.zeros((12, 16), dtype=bool)
        keep[:gh, :gw] = True
        for query, target in zip(qi.tolist(), gi.tolist()):
            t = gt[i][target, ::4, ::4].numpy()*keep
            p = (previous[i, query].detach().numpy() >= 0)*keep
            selected[i, query] = (oracle.count_boundary(t[None])[0] |
                                  oracle.count_boundary(p[None])[0]) & keep
            targets[i, query] = t
    count = int(selected.sum())
    z = final.detach().numpy().astype(float)
    expected = ((np.logaddexp(0, z)-z*targets)*selected).sum()/count
    np.testing.assert_allclose(loss.item(), expected, rtol=1e-6, atol=1e-6)
    assert info['matched'] == 3 and info['selected_pixels'] == count
    loss.backward()
    assert previous.grad is None  # the hard selection branch is detached
    expected_grad = (1/(1+np.exp(-z))-targets)*selected/count
    np.testing.assert_allclose(final.grad.numpy(), expected_grad, atol=1e-7, rtol=1e-5)
    assert not final.grad.numpy()[~selected].any()
    assert final.grad.numpy()[selected].any()
    assert all(torch.equal(a, b) for a, b in zip(snapshots, [previous, final, *gt]))
    checks = ['shuffled_final_gt_mapping_and_independent_oracle',
              'analytic_final_gradients', 'no_previous_selection_gradient',
              'unmatched_and_padding_zero_gradients', 'inputs_unchanged']

    # Reindex all queries and transform supplied matches: exact same instances.
    permutation = torch.tensor([3, 0, 2, 1])
    inverse = torch.argsort(permutation)
    perm_matches = [(inverse[q], g) for q, g in matches]
    perm_loss, _ = query_boundary_loss(previous[:, permutation], final[:, permutation],
                                       gt, perm_matches, valid, loss_weight=1)
    torch.testing.assert_close(perm_loss, loss)
    checks.append('query_permutation_invariant')
    # Relabel GT array positions, preserving actual final correspondence.
    reordered_gt = [gt[0].flip(0), gt[1]]
    gt_matches = [(matches[0][0], 1-matches[0][1]), matches[1]]
    gt_loss, _ = query_boundary_loss(previous, final, reordered_gt, gt_matches, valid, loss_weight=1)
    torch.testing.assert_close(gt_loss, loss)
    checks.append('gt_permutation_invariant')
    # A wrong GT association must be observable; this fixture is not degenerate.
    wrong = [(matches[0][0], matches[0][1].flip(0)), matches[1]]
    wrong_loss, _ = query_boundary_loss(previous, final, gt, wrong, valid, loss_weight=1)
    assert abs(wrong_loss.item()-loss.item()) > 1e-5
    checks.append('wrong_association_detectable')

    p2, f2 = previous.detach().clone(), final.detach().clone()
    p2[1, :, 9:, :] = 99; p2[1, :, :, 13:] = -99
    f2[1, :, 9:, :] = -99; f2[1, :, :, 13:] = 99
    padded, _ = query_boundary_loss(p2, f2, gt, matches, valid, loss_weight=1)
    torch.testing.assert_close(padded, loss)
    checks.append('padding_perturbation_invariant_including_partial_cells')

    # Disabled helper makes no assignments/targets, consumes no RNG, adds zero.
    x = final.detach().clone().requires_grad_()
    y = final.detach().clone().requires_grad_()
    state = torch.random.get_rng_state().clone()
    disabled, meta = query_boundary_loss(None, x, None, None, None, loss_weight=0)
    assert torch.equal(state, torch.random.get_rng_state()) and meta['disabled']
    base = y.square().mean()
    combined = x.square().mean()+disabled
    base.backward(); combined.backward()
    assert torch.equal(base, combined) and torch.equal(x.grad, y.grad)
    checks.append('weight_zero_helper_value_gradient_rng_bypass')

    empty_indices = torch.empty(0, dtype=torch.long)
    f3 = final.detach().clone().requires_grad_()
    empty, _ = query_boundary_loss(previous, f3, [g[:0] for g in gt],
                                   [(empty_indices, empty_indices)]*2, valid, loss_weight=1)
    empty.backward()
    assert empty.item() == 0 and f3.grad is not None and not f3.grad.any()
    checks.append('zero_matches_connected_zero')
    blank = torch.zeros(1, 48, 64)
    f4 = torch.randn(1, 1, 12, 16, requires_grad=True)
    no_region, _ = query_boundary_loss(torch.full_like(f4, -3), f4, [blank],
                                       [(torch.tensor([0]), torch.tensor([0]))],
                                       [(48, 64)], loss_weight=1)
    no_region.backward()
    assert no_region.item() == 0 and not f4.grad.any()
    checks.append('zero_region_connected_zero')
    no_cell, _ = query_boundary_loss(torch.full_like(f4, -3), f4, [blank],
                                     [(torch.tensor([0]), torch.tensor([0]))],
                                     [(2, 2)], loss_weight=1)
    assert no_cell.item() == 0
    checks.append('image_smaller_than_grid_stride')

    def rejected(name, p=previous, f=final, g=gt, m=matches, v=valid, weight=1):
        try:
            query_boundary_loss(p, f, g, m, v, loss_weight=weight)
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: '+name)
    rejected('duplicate_query_rejected', m=[(torch.tensor([1, 1]), torch.tensor([0, 1])), matches[1]])
    rejected('duplicate_gt_rejected', m=[(torch.tensor([1, 3]), torch.tensor([0, 0])), matches[1]])
    rejected('out_of_range_gt_rejected', m=[(torch.tensor([1]), torch.tensor([2])), matches[1]])
    rejected('different_grid_rejected', p=previous[..., :-1])
    rejected('negative_weight_rejected', weight=-1)
    invalid_gt = [gt[0], gt[1].clone()]
    invalid_gt[1][0, 47, 63] = 1
    rejected('nonzero_gt_padding_rejected', g=invalid_gt)
    return dict(passed=True, checks=checks, check_count=len(checks),
                loss=loss.item(), oracle_loss=expected, wrong_association_loss=wrong_loss.item(),
                selected_pixels=count, matched=3,
                limitations=['Standalone helper only; no actual assignment or MMDetection head executed',
                             'weight=0 tested with a synthetic base loss, not original head BCE/Dice',
                             'FP32 single-process global normalization; distributed/AMP unverified',
                             'Native-grid nearest GT resizing can lose thin objects; width=3 is a probe setting, not E1 selection'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/query-bll-synthetic/validation.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite '+str(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=False)
    try:
        result = run()
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        files = ['scripts/query_bll.py','scripts/validate-query-bll.py',
                 'scripts/bll_core.py','scripts/validate-bll.py']
        result.update(python=platform.python_version(), torch=torch.__version__, device='cpu',
                      seed=20261021, source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
                      base_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip())
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['passed','check_count','matched','selected_pixels']}))


if __name__ == '__main__':
    main()
