"""Synthetic CPU BLL verification; never trains or reads UIIS predictions."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from bll_core import boundary, three_stage_bll

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference():
    """Load only the two pinned upstream AST definitions, not MMDetection.

    Adapter for upstream binary_cross_entropy: for this test's binary targets,
    unit weights and default mean reduction, it is exactly PyTorch mean BCE.
    This does not validate MMDetection's general weighting/ignore/label APIs.
    """
    manifest = json.loads((ROOT / 'docs/p3-source-manifest.json').read_text(encoding='utf-8-sig'))
    entry = next(r for r in manifest if r.get('revision') ==
                 '07042d254ffbdbf584ee0a5725e3c5e5e57e27c5' and
                 r['path'].endswith('/losses/cross_entropy_loss.py'))
    path = ROOT / entry['local_path']
    assert sha(path) == entry['sha256'].lower()
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if getattr(n, 'name', '') in
             ('generate_block_target', 'LaplacianCrossEntropyLoss')]
    assert len(nodes) == 2
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = []
    env = dict(torch=torch, nn=torch.nn, F=F,
               binary_cross_entropy=F.binary_cross_entropy_with_logits)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env['generate_block_target'], env['LaplacianCrossEntropyLoss'](), entry


def count_boundary(a, width=3, threshold=0.15):
    """Independent oracle: count unlike neighbors, with explicit outside zeros."""
    n, h, w = a.shape
    result = np.zeros_like(a, dtype=bool)
    for i in range(n):
        for y in range(h):
            for x in range(w):
                unlike = 0
                for dy in range(-width, width + 1):
                    for dx in range(-width, width + 1):
                        yy, xx = y + dy, x + dx
                        value = a[i, yy, xx] if 0 <= yy < h and 0 <= xx < w else 0
                        unlike += value != a[i, y, x]
                result[i, y, x] = unlike > threshold * (2 * width + 1) ** 2
    return result


def resize(a, shape):
    """Independent NumPy bilinear interpolation with align_corners=True."""
    h, w = a.shape[-2:]
    ys = np.linspace(0, h - 1, shape[0])
    xs = np.linspace(0, w - 1, shape[1])
    y0, x0 = ys.astype(int), xs.astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0)[None, :, None], (xs - x0)[None, None, :]
    return ((1-wy) * ((1-wx)*a[:, y0[:, None], x0] + wx*a[:, y0[:, None], x1]) +
            wy * ((1-wx)*a[:, y1[:, None], x0] + wx*a[:, y1[:, None], x1]))


def mask(kind, h, w, index):
    a = np.zeros((h, w), dtype=np.float32)
    if kind in ('rectangle', 'shifted'):
        a[h//4:3*h//4, w//4:3*w//4] = 1
        if index % 2:
            a[:] = 0
            a[h//3:h//2+1, w//3:w//2+1] = 1
    elif kind == 'line':
        a[:, w//2] = 1
    elif kind == 'edge':
        a[:h//2, :w//3] = 1
    elif kind == 'full':
        a[:] = 1
    elif kind == 'empty':
        pass
    else:
        raise ValueError(kind)
    return a


def run():
    torch.set_num_threads(1)
    torch.manual_seed(20261021)
    official_boundary, official_loss, source = reference()
    cases = []
    for batch in (1, 3):
        for ratio in (1, 2):
            shapes = [(s, s if ratio == 1 else s + s//2) for s in (14, 28, 56)]
            for kind in ('rectangle', 'shifted', 'line', 'edge', 'full', 'empty'):
                arrays = [np.stack([mask(kind, h, w, i) for i in range(batch)]) for h, w in shapes]
                targets = [torch.from_numpy(a) for a in arrays]
                preds = []
                for a in arrays:
                    p = np.roll(a, 2, axis=2) if kind == 'shifted' else a
                    logits = torch.tensor((p * 2 - 1)[:, None] * 2.)
                    logits = logits + torch.rand_like(logits) * 0.4 - 0.2
                    preds.append(logits.requires_grad_())
                region_small = count_boundary((preds[1][:, 0].detach().numpy() >= 0)) | count_boundary(arrays[1])
                expected_region = resize(region_small.astype(float), shapes[2]) >= 0.5
                for a in arrays[:2]:
                    expected = count_boundary(a)
                    assert np.array_equal(boundary(torch.from_numpy(a)).numpy(), expected)
                    assert np.array_equal((official_boundary(torch.from_numpy(a)) == 1).numpy(), expected)
                total, components, region = three_stage_bll(preds, targets)
                assert np.array_equal(region.numpy(), expected_region)
                expected_losses = []
                expected_grads = []
                for idx, (p, a) in enumerate(zip(preds, arrays)):
                    z = p.detach().numpy()[:, 0].astype(np.float64)
                    pixel_loss = np.logaddexp(0., z) - z*a
                    weight = np.ones_like(a) / a.size if idx < 2 else expected_region / max(1, expected_region.sum())
                    expected_losses.append(float((pixel_loss * weight).sum()))
                    expected_grads.append(((1/(1+np.exp(-z))-a)*weight*(0.25, 0.65, 1.)[idx])[:, None])
                np.testing.assert_allclose([c.item() for c in components], expected_losses, atol=1e-6, rtol=1e-6)
                other = [p.detach().clone().requires_grad_() for p in preds]
                ref = official_loss(other, targets)
                torch.testing.assert_close(total, ref, atol=1e-6, rtol=1e-6)
                total.backward()
                ref.backward()
                for p, q, grad in zip(preds, other, expected_grads):
                    assert torch.isfinite(p.grad).all()
                    np.testing.assert_allclose(p.grad.numpy(), grad, atol=1e-7, rtol=1e-5)
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5)
                assert not region.requires_grad
                assert torch.count_nonzero(preds[2].grad[:, 0][~region]) == 0
                if region.any():
                    assert (preds[2].grad[:, 0][region] != 0).all()
                for idx in (0, 1):
                    assert (preds[idx].grad != 0).all()
                better = [p.detach().clone() for p in preds]
                better[-1] = (targets[-1][:, None]*2-1)*6
                improved = three_stage_bll(better, targets)[1][-1].item()
                if region.any():
                    assert improved < components[-1].item()
                else:
                    assert improved == components[-1].item() == 0
                cases.append(dict(kind=kind, batch=batch, shapes=shapes,
                                  selected_pixels=int(region.sum()), loss=total.item(),
                                  final_loss=components[-1].item(), improved_final_loss=improved,
                                  reference_abs_error=abs(total.item()-ref.item())))
    # Threshold hand checks: 7 vs 8 unlike neighbors at the central pixel.
    for neighbors, expected in ((7, False), (8, True)):
        a = np.zeros((1, 7, 7), np.float32)
        coords = [(y, x) for y in range(7) for x in range(7) if (y, x) != (3, 3)]
        for y, x in coords[:neighbors]:
            a[0, y, x] = 1
        assert boundary(torch.from_numpy(a))[0, 3, 3].item() == expected
        assert (official_boundary(torch.from_numpy(a)) == 1)[0, 3, 3].item() == expected
    # Exact threshold: response 8 is NOT selected when threshold*49 == 8.
    assert not boundary(torch.from_numpy(a), threshold=8/49)[0, 3, 3]
    assert (official_boundary(torch.from_numpy(a), u=8/49) == 1)[0, 3, 3].item() is False
    assert boundary(torch.ones(1, 14, 14)).any()  # zero-padding border
    empty_preds = [torch.empty(0, 1, s, s, requires_grad=True) for s in (14, 28, 56)]
    empty_targets = [torch.empty(0, s, s) for s in (14, 28, 56)]
    zero, _, _ = three_stage_bll(empty_preds, empty_targets)
    assert zero.item() == 0 and zero.requires_grad
    zero.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in empty_preds)
    try:
        original_empty = official_loss(empty_preds, empty_targets)
        original_empty_status = 'finite' if torch.isfinite(original_empty) else 'nonfinite'
    except Exception as exc:
        original_empty_status = type(exc).__name__ + ': ' + str(exc)
    # Invalid-target checks are our explicit pilot contract, not upstream behavior.
    try:
        boundary(torch.full((1, 7, 7), .5))
    except ValueError:
        pass
    else:
        raise AssertionError('Nonbinary target accepted')
    return dict(passed=True, cases=cases, case_count=len(cases),
                threshold_checks=3, zero_instances='graph_connected_zero_passed',
                upstream_zero_instances=original_empty_status,
                source=source,
                limitations=['Reference uses extracted upstream AST plus mean-BCE adapter for binary unit-weight targets',
                             'Zero-instance protection is a local extension, not upstream equivalence',
                             'CPU FP32 only; no Mask2Former, UIIS, GPU, training or AP validation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/bll-synthetic/validation.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=False)
    try:
        result = run()
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result.update(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                      device='cpu', seed=20261021,
                      isolated_dependencies={d.metadata['Name']:d.version for d in
                                             importlib.metadata.distributions(path=[str(ROOT/'.local/bll-libs')])},
                      source_sha256={p:sha(ROOT/p) for p in ['scripts/bll_core.py','scripts/validate-bll.py']},
                      base_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip())
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['passed','case_count','upstream_zero_instances','torch']}))


if __name__ == '__main__':
    main()
