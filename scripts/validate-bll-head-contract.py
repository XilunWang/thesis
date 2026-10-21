"""Local adapter contract harness. NOT an installed MMDetection/model test."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.query_bll import query_boundary_loss


def main():
    output = ROOT / 'reports/bll-head-contract/validation.json'
    if output.exists():
        raise FileExistsError(output)
    parent_path = ROOT / 'work/p3-reference/mmdet-3.3.0/maskformer_head.py'
    adapter_path = ROOT / 'scripts/bll_mask2former_head.py'
    tree = ast.parse(parent_path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'loss_by_feat')
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    ns = {'multi_apply': lambda f, *xs: tuple(map(list, zip(*map(f, *xs))))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(parent_path), 'exec'), ns)

    class Parent:
        loss_by_feat = ns['loss_by_feat']

        def __init__(self):
            self.calls = 0
            self.fail = False

        def _get_targets_single(self, layer, gt):
            self.calls += 1
            torch.rand(3)  # Stand-in for original stochastic matching/point sampling.
            if self.fail:
                raise RuntimeError('injected parent failure')
            # Deliberately different earlier-layer assignments.
            q = torch.tensor([layer % 2])
            return (None,) * 6 + (SimpleNamespace(pos_inds=q, pos_assigned_gt_inds=torch.tensor([0])),)

        def _loss_by_feat_single(self, cls, mask, gts, metas):
            for gt in gts:
                self._get_targets_single(int(cls.item()), gt)
            return mask.square().mean(), mask.mean(), mask.sigmoid().mean()

    tree = ast.parse(adapter_path.read_text(encoding='utf-8'))
    adapter = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    adapter.decorator_list = []
    import math
    ns = dict(Mask2FormerHead=Parent, math=math, torch=torch,
              query_boundary_loss=query_boundary_loss)
    exec(compile(ast.Module(body=[adapter], type_ignores=[]), str(adapter_path), 'exec'), ns)
    Candidate = ns['BLLMask2FormerHead']
    torch.manual_seed(21)
    masks = [torch.randn(1, 2, 8, 8, requires_grad=True) for _ in range(3)]
    scores = [torch.tensor(float(i)) for i in range(3)]
    gt = torch.zeros(1, 32, 32)
    gt[:, 4:24, 8:24] = 1
    gts = [SimpleNamespace(masks=gt)]
    metas = [dict(img_shape=(29, 31))]

    def run(head):
        torch.manual_seed(42)
        losses = head.loss_by_feat(scores, masks, gts, metas)
        rng = torch.get_rng_state().clone()
        grads = torch.autograd.grad(sum(losses.values()), masks, retain_graph=True)
        return losses, rng, grads

    base = Parent()
    original, rng, grads = run(base)
    off = Candidate(bll_loss_weight=0)
    disabled, off_rng, off_grads = run(off)
    assert original.keys() == disabled.keys()
    assert all(torch.equal(original[k], disabled[k]) for k in original)
    assert all(torch.equal(a, b) for a, b in zip(grads, off_grads))
    assert torch.equal(rng, off_rng) and base.calls == off.calls == 3
    on = Candidate(bll_loss_weight=1)
    enabled, on_rng, on_grads = run(on)
    assert set(enabled) == set(original) | {'loss_bll'}
    assert all(torch.equal(original[k], enabled[k]) for k in original)
    assert torch.equal(rng, on_rng) and on.calls == 3
    expected, _ = query_boundary_loss(masks[-2], masks[-1], [gt],
        [(torch.tensor([0]), torch.tensor([0]))], [(29, 31)], loss_weight=1)
    assert torch.equal(expected, enabled['loss_bll']) and expected.item() > 0
    assert all(torch.equal(a, b) for a, b in zip(grads[:-1], on_grads[:-1]))
    assert not torch.equal(grads[-1], on_grads[-1])
    assert all(torch.isfinite(g).all() for g in on_grads)
    on.fail = True
    try:
        run(on)
    except RuntimeError as error:
        assert str(error) == 'injected parent failure'
    else:
        raise AssertionError('Parent failure swallowed')
    assert on._bll_matches is None
    on.fail = False
    run(on)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=True, scope='AST adapter + upstream orchestration; fake single-layer loss/matching',
        checks=['disabled loss/gradient/RNG equality', 'enabled original losses/RNG unchanged',
                'one assignment per image per layer', 'final-layer match used',
                'additional gradient only in final logits', 'exception cleanup and retry'],
        loss_bll=expected.item(), torch=torch.__version__,
        sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [parent_path, adapter_path, Path(__file__)]})
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
