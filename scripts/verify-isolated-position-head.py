"""Fixed-cache and frozen-model acceptance. No optimizer or training updates."""
import fcntl
import hashlib
import json
import random
import sys
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from pycocotools import mask as mu
from scripts.isolated_position_head import IsolatedPositionHead, make_targets, position_loss
OUT = ROOT/'reports/isolated-position-head-20261101'
SRC = ROOT/'reports/identity-augmentation-check-20261101'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def seed(n):
    random.seed(n); np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n)


def cached_targets(d):
    masks = mu.decode(d['gt_rles']).transpose(2, 0, 1) if d['gt_rles'] else np.zeros((0, *d['padded_shape']), dtype=np.uint8)
    return make_targets(masks, d['annotation_ids'], d['original_areas'], d['valid_shape'])


def main(result):
    torch.set_num_threads(2); seed(20261101)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    previous = json.loads((ROOT/'reports/query-cell-assignment-20261101/result.json').read_text())
    count = 0
    for run in previous['runs']:
        if run['filtered']:
            continue
        d = json.loads((SRC/f"image-{run['image_id']}-seed-{run['seed']}.json").read_text())
        t, v, assignments = cached_targets(d)
        assert assignments == {r['annotation_id']: r['cell'] for r in run['assignments']}
        assert int(t.sum()) == run['original_small']
        count += len(assignments)
        # Invalid padding logits have no effect, and exactly zero gradient.
        p = torch.zeros_like(t).unsqueeze(0).requires_grad_()
        a = position_loss(p, t[None], v[None]); a.backward()
        assert not p.grad[~v[None]].any()
        altered = p.detach().clone(); altered[~v[None]] = float('nan')
        assert torch.equal(a.detach(), position_loss(altered, t[None], v[None]))
    assert count == 1076
    for value in [-1000., 0., 1000.]:
        p = torch.full((1, 2, 2), value, requires_grad=True)
        loss = position_loss(p, torch.tensor([[[1., 0.], [0., 0.]]]), torch.ones_like(p, dtype=torch.bool))
        loss.backward(); assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
    empty = np.zeros((0, 16, 16), dtype=np.uint8)
    t, v, assignment = make_targets(empty, [], [], [9, 9]); assert v.all() and not assignment
    try:
        make_targets(np.ones((2, 8, 8), dtype=np.uint8), [1, 2], [1, 1], [8, 8])
    except ValueError:
        pass
    else:
        raise AssertionError('Collision deficit was silently accepted')
    branch = IsolatedPositionHead()
    c3 = torch.randn(1, 512, 2, 2, requires_grad=True)
    c4 = torch.randn(1, 1024, 1, 1, requires_grad=True)
    position_loss(branch(c3, c4), t[None], v[None]).backward()
    assert c3.grad is None and c4.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
    result['cache_checks'] = dict(matched_targets=count, padding_loss_and_gradient_excluded=True, empty_target_finite=True, deficit_rejected=True, detached_inputs=True)
    print('CACHE_ACCEPTED', flush=True)

    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import DATASETS, MODELS
    from mmdet.utils import register_all_modules
    from scripts.identity_data_pipeline import with_identity
    register_all_modules()
    cfg = Config.fromfile(str(ROOT/'reports/uiis-baseline-36e-evaluation/evaluation-config.py'))
    ck = ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth'
    assert digest(ck) == '53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
    initial = torch.load(ck, map_location='cpu')['state_dict']
    model = MODELS.build(cfg.model); model.load_state_dict(initial, strict=True)
    model.requires_grad_(False); model.cuda().eval()
    branch = IsolatedPositionHead().cuda()
    branch_state = {k: v.detach().clone() for k, v in branch.state_dict().items()}
    traincfg = Config.fromfile(str(ROOT/'work_dirs/uiis_baseline_r50_36e/config-resume.py'))
    ds = DATASETS.build(with_identity(traincfg.train_dataloader.dataset)); ds.full_init()
    indices = {ds.get_data_info(i)['img_id']: i for i in range(len(ds))}
    result['real_samples'] = []
    for iid, n in [(2968, 20263980), (2355, 20263367), (976, 21261988)]:
        seed(n); item = ds.prepare_data(indices[iid]); assert item is not None
        data = model.data_preprocessor(pseudo_collate([item]), training=True)
        sample = data['data_samples'][0]; gt = sample.gt_instances
        d = json.loads((SRC/f'image-{iid}-seed-{n}.json').read_text())
        assert gt.annotation_ids.tolist() == d['annotation_ids']
        assert gt.original_areas.tolist() == d['original_areas']
        masks = gt.masks.masks
        expected = mu.decode(d['gt_rles']).transpose(2, 0, 1) if d['gt_rles'] else np.zeros_like(masks)
        assert np.array_equal(masks, expected)
        t, v, assignments = make_targets(masks, gt.annotation_ids.tolist(), gt.original_areas.tolist(), sample.img_shape)
        with torch.no_grad():
            features = model.extract_feat(data['inputs'])
            before = model.panoptic_head(features, data['data_samples'])
            before = [[x.cpu().clone() for x in part] for part in before]
        branch.zero_grad(set_to_none=True)
        logits = branch(features[1], features[2])
        loss = position_loss(logits, t[None].cuda(), v[None].cuda()); loss.backward()
        grads = {k: float(p.grad.norm()) for k, p in branch.named_parameters()}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
        assert all(g > 0 for g in grads.values()) and torch.isfinite(loss)
        assert all(p.grad is None for p in model.parameters())
        assert all(torch.equal(branch_state[k], x) for k, x in branch.state_dict().items())
        with torch.no_grad():
            after = model.panoptic_head(model.extract_feat(data['inputs']), data['data_samples'])
        assert all(torch.equal(a, b.cpu()) for pa, pb in zip(before, after) for a, b in zip(pa, pb))
        assert all(torch.equal(x.cpu(), initial[k]) for k, x in model.state_dict().items())
        result['real_samples'].append(dict(image_id=iid, seed=n, positives=len(assignments), valid_cells=int(v.sum()), loss=float(loss.detach()), branch_gradient_norms=grads, original_logits_exact=True, base_gradients_absent=True, all_states_unchanged=True))
        print('REAL_SAMPLE_ACCEPTED', iid, len(assignments), flush=True)
        del features, before, after, logits, loss, data
    result.update(passed=True, parameters=sum(p.numel() for p in branch.parameters()), training_updates=0, checkpoint_sha256=digest(ck), source_sha256={p.name: digest(p) for p in [Path(__file__), ROOT/'scripts/isolated_position_head.py', ROOT/'scripts/identity_data_pipeline.py', ROOT/'reports/query-cell-assignment-20261101/result.json']})


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'result.json').exists()
    result = dict(passed=False, training_updates=0)
    try:
        with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main(result)
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        (OUT/'result.json').write_text(json.dumps(result, indent=2))
