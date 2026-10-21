"""Four fixed UIIS train images, E0 weights, full-model forward/backward only."""
import copy
import hashlib
import json
import random
import runpy
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(result, deterministic_cudnn):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import mmdet
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import MODELS, DATASETS
    from mmdet.utils import register_all_modules
    import scripts.bll_mask2former_head as adapter

    assert mmdet.__version__ == '3.3.0'
    register_all_modules()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = deterministic_cudnn
    torch.backends.cudnn.benchmark = False
    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    cfg = runpy.run_path(str(ROOT/'scripts/prepare-uiis-baseline.py'))['build_config']()
    checkpoint = ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth'
    expected = '53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
    assert sha(checkpoint) == expected
    saved = torch.load(checkpoint, map_location='cpu')
    state = saved['state_dict']
    del saved
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    dataset.full_init()
    indices = sorted(range(len(dataset)), key=lambda i: dataset.get_data_info(i)['img_id'])[:4]
    packed = []
    result.update(scope='Four fixed train images: E0 train augmentation plus two controlled padding views; no optimizer',
                  deterministic_cudnn=deterministic_cudnn,
                  torch=torch.__version__, mmdet=mmdet.__version__, checkpoint_sha256=expected,
                  train_annotation_sha256=sha(ROOT/'data/UIIS/annotations/train.json'),
                  gradient_tolerance=dict(atol=1e-6, rtol=1e-5), images=[], batches=[])
    padding_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    for transform in padding_cfg.pipeline:
        if transform.type == 'RandomResize':
            transform.ratio_range = (0.503, 0.503)
    padding_dataset = DATASETS.build(padding_cfg)
    padding_dataset.full_init()
    views = [(dataset,index,'original_train_pipeline') for index in indices]
    views += [(padding_dataset,index,'controlled_ratio_0.503_padding_probe') for index in indices[:2]]
    for i, (source, index, profile) in enumerate(views):
        info = source.get_data_info(index)
        seed(20261021+i)
        # prepare_data avoids silent replacement of the requested image on rejection.
        item = source.prepare_data(index)
        assert item is not None and item['data_samples'].img_id == info['img_id']
        packed.append(item)
        result['images'].append(dict(index=index, id=info['img_id'], profile=profile,
            filename=Path(info['img_path']).name, sha256=sha(info['img_path']),
            seed=20261021+i, source_instances=len(info['instances']),
            transformed_instances=len(item['data_samples'].gt_instances),
            input_shape=list(item['inputs'].shape),
            metainfo={k:v for k,v in item['data_samples'].metainfo.items() if k != 'img_path'}))
    original = MODELS.build(cfg.model).cuda()
    candidate_cfg = copy.deepcopy(cfg.model)
    candidate_cfg.panoptic_head.update(type='BLLMask2FormerHead', bll_loss_weight=0., bll_boundary_width=3)
    candidate = MODELS.build(candidate_cfg).cuda()
    original_helper = adapter.query_boundary_loss
    current_audit = []

    def audited_helper(previous, final, gts, matches, shapes, **kwargs):
        loss, diag = original_helper(previous, final, gts, matches, shapes, **kwargs)
        rows = []
        with torch.no_grad():
            for gt, (qi, gi), (vh, vw) in zip(gts, matches, shapes):
                h, w = final.shape[-2:]
                hp, wp = gt.shape[-2:]
                assert hp % h == 0 and wp % w == 0
                gh, gw = vh//(hp//h), vw//(wp//w)
                small = (F.interpolate(gt[:, None].float(), (h,w), mode='nearest')[:,0]
                         if len(gt) else gt.new_empty((0,h,w)))
                before = gt.flatten(1).sum(1)
                after = small[:, :gh, :gw].flatten(1).sum(1)
                lost = ((before > 0) & (after == 0)).nonzero().flatten()
                rows.append(dict(gt_pixels=before.tolist(), bll_grid_pixels=after.tolist(),
                    vanished_gt_indices=lost.tolist(), matched_queries=qi.tolist(),
                    matched_gt_indices=gi.tolist(), mask_grid=[h,w], valid_grid=[gh,gw]))
        current_audit.append(dict(diagnostics=diag, images=rows))
        return loss, diag

    adapter.query_boundary_loss = audited_helper
    def tensor_hash(t):
        return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def run(model, batch, weight, run_seed):
        model.load_state_dict(state, strict=True)
        model.train()
        model.zero_grad(set_to_none=True)
        if hasattr(model.panoptic_head, 'bll_loss_weight'):
            model.panoptic_head.bll_loss_weight = weight
        seed(run_seed)
        current_audit.clear()
        data = model.data_preprocessor(copy.deepcopy(batch), training=True)
        inputs = data['inputs']
        checks = []
        for sample in data['data_samples']:
            gt = sample.gt_instances
            masks = gt.masks.to_tensor(dtype=torch.bool, device=inputs.device)
            vh, vw = sample.img_shape[:2]
            assert masks.shape[-2:] == inputs.shape[-2:]
            assert not masks[:, vh:, :].any() and not masks[:, :, vw:].any()
            assert ((gt.labels >= 0) & (gt.labels < 7)).all()
            checks.append(dict(id=sample.img_id, valid_shape=[vh,vw],
                padded_shape=list(masks.shape[-2:]), instances=len(gt),
                labels=gt.labels.tolist(), mask_sha256=tensor_hash(masks)))
        assert inputs.dtype == torch.float32 and torch.isfinite(inputs).all()
        outputs = {}
        def capture(module, args, out):
            outputs.update(final_class_sha256=tensor_hash(out[0][-1]),
                           final_mask_sha256=tensor_hash(out[1][-1]),
                           final_mask_shape=list(out[1][-1].shape))
        hook = model.panoptic_head.register_forward_hook(capture)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            losses = model(**data, mode='loss')
            assert all(torch.isfinite(v).all() for v in losses.values())
            sum(losses.values()).backward()
        finally:
            hook.remove()
        torch.cuda.synchronize()
        gradients = {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
        # No optimizer exists; check parameters directly against the loaded E0 tensors.
        assert all(torch.equal(p.detach().cpu(), state[n]) for n,p in model.named_parameters())
        return dict(losses={k:float(v.detach()) for k,v in losses.items()},
            gradients=gradients, outputs=outputs, preprocessing=checks,
            input_sha256=tensor_hash(inputs), input_shape=list(inputs.shape),
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
            audit=copy.deepcopy(current_audit), seconds=time.perf_counter()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())

    def compare(a, b):
        assert a['losses'] == b['losses']
        assert a['outputs'] == b['outputs']
        assert a['preprocessing'] == b['preprocessing'] and a['input_sha256'] == b['input_sha256']
        assert torch.equal(a['cpu_rng'], b['cpu_rng']) and torch.equal(a['cuda_rng'], b['cuda_rng'])
        assert a['gradients'].keys() == b['gradients'].keys()
        errors = {n: float((g-b['gradients'][n]).abs().max()) for n,g in a['gradients'].items()}
        failures = [n for n,g in a['gradients'].items()
                    if not torch.allclose(g, b['gradients'][n], atol=1e-6, rtol=1e-5)]
        return dict(passed=not failures, max_absolute_gradient_difference=max(errors.values()),
                    outside_tolerance=failures)

    try:
        for offset in [0,2,4]:
            print('BATCH_START', offset//2, flush=True)
            batch = pseudo_collate(packed[offset:offset+2])
            baseline = run(original, batch, 0, 20261100+offset)
            repeat = run(original, batch, 0, 20261100+offset)
            repeat_check = compare(baseline, repeat)
            del repeat
            off = run(candidate, batch, 0, 20261100+offset)
            off_check = compare(baseline, off)
            del off
            on = run(candidate, batch, 1, 20261100+offset)
            assert set(on['losses']) == set(baseline['losses']) | {'loss_bll'}
            assert all(on['losses'][k] == v for k,v in baseline['losses'].items())
            assert on['outputs'] == baseline['outputs']
            assert on['preprocessing'] == baseline['preprocessing']
            assert on['input_sha256'] == baseline['input_sha256']
            assert torch.equal(on['cpu_rng'], baseline['cpu_rng']) and torch.equal(on['cuda_rng'], baseline['cuda_rng'])
            assert on['losses']['loss_bll'] > 0
            assert any(not torch.equal(g, on['gradients'][n]) for n,g in baseline['gradients'].items())
            assert any(n.startswith('backbone.') for n in on['gradients'])
            result['batches'].append(dict(ids=[r['id'] for r in on['preprocessing']],
                preprocessing=on['preprocessing'], input_shape=on['input_shape'],
                outputs=on['outputs'], original_repeat=repeat_check, disabled=off_check,
                losses=on['losses'], audit=on['audit'],
                gradient_tensor_count=len(on['gradients']), parameters_unchanged=True,
                measurements={k:dict(seconds=v['seconds'], peak_allocated_bytes=v['peak_allocated_bytes'])
                              for k,v in [('baseline',baseline),('enabled',on)]}))
            print('BATCH_DONE', offset//2, 'loss_bll', on['losses']['loss_bll'], flush=True)
            del baseline, on
    finally:
        adapter.query_boundary_loss = original_helper
    assert sha(checkpoint) == expected
    result['passed'] = all(b['original_repeat']['passed'] and b['disabled']['passed'] for b in result['batches'])
    result['limitations'] = ['four unique images/six views, no AP or generalization conclusion',
        'third batch forces resize ratio 0.503 for padding coverage; not an E0 training-strategy change',
        'no optimizer, no full-model checkpoint restore test, no AMP/DDP',
        'both models resident and diagnostic copies/hooks; not an efficiency benchmark',
        'BLL width=3/weight=1 are probes; vanished instances must be reviewed before training']


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/bll-real-images/validation.json')
    parser.add_argument('--deterministic-cudnn', action='store_true')
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=False)
    try:
        verify(result, args.deterministic_cudnn)
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['source_sha256'] = {f:sha(ROOT/f) for f in [
            'scripts/verify-bll-real-images.py','scripts/bll_mask2former_head.py',
            'scripts/query_bll.py','scripts/bll_core.py','scripts/prepare-uiis-baseline.py']}
        output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('REAL_IMAGES_CHECK', result['passed'], flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
