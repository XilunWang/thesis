"""One synthetic feature batch through real MMDetection head; no optimizer/training.

Requires the project server environment. Does not test backbone/data pipeline,
UIIS images, E0 checkpoint, optimizer restore, AMP, DDP or AP improvement.
"""
import argparse
import copy
import hashlib
import io
import json
import runpy
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(device):
    import mmdet
    import torch
    from mmengine.config import Config
    from mmengine.structures import InstanceData
    from mmdet.registry import MODELS
    from mmdet.structures import DetDataSample
    from mmdet.utils import register_all_modules
    from scripts.bll_mask2former_head import BLLMask2FormerHead

    if mmdet.__version__ != '3.3.0':
        raise RuntimeError('Probe reviewed for mmdet 3.3.0 only')
    register_all_modules()
    cfg = runpy.run_path(str(ROOT / 'scripts/prepare-uiis-baseline.py'))['build_config']()
    head_cfg = copy.deepcopy(cfg.model.panoptic_head)
    head_cfg.train_cfg = copy.deepcopy(cfg.model.train_cfg)
    overlay = Config.fromfile(str(ROOT / 'configs/bll-head-probe.py'))
    candidate_cfg = copy.deepcopy(head_cfg)
    candidate_cfg.update(overlay.panoptic_head)
    torch.manual_seed(20261021)
    original = MODELS.build(head_cfg).to(device).train()
    candidate = MODELS.build(candidate_cfg).to(device).train()
    candidate.load_state_dict(original.state_dict(), strict=True)
    assert isinstance(candidate, BLLMask2FormerHead)
    features = tuple(torch.randn(1, channels, 32 // 2**i, 32 // 2**i, device=device)
                     for i, channels in enumerate(head_cfg.in_channels))
    meta = dict(img_shape=(117, 125), batch_input_shape=(128, 128),
                pad_shape=(128, 128), ori_shape=(117, 125))
    samples = [DetDataSample(metainfo=meta)]
    masks = torch.zeros(2, 128, 128, device=device, dtype=torch.long)
    masks[0, 12:68, 20:80] = 1
    masks[1, 70:110, 86:119] = 1
    gts = [InstanceData(labels=torch.tensor([0, 2], device=device), masks=masks)]

    def run(head):
        head.zero_grad(set_to_none=True)
        torch.manual_seed(20261021)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        scores, predictions = head(features, samples)
        losses = head.loss_by_feat(scores, predictions, gts, [meta])
        sum(losses.values()).backward()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        assert all(torch.isfinite(v).all() for v in losses.values())
        gradients = {n: p.grad.detach().cpu().clone() for n, p in head.named_parameters() if p.grad is not None}
        assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
        return dict(losses={k: v.detach().cpu() for k, v in losses.items()},
                    outputs=[v.detach().cpu() for v in list(scores)+list(predictions)],
                    gradients=gradients, rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
                    seconds=time.perf_counter()-start,
                    peak_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)

    def equal(a, b, key):
        assert a[key].keys() == b[key].keys(), key
        for name in a[key]:
            torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0, msg=name)

    baseline = run(original)
    candidate.bll_loss_weight = 0
    disabled = run(candidate)
    equal(baseline, disabled, 'losses')
    equal(baseline, disabled, 'gradients')
    assert len(baseline['outputs']) == len(disabled['outputs'])
    for a, b in zip(baseline['outputs'], disabled['outputs']):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(baseline['rng'], disabled['rng'])
    if baseline['cuda_rng'] is not None:
        assert torch.equal(baseline['cuda_rng'], disabled['cuda_rng'])
    candidate.bll_loss_weight = 1
    enabled = run(candidate)
    assert set(enabled['losses']) == set(baseline['losses']) | {'loss_bll'}
    assert enabled['losses']['loss_bll'].item() > 0
    for name, value in baseline['losses'].items():
        torch.testing.assert_close(value, enabled['losses'][name], rtol=0, atol=0)
    assert torch.equal(baseline['rng'], enabled['rng'])
    if baseline['cuda_rng'] is not None:
        assert torch.equal(baseline['cuda_rng'], enabled['cuda_rng'])
    assert any(not torch.equal(g, enabled['gradients'][n]) for n, g in baseline['gradients'].items())
    buffer = io.BytesIO()
    torch.save(candidate.state_dict(), buffer)
    buffer.seek(0)
    candidate.load_state_dict(torch.load(buffer, map_location=device, weights_only=True), strict=True)
    restored = run(candidate)
    equal(enabled, restored, 'losses')
    equal(enabled, restored, 'gradients')
    return dict(passed=True, scope='real head, synthetic features/GT, no backbone or data pipeline',
        mmdet=mmdet.__version__, torch=torch.__version__, device=device,
        checks=['weight=0 exact output/loss/parameter-gradient/RNG equality',
                'weight=1 original losses/RNG unchanged, finite changed gradients',
                'state_dict in-memory roundtrip loss/gradient equality'],
        loss_bll=enabled['losses']['loss_bll'].item(),
        measurements={k: {field: v[field] for field in ['seconds', 'peak_bytes']}
                      for k, v in [('baseline', baseline), ('disabled', disabled), ('enabled', enabled)]},
        measurement_limit='single cold/warm runs with both heads resident; not a throughput or incremental-memory benchmark')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/bll-head-framework/validation.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=False)
    try:
        result = verify(args.device)
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        files = ['scripts/bll_mask2former_head.py', 'scripts/query_bll.py',
                 'scripts/bll_core.py', 'scripts/verify-bll-head-framework.py',
                 'configs/bll-head-probe.py', 'scripts/prepare-uiis-baseline.py']
        result['sha256'] = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['passed', 'scope', 'loss_bll']}))


if __name__ == '__main__':
    main()
