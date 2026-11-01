"""Bounded frozen-E0 probe; no optimizer, backward, or candidate search."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'reports/position-query-probe-20261101'
CASES = [(2, 1106, 21262118, [(5354, 18015)]),
         (3, 2271, 21263283, [(13226, 18173), (13227, 649)]),
         (8, 730, 20261742, [(3411, 60569)])]
OFFSETS = [('hint', 0, 0), ('left', -32, 0), ('right', 32, 0),
           ('up', 0, -32), ('down', 0, 32)]


def window(cell, valid, padded, dx=0, dy=0):
    """Half-open valid-image intersection; retain partial boundary tokens."""
    vh, vw = valid[:2]
    ph, pw = padded[:2]
    x, y = 4 * (cell % 256) + 2, 4 * (cell // 256) + 2
    assert pw == 1024 and 0 <= x < vw and 0 <= y < vh
    x, y = min(max(x + dx, 0), vw - 1), min(max(y + dy, 0), vh - 1)
    gx, gy = x // 8, y // 8
    coords = [(yy, xx) for yy in range(max(0, gy - 1), min(math.ceil(vh / 8), gy + 2))
              for xx in range(max(0, gx - 1), min(math.ceil(vw / 8), gx + 2))]
    assert coords and ph % 8 == pw % 8 == 0
    return dict(center=[x, y], tokens=coords, memory_shape=[ph // 8, pw // 8])


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(result):
    import numpy as np
    import torch
    from pycocotools import mask as mu
    from mmengine.config import Config
    from mmdet.registry import MODELS, DATASETS
    from mmdet.utils import register_all_modules
    from scripts.identity_data_pipeline import with_identity

    base = load_module('probe_base', 'run-position-grid4-pilot.py')
    analysis = load_module('probe_matching', 'analyze-fixed12-e0.py')
    exporter = load_module('probe_encoding', 'export-fixed12-e0.py')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules()
    # Reuse the exact audited data method, without Pilot.__init__/head/optimizer.
    pilot = base.Pilot.__new__(base.Pilot)
    cp = ROOT / 'work_dirs/uiis_baseline_r50_36e/epoch_36.pth'
    assert sha(cp) == '53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
    pilot.initial = torch.load(cp, map_location='cpu')['state_dict']
    cfg = Config.fromfile(str(ROOT / 'reports/uiis-baseline-36e-evaluation/evaluation-config.py'))
    pilot.model = MODELS.build(cfg.model)
    pilot.model.load_state_dict(pilot.initial, strict=True)
    pilot.model.requires_grad_(False).cuda().eval()
    config = ROOT / 'work_dirs/uiis_baseline_r50_36e/config-resume.py'
    preflight = json.loads((ROOT / 'reports/position-learning-preflight-20261101/result.json').read_text())
    assert sha(config) == preflight['config_sha256']
    traincfg = Config.fromfile(str(config))
    pilot.ds = DATASETS.build(with_identity(traincfg.train_dataloader.dataset))
    pilot.ds.full_init()
    pilot.index = {pilot.ds.get_data_info(i)['img_id']: i for i in range(len(pilot.ds))}
    vp = ROOT / 'reports/position-learning-plan-grid4-20261101/views.json'
    manifest = json.loads((vp.parent / 'manifest.json').read_text())
    assert sha(vp) == manifest['views_sha256']
    assert sha(ROOT / 'data/UIIS/annotations/train.json') == manifest['source_sha256']
    views = json.loads(vp.read_text())['runs']
    head = pilot.model.panoptic_head
    fusion = pilot.model.panoptic_fusion_head
    result.update(script_sha256=sha(Path(__file__)), checkpoint_sha256=sha(cp), runs=[])
    source = ROOT / 'reports/fixed12-e0-complementarity-20261101/attempt-2'
    proof = json.loads((source / 'result.json').read_text())
    assert proof['passed']
    result['source_result_sha256'] = sha(source / 'result.json')
    with torch.no_grad():
        for case, image_id, seed, targets in CASES:
            oldpath = source / f'case-{case:02d}.json'
            assert sha(oldpath) == next(r['sha256'] for r in proof['runs'] if r['case_id'] == case)
            old = json.loads(oldpath.read_text())
            row = next(r for r in views if r['split'] == 'holdout' and (r['image_id'], r['seed']) == (image_id, seed))
            assert all(row[k] == old[k] for k in ['input_sha256', 'masks_sha256'])
            data, _, _ = pilot.data(row)
            vh, vw = old['valid_shape']
            gt = data['data_samples'][0].gt_instances
            assert gt.annotation_ids.tolist() == old['annotation_ids']
            assert np.array_equal(gt.masks.masks[:, :vh, :vw], mu.decode(old['gt_rles']).transpose(2, 0, 1))
            torch.cuda.synchronize()
            start = time.perf_counter()
            features = pilot.model.extract_feat(data['inputs'])
            torch.cuda.synchronize()
            feature_seconds = time.perf_counter() - start
            branches = [('baseline', None, None)]
            for gid, cell in targets:
                reference = window(cell, row['valid_shape'], row['padded_shape'])
                for name, dx, dy in OFFSETS:
                    win = window(cell, row['valid_shape'], row['padded_shape'], dx, dy)
                    win['valid_control'] = name == 'hint' or win['tokens'] != reference['tokens']
                    branches.append((name, gid, win))
            for name, gid, win in branches:
                calls = []
                def hook(module, args, kwargs):
                    mask = kwargs['cross_attn_mask']
                    mh, mw = win['memory_shape']
                    assert mask.shape[-1] == mh * mw and mask.shape[-2] == 100
                    replacement = torch.ones_like(mask, dtype=torch.bool)
                    for yy, xx in win['tokens']:
                        replacement[..., yy * mw + xx] = False
                    assert (~replacement).sum(-1).eq(len(win['tokens'])).all()
                    kwargs = dict(kwargs, cross_attn_mask=replacement)
                    calls.append(dict(shape=list(mask.shape), allowed=len(win['tokens'])))
                    return args, kwargs
                handle = head.transformer_decoder.layers[2].register_forward_pre_hook(hook, with_kwargs=True) if win else None
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                try:
                    cls, padded = head.predict(features, data['data_samples'])
                    actual = fusion.predict(cls, padded, data['data_samples'], rescale=False)[0]['ins_results']
                finally:
                    if handle:
                        handle.remove()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                assert len(calls) == (1 if win else 0)
                assert torch.isfinite(cls).all() and torch.isfinite(padded).all()
                query_rles = exporter.encode((padded[0, :, :vh, :vw] > 0).cpu().numpy())
                probs = cls[0].softmax(-1).cpu().numpy()
                pred_rles = exporter.encode(actual.masks.cpu().numpy())
                labels, scores = actual.labels.tolist(), actual.scores.tolist()
                if win is None:
                    top = np.asarray(old['selected_query_class_flat_indices'])
                    assert query_rles == old['query_rles']
                    assert probs.tolist() == old['class_probabilities']
                    assert labels == (top % 7).tolist() and scores == old['selected_scores']
                    assert pred_rles == [old['query_rles'][q] for q in top // 7]
                qiou = mu.iou(query_rles, old['gt_rles'], [0] * len(old['labels']))
                piou = mu.iou(pred_rles, old['gt_rles'], [0] * len(old['labels']))
                matched = analysis.detection_match(piou, labels, scores, old['labels'], old['annotation_ids'])
                detected = [old['annotation_ids'][j] for j in matched]
                rows = [dict(gt_id=g, best_iou=float(qiou[:, j].max()), best_query=int(qiou[:, j].argmax()),
                             detected=j in matched) for j, g in enumerate(old['annotation_ids'])]
                record = dict(case_id=case, image_id=image_id, seed=seed, branch=name, target_gt=gid,
                              window=win, hook_calls=calls, input_sha256=row['input_sha256'], masks_sha256=row['masks_sha256'],
                              seconds=seconds, feature_seconds=feature_seconds, peak_memory_bytes=torch.cuda.max_memory_allocated(),
                              detected_gt=detected, unmatched_predictions_above_05=sum(s >= .5 for s in scores)-len(matched),
                              per_gt=rows, query_rles=query_rles, class_probabilities=probs.tolist(),
                              prediction_rles=pred_rles, prediction_labels=labels, prediction_scores=scores)
                filename = f'case-{case:02d}-{gid or 0}-{name}.json'
                (OUT / filename).write_text(json.dumps(record))
                result['runs'].append(dict(file=filename, sha256=sha(OUT / filename)))
                print('DONE', filename, flush=True)
    pilot.protected()
    assert len(result['runs']) == 23
    result.update(passed=True, model_state_unchanged=True, optimizer_created=False)


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        for _, _, _, targets in CASES:
            for _, cell in targets:
                w = window(cell, [1024, 1024], [1024, 1024])
                assert 1 <= len(w['tokens']) <= 9
                assert all(window(cell, [1024, 1024], [1024, 1024], dx, dy)['tokens'] != w['tokens'] for _, dx, dy in OFFSETS[1:])
        assert window(0, [3, 3], [1024, 1024])['tokens'] == [(0, 0)]
        assert len(window(0, [9, 9], [1024, 1024])['tokens']) == 4
        assert window(0, [9, 9], [1024, 1024], -32, 0)['center'] == [0, 2]
        print('WINDOW_SELF_TEST_PASSED')
    else:
        import fcntl
        OUT.mkdir(exist_ok=False)
        result = dict(passed=False, training_updates=0)
        try:
            with (ROOT / 'work_dirs/uiis-training.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                main(result)
        except BaseException:
            result['error'] = traceback.format_exc()
            raise
        finally:
            (OUT / 'result.json').write_text(json.dumps(result, indent=2))
