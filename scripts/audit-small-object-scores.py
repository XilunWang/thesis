"""Existing E0 JSON only: score limits, threshold cost and false positives."""
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/small-score-audit-20261029'
THRESHOLDS = [0., .01, .05, .1, .2, .25, .3, .5, .7]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('matcher', ROOT / 'scripts/analyze-uiis-errors.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.self_test()
    ann = ROOT / 'data/UIIS/annotations/val.json'
    pred = ROOT / 'reports/uiis-baseline-36e-evaluation/predictions.segm.json'
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [ann, pred]}
    coco = COCO(str(ann))
    predictions = json.loads(pred.read_text())
    grouped = defaultdict(list)
    for p in predictions:
        grouped[p['image_id']].append(p)
    totals = {str(t): Counter() for t in THRESHOLDS}
    candidates = []
    low_gt = []
    for index, (iid, im) in enumerate(coco.imgs.items()):
        gt = [a for a in coco.imgToAnns[iid] if not a.get('iscrowd', 0)]
        crowd = [a for a in coco.imgToAnns[iid] if a.get('iscrowd', 0)]
        pp = sorted(grouped[iid], key=lambda p: -p['score'])
        gc = [a['category_id'] for a in gt]
        iou = mu.iou([p['segmentation'] for p in pp], [coco.annToRLE(a) for a in gt], [0] * len(gt)) if pp and gt else np.zeros((len(pp), len(gt)))
        ci = mu.iou([p['segmentation'] for p in pp], [coco.annToRLE(a) for a in crowd], [1] * len(crowd)) if pp and crowd else np.zeros((len(pp), len(crowd)))
        areas = [int(mu.area(p['segmentation'])) for p in pp]
        labels, matches, _ = m.match(iou, [p['category_id'] for p in pp], gc)
        for k, label in enumerate(labels):
            if label != 'matched' and any(a['category_id'] == pp[k]['category_id'] and ci[k, j] >= .5 for j, a in enumerate(crowd)):
                labels[k] = 'ignored_crowd'
        used05 = {j for k, j in enumerate(matches) if j is not None and pp[k]['score'] >= .5}
        for j, a in enumerate(gt):
            if a['area'] >= 1024 or j in used05:
                continue
            valid = [k for k, p in enumerate(pp) if p['category_id'] == a['category_id'] and iou[k, j] >= .5]
            if valid and max(pp[k]['score'] for k in valid) < .5:
                k = max(valid, key=lambda k: pp[k]['score'])
                s = pp[k]['score']
                # q >= .5*A/(A+1e-6) for nonempty positive-logit masks.
                upper = min(1., s * 2 * (1 + 1e-6 / areas[k]))
                low_gt.append(dict(gt_id=a['id'], image_id=iid, category_id=a['category_id'],
                    highest_qualifying_score=s, corresponding_iou=float(iou[k, j]),
                    predicted_area=areas[k], classification_upper_bound=upper,
                    classification_necessarily_below_05=upper < .5))
        for k, (p, label, j) in enumerate(zip(pp, labels, matches)):
            candidates.append(dict(image_id=iid, category_id=p['category_id'], score=p['score'],
                predicted_area=areas[k], label=label,
                matched_gt_size=None if j is None else ('small' if gt[j]['area'] < 1024 else 'other')))
        for t in THRESHOLDS:
            c = totals[str(t)]
            for k, (p, label, j) in enumerate(zip(pp, labels, matches)):
                if p['score'] < t:
                    continue
                c['predictions'] += 1
                c[label] += 1
                if label == 'matched':
                    c['small_matched' if gt[j]['area'] < 1024 else 'other_matched'] += 1
                elif label != 'ignored_crowd':
                    c['false_positive'] += 1
                if 0 < areas[k] < 1024:
                    c['predicted_small_' + ('tp' if label == 'matched' else 'ignored' if label == 'ignored_crowd' else 'fp')] += 1
        if index % 150 == 0:
            print('IMAGES', index + 1, flush=True)
    assert len(low_gt) == 176
    assert totals['0.5']['small_matched'] == 655 and totals['0.5']['matched'] == 2509
    assert totals['0.5']['false_positive'] == 2435
    assert totals['0.3']['matched'] == 2664
    bins = []
    for lo, hi in zip([0., .01, .05, .1, .2, .25, .3], [.01, .05, .1, .2, .25, .3, .5]):
        cc = [p for p in candidates if lo <= p['score'] < hi and 0 < p['predicted_area'] < 1024]
        counts = Counter(p['label'] for p in cc)
        bins.append(dict(score_range=[lo, hi], counts=dict(counts),
                         low_score_gt_count=sum(lo <= x['highest_qualifying_score'] < hi for x in low_gt)))
    valid = [p for p in candidates if 0 < p['predicted_area'] < 1024 and p['score'] < .5 and p['label'] != 'ignored_crowd']
    distributions = {}
    for name, accept in [('tp', lambda p: p['label'] == 'matched'), ('fp', lambda p: p['label'] != 'matched')]:
        ss = [p['score'] for p in valid if accept(p)]
        distributions[name] = dict(n=len(ss), score_quantiles=np.quantile(ss, [0, .25, .5, .75, 1]).tolist())
    result = dict(scope='Exploratory held-out-test JSON diagnosis, no score fitting or threshold selection; not AP.',
        images=len(coco.imgs), prediction_fields=sorted(predictions[0]), input_sha256=hashes,
        empty_mask_predictions=sum(p['predicted_area'] == 0 for p in candidates),
        thresholds={t: dict(c) for t, c in totals.items()},
        low_score_small_gt=dict(n=len(low_gt), class_below05_necessary=sum(x['classification_necessarily_below_05'] for x in low_gt),
            highest_qualifying_score_quantiles=np.quantile([x['highest_qualifying_score'] for x in low_gt], [0,.25,.5,.75,1]).tolist(),
            categories=dict(Counter(coco.cats[x['category_id']]['name'] for x in low_gt))),
        predicted_small_low_score_bins=bins, predicted_small_low_score_distributions=distributions,
        limitations=['Only saved top100 query-class candidates; raw class logits, mask probabilities and query IDs absent.',
            'TP/FP are sequential one-to-one IoU0.5 labels, crowd ignored; predicted size and GT size differ.',
            'A score below0.5 is a diagnostic miss, not automatically excluded from original COCO AP.',
            'Classification upper bound is inferred from pinned framework formula, not measured decomposition.'])
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in [ann, pred]}
    for c in totals.values():
        assert c['predictions'] == c['matched'] + c['false_positive'] + c['ignored_crowd']
    (OUT/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT/'low-score-small-gt.json').write_text(json.dumps(low_gt, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
