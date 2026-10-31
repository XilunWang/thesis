"""Read-only COCO ranking and large-instance diagnostics from frozen predictions."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.local/analysis-libs'))
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

OUT = ROOT / 'reports/e1p-ap-drop-20261031'
LABELS = ['E0', 'E1', 'E1-P']
FOLDERS = ['uiis-baseline-36e-evaluation', 'uiis-e1-bll-36e-evaluation',
           'uiis-e1p-pixel-bll-36e-evaluation']


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def mean(a):
    return float(np.mean(a)) if len(a) else None


def valid(a):
    return mean(a[a >= 0] * 100)


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'result.json').exists(), 'Preserve existing completed result'
    src = ROOT / 'reports/e1p-final-comparison'
    summary = json.loads((src / 'summary.json').read_text(encoding='utf-8'))
    rows = json.loads((src / 'instances.json').read_text(encoding='utf-8'))
    ann = ROOT / 'data/UIIS/annotations/val.json'
    assert sha(ann) == summary['annotation_sha256']
    gt = COCO(str(ann))
    result = {'models': {}, 'input_hashes': {'annotations': sha(ann)},
              'scope': 'Saved predictions only; no inference or optimization. COCO maxDets=[100,300,1000].'}
    for label, folder in zip(LABELS, FOLDERS):
        path = ROOT / 'reports' / folder / 'predictions.segm.json'
        digest = sha(path)
        assert digest == summary['inputs'][label]['sha256']
        result['input_hashes'][label] = digest
        preds = json.loads(path.read_text(encoding='utf-8'))
        for p in preds:
            p.pop('bbox', None)
        dt = gt.loadRes(preds)
        ev = COCOeval(gt, dt, 'segm')
        ev.params.maxDets = [100, 300, 1000]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        assert np.allclose(ev.stats[:6], summary['coco'][label]['stats'][:6], rtol=0, atol=1e-10)
        model = {'classes': {}}
        for k, cid in enumerate(ev.params.catIds):
            name = gt.cats[cid]['name']
            large = [a for a in gt.anns.values() if a['category_id'] == cid
                     and not a.get('iscrowd', 0) and 9216 <= a['area'] <= 1e10]
            c = {'large_gt': len(large), 'all_ap': valid(ev.eval['precision'][:, :, k, 0, -1]),
                 'large_ap': valid(ev.eval['precision'][:, :, k, 3, -1]), 'iou': {}}
            ee = [e for e in ev.evalImgs if e and e['category_id'] == cid
                  and e['aRng'] == ev.params.areaRng[3]]
            ng = sum(int(np.count_nonzero(~np.asarray(e['gtIgnore'], dtype=bool))) for e in ee)
            assert ng == len(large)
            for t, thr in enumerate(ev.params.iouThrs):
                scores = np.concatenate([np.array(e['dtScores']) for e in ee])
                ids = np.concatenate([np.array(e['dtIds'], dtype=int) for e in ee])
                match = np.concatenate([e['dtMatches'][t] > 0 for e in ee])
                ignored = np.concatenate([e['dtIgnore'][t] for e in ee])
                order = np.argsort(-scores, kind='mergesort')
                order = order[~ignored[order]]
                scores, ids, match = scores[order], ids[order], match[order]
                tp, fp = np.cumsum(match), np.cumsum(~match)
                recalls = tp / ng if ng else np.zeros(len(tp))
                precisions = tp / np.maximum(tp + fp, 1)
                # Independent 101-point interpolated AP, same category/area/IoU.
                envelope = np.maximum.accumulate(precisions[::-1])[::-1]
                grid = np.zeros(101)
                for j, r in enumerate(ev.params.recThrs):
                    z = np.searchsorted(recalls, r)
                    if z < len(envelope): grid[j] = envelope[z]
                official = valid(ev.eval['precision'][t, :, k, 3, -1])
                if ng: assert abs(grid.mean() * 100 - official) < 1e-8
                checkpoints = {}
                for r in [.25, .5, .75]:
                    at = np.flatnonzero(recalls >= r)
                    j = int(at[0]) if len(at) else None
                    checkpoints[str(r)] = None if j is None else dict(
                        score=float(scores[j]), tp=int(tp[j]), fp=int(fp[j]), precision=float(precisions[j]))
                top = []
                if round(float(thr), 2) in [.5, .75, .9]:
                    for j in range(min(12, len(ids))):
                        pred = dt.anns[int(ids[j])]
                        im, cat = pred['image_id'], pred['category_id']
                        gi = [a for a in gt.imgToAnns[im] if not a.get('iscrowd', 0)]
                        # IoUs already computed by COCO in sorted detection order.
                        from pycocotools import mask as mu
                        overlaps = mu.iou([pred['segmentation']], [gt.annToRLE(a) for a in gi], [0]*len(gi))[0] if gi else []
                        same = [(float(v), a['id']) for a, v in zip(gi, overlaps) if a['category_id'] == cat]
                        best = max(same, default=(0., None))
                        other = max([float(v) for a, v in zip(gi, overlaps) if a['category_id'] != cat], default=0.)
                        reason = 'tp' if match[j] else ('duplicate_or_competition' if best[0] >= thr else
                                 'wrong_class_overlap' if other >= thr else 'localization' if best[0] >= .1 else 'background_or_very_low_overlap')
                        top.append(dict(rank=j+1, dt_id=int(ids[j]), image_id=im, score=float(scores[j]),
                                        tp=bool(match[j]), best_same_iou=best[0], best_gt=best[1], reason=reason))
                c['iou'][f'{thr:.2f}'] = dict(ap=official, recall=float(recalls[-1]) if len(recalls) else 0.,
                    tp=int(tp[-1]) if len(tp) else 0, fp=int(fp[-1]) if len(fp) else 0,
                    at_recall=checkpoints, top=top,
                    tp_score_median=float(np.median(scores[match])) if match.any() else None)
            model['classes'][name] = c
        result['models'][label] = model
        assert sha(path) == digest
        print(label, 'RANKING_CHECKED', flush=True)
    result['large_class_contributions'] = {}
    for ref in ['E0', 'E1']:
        changes = {c: (result['models']['E1-P']['classes'][c]['large_ap'] - v['large_ap'])
                   for c, v in result['models'][ref]['classes'].items()}
        assert abs(sum(changes.values()) / 7 - 100*(summary['coco']['E1-P']['stats'][5]-summary['coco'][ref]['stats'][5])) < 1e-8
        result['large_class_contributions'][ref] = {c: dict(delta=v, macro_contribution=v/7) for c,v in changes.items()}
    rr = [r for r in rows if r['size'] == 'large']
    result['large_fixed_threshold'] = {}
    for ref in ['E0', 'E1']:
        common = [r for r in rr if r[ref]['matched'] and r['E1-P']['matched']]
        lost = [r for r in rr if r[ref]['matched'] and not r['E1-P']['matched']]
        result['large_fixed_threshold'][ref] = dict(common=len(common), lost=len(lost),
            lost_reasons=dict(Counter(r['E1-P']['miss_reason'] for r in lost)),
            crossed_iou={str(t):dict(down=sum(r[ref]['mask_iou'] >= t > r['E1-P']['mask_iou'] for r in common),
                                    up=sum(r['E1-P']['mask_iou'] >= t > r[ref]['mask_iou'] for r in common)) for t in [.75,.85,.9,.95]},
            worst_mask_gt_ids=[r['gt_id'] for r in sorted(common,key=lambda r:r['E1-P']['mask_iou']-r[ref]['mask_iou'])[:8]])
    result['passed'] = True
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('E1P_AP_DROP_DIAGNOSTIC_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
