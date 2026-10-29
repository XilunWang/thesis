"""Explain aggregate AP versus paired boundary changes from saved outputs only."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.local/analysis-libs'))
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'reports/e1-offline-comparison'
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    rows = json.loads((source / 'instances.json').read_text(encoding='utf-8'))
    assert len(rows) == 3748 and len({r['gt_id'] for r in rows}) == len(rows)
    annotation = ROOT / 'data/UIIS/annotations/val.json'
    assert sha(annotation) == summary['annotation_sha256']
    gt = COCO(str(annotation))
    result = {'scope': 'existing predictions and fixed matched instances; no inference',
              'input_hashes': {}, 'models': {}, 'paired': {}}
    for label, folder in [('E0', 'uiis-baseline-36e-evaluation'),
                          ('E1', 'uiis-e1-bll-36e-evaluation')]:
        path = ROOT / 'reports' / folder / 'predictions.segm.json'
        digest = sha(path)
        assert digest == summary['inputs'][label]['sha256']
        result['input_hashes'][label] = digest
        predictions = json.loads(path.read_text(encoding='utf-8'))
        for p in predictions:
            p.pop('bbox', None)
        ev = COCOeval(gt, gt.loadRes(predictions), 'segm')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        assert np.allclose(ev.stats[:6], summary['coco'][label]['stats'][:6], atol=1e-10, rtol=0)
        precision = ev.eval['precision']
        by_class = {}
        for k, cid in enumerate(ev.params.catIds):
            values = precision[:, :, k, 0, -1]
            by_class[gt.cats[cid]['name']] = float(values[values >= 0].mean() * 100)
        result['models'][label] = {'ap': float(ev.stats[0] * 100), 'by_class': by_class}
        assert sha(path) == digest
    delta = result['models']['E1']['ap'] - result['models']['E0']['ap']
    changes = {c: result['models']['E1']['by_class'][c] - v
               for c, v in result['models']['E0']['by_class'].items()}
    assert abs(sum(changes.values()) / len(changes) - delta) < 1e-10
    result['ap_delta'] = delta
    result['class_delta_and_macro_contribution'] = {
        c: {'delta': v, 'contribution': v / len(changes)} for c, v in changes.items()}
    for size in ['small', 'medium', 'large', 'all']:
        common = [r for r in rows if (size == 'all' or r['size'] == size)
                  and r['E0']['matched'] and r['E1']['matched']]
        counts = {'boundary_down_mask_up': 0, 'boundary_down_mask_not_up': 0,
                  'boundary_not_down_mask_up': 0, 'boundary_not_down_mask_not_up': 0}
        for r in common:
            down = r['E1']['boundary_iou'] < r['E0']['boundary_iou'] - 1e-12
            up = r['E1']['mask_iou'] > r['E0']['mask_iou'] + 1e-12
            counts[('boundary_down' if down else 'boundary_not_down') +
                   ('_mask_up' if up else '_mask_not_up')] += 1
        assert sum(counts.values()) == len(common)
        result['paired'][size] = {'n': len(common), 'counts': counts,
            'mean_delta_percentage_points': {key: float(np.mean([
                r['E1'][key] - r['E0'][key] for r in common]) * 100)
                for key in ['boundary_iou', 'mask_iou', 'over_fraction', 'under_fraction']},
            'boundary_declines_with_more_overfill': sum(
                r['E1']['boundary_iou'] < r['E0']['boundary_iou'] - 1e-12
                and r['E1']['over_fraction'] > r['E0']['over_fraction'] + 1e-12
                for r in common)}
    for name in ['summary.json', 'instances.json']:
        result['input_hashes'][name] = sha(source / name)
    out = ROOT / 'reports/e1-ap-boundary-analysis-20261029'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
