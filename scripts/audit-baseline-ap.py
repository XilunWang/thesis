"""Offline independent COCOeval audit; never modify data or checkpoints."""
import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.local/analysis-libs'))
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    out = ROOT / 'reports/baseline-ap-audit-20261029'
    out.mkdir(exist_ok=True)
    data = ROOT / 'data/UIIS'
    manifest = json.loads((data / 'download-manifest.json').read_text())
    result = {'scope': 'saved predictions, not fresh model inference', 'splits': {}}
    hashes = {}
    datasets = {}
    for split in ['train', 'val']:
        path = data / 'annotations' / (split + '.json')
        ds = datasets[split] = json.loads(path.read_text())
        expected = next(x['lfs']['oid'] for x in manifest['files'] if x['path'] == 'annotations/' + split + '.json')
        actual = sha(path)
        assert actual == expected
        ids = {x['id'] for x in ds['images']}
        cats = {x['id'] for x in ds['categories']}
        assert len(ids) == len(ds['images'])
        assert len({a['id'] for a in ds['annotations']}) == len(ds['annotations'])
        assert all(a['image_id'] in ids and a['category_id'] in cats for a in ds['annotations'])
        groups = hashes[split] = defaultdict(list)
        for im in ds['images']:
            groups[sha(data / split / im['file_name'])].append({'id': im['id'], 'file_name': im['file_name']})
        result['splits'][split] = dict(images=len(ids), annotations=len(ds['annotations']), categories=ds['categories'], sha256=actual, matches_saved_source_manifest=True)
    assert datasets['train']['categories'] == datasets['val']['categories']
    overlaps = sorted(set(hashes['train']) & set(hashes['val']))
    result['exact_cross_split_duplicates'] = [dict(train=hashes['train'][h], val=hashes['val'][h]) for h in overlaps]
    excluded = {im['id'] for h in overlaps for im in hashes['val'][h]}
    predpath = ROOT / 'reports/uiis-baseline-36e-evaluation/predictions.segm.json'
    predictions = json.loads(predpath.read_text())
    gt = COCO(str(data / 'annotations/val.json'))
    assert {p['image_id'] for p in predictions} == set(gt.getImgIds())
    assert all(p['category_id'] in gt.cats and np.isfinite(p['score']) and 0 <= p['score'] <= 1 for p in predictions)
    assert all(p['segmentation']['size'] == [gt.imgs[p['image_id']]['height'], gt.imgs[p['image_id']]['width']] for p in predictions)
    # Match MMDetection segm evaluation: mask area, not predicted box area.
    for p in predictions:
        p.pop('bbox', None)
    dt = gt.loadRes(predictions)
    result['prediction_sha256'] = sha(predpath)
    result['prediction_count'] = len(predictions)
    for name, ids in [('official_691', gt.getImgIds()), ('sensitivity_without_exact_overlap', sorted(set(gt.getImgIds()) - excluded))]:
        ev = COCOeval(gt, dt, 'segm')
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        result[name] = dict(images=len(ids), stats=ev.stats.tolist(), useCats=ev.params.useCats, maxDets=ev.params.maxDets, iouThrs=ev.params.iouThrs.tolist())
        if name == 'official_691':
            result['per_category_ap'] = {}
            for k, cid in enumerate(ev.params.catIds):
                values = ev.eval['precision'][:, :, k, 0, -1]
                result['per_category_ap'][gt.cats[cid]['name']] = float(values[values >= 0].mean())
    previous = json.loads((ROOT / 'reports/uiis-baseline-36e-evaluation/metrics.json').read_text())
    keys = ['coco/segm_mAP', 'coco/segm_mAP_50', 'coco/segm_mAP_75', 'coco/segm_mAP_s', 'coco/segm_mAP_m', 'coco/segm_mAP_l']
    result['matches_all_six_reported_rounded_metrics'] = all(round(v, 3) == previous['metrics'][k] for v, k in zip(result['official_691']['stats'], keys))
    assert result['matches_all_six_reported_rounded_metrics']
    result['duplicate_sensitivity_ap_points'] = 100 * (result['sensitivity_without_exact_overlap']['stats'][0] - result['official_691']['stats'][0])
    (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('AUDIT_OK', result['official_691']['stats'][:3], 'duplicate_delta', result['duplicate_sensitivity_ap_points'])

if __name__ == '__main__':
    main()
