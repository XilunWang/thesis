"""P2 boundary pilot: synthetic checks and fixed first six images; never AP.

Boundary convention: bowenc0221/boundary-iou-api @
37d25586a677b043ed585f10e5c42d4e80176ea9, utils/boundary_utils.py.
Independent NumPy implementation; OpenCV equivalence remains a separate gate.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATIO = 0.02


def binary(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not all(mask.shape) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected a nonzero-size 2D binary mask')
    return mask.astype(bool)


def boundary(mask, ratio=RATIO):
    """Inner band; square erosion, image exterior is background."""
    mask = binary(mask)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError('ratio must be positive and finite')
    h, w = mask.shape
    radius = max(1, int(round(ratio * np.hypot(h, w))))
    if 2 * radius + 1 > min(h, w):
        return mask.copy()
    padded = np.pad(mask.astype(np.int64), radius)
    summed = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    k = 2 * radius + 1
    sums = summed[k:, k:] - summed[:-k, k:] - summed[k:, :-k] + summed[:-k, :-k]
    return mask & (sums != k * k)


def iou(a, b):
    a, b = binary(a), binary(b)
    if a.shape != b.shape:
        raise ValueError('Mask shapes differ')
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else None


def scores(a, b):
    mi = iou(a, b)
    bi = iou(boundary(a), boundary(b))
    return dict(mask_iou=mi, boundary_iou=bi,
                combined_iou=min(mi, bi) if mi is not None and bi is not None else None)


def synthetic():
    base = np.zeros((100, 100), dtype=bool)
    base[20:80, 20:80] = True
    shifted = np.zeros_like(base)
    shifted[20:80, 23:83] = True
    empty = np.zeros_like(base)
    disjoint = np.zeros_like(base)
    disjoint[0:10, 0:10] = True
    full = np.ones_like(base)
    thin = np.zeros_like(base)
    thin[50, 10:90] = True
    cases = {name: scores(a, b) for name, a, b in [
        ('identical', base, base), ('shift_3px', base, shifted),
        ('disjoint', base, disjoint), ('one_empty', base, empty),
        ('both_empty', empty, empty), ('full_image', full, full),
        ('thin', thin, thin)]}
    assert cases['identical']['boundary_iou'] == 1
    assert cases['disjoint']['boundary_iou'] == 0
    assert cases['one_empty']['boundary_iou'] == 0
    assert cases['both_empty']['boundary_iou'] is None
    assert np.count_nonzero(boundary(base)) == 60**2 - 54**2
    assert np.count_nonzero(boundary(full)) == 100**2 - 94**2
    assert np.array_equal(boundary(thin), thin)
    assert abs(cases['shift_3px']['mask_iou'] - 3420/3780) < 1e-12
    assert 0 < cases['shift_3px']['boundary_iou'] < cases['shift_3px']['mask_iou']
    assert scores(base, shifted) == scores(shifted, base)
    # Independent slow neighborhood oracle, including borders and holes.
    rng = np.random.default_rng(20261021)
    comparisons = 0
    for shape in [(1, 1), (7, 9), (20, 25)]:
        for ratio in [0.02, 0.12]:
            for mask in [rng.random(shape) > .2, np.ones(shape, dtype=bool)]:
                radius = max(1, round(ratio * np.hypot(*shape)))
                padded = np.pad(mask, radius)
                expected = np.zeros(shape, dtype=bool)
                for y, x in np.ndindex(shape):
                    expected[y, x] = mask[y, x] and not padded[y:y+2*radius+1, x:x+2*radius+1].all()
                assert np.array_equal(boundary(mask, ratio), expected)
                comparisons += 1
    for invalid in [np.array([[2]]), np.zeros((0, 1)), np.zeros((2, 2, 1))]:
        try:
            boundary(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')
    return dict(passed=True, cases=cases, oracle_comparisons=comparisons,
                official_opencv_runtime_comparison='not_run')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pilot():
    # No custom polygon/RLE decoder fallback: use the established COCO decoder.
    from pycocotools.coco import COCO
    from pycocotools import mask as mu
    spec = importlib.util.spec_from_file_location('error_analysis', ROOT/'scripts/analyze-uiis-errors.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    analysis.self_test()
    ann = ROOT/'data/UIIS/annotations/val.json'
    pred = ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json'
    diag = json.loads((ROOT/'reports/uiis-error-analysis/diagnostics.json').read_text(encoding='utf8'))
    assert sha(ann) == diag['annotation_sha256'] and sha(pred) == diag['prediction_sha256']
    coco = COCO(str(ann))
    selected = list(coco.imgs)[:6]
    assert [coco.imgs[i]['file_name'] for i in selected] == [
        'L_1000.jpg', 'L_1001.jpg', 'L_1025.jpg', 'L_1029.jpg', 'L_1032.jpg', 'L_1046.jpg']
    predictions = json.loads(pred.read_text())
    expected = {row['image_id']: row for row in diag['per_image']}
    rows = []
    for iid in selected:
        assert not any(a.get('ignore', 0) and not a.get('iscrowd', 0) for a in coco.imgToAnns[iid])
        gt = [a for a in coco.imgToAnns[iid] if not a.get('iscrowd', 0)]
        crowd = [a for a in coco.imgToAnns[iid] if a.get('iscrowd', 0)]
        shape = (coco.imgs[iid]['height'], coco.imgs[iid]['width'])
        for a in gt + crowd:
            decoded = binary(coco.annToMask(a))
            assert decoded.shape == shape and decoded.any(), ('Invalid GT mask', a['id'])
        pp = sorted([p for p in predictions if p['image_id'] == iid and p['score'] >= .5], key=lambda p: -p['score'])
        assert all(binary(mu.decode(p['segmentation'])).shape == shape for p in pp)
        matrix = mu.iou([p['segmentation'] for p in pp], [coco.annToRLE(a) for a in gt], [0]*len(gt)) if pp and gt else np.zeros((len(pp), len(gt)))
        labels, matches, used = analysis.match(matrix, [p['category_id'] for p in pp], [a['category_id'] for a in gt])
        ciou = mu.iou([p['segmentation'] for p in pp], [coco.annToRLE(a) for a in crowd], [1]*len(crowd)) if pp and crowd else np.zeros((len(pp), len(crowd)))
        ignored = sum(label != 'matched' and any(a['category_id'] == pp[k]['category_id'] and ciou[k, j] >= .5 for j, a in enumerate(crowd)) for k, label in enumerate(labels))
        counts = dict(gt=len(gt), matched=len(used), missed=len(gt)-len(used), predictions=len(pp), ignored_crowd=ignored)
        for key, value in counts.items():
            assert value == expected[iid]['thresholds']['0.5'].get(key, 0), (iid, key)
        counts['unmatched_nonignored_predictions'] = len(pp)-len(used)-ignored
        pairs = []
        for k, j in enumerate(matches):
            if j is None:
                continue
            pm, gm = mu.decode(pp[k]['segmentation']), coco.annToMask(gt[j])
            result = scores(pm, gm)
            assert abs(result['mask_iou']-float(matrix[k, j])) < 1e-10
            pairs.append(dict(gt_id=gt[j]['id'], prediction_rank=k+1, category_id=gt[j]['category_id'], gt_area=gt[j]['area'], score=pp[k]['score'], **result))
        rows.append(dict(image_id=iid, file_name=coco.imgs[iid]['file_name'], counts=counts, pairs=pairs))
    values = [p['boundary_iou'] for row in rows for p in row['pairs'] if p['boundary_iou'] is not None]
    return dict(passed=True, annotation_sha256=sha(ann), prediction_sha256=sha(pred),
                mean_matched_boundary_iou=float(np.mean(values)) if values else None,
                matched_pair_count=len(values), images=rows, metric='conditional pair diagnostic, not AP')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/uiis-boundary-pilot-v1')
    args = parser.parse_args()
    if args.output.resolve() == ROOT.resolve() or args.output.exists():
        raise FileExistsError('Use a new dedicated output directory; existing results are protected')
    result = dict(protocol='docs/boundary-evaluation-protocol.md', ratio=RATIO,
                  python=platform.python_version(), numpy=np.__version__, script_sha256=sha(Path(__file__)),
                  base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  synthetic=synthetic())
    code = 0
    try:
        result['fixed_six'] = pilot()
    except (ImportError, PermissionError) as error:
        result['fixed_six'] = dict(passed=False, status='blocked', error=f'{type(error).__name__}: {error}')
        code = 2
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
