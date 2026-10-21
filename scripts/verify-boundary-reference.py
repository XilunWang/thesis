"""Compare local boundaries against the unmodified, pinned upstream function.

Fetch SOURCE_URL into work/boundary-reference/boundary_utils.py before running.
Requires numpy, Pillow, pycocotools and opencv-python-headless.
Writes a separate report inside the existing pilot directory, never overwrites.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = 'https://raw.githubusercontent.com/bowenc0221/boundary-iou-api/37d25586a677b043ed585f10e5c42d4e80176ea9/boundary_iou/utils/boundary_utils.py'
SOURCE_HASH = 'a495cc15fdbded38681edef30b29c62014e81e864f0e68ca6062ce42db51fd9c'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT/'work/boundary-reference/boundary_utils.py')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/uiis-boundary-pilot/reference-comparison.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite reference comparison')
    local = load('boundary_pilot', ROOT/'scripts/validate-uiis-boundary.py')
    assert local.sha(args.reference) == SOURCE_HASH, 'Upstream source hash mismatch'
    official = load('official_boundary_utils', args.reference)
    tests = []

    def compare(name, mask, ratio=.02):
        mask = np.asarray(mask, dtype=np.uint8)
        a = local.boundary(mask, ratio)
        b = official.mask_to_boundary(mask, ratio)
        difference = int(np.count_nonzero(a != b))
        tests.append(dict(name=name, shape=list(mask.shape), ratio=ratio, differing_pixels=difference))
        return b

    rng = np.random.default_rng(20261021)
    for shape in [(1, 1), (1, 17), (7, 9), (20, 25), (100, 100), (255, 341)]:
        masks = {'empty': np.zeros(shape, np.uint8), 'full': np.ones(shape, np.uint8),
                 'random': (rng.random(shape) > .2).astype(np.uint8)}
        m = np.zeros(shape, np.uint8)
        m[shape[0]//2, :] = 1
        masks['thin'] = m
        m = np.zeros(shape, np.uint8)
        m[:max(1, shape[0]//2), :max(1, shape[1]//2)] = 1
        masks['touches_border'] = m
        m = np.ones(shape, np.uint8)
        m[shape[0]//3:2*shape[0]//3, shape[1]//3:2*shape[1]//3] = 0
        masks['hole'] = m
        for name, m in masks.items():
            for ratio in [.005, .02, .12]:
                compare(f'synthetic/{shape}/{name}', m, ratio)
    synthetic_count = len(tests)
    # Recheck the established pilot, including the original synthetic IoU tests.
    local.synthetic()
    stored_path = ROOT/'reports/uiis-boundary-pilot/validation.json'
    stored_hash = local.sha(stored_path)
    stored = json.loads(stored_path.read_text(encoding='utf8'))['fixed_six']
    current = local.pilot()
    assert current == stored, 'Existing pilot result changed'
    coco = COCO(str(ROOT/'data/UIIS/annotations/val.json'))
    predictions = json.loads((ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json').read_text())
    pair_checks = []
    for row in stored['images']:
        iid = row['image_id']
        pp = sorted([p for p in predictions if p['image_id'] == iid and p['score'] >= .5], key=lambda p: -p['score'])
        gb = {a['id']: compare(f'image/{iid}/gt/{a["id"]}', coco.annToMask(a)) for a in coco.imgToAnns[iid]}
        pb = {k+1: compare(f'image/{iid}/prediction/{k+1}', mu.decode(p['segmentation'])) for k, p in enumerate(pp)}
        for pair in row['pairs']:
            g, p = gb[pair['gt_id']], pb[pair['prediction_rank']]
            # Independent IoU via official COCO RLE, not local scalar routine.
            score = float(mu.iou([mu.encode(np.asfortranarray(p))], [mu.encode(np.asfortranarray(g))], [0])[0, 0])
            pair_checks.append(dict(image_id=iid, gt_id=pair['gt_id'], boundary_iou=score,
                                    absolute_error=abs(score-pair['boundary_iou'])))
    passed = all(t['differing_pixels'] == 0 for t in tests) and all(p['absolute_error'] <= 1e-12 for p in pair_checks)
    assert local.sha(stored_path) == stored_hash
    result = dict(passed=passed, source_url=SOURCE_URL, source_sha256=SOURCE_HASH,
                  script_sha256=local.sha(Path(__file__)), local_script_sha256=local.sha(ROOT/'scripts/validate-uiis-boundary.py'),
                  base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  environment=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                                   pycocotools=importlib.metadata.version('pycocotools')),
                  pilot_sha256=stored_hash, annotation_sha256=current['annotation_sha256'],
                  prediction_sha256=current['prediction_sha256'], synthetic_mask_comparisons=synthetic_count,
                  real_mask_comparisons=len(tests)-synthetic_count, matched_pair_comparisons=len(pair_checks),
                  differing_pixels=sum(t['differing_pixels'] for t in tests),
                  max_pair_absolute_error=max(p['absolute_error'] for p in pair_checks),
                  tests=tests, pairs=pair_checks,
                  scope='Boundary extraction equivalence and fixed-six paired IoU only; not Boundary AP or full dataset evaluation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('tests', 'pairs')}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
