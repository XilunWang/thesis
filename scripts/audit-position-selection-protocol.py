"""Offline reference selector and budget bounds; no learned proposal evaluation."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.local/analysis-libs'))
import numpy as np


def select(logits, valid_shape, k=100):
    """Rank valid stride8 cells, no threshold/NMS, row-major tie break.

    Coordinates are centers of each cell's intersection with the valid image,
    normalized by valid width/height. They are references, not foreground pixels.
    """
    if logits.ndim != 2 or k <= 0:
        raise ValueError('Expected 2D logits and positive budget')
    h, w = valid_shape
    gh, gw = logits.shape
    if not (0 < h <= gh*8 and 0 < w <= gw*8):
        raise ValueError('Invalid image extent')
    yy, xx = np.indices(logits.shape)
    cells = np.flatnonzero((yy*8 < h) & (xx*8 < w))
    scores = logits.ravel()[cells]
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite valid score')
    chosen = cells[np.lexsort((cells, -scores))[:k]]
    cy, cx = chosen//gw, chosen % gw
    xy = np.stack(((cx*8+np.minimum(cx*8+8, w))/(2*w),
                   (cy*8+np.minimum(cy*8+8, h))/(2*h)), axis=1)
    return chosen, xy


def main():
    out = ROOT/'reports/position-selection-protocol-20261101'
    out.mkdir(exist_ok=False)
    src = ROOT/'reports/query-cell-assignment-20261101/result.json'
    proof = json.loads(src.read_text()); assert proof['passed']
    rows = []; total = 0
    for run in proof['runs']:
        if run['filtered']:
            continue
        d = json.loads((ROOT/'reports/identity-augmentation-check-20261101'/f"image-{run['image_id']}-seed-{run['seed']}.json").read_text())
        h, w = d['padded_shape']; grid = np.zeros((h//8, w//8), dtype=np.float32)
        positives = set()
        for a in run['assignments']:
            x, y = a['cell']; grid[y, x] = 1; positives.add(y*(w//8)+x)
        total += len(positives)
        bound = {}
        for k in [20, 50, 100, 200]:
            cells, xy = select(grid, d['valid_shape'], k)
            hit = len(positives.intersection(cells.tolist()))
            assert hit == min(k, len(positives))
            assert len(set(cells)) == len(cells) and np.all((xy >= 0) & (xy < 1))
            bound[str(k)] = hit
        rows.append(dict(image_id=run['image_id'], seed=run['seed'], small=len(positives), oracle_hits=bound))
    # Adjacent unequal peaks survive; no local-maximum suppression.
    grid = np.array([[9., 8., 1.], [0., 0., 0.]])
    assert select(grid, [16, 24], 2)[0].tolist() == [0, 1]
    # Fully invalid cells can even contain NaN; partial edge cells remain valid.
    grid = np.full((3, 3), np.nan); grid[:2, :2] = 0
    chosen, xy = select(grid, [9, 9], 100)
    assert chosen.tolist() == [0, 1, 3, 4] and np.all(xy < 1)
    grid[0, 0] = np.nan
    try:
        select(grid, [9, 9])
    except ValueError:
        pass
    else:
        raise AssertionError('Valid NaN accepted')
    summary = {str(k): dict(oracle_covered=sum(r['oracle_hits'][str(k)] for r in rows),
                           instances=total, capped_samples=sum(r['small'] > k for r in rows)) for k in [20, 50, 100, 200]}
    result = dict(passed=True, training_updates=0, scope='GT-assigned oracle budget bounds only; not learned recall or AP',
                  primary_pool_k=100, diagnostic_k=[20, 50, 100, 200], summary=summary, runs=rows,
                  adjacent_preserved=True, padding_excluded=True, deterministic_ties=True,
                  source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
