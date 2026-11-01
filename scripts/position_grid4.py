"""Revision grid4-v1: four spatial subcells per detached stride8 feature."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.isolated_position_head import IsolatedPositionHead, position_loss

REPRESENTATION = 'grid4-v1'
STRIDE = 4


class Grid4PositionHead(IsolatedPositionHead):
    def __init__(self, width=128):
        super().__init__(width)
        self.predictor[-1] = nn.Conv2d(width, 4, 1)

    def forward(self, c3, c4):
        a = self.c3(self.normalize(c3))
        b = self.c4(self.normalize(c4))
        hidden = F.relu(a + F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False))
        # Channel 2*dy+dx maps to fine pixel [2*y+dy, 2*x+dx].
        return F.pixel_shuffle(self.predictor(hidden), 2)[:, 0]


def make_targets(masks, annotation_ids, original_areas, valid_shape):
    n, h, w = masks.shape
    assert h % STRIDE == w % STRIDE == 0
    assert n == len(annotation_ids) == len(original_areas)
    assert len(set(annotation_ids)) == n
    vh, vw = valid_shape[:2]
    assert 0 < vh <= h and 0 < vw <= w
    assert not masks[:, vh:, :].any() and not masks[:, :, vw:].any()
    candidates = {}
    gw = w // STRIDE
    for gid, area, mask in zip(annotation_ids, original_areas, masks):
        if not 0 < area <= 1024:
            continue
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError(f'Empty retained small target: {gid}')
        j = np.argmin((xx-xx.mean())**2 + (yy-yy.mean())**2)
        cx, cy = int(xx[j])//STRIDE, int(yy[j])//STRIDE
        counts = mask.reshape(h//STRIDE, STRIDE, gw, STRIDE).sum((1, 3))
        candidates[int(gid)] = sorted(map(int, np.flatnonzero(counts)), key=lambda c: (
            (c//gw-cy)**2 + (c % gw-cx)**2, -int(counts.flat[c]), c))
    owner = {}

    def visit(gid, seen):
        for cell in candidates[gid]:
            if cell in seen:
                continue
            seen.add(cell)
            if cell not in owner or visit(owner[cell], seen):
                owner[cell] = gid
                return True
        return False

    for gid in sorted(candidates):
        if not visit(gid, set()):
            raise ValueError('Grid4 occupied-cell assignment deficit; no target was discarded')
    target = torch.zeros(h//STRIDE, gw)
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[:(vh+STRIDE-1)//STRIDE, :(vw+STRIDE-1)//STRIDE] = True
    for cell in owner:
        target.view(-1)[cell] = 1
    assert not target[~valid].any()
    return target, valid, {gid: [cell % gw, cell//gw] for cell, gid in owner.items()}


def select(logits, valid_shape, k=100):
    """Fine-grid top-k; no score threshold, distance suppression or GT input."""
    if logits.ndim != 2 or k <= 0:
        raise ValueError('Expected 2D logits and positive budget')
    h, w = valid_shape[:2]
    gh, gw = logits.shape
    if not (0 < h <= gh*STRIDE and 0 < w <= gw*STRIDE):
        raise ValueError('Invalid image extent')
    yy, xx = np.indices(logits.shape)
    cells = np.flatnonzero((yy*STRIDE < h) & (xx*STRIDE < w))
    scores = logits.ravel()[cells]
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite valid score')
    chosen = cells[np.lexsort((cells, -scores))[:k]]
    cy, cx = chosen//gw, chosen % gw
    xy = np.stack(((cx*STRIDE+np.minimum(cx*STRIDE+STRIDE, w))/(2*w),
                   (cy*STRIDE+np.minimum(cy*STRIDE+STRIDE, h))/(2*h)), axis=1)
    return chosen, xy
