"""Experimental detached C3/C4 seed head; no queries, masks or output merging."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_targets(masks, annotation_ids, original_areas, valid_shape):
    """One positive per original-small instance, using occupied-cell matching.

    A partial edge cell is valid when it intersects the unpadded image.
    Fail explicitly on a matching deficit; never silently drop a target.
    """
    n, h, w = masks.shape
    assert h % 8 == w % 8 == 0
    assert n == len(annotation_ids) == len(original_areas)
    assert len(set(annotation_ids)) == n
    vh, vw = valid_shape[:2]
    assert 0 < vh <= h and 0 < vw <= w
    assert not masks[:, vh:, :].any() and not masks[:, :, vw:].any()
    candidates = {}
    for gid, area, mask in zip(annotation_ids, original_areas, masks):
        if not 0 < area <= 1024:
            continue
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError(f'Empty retained small target: {gid}')
        j = np.argmin((xx-xx.mean())**2 + (yy-yy.mean())**2)
        cx, cy = int(xx[j])//8, int(yy[j])//8
        counts = mask.reshape(h//8, 8, w//8, 8).sum((1, 3))
        candidates[int(gid)] = sorted(map(int, np.flatnonzero(counts)), key=lambda c: (
            (c//(w//8)-cy)**2 + (c % (w//8)-cx)**2, -int(counts.flat[c]), c))
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
            raise ValueError('Occupied-cell assignment deficit; no target was discarded')
    target = torch.zeros(h//8, w//8)
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[:(vh+7)//8, :(vw+7)//8] = True
    for cell in owner:
        target.view(-1)[cell] = 1
    assert not target[~valid].any()
    return target, valid, {gid: [cell % (w//8), cell//(w//8)] for cell, gid in owner.items()}


class IsolatedPositionHead(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.c3 = nn.Conv2d(512, width, 1)
        self.c4 = nn.Conv2d(1024, width, 1)
        self.predictor = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, 1, 1))

    @staticmethod
    def normalize(x):
        x = x.detach()
        return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)

    def forward(self, c3, c4):
        a = self.c3(self.normalize(c3))
        b = self.c4(self.normalize(c4))
        return self.predictor(F.relu(a + F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)))[:, 0]


def position_loss(logits, targets, valid):
    """Per-image equally weighted positive/negative means; empty images use negatives.

    Non-seed valid cells are negatives, not semantic background labels.
    Index first so invalid padding logits cannot influence the loss.
    """
    assert logits.shape == targets.shape == valid.shape
    losses = []
    for pred, target, keep in zip(logits, targets, valid):
        assert keep.any() and not target[~keep].any()
        p, t = pred[keep], target[keep]
        raw = F.binary_cross_entropy_with_logits(p, t, reduction='none')
        terms = [raw[t == value].mean() for value in (0, 1) if (t == value).any()]
        losses.append(torch.stack(terms).mean())
    return torch.stack(losses).mean()
