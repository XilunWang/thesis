"""Standalone query-aligned BLL candidate; not registered into MMDetection.

Final-layer assignment is supplied, never recomputed. The previous layer uses
the same query indices even if its own training assignment was different.
This changes WaterMask's RoI/multiresolution setting to a shared query grid.
"""
import math
import torch
import torch.nn.functional as F
if __package__:
    from .bll_core import boundary
else:
    from bll_core import boundary


def query_boundary_loss(previous, final, gt_masks, final_matches, valid_shapes,
                        *, loss_weight, boundary_width=3):
    """Return weighted loss and diagnostic counts (CPU/FP32 candidate).

    previous/final: [B,Q,h,w]; gt_masks: B padded binary [Gi,Hp,Wp].
    final_matches: B tuples (positive query indices, corresponding GT indices),
    directly from the SAME final-layer assignment used for baseline losses.
    valid_shapes: B resized, unpadded (H,W), with top-left image placement.
    Integer stride and equal previous/final grids are required; partial cells
    at the valid-image edge are excluded. No input tensor is modified.
    """
    if not math.isfinite(loss_weight) or loss_weight < 0:
        raise ValueError('Expected finite nonnegative loss weight')
    # Bypass matching and boundary computation when disabled. This alone is not
    # a whole-head lambda=0 equivalence test (the head is not integrated yet).
    if loss_weight == 0:
        return final.sum() * 0, dict(disabled=True, matched=0, selected_pixels=0)
    if (final.ndim != 4 or previous.shape != final.shape or
            final.dtype != torch.float32 or previous.dtype != torch.float32 or
            previous.device != final.device or min(final.shape) < 1):
        raise ValueError('Expected same-grid nonempty FP32 [B,Q,h,w] tensors')
    if not torch.isfinite(final).all() or not torch.isfinite(previous).all():
        raise ValueError('Nonfinite logits')
    b, q, h, w = final.shape
    if not (len(gt_masks) == len(final_matches) == len(valid_shapes) == b):
        raise ValueError('Batch metadata lengths differ')
    numerator = final.sum() * 0
    denominator = 0
    rows = []
    for i, (gt, pair, valid) in enumerate(zip(gt_masks, final_matches, valid_shapes)):
        if gt.ndim != 3 or gt.device != final.device:
            raise ValueError('GT must be colocated [G,Hp,Wp]')
        hp, wp = gt.shape[-2:]
        if hp < h or wp < w or hp % h or wp % w:
            raise ValueError('Padded GT dimensions must have integer grid stride')
        vh, vw = valid
        if not (isinstance(vh, int) and isinstance(vw, int) and 0 < vh <= hp and 0 < vw <= wp):
            raise ValueError('Invalid unpadded image shape')
        qi, gi = pair
        if (qi.ndim != 1 or gi.ndim != 1 or qi.shape != gi.shape or
                qi.dtype != torch.long or gi.dtype != torch.long or
                qi.device != final.device or gi.device != final.device):
            raise ValueError('Matched indices must be colocated int64 vectors')
        if (qi.unique().numel() != qi.numel() or gi.unique().numel() != gi.numel() or
                (qi < 0).any() or (qi >= q).any() or
                (gi < 0).any() or (gi >= gt.shape[0]).any()):
            raise ValueError('Invalid or non-one-to-one final assignment')
        if not torch.all((gt == 0) | (gt == 1)):
            raise ValueError('Binary GT required')
        if gt[:, vh:, :].any() or gt[:, :, vw:].any():
            raise ValueError('GT padding must be zero')
        n = qi.numel()
        gh, gw = vh // (hp // h), vw // (wp // w)
        if n == 0:
            rows.append(dict(matched=0, selected_pixels=0, valid_grid=[gh, gw]))
            continue
        with torch.no_grad():
            target = F.interpolate(gt[gi, None].float(), size=(h, w), mode='nearest')[:, 0]
            valid_grid = torch.zeros((h, w), dtype=torch.bool, device=final.device)
            valid_grid[:gh, :gw] = True
            target = target * valid_grid
            prev_binary = (previous[i, qi].detach().sigmoid() >= .5) & valid_grid
            region = (boundary(prev_binary, boundary_width) |
                      boundary(target, boundary_width)) & valid_grid
        pixels = F.binary_cross_entropy_with_logits(final[i, qi], target, reduction='none')
        numerator = numerator + pixels[region].sum()
        count = int(region.sum())
        denominator += count
        rows.append(dict(matched=n, selected_pixels=count, valid_grid=[gh, gw]))
    loss = loss_weight * numerator / max(denominator, 1)
    return loss, dict(disabled=False, matched=sum(r['matched'] for r in rows),
                     selected_pixels=denominator, per_image=rows,
                     normalization='single-process global selected pixel mean')
