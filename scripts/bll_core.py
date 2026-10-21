"""CPU-testable three-stage WaterMask BLL semantics, not a Mask2Former adapter.

Reference: WaterMask mmdet2 commit 07042d254ffbdbf584ee0a5725e3c5e5e57e27c5,
cross_entropy_loss.py generate_block_target / LaplacianCrossEntropyLoss.
This independently written implementation restricts targets to binary masks and
adds an explicit graph-connected zero for zero instances. No model imports it.
"""
import torch
import torch.nn.functional as F


def boundary(mask, width=3, threshold=0.15):
    """Two-sided boundary, zero padding; unlike the evaluation inner boundary."""
    if mask.ndim != 3 or min(mask.shape[-2:]) < 1:
        raise ValueError('Expected [N,H,W] with nonzero spatial dimensions')
    if width < 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid boundary settings')
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('This pilot requires binary targets')
    if mask.shape[0] == 0:
        return torch.zeros_like(mask, dtype=torch.bool)
    mask = mask.float()
    size = 2 * width + 1
    kernel = -torch.ones((1, 1, size, size), device=mask.device)
    kernel[0, 0, width, width] = size * size - 1
    response = F.conv2d(mask[:, None], kernel, padding=width)[:, 0]
    return response.abs() > threshold * size * size


def three_stage_bll(predictions, targets):
    """Returns total, three unweighted components, and final selection mask.

    Early stages: mean BCE over all instances/pixels. Final: global selected
    pixel mean, not per-instance mean. Stage weights remain (0.25,0.65,1).
    """
    if len(predictions) != 3 or len(targets) != 3:
        raise ValueError('Exactly three stages required')
    batch = predictions[0].shape[0]
    for pred, target in zip(predictions, targets):
        if (pred.ndim != 4 or pred.shape[1] != 1 or
                target.shape != pred[:, 0].shape or pred.shape[0] != batch or
                min(pred.shape[-2:]) < 1):
            raise ValueError('Stage shape mismatch')
        if pred.device != target.device or pred.dtype != torch.float32:
            raise ValueError('Pilot requires colocated FP32 inputs')
        if not torch.isfinite(pred).all() or not torch.all((target == 0) | (target == 1)):
            raise ValueError('Expected finite logits and binary targets')
    if batch == 0:
        components = [p.sum() * 0 for p in predictions]
        region = torch.zeros_like(targets[-1], dtype=torch.bool)
    else:
        with torch.no_grad():
            previous = predictions[1][:, 0].sigmoid() >= 0.5
            selected = boundary(previous) | boundary(targets[1])
            region = F.interpolate(selected[:, None].float(),
                                   size=targets[2].shape[-2:], mode='bilinear',
                                   align_corners=True)[:, 0] >= 0.5
        components = [F.binary_cross_entropy_with_logits(p[:, 0], t.float())
                      for p, t in zip(predictions[:2], targets[:2])]
        pixels = F.binary_cross_entropy_with_logits(
            predictions[2][:, 0], targets[2].float(), reduction='none')
        components.append(pixels[region].sum() / region.sum().clamp(min=1))
    total = sum(w * value for w, value in zip((0.25, 0.65, 1.), components))
    return total, components, region
