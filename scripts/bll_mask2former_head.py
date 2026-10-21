"""Opt-in MMDetection 3.3.0 query BLL probe, single-process FP32 only."""
import math

import torch
from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.registry import MODELS

from .query_bll import query_boundary_loss


@MODELS.register_module()
class BLLMask2FormerHead(Mask2FormerHead):
    def __init__(self, *args, bll_loss_weight=0.0, bll_boundary_width=3, **kwargs):
        if not math.isfinite(bll_loss_weight) or bll_loss_weight < 0:
            raise ValueError('BLL weight must be finite and nonnegative')
        if not isinstance(bll_boundary_width, int) or bll_boundary_width < 1:
            raise ValueError('BLL width must be a positive integer')
        super().__init__(*args, **kwargs)
        self.bll_loss_weight = bll_loss_weight
        self.bll_boundary_width = bll_boundary_width
        self._bll_matches = None

    def _get_targets_single(self, *args, **kwargs):
        result = super()._get_targets_single(*args, **kwargs)
        if self._bll_matches is not None:
            sampling = result[6]
            self._bll_matches.append((sampling.pos_inds,
                                      sampling.pos_assigned_gt_inds))
        return result

    def loss_by_feat(self, all_cls_scores, all_mask_preds,
                     batch_gt_instances, batch_img_metas):
        if self.bll_loss_weight == 0:
            return super().loss_by_feat(all_cls_scores, all_mask_preds,
                                        batch_gt_instances, batch_img_metas)
        if self._bll_matches is not None:
            raise RuntimeError('Reentrant BLL loss calls are unsupported')
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError('BLL probe supports a single process only')
        if len(all_mask_preds) < 2 or len(all_mask_preds) != len(all_cls_scores):
            raise ValueError('BLL requires at least two matching decoder layers')
        if any(p.dtype != torch.float32 for p in all_mask_preds):
            raise ValueError('BLL probe requires FP32 logits; disable AMP')
        batch = len(batch_gt_instances)
        if batch == 0 or len(batch_img_metas) != batch:
            raise ValueError('Invalid BLL batch metadata')
        # The pinned parent traverses layers in order, then images in order.
        # Observe the existing assignments: never call get_targets a second time.
        matches = []
        self._bll_matches = matches
        try:
            losses = super().loss_by_feat(all_cls_scores, all_mask_preds,
                                          batch_gt_instances, batch_img_metas)
        finally:
            self._bll_matches = None
        if len(matches) != batch * len(all_mask_preds):
            raise RuntimeError('Unexpected parent assignment traversal; stop BLL probe')
        loss, _ = query_boundary_loss(
            all_mask_preds[-2], all_mask_preds[-1],
            [gt.masks for gt in batch_gt_instances], matches[-batch:],
            [tuple(meta['img_shape'][:2]) for meta in batch_img_metas],
            loss_weight=self.bll_loss_weight, boundary_width=self.bll_boundary_width)
        losses['loss_bll'] = loss
        return losses
