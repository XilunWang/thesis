"""MMDetection 3.3.0: optional valid-region sampling for matching only."""
import torch
from mmcv.ops import point_sample
from mmengine.structures import InstanceData
from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.registry import MODELS


@MODELS.register_module()
class ValidRegionMask2FormerHead(Mask2FormerHead):
    """No new parameters; inference and mask-loss sampling are inherited."""

    def __init__(self, *args, match_valid_region=False, **kwargs):
        if type(match_valid_region) is not bool:
            raise TypeError('match_valid_region must be bool')
        super().__init__(*args, **kwargs)
        self.match_valid_region = match_valid_region

    def _get_targets_single(self, cls_score, mask_pred, gt_instances, img_meta):
        if not self.match_valid_region:
            return super()._get_targets_single(
                cls_score, mask_pred, gt_instances, img_meta)
        gt_labels, gt_masks = gt_instances.labels, gt_instances.masks
        height, width = gt_masks.shape[-2:]
        valid_height, valid_width = img_meta['img_shape'][:2]
        if not (0 < valid_height <= height and 0 < valid_width <= width):
            raise ValueError('img_shape must fit the padded GT mask canvas')
        # Preserve the exact parent path when there is no padding (or no GT).
        if (valid_height, valid_width) == (height, width) or len(gt_labels) == 0:
            return super()._get_targets_single(
                cls_score, mask_pred, gt_instances, img_meta)

        # Same point count/RNG consumption as upstream; only coordinates differ.
        num_queries, num_gts = cls_score.shape[0], gt_labels.shape[0]
        point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)
        point_coords = point_coords * point_coords.new_tensor(
            [valid_width / width, valid_height / height])
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1)).squeeze(1)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1, 1)).squeeze(1)
        sampled_gt = InstanceData(labels=gt_labels, masks=gt_points_masks)
        sampled_pred = InstanceData(scores=cls_score, masks=mask_points_pred)
        assignment = self.assigner.assign(
            pred_instances=sampled_pred, gt_instances=sampled_gt, img_meta=img_meta)
        sampling = self.sampler.sample(
            assign_result=assignment,
            pred_instances=InstanceData(scores=cls_score, masks=mask_pred),
            gt_instances=gt_instances)
        pos_inds, neg_inds = sampling.pos_inds, sampling.neg_inds
        labels = gt_labels.new_full((self.num_queries,), self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries,))
        mask_targets = gt_masks[sampling.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries,))
        mask_weights[pos_inds] = 1.0
        return (labels, label_weights, mask_targets, mask_weights,
                pos_inds, neg_inds, sampling)
