"""Optional R50 C3/C4 residual access for original Mask2Former query memory.

Only forward memory changes. Loss, assignment, heads and prediction are inherited.
"""
from typing import List, Tuple
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.registry import MODELS
from mmdet.structures import SampleList


@MODELS.register_module()
class DirectQueryFeatureHead(Mask2FormerHead):
    def __init__(self, in_channels, *args, direct_query_features=False, **kwargs):
        if type(direct_query_features) is not bool:
            raise TypeError('direct_query_features must be bool')
        if list(in_channels) != [256, 512, 1024, 2048]:
            raise ValueError('This experiment is defined for the existing R50 features')
        super().__init__(in_channels, *args, **kwargs)
        if self.decoder_embed_dims != 256 or self.num_transformer_feat_level != 3:
            raise ValueError('Expected original 256-channel, 3-scale decoder')
        self.direct_query_features = direct_query_features
        # These CPU-created projections are zeroed immediately. Do not let their
        # discarded random initialization perturb the original training RNG.
        with torch.random.fork_rng(devices=[]):
            self.direct_c3 = nn.Conv2d(512, 256, 1, bias=False)
            self.direct_c4 = nn.Conv2d(1024, 256, 1, bias=False)
            self._zero_direct_weights()

    def _zero_direct_weights(self):
        nn.init.zeros_(self.direct_c3.weight)
        nn.init.zeros_(self.direct_c4.weight)

    def init_weights(self):
        super().init_weights()
        self._zero_direct_weights()

    def add_direct_features(self, memory, features, level):
        if level == 0:
            return memory
        if level not in (1, 2):
            raise ValueError('Unexpected memory index')
        raw = features[2] if level == 1 else features[1]
        projection = self.direct_c4 if level == 1 else self.direct_c3
        if raw.shape[0] != memory.shape[0] or raw.shape[-2:] != memory.shape[-2:]:
            raise ValueError('Raw R50 and query memory must align; no implicit resizing')
        normalized = F.layer_norm(raw.permute(0, 2, 3, 1), (raw.shape[1],), eps=1e-5)
        residual = projection(normalized.permute(0, 3, 1, 2).contiguous())
        return memory + residual

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        if not self.direct_query_features:
            return super().forward(x, batch_data_samples)
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = self.add_direct_features(decoder_input, x, i)
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

