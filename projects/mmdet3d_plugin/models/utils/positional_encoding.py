# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule

@POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding3D(BaseModule):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding3D, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        # mask的形状是[B,N,H,W]
        # mask的数值转成整型，0或1
        mask = mask.to(torch.int)
        # mask取反
        not_mask = 1 - mask  # logical_not
        # 沿着第1维度进行累加，形状不变，但每个元素是该位置及之前所有元素的累积和
        n_embed = not_mask.cumsum(1, dtype=torch.float32)
        # 沿着第2维度进行累加，形状不变，但每个元素是该位置及之前所有元素的累积和
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        # 沿着第3维度进行累加，形状不变，但每个元素是该位置及之前所有元素的累积和
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        # 分别对n_embed, y_embed, x_embed进行归一化，然后乘以scale
        # scale是2*pi，相当于是把n_embed, y_embed, x_embed归一化到[0, 2*pi]之间，为后续计算sin和cos做准备
        if self.normalize:
            n_embed = (n_embed + self.offset) / \
                      (n_embed[:, -1:, :, :] + self.eps) * self.scale
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, :, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, :, -1:] + self.eps) * self.scale
        # 构造[0~128)的等差数列，num_feats=128
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        # 构造以10000为底，以dim_t/128为指数的数列，实际是0~8000左右的一个数列，形状像底数大于1的指数函数，刚开始增加的慢，后面增加的快
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        # 将n_embed扩展一个维度，形状变为[B, N, H, W, 1]，
        # 最后一个维度的每个值除以dim_t中的每个值，形状变成[B, N, H, W, num_feats]
        # 由于dim_t是一个递增序列，pos_n的最后一个维度是一个递减序列
        pos_n = n_embed[:, :, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, N, H, W = mask.size()
        # pos_n的形状是[B, N, H, W, num_feats]
        # 取pos_n最后一个维度的偶数位和奇数位，分别进行sin和cos运算，然后在最后一个维度上进行叠加
        # 最后将形状重新设定为[B, N, H, W, -1]，使偶数的sin和奇数的cos叠加到一个维度中，形状不变
        pos_n = torch.stack(
            (pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        # 将pos_n, pos_y, pos_x在最后一个维度上进行拼接，形状变为[B, N, H, W, num_feats*3]
        # 然后对维度顺序进行调整，形状变成[B, N, num_feats*3, H, W]，其中num_feats*3=384
        pos = torch.cat((pos_n, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding3D(BaseModule):
    """Position embedding with learnable embedding weights.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding3D, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str