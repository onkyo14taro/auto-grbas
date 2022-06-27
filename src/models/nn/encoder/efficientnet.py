r"""Module for EfficientNet for variable-size images.

This implementation is based on [1].

[1] https://github.com/rwightman/pytorch-image-models/tree/v0.5.4
    Copyright (c) 2019 Ross Wightman
    Apache License 2.0
    https://opensource.org/licenses/Apache-2.0
"""

from __future__ import annotations
from typing import Callable, Literal, Union

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.efficientnet import EfficientNet, EfficientNetFeatures
from timm.models.layers import drop_path
from timm.models.efficientnet_blocks import (
    ConvBnAct, EdgeResidual, InvertedResidual, SqueezeExcite)

from ..varlen import (
    convert_varlen_module, varlen_apply_func, varlen_add)


__all__ = [
    'VariableLengthEfficientNet',
    'VariableLengthEfficientNetFeatures',
    'create_variable_length_efficientnet',
]


def create_variable_length_efficientnet(
    in_chans: int,
    *,
    model_name: Literal[
        "tf_efficientnetv2_b0",
        "tf_efficientnetv2_b1",
        "tf_efficientnetv2_b2",
        "tf_efficientnetv2_b3",
        "tf_efficientnetv2_s",
        "tf_efficientnetv2_m",
        "tf_efficientnetv2_l",
    ] = "tf_efficientnetv2_b0",
    features_only: bool = True,
    num_classes: int = 1000,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> Union[EfficientNet, EfficientNetFeatures]:
    r"""Create an EfficientNet that can handle variable-length inputs.

    Parameters
    ----------
    in_chans : int
        Number of input channels.
    model_name : str
        CNN model name.

        Available models are as follows:

        - `"tf_efficientnetv2_b0"`
        - `"tf_efficientnetv2_b1"`
        - `"tf_efficientnetv2_b2"`
        - `"tf_efficientnetv2_b3"`
        - `"tf_efficientnetv2_s"`
        - `"tf_efficientnetv2_m"`
        - `"tf_efficientnetv2_l"`

        By default, `"tf_efficientnetv2_b0"`.
    features_only : bool
        Whether to remove the classification head or not.

        By default, `True`.
    num_classes : int
        Number of output classes.

        By default, `1000`.
    drop_rate : float
        Dropout rate for training.

        By default, `0.0`.
    drop_path_rate : float
        Drop rate of the stochastic depth.

        By default, `0.0`.

    Returns
    -------
    model : EfficientNet or EfficientNetFeatures
        If `features_only == True`, return an instance of `EfficientNet`.
        Otherwise, return an instance of `EfficientNetFeatures`.
    """
    if not model_name.startswith('tf_efficientnetv2'):
        raise NotImplementedError(
            f'The current implemantation only handles models with names '
            f'starting with `tf_efficientnetv2`; found {model_name}'
        )
    net = timm.create_model(
        model_name,
        in_chans=in_chans,
        features_only=features_only,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    net.conv_stem = convert_varlen_module(net.conv_stem)
    net.bn1 = convert_varlen_module(net.bn1)
    net.act1 = convert_varlen_module(net.act1)
    if not features_only:
        net.conv_head = convert_varlen_module(net.conv_head)
        net.bn2 = convert_varlen_module(net.bn2)
        net.act2 = convert_varlen_module(net.act2)
        net.global_pool = convert_varlen_module(net.global_pool)
    _replace_module(net, ConvBnAct, VariableLengthConvBnAct)
    _replace_module(net, EdgeResidual, VariableLengthEdgeResidual)
    _replace_module(net, InvertedResidual, VariableLengthInvertedResidual)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in net.modules())
    if features_only:
        return VariableLengthEfficientNetFeatures(net)
    else:
        return VariableLengthEfficientNet(net)


def _replace_module(
    net: nn.Module,
    target_module_cls: type,
    func: Callable[[nn.Module], nn.Module]=lambda m: nn.Identity()
) -> None:
    for name, m in net.named_modules():
        m_ = net
        if isinstance(m, target_module_cls):
            parts = name.split('.')
            for part in parts[:-1]:
                if part.isnumeric():
                    m_ = m_[int(part)]
                else:
                    m_ = getattr(m_, part)
            part_last = parts[-1]
            if part_last.isnumeric():
                assert m_[int(part_last)] is m
                m_[int(part_last)] = func(m)
            else:
                assert getattr(m_, part_last) is m
                setattr(m_, part_last, func(m))


class VariableLengthEfficientNet(nn.Module):
    def __init__(self, efficientnet: EfficientNet) -> None:
        super().__init__()
        self.num_classes = efficientnet.num_classes
        self.num_features = efficientnet.num_features
        self.drop_rate = efficientnet.drop_rate
        self.conv_stem = efficientnet.conv_stem
        self.bn1 = efficientnet.bn1
        self.act1 = efficientnet.act1
        self.blocks = efficientnet.blocks
        self.feature_info = efficientnet.feature_info
        self.conv_head = efficientnet.conv_head
        self.bn2 = efficientnet.bn2
        self.act2 = efficientnet.act2
        self.global_pool = efficientnet.global_pool
        self.classifier = efficientnet.classifier
        efficientnet.drop_rate = None
        efficientnet.conv_stem = None
        efficientnet.bn1 = None
        efficientnet.act1 = None
        efficientnet.blocks = None
        efficientnet.feature_info = None
        efficientnet.conv_head = None
        efficientnet.bn2 = None
        efficientnet.act2 = None
        efficientnet.global_pool = None
        efficientnet.classifier = None

    def forward_features(self, x: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        x = self.forward_features(x)
        x = self.global_pool(x)
        if isinstance(x, list):
            x = torch.concat(x, dim=0)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


class VariableLengthEfficientNetFeatures(nn.Module):
    def __init__(self, efficientnet_features: EfficientNetFeatures) -> None:
        super().__init__()
        self.drop_rate = efficientnet_features.drop_rate
        self.conv_stem = efficientnet_features.conv_stem
        self.bn1 = efficientnet_features.bn1
        self.act1 = efficientnet_features.act1
        self.blocks = efficientnet_features.blocks
        self.feature_info = efficientnet_features.feature_info
        self._stage_out_idx = efficientnet_features._stage_out_idx
        self.feature_hooks = efficientnet_features.feature_hooks
        efficientnet_features.drop_rate = None
        efficientnet_features.conv_stem = None
        efficientnet_features.bn1 = None
        efficientnet_features.act1 = None
        efficientnet_features.blocks = None
        efficientnet_features.feature_info = None
        efficientnet_features._stage_out_idx = None
        efficientnet_features.feature_hooks = None

    def forward(self, x: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x


class VariableLengthConvBnAct(nn.Module):
    """ Conv + Norm Layer + Activation w/ optional skip connection
    """
    def __init__(self, conv_bn_act: ConvBnAct) -> None:
        super().__init__()
        self.has_residual = conv_bn_act.has_residual
        self.drop_path_rate = conv_bn_act.drop_path_rate

        self.conv = convert_varlen_module(conv_bn_act.conv)
        self.bn1 = convert_varlen_module(conv_bn_act.bn1)
        self.act1 = convert_varlen_module(conv_bn_act.act1)

        conv_bn_act.conv = None
        conv_bn_act.bn1 = None
        conv_bn_act.act1 = None

    def feature_info(self, location):
        if location == 'expansion':  # output of conv after act, same as block coutput
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = varlen_apply_func(x, drop_path, self.drop_path_rate, self.training)
            x = varlen_add(x, shortcut)
        return x


class VariableLengthEdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride
    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
        - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html
    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
      * MobileDet - https://arxiv.org/abs/2004.14525
      * EfficientNet-X - https://arxiv.org/abs/2102.05610
      * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """

    def __init__(self, edge_res: EdgeResidual) -> None:
        super().__init__()
        self.has_residual = edge_res.has_residual
        self.drop_path_rate = edge_res.drop_path_rate

        # Expansion convolution
        self.conv_exp = convert_varlen_module(edge_res.conv_exp)
        self.bn1 = convert_varlen_module(edge_res.bn1)
        self.act1 = convert_varlen_module(edge_res.act1)

        # Squeeze-and-excitation
        if isinstance(edge_res.se, (nn.Identity, SqueezeExcite)):
            self.se = convert_varlen_module(edge_res.se)
        else:
            raise NotImplementedError(
                'inv_res.se must be SqueezeExcite or Identity.')

        # Point-wise linear projection
        self.conv_pwl = convert_varlen_module(edge_res.conv_pwl)
        self.bn2 = convert_varlen_module(edge_res.bn2)

        edge_res.conv_exp = None
        edge_res.bn1 = None
        edge_res.act1 = None
        edge_res.se = None
        edge_res.conv_pwl = None
        edge_res.bn2 = None

    def feature_info(self, location):
        if location == 'expansion':  # after SE, before PWL
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        shortcut = x

        # Expansion convolution
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = varlen_apply_func(x, drop_path, self.drop_path_rate, self.training)
            x = varlen_add(x, shortcut)

        return x


class VariableLengthInvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE
    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(self, inv_res: InvertedResidual) -> None:
        super().__init__()
        self.has_residual = inv_res.has_residual
        self.drop_path_rate = inv_res.drop_path_rate

        # Point-wise expansion
        self.conv_pw = convert_varlen_module(inv_res.conv_pw)
        self.bn1 = convert_varlen_module(inv_res.bn1)
        self.act1 = convert_varlen_module(inv_res.act1)

        # Depth-wise convolution
        self.conv_dw = convert_varlen_module(inv_res.conv_dw)
        self.bn2 = convert_varlen_module(inv_res.bn2)
        self.act2 = convert_varlen_module(inv_res.act2)

        # Squeeze-and-excitation
        if isinstance(inv_res.se, (nn.Identity, SqueezeExcite)):
            self.se = convert_varlen_module(inv_res.se)
        else:
            raise NotImplementedError(
                'inv_res.se must be SqueezeExcite or Identity.')

        # Point-wise linear projection
        self.conv_pwl = convert_varlen_module(inv_res.conv_pwl)
        self.bn3 = convert_varlen_module(inv_res.bn3)

        inv_res.conv_pw = None
        inv_res.bn1 = None
        inv_res.act1 = None
        inv_res.conv_dw = None
        inv_res.bn2 = None
        inv_res.act2 = None
        inv_res.se = None
        inv_res.conv_pwl = None
        inv_res.bn3 = None

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        shortcut = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = varlen_apply_func(x, drop_path, self.drop_path_rate, self.training)
            x = varlen_add(x, shortcut)

        return x
