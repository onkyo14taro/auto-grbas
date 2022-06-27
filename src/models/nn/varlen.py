from __future__ import annotations
from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor
import torch.nn as nn


################################################################################
################################################################################
### Main functions and classes
################################################################################
################################################################################
class VariableLengthBatchNorm2D(nn.Module):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) [1].

    This implementation is based on PyTorch's batch normalization [2].

    Parameters
    ----------
    num_features : int
        Number of features or channels of the input.

    eps : float
        Value added to the denominator for numerical stability, by default
        `1e-5`.

    momentum : float
        Value used for the `running_ma` computation. Can be set to `None` for
        cumulative moving average (i.e. simple average).

        By default `0.1`.

    affine : bool
        Boolean value that when set to `True`, this module has learnable
        affine parameters.

        By default `True`.

    track_running_stats : bool
        Boolean value that when set to `True`, this module tracks the running
        mean and variance, and when set to `False`, this module does not track
        such statistics, and initializes statistics buffers `running_mean` and
        `running_var` as `None`. When these buffers are `None`, this module
        always uses batch statistics. in both training and eval modes.

        By default `True`.

    References
    ----------
    [1] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network
        training by reducing internal covariate shift," in ICML, 2015,
        pp. 448--456.
    [2] https://pytorch.org/docs/1.10.1/_modules/torch/nn/modules/batchnorm.html
        BSD-style license
        https://github.com/pytorch/pytorch/blob/master/LICENSE
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats 
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[torch.Tensor]
            self.running_var: Optional[torch.Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
            self.num_batches_tracked: Optional[torch.Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def forward(
        self,
        x: Union[Tensor, list[Tensor]],
    ) -> Union[Tensor, list[Tensor]]:
        variable_length = isinstance(x, list)

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0/self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        training = self.training or not self.track_running_stats
        redux = [0, 2, 3]
        vdim = [1, self.num_features, 1, 1]

        if training:
            if variable_length:
                mean = x[0].float().mean(redux, keepdim=True)
                for x_ in x[1:]:
                    if x_.size(0) != 1:
                        raise ValueError(
                            'The first dimension (batch) of each element of x '
                            'must be `1`.')
                    mean = mean + x_.float().mean(redux, keepdim=True)
                mean = mean / len(x)

                var = (x[0].float()-mean).square().mean(redux, keepdim=True)
                for x_ in x[1:]:
                    var = var + (x_.float()-mean).square().mean(redux, keepdim=True)
                var = var / len(x)
            else:
                mean = x.float().mean(redux, keepdim=True)
                var = x.float().var(redux, keepdim=True, unbiased=False)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean.lerp_(mean.view(self.running_mean.shape),
                                            exponential_average_factor)
                    self.running_var.lerp_(var.view(self.running_var.shape),
                                           exponential_average_factor)
        else:
            mean = self.running_mean.float().view(vdim)
            var = self.running_var.float().view(vdim)

        if variable_length:
            x = [self._normalize(x_, mean, var) for x_ in x]
        else:
            x = self._normalize(x, mean, var)

        return x

    def _normalize(
        self,
        x: Tensor,
        mean: Tensor,
        var: Tensor,
    ) -> Tensor:
        if self.affine:
            weight = self.weight[None, :, None, None]
            bias = self.bias[None, :, None, None]
            x = (x - mean) * (weight / torch.sqrt(var + self.eps)) + bias
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class VariableLengthWrapper(nn.Module):
    r"""Wrapper module to make modules variable-length compatible.

    Parameters
    ----------
    module : torch.nn.Module
        Module with fixed-length.
    """
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._m = module

    def forward(self, x: Union[Tensor, list[Tensor]]) -> Union[Tensor, list[Tensor]]:
        return [self._m(x_) for x_ in x] if isinstance(x, list) else self._m(x)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except:
            if hasattr(self._m, name):
                return getattr(self._m, name)
            else:
                raise AttributeError


def varlen_apply_func(
    x: Union[Tensor, list[Tensor]],
    func: Callable[[Tensor], Tensor],
    *args,
    **kwargs,
) -> Union[Tensor, list[Tensor]]:
    r"""Apply a function to variable-length tensors.

    Parameters
    ----------
    x : Tensor or list[Tensor]
        A tensor or a list of tensors.
    func : Callable[[Tensor], Tensor]
        A tensor or a list of tensors. If `x` is a list, then `y` must be a list
        of the same length as `x`.
    args
        Second and subsequent positional arguments for `func`.
    kwargs
        Keyword arguments for `func`.

    Returns
    -------
    y : Tensor or list[Tensor]
        Output.
    """
    if isinstance(x, list):
        return [func(x_, *args, **kwargs) for x_ in x]
    else:
        return func(x, *args, **kwargs)


def varlen_add(
    x: Union[Tensor, list[Tensor]],
    y: Union[Tensor, list[Tensor]]
) -> Union[Tensor, list[Tensor]]:
    r"""Addition of variable-length tensors.

    Parameters
    ----------
    x : Tensor or list[Tensor]
        A tensor or a list of tensors.
    y : Tensor or list[Tensor]
        A tensor or a list of tensors. If `x` is a list, then `y` must be a list
        of the same length as `x`.

    Returns
    -------
    z : Tensor or list[Tensor]
        Sum of `x` and `y`.
    """
    if isinstance(x, list):
        return [x_ + y_ for x_, y_ in zip(x, y)]
    else:
        return x + y


def varlen_mul(
    x: Union[Tensor, list[Tensor]],
    y: Union[Tensor, list[Tensor]]
) -> Union[Tensor, list[Tensor]]:
    r"""Multiplication of variable-length tensors.

    Parameters
    ----------
    x : Tensor or list[Tensor]
        A tensor or a list of tensors.
    y : Tensor or list[Tensor]
        A tensor or a list of tensors. If `x` is a list, then `y` must be a list
        of the same length as `x`.

    Returns
    -------
    z : Tensor or list[Tensor]
        Product of `x` and `y`.
    """
    if isinstance(x, list):
        return [x_ * y_ for x_, y_ in zip(x, y)]
    else:
        return x * y


def convert_varlen_module(fixedlen_module: nn.Module) -> nn.Module:
    r"""Convert a module with fixed-length input to variable-length input.

    Parameters
    ----------
    fixedlen_module : torch.nn.Module
        Module with fixed-length.

    Returns
    -------
    varlen_module : torch.nn.Module
        Module with variable-length.
    """
    if isinstance(fixedlen_module, nn.Identity):
        return fixedlen_module
    elif isinstance(fixedlen_module, nn.BatchNorm2d):
        return _varlen_batchnorm_2d(fixedlen_module)
    elif not isinstance(fixedlen_module, nn.modules.batchnorm._NormBase):
        return VariableLengthWrapper(fixedlen_module)
    else:
        raise NotImplementedError(type(fixedlen_module))


################################################################################
################################################################################
### Main functions and classes
################################################################################
################################################################################
def _varlen_batchnorm_2d(batchnorm_2d: nn.BatchNorm2d) -> VariableLengthBatchNorm2D:
    assert isinstance(batchnorm_2d, nn.BatchNorm2d)
    return VariableLengthBatchNorm2D(
        num_features=batchnorm_2d.num_features,
        eps=batchnorm_2d.eps,
        momentum=batchnorm_2d.momentum,
        affine=batchnorm_2d.affine,
        track_running_stats=batchnorm_2d.track_running_stats,
        device=batchnorm_2d.weight.device if batchnorm_2d.affine else None,
        dtype=batchnorm_2d.weight.dtype if batchnorm_2d.affine else None,
    )
