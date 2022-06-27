from __future__ import annotations
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn


################################################################################
################################################################################
### Main functions and classes
################################################################################
################################################################################
class PowerCompressor(nn.Module):
    r"""Power log compressor.

    Parameters
    ----------
    threshold : float
        Threshold value, by default `1e-6`.
    """
    def __init__(
        self,
        *,
        threshold: float = 1e-6,
    ) -> None: 
        super().__init__()
        if torch.tensor(threshold, dtype=torch.float32) <= 0:
            raise ValueError("threshold must be a positive value.")
        self.threshold = threshold

    def forward(
        self,
        power_specgram: Union[Tensor, list[Tensor]],
    ) -> Union[Tensor, list[Tensor]]:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        power_specgram : Tensor [shape=(batch_size, num_filters, num_frames)] or
        list[Tensor [shape=(1, num_filters, num_frames)]]
            Power spectrogram(s).

        Returns
        -------
        log_power_specgram :
        Tensor [shape=(batch_size, num_filters, num_frames)] or
        list[Tensor [shape=(1, num_filters, num_frames)]]
            Log-compressed power spectrogram(s).
        """
        if isinstance(power_specgram, list):
            return [self._forward_batch(p) for p in power_specgram]
        else:
            return self._forward_batch(power_specgram)

    def _forward_batch(
        self,
        power_specgram: Tensor,
    ) -> Tensor:
        threshold = torch.tensor(
            self.threshold, device=power_specgram.device,
            dtype=power_specgram.dtype)
        if threshold <= 0:
            raise ValueError("threshold must be a positive value.")

        power_specgram = power_specgram.clamp(min=threshold).log()
        power_specgram = power_specgram - threshold.log()
        return power_specgram

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


class PhaseDerivativeCompressor(nn.Module):
    r"""Phase derivative compressor.

    This implementation is based on PyTorch's batch normalization [1].

    Parameters
    ----------
    num_filters : int
        Number of filters, by default `96`.
    eps : float
        Value added to the denominator for numerical stability, by default
        `1e-5`.
    momentum : float
        Value used for the `running_ma` computation. Can be set to `None` for
        cumulative moving average (i.e. simple average).

        By default `0.1`.
    scaling : bool
        Boolean value that when set to `True`, this module has learnable
        scaling parameters.

        By default `True`.
    track_running_stats : bool
        Boolean value that when set to `True`, this module tracks the running
        mean absolute value, and when set to `False`, this module does not track
        such statistics, and initializes statistics buffers `running_ma` as
        `None`. When these buffers are `None`, this module always uses batch
        statistics in both training and eval modes.

        By default `True`.

    References
    ----------
    [1] https://pytorch.org/docs/1.10.1/_modules/torch/nn/modules/batchnorm.html
        BSD-style license
        https://github.com/pytorch/pytorch/blob/master/LICENSE
    """
    def __init__(
        self,
        *,
        num_filters: int = 128,
        eps: float = 1e-5,
        momentum: float = 0.1,
        scaling: bool = True,
        track_running_stats: bool = True,
        device = None,
        dtype = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_filters = num_filters
        self.eps = eps
        self.momentum = momentum
        self.scaling = scaling
        self.track_running_stats = track_running_stats 

        if self.scaling:
            self.gamma = nn.Parameter(torch.empty(num_filters))
        else:
            self.register_parameter("gamma", None)

        if self.track_running_stats:
            self.register_buffer("running_ma",
                torch.ones(num_filters, **factory_kwargs))
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0, dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"})
            )
        else:
            self.register_buffer("running_ma", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_ma.fill_(1.0)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.scaling:
            nn.init.ones_(self.gamma)

    def _validate_input(self,
        x: Union[Tensor, list[Tensor]],
    ) -> None:
        if isinstance(x, list):
            if not all(
                isinstance(y, Tensor) \
                    and y.ndim == 3 \
                    and y.size(0) == 1 \
                    and y.size(1) == self.num_filters
                for y in x
            ):
                raise ValueError(
                    "If x is a list of Tensors, the shape of each tensor in x "
                    "must be (1, self.num_filters, time).")
        else:
            if not (
                isinstance(x, Tensor) \
                    and x.ndim == 3 \
                    and x.size(1) == self.num_filters
            ):
                raise ValueError(
                    "If x is a Tensor, x.shape must be "
                    "(batch_size, self.num_filters, time).")

    def forward(
        self,
        x: Union[Tensor, list[Tensor]],
    ) -> Union[Tensor, list[Tensor]]:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        x : Tensor [shape=(batch_size, num_filters, num_frames)] or
        list[Tensor [shape=(1, num_filters, num_frames)]]
            Phase-derivative time-frequency representation(s).

        Returns
        -------
        x : Tensor [shape=(batch_size, num_filters, num_frames)] or
        list[Tensor [shape=(1, num_filters, num_frames)]]
            Compressed phase-derivative time-frequency representation(s).
        """
        self._validate_input(x)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0/self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        training = self.training or not self.track_running_stats
        redux = [0, 2]
        vdim = [1, self.num_filters, 1]

        if training:
            if isinstance(x, list):
                ma = x[0].float().abs().mean(redux, keepdim=True)
                for x_ in x[1:]:
                    ma = ma + x_.float().abs().mean(redux, keepdim=True)
                ma = ma / len(x)
            else:
                ma = x.float().abs().mean(redux, keepdim=True)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_ma.lerp_(ma.view(self.running_ma.shape),
                                          exponential_average_factor)
        else:
            ma = self.running_ma.float().view(vdim)

        if isinstance(x, list):
            x = [self._normalize(x_, ma) for x_ in x]
        else:
            x = self._normalize(x, ma)

        return x

    def _normalize(
        self,
        x: Tensor,
        ma: Tensor,
    ) -> Tensor:
        if self.scaling:
            gamma = self.gamma[None, :, None]
            x = x * (gamma / (10.0*(ma + self.eps)))
        else:
            x = x / (10.0*(ma + self.eps))
        return torch.tanh(x)

    def extra_repr(self) -> str:
        return f"num_filters={self.num_filters}, eps={self.eps}, " \
               f"momentum={self.momentum}, scaling={self.scaling}, " \
               f"track_running_stats={self.track_running_stats}"
