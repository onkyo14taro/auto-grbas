r"""Module for SpecAugment.

This implementation is almost the same as the torchaudio implementation [1]. The
differences are as follows:

- No masking in the evaluation mode.
- Can handle lists of spectrograms of different lengths for each sample.
- "i.i.d." of torchaudio is for both batches and channels, whereas "i.i.d." of
  this implementation is for batches only (shared across channels).

[1] https://pytorch.org/audio/0.11.0/_modules/torchaudio/transforms.html
    Copyright (c) 2017 Facebook Inc. (Soumith Chintala)
    BSD 2-Clause License
    https://opensource.org/licenses/BSD-2-Clause
"""

from __future__ import annotations
from typing import Union

import torch
from torch import Tensor


__all__ = [
    "FrequencyMasking",
    "TimeMasking",
]


class _AxisMasking(torch.nn.Module):
    r"""Apply masking to a spectrogram.

    Args:
        mask_param (int): Maximum possible length of the mask.
        axis (int): What dimension the mask is applied on.
        batch_iid_masks (bool): Applies iid masks to each of the examples in the batch dimension.
            This option is applicable only when the input tensor is 4D.
        p (float, optional): maximum proportion of columns that can be masked. (Default: 1.0)
    """
    __constants__ = ["mask_param", "axis", "batch_iid_masks", "p"]

    def __init__(self, mask_param: int, axis: int, batch_iid_masks: bool, p: float = 1.0) -> None:

        super(_AxisMasking, self).__init__()
        self.mask_param = mask_param
        self.axis = axis
        self.batch_iid_masks = batch_iid_masks
        self.p = p

    def forward(
        self,
        specgram: Union[Tensor, list[Tensor]],
        mask_value: float = 0.0,
    ) -> Union[Tensor, list[Tensor]]:
        r"""
        Args:
            specgram (Tensor): Tensor of dimension `(..., freq, time)`.
            mask_value (float): Value to assign to the masked columns.

        Returns:
            Tensor: Masked spectrogram of dimensions `(..., freq, time)`.
        """
        if not self.training:
            return specgram

        # if batch_iid_masks flag marked and specgram has a batch dimension
        if (isinstance(specgram, Tensor) and specgram.ndim != 4) \
            or (isinstance(specgram, list) and specgram[0].ndim != 4):
            raise ValueError('specgram must be a 4D tensor or list of them.')

        if self.batch_iid_masks:
            return mask_along_axis_batch_iid(specgram, self.mask_param, mask_value, self.axis + 1, p=self.p)
        else:
            if isinstance(specgram, list):
                raise ValueError('If specgram is a list or tensors, you must set batch_iid_masks to True.')
            return mask_along_axis(specgram, self.mask_param, mask_value, self.axis, p=self.p)


class FrequencyMasking(_AxisMasking):
    r"""Apply masking to a spectrogram in the frequency domain.

    Proposed in *SpecAugment* [:footcite:`specaugment`].

    Args:
        freq_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, freq_mask_param).
        batch_iid_masks (bool, optional): whether to apply different masks to each
            example/channel in the batch. (Default: ``False``)
            This option is applicable only when the input tensor is 4D.

    Example
        >>> spectrogram = torchaudio.transforms.Spectrogram()
        >>> masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=80)
        >>>
        >>> original = spectrogram(waveform)
        >>> masked = masking(original)

        .. image::  https://download.pytorch.org/torchaudio/doc-assets/specaugment_freq_masking1.png
           :alt: The original spectrogram

        .. image::  https://download.pytorch.org/torchaudio/doc-assets/specaugment_freq_masking2.png
           :alt: The spectrogram masked along frequency axis
    """

    def __init__(self, freq_mask_param: int, batch_iid_masks: bool = False) -> None:
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, batch_iid_masks)

    def extra_repr(self) -> str:
        return f'mask_param={self.mask_param}, batch_iid_masks={self.batch_iid_masks}'


class TimeMasking(_AxisMasking):
    r"""Apply masking to a spectrogram in the time domain.

    Proposed in *SpecAugment* [:footcite:`specaugment`].

    Args:
        time_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, time_mask_param).
        batch_iid_masks (bool, optional): whether to apply different masks to each
            example/channel in the batch. (Default: ``False``)
            This option is applicable only when the input tensor is 4D.
        p (float, optional): maximum proportion of time steps that can be masked.
            Must be within range [0.0, 1.0]. (Default: 1.0)

    Example
        >>> spectrogram = torchaudio.transforms.Spectrogram()
        >>> masking = torchaudio.transforms.TimeMasking(time_mask_param=80)
        >>>
        >>> original = spectrogram(waveform)
        >>> masked = masking(original)

        .. image::  https://download.pytorch.org/torchaudio/doc-assets/specaugment_time_masking1.png
           :alt: The original spectrogram

        .. image::  https://download.pytorch.org/torchaudio/doc-assets/specaugment_time_masking2.png
           :alt: The spectrogram masked along time axis
    """

    def __init__(self, time_mask_param: int, batch_iid_masks: bool = False, p: float = 1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")
        super(TimeMasking, self).__init__(time_mask_param, 2, batch_iid_masks, p=p)

    def extra_repr(self) -> str:
        return f'mask_param={self.mask_param}, batch_iid_masks={self.batch_iid_masks}, ' \
               f'p={self.p}'


def mask_along_axis(
    specgram: Tensor,
    mask_param: int,
    mask_value: float,
    axis: int,
    p: float = 1.0,
) -> Tensor:
    r"""
    Apply a mask along ``axis``. Mask will be applied from indices ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, max_v)`` and ``v_0`` from ``uniform(0, specgrams.size(axis) - v)``, with
    ``max_v = mask_param`` when ``p = 1.0`` and ``max_v = min(mask_param, floor(specgrams.size(axis) * p))``
    otherwise. All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectrogram `(channel, freq, time)`
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (1 -> frequency, 2 -> time)
        p (float, optional): maximum proportion of columns that can be masked. (Default: 1.0)

    Returns:
        Tensor: Masked spectrogram of dimensions `(channel, freq, time)`
    """
    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    mask_param = _get_mask_param(mask_param, p, specgram.shape[axis])
    if mask_param < 1:
        return specgram

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))
    value = torch.rand(
        1, device=specgram.device, dtype=specgram.dtype) * mask_param
    min_value = torch.rand(
        1, device=specgram.device, dtype=specgram.dtype
    ) * (specgram.size(axis) - value)

    mask_start = (min_value.long()).squeeze()
    mask_end = (min_value.long() + value.long()).squeeze()
    mask = torch.arange(0, specgram.shape[axis], device=specgram.device, dtype=specgram.dtype)
    mask = (mask >= mask_start) & (mask < mask_end)
    if axis == 1:
        mask = mask.unsqueeze(-1)

    assert mask_end - mask_start < mask_param

    specgram = specgram.masked_fill(mask, mask_value)

    # unpack batch
    specgram = specgram.reshape(shape[:-2] + specgram.shape[-2:])

    return specgram


def mask_along_axis_batch_iid(
    specgrams: Union[Tensor, list[Tensor]],
    mask_param: int,
    mask_value: float,
    axis: int,
    p: float = 1.0,
) -> Union[Tensor, list[Tensor]]:
    r"""
    Apply a mask along ``axis``. Mask will be applied from indices ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, max_v)`` and ``v_0`` from ``uniform(0, specgrams.size(axis) - v)``, with
    ``max_v = mask_param`` when ``p = 1.0`` and ``max_v = min(mask_param, floor(specgrams.size(axis) * p))``
    otherwise.

    Args:
        specgrams (Tensor): Real spectrograms `(batch, channel, freq, time)`
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (2 -> frequency, 3 -> time)
        p (float, optional): maximum proportion of columns that can be masked. (Default: 1.0)

    Returns:
        Tensor: Masked spectrograms of dimensions `(batch, channel, freq, time)`
    """

    if axis not in [2, 3]:
        raise ValueError("Only Frequency and Time masking are supported")

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    if isinstance(specgrams, list):
        if len(specgrams) == 0:
            return specgrams
        device = specgrams[0].device
        dtype = specgrams[0].dtype

        axis_length = torch.tensor(
            [spec.shape[axis] for spec in specgrams],
            device=device, dtype=dtype
        )
        mask_param = torch.minimum(torch.tensor(mask_param, device=device, dtype=dtype), torch.round(axis_length * p).long())

        device = specgrams[0].device
        dtype = specgrams[0].dtype

        batch_size = len(specgrams)
        value = torch.rand(batch_size, device=device, dtype=dtype) * mask_param
        min_value = torch.rand(batch_size, device=device, dtype=dtype) * (axis_length - value)

        # Create broadcastable mask
        mask_start = min_value.long()[..., None, None]
        mask_end = (min_value.long() + value.long())[..., None, None]

        # Per batch example masking
        specgrams_masked = []
        for i, spec in enumerate(specgrams):
            spec = spec.transpose(axis, -1)
            mask = torch.arange(0, spec.size(-1), device=device, dtype=dtype)
            spec = spec.masked_fill((mask >= mask_start[i:i+1]) & (mask < mask_end[i:i+1]), mask_value)
            spec = spec.transpose(axis, -1)
            specgrams_masked.append(spec)

        return specgrams_masked

    else:
        axis_length = specgrams.shape[axis]
        mask_param = min(mask_param, int(axis_length * p))
        if mask_param < 1:
            return specgrams

        device = specgrams.device
        dtype = specgrams.dtype

        batch_size = specgrams.shape[0]
        value = torch.rand(batch_size, device=device, dtype=dtype) * mask_param
        min_value = torch.rand(batch_size, device=device, dtype=dtype) * (axis_length - value)

        # Create broadcastable mask
        mask_start = min_value.long()[..., None, None, None]
        mask_end = (min_value.long() + value.long())[..., None, None, None]
        mask = torch.arange(0, specgrams.size(axis), device=device, dtype=dtype)

        # Per batch example masking
        specgrams = specgrams.transpose(axis, -1)
        specgrams = specgrams.masked_fill((mask >= mask_start) & (mask < mask_end), mask_value)
        specgrams = specgrams.transpose(axis, -1)

        return specgrams


def _get_mask_param(mask_param: int, p: float, axis_length: int) -> int:
    return min(mask_param, int(axis_length * p))
