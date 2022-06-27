from __future__ import annotations
from typing import Collection, Literal, Optional, Union

import torch
import torch.nn as nn

from .mel import MelSpectrogram
from .specaug import FrequencyMasking, TimeMasking
from ..varlen import VariableLengthBatchNorm2D


class Frontend(nn.Module):
    r"""Frontend that calculate time-frequency representations from waveforms.

    Parameters
    ----------
    fs : float
        Sampling frequency, by default `16000` Hz.
    num_filters : int
        Number of filters, by default `128`.
    window_length : int
        Window length in samples, by default `800` (if `fs == 16000`, 50 ms).
    shift_length : int
        Shift length in samples, by default `160` (if `fs == 16000`, 10 ms).
    f_min : float
        Minimum frequency in Hz, by default `0.0` Hz.
    f_max : float or None
        Maximum frequency in Hz. If `None`, this will be set to `fs/2` (Nyquist
        frequency).

        By default, `None`.
    num_fft : int, None
        Number of FFT points. If `None`, this will be set to `window_length*5`.

        By default, `None`.
    window_name : str
        Window name, by default "hann".
    power_threshold : float
        Lower bound of power spectrogram in arbitary unit (NOT in dB).
        Values that are smaller than `power_threshold` will be clipped.

        By default, `1e-6`.
    phase_comp_eps : float
        Value added to the denominator for numerical stability of the phase-
        derivative compressors, by default `1e-5`.
    phase_comp_momentum : float
        Value used for the `running_ma` computation of the phase-derivative
        compressors. Can be set to `None` for cumulative moving average (i.e.
        simple average).

        By default `0.1`.
    phase_comp_trainable : bool
        Boolean value that when set to `True`, this module has learnable
        scaling parameters of the phase-derivative compressor.

        By default `True`.
    features : "power", "ist_frq", "grp_dly", or Collection of them
        Feature or collection of features to be calculated.

        Available features are as follows:

        - `"power"` : log power
        - `"ist_frq"` : instantaneous frequency, which is the time-derivative of
        the phase of the complex spectrogram
        - `"grp_dly"` : group delay, which is the frequency-derivative of the
        phase of the complex spectrogram

        For example, if you want to calculate all features, set `features` to
        `["power", "ist_frq", "grp_dly"]`.

        By default, "power".
    time_mask : bool
        Whether to use the time masking or not, by default `False`.
    time_mask_param : int
        Maximum possible length of the time mask. Indices uniformly sampled from
        [0, time_mask_param).

        By default, `20`.
    time_mask_batch_iid_masks : bool
        Whether to apply different time masks to each example in the batch, by
        default `True`.
    time_mask_p : float
        Maximum proportion of time steps that can be masked. Must be within
        range [0.0, 1.0].

        By default, `1.0`.
    freq_mask : bool
        Whether to use the frequency masking or not, by default `True`.
    freq_mask_param : int
        Maximum possible length of the frequency mask. Indices uniformly sampled
        from [0, freq_mask_param).

        By default, `26`.
    freq_mask_batch_iid_masks : bool
        Whether to apply different frequency masks to each example in the batch,
        by default `True`.
    batch_norm : bool
        Whether to use the 2-D batch normalization or not, by default `True`.
    batch_norm_eps : float
        Value added to the denominator for numerical stability of the batch
        normalization, by default `1e-5`.
    batch_norm_momentum : float
        Value used for the `running_ma` computation of the batch normalization.
        Can be set to `None` for cumulative moving average (i.e. simple
        average).

        By default `0.1`.
    batch_norm_trainable : bool
        Whether the batch normalization is trainable or not, by default `True`.
    """
    def __init__(
        self,
        *,
        fs: float = 16000,
        num_filters: int = 128,
        window_length: int = 800,
        shift_length: int = 160,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        num_fft: Optional[int] = None,
        window_name: str = "hann",
        power_threshold: float = 1e-6,
        phase_comp_eps: float = 1e-5,
        phase_comp_momentum: float = 0.1,
        phase_comp_trainable: bool = True,
        features: Union[
            Literal["power", "ist_frq", "grp_dly"],
            Collection[Literal["power", "ist_frq", "grp_dly"]]
        ] = "power",
        time_mask: bool = False,
        time_mask_param: int = 20,
        time_mask_batch_iid_masks: bool = True,
        time_mask_p: float = 1.0,
        freq_mask: bool = True,
        freq_mask_param: int = 26,
        freq_mask_batch_iid_masks: bool = True,
        batch_norm: bool = True,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
        batch_norm_trainable: bool = True,
        device = None,
        dtype = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.tfr_calculator = MelSpectrogram(
            fs=fs,
            num_filters=num_filters,
            window_length=window_length,
            shift_length=shift_length,
            f_min=f_min,
            f_max=f_max,
            num_fft=num_fft,
            window_name=window_name,
            power_threshold=power_threshold,
            phase_comp_eps=phase_comp_eps,
            phase_comp_momentum=phase_comp_momentum,
            phase_comp_trainable=phase_comp_trainable,
            features=features,
            **factory_kwargs,
        )
        self.out_channels = len(self.tfr_calculator.features)
        self.time_mask = TimeMasking(
            time_mask_param=time_mask_param,
            batch_iid_masks=time_mask_batch_iid_masks,
            p=time_mask_p,
        ) if time_mask else None
        self.freq_mask = FrequencyMasking(
            freq_mask_param=freq_mask_param,
            batch_iid_masks=freq_mask_batch_iid_masks,
        ) if freq_mask else None
        self.batch_norm_2d = VariableLengthBatchNorm2D(
            self.out_channels,
            eps=batch_norm_eps,
            momentum=batch_norm_momentum,
            affine=batch_norm_trainable,
        ) if batch_norm else None

    def forward(
        self,
        waveform: Union[torch.Tensor, list[torch.Tensor]],
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        x : Tensor [shape=(batch_size, 1, time)] or
        list[Tensor [shape=(1, 1, time)]]
            Waveform(s).

        Returns
        -------
        tfr :
        Tensor [shape=(batch_size, num_features, num_filters, num_frames)] or
        list[Tensor [shape=(1, num_features, num_filters, num_frames)]]
            Time-frequency representation(s).
        """
        tfr = self.tfr_calculator(waveform)
        if self.time_mask is not None:
            tfr = self.time_mask(tfr)
        if self.freq_mask is not None:
            tfr = self.freq_mask(tfr)
        if self.batch_norm_2d is not None:
            tfr = self.batch_norm_2d(tfr)
        return tfr

    def extra_repr(self) -> str:
        return f"out_channels={self.out_channels}"
