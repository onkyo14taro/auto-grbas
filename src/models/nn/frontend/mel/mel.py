from __future__ import annotations
import math
from typing import Collection, Literal, Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
import torchaudio.functional as AF

from .compressor import PowerCompressor, PhaseDerivativeCompressor


################################################################################
################################################################################
### Main functions and classes
################################################################################
################################################################################
class MelSpectrogram(nn.Module):
    r"""Calculate mel-frequency time-frequency representations.

    This class can calculate not only power but also phase-derivative time-
    frequency representations (instantaneous frequency and group delay). To
    calculate the phase-derivatives, this class uses Auger-Flandrin equation
    [1]. Also, to prevent divergence of the phase derivative in elements with
    zero power [2], this class uses power spectral weighted smoothing [3].

    Note that this class uses the HTK formula [6] as a mel scale.

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

    References
    ----------
    [1] F. Auger and P. Flandrin, "Improving the Readability of Time-Frequency
        and Time-Scale Representations by the Reassignment Method," IEEE Trans.
        Signal Process., vol. 43, no. 5, pp. 1068--1089, 1995.
    [2] P. Balazs, D. Bayer, F. Jaillet, and P. Søndergaard, "The pole behavior
        of the phase derivative of the short-time Fourier transform," Appl.
        Comput. Harmon. Anal., vol. 40, no. 3, pp. 610--621, 2016.
    [3] H. Kawahara, K. Sakakibara, M. Morise, H. Banno, and T. Toda, "A
        modulation property of time-frequency derivatives of filtered phase and
        its application to aperiodicity and fo estimation," Interspeech, 2017.
    """
    features: tuple[Literal["power", "ist_frq", "grp_dly"]]

    def __init__(
        self,
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
        device = None,
        dtype = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(features, str):
            if features not in _FEATURES:
                raise ValueError(
                    'features must be either "power", "ist_frq", "grp_dly", or '
                    'a tuple of them.')
            features = (features, )
        else:
            if not (isinstance(features, Collection) \
               and len(features) and all((f in _FEATURES) for f in features)):
                raise ValueError(
                    'features must be either "power", "ist_frq", "grp_dly", or '
                    'a tuple of them.')
            features = tuple(sorted(set(features)))

        if num_fft is None:
            num_fft = window_length * 5

        self.fs = fs
        self.num_filters = num_filters
        self.window_length = window_length
        self.shift_length = shift_length
        self.f_min = f_min
        self.f_max = f_max
        self.num_fft = num_fft
        self.window_name = window_name
        self.power_threshold = power_threshold
        self.features = features

        self.mel_scale = MelScale(
            n_mels=self.num_filters,
            sample_rate=self.fs,
            f_min=self.f_min,
            f_max=self.f_max,
            n_stft=self.num_fft//2 + 1,
            norm=1,
            mel_scale="htk"
        )

        compressor = {}
        for feature in self.features:
            if feature == "power":
                compressor[feature] = PowerCompressor(
                    threshold=self.power_threshold,
                )
            elif feature == "ist_frq" or feature == "grp_dly":
                compressor[feature] = PhaseDerivativeCompressor(
                    num_filters=num_filters,
                    eps=phase_comp_eps,
                    momentum=phase_comp_momentum,
                    scaling=phase_comp_trainable,
                    **factory_kwargs,
                )
        self.compressor = nn.ModuleDict(compressor)

        if "power" in self.features:
            features = self.features
        else:
            features = sorted(list(self.features) + ["power"])
        for feature in features:
            self.register_buffer(
                f"window_{feature}",
                _generate_general_cosine_window(
                    self.window_length, self.window_name, mode=feature),
            )

    def _validate_input(self,
        waveform: Union[Tensor, list[Tensor]],
    ) -> None:
        if isinstance(waveform, list):
            if not all(
                isinstance(w, Tensor) \
                    and w.ndim == 3 \
                    and w.size(0) == w.size(1) == 1 
                for w in waveform
            ):
                raise ValueError(
                    "If waveform is a list of Tensors, the shape of each tensor "
                    "in waveform must be (1, 1, time).")
        else:
            if not (
                isinstance(waveform, Tensor) \
                    and waveform.ndim == 3 \
                    and waveform.size(1) == 1
            ):
                raise ValueError(
                    "If waveform is a Tensor, waveform.shape must be "
                    "(batch_size, 1, time).")

    def forward(
        self,
        waveform: Union[Tensor, list[Tensor]],
    ) -> Union[Tensor, list[Tensor]]:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        waveform : Tensor [shape=(batch_size, 1, time)] or
        list[Tensor [shape=(1, 1, time)]]
            Waveform(s).

        Returns
        -------
        tfr :
        Tensor [shape=(batch_size, num_features, num_filters, num_frames)] or
        list[Tensor [shape=(1, num_features, num_filters, num_frames)]]
            Time-frequency representation(s).
        """
        self._validate_input(waveform)
        specgram_dict = self._calculate_raw_features(waveform)

        # Compress.
        specgram = [self.compressor[k](v) for k, v in specgram_dict.items()]

        # Stack as 2D images.
        if isinstance(waveform, list):
            specgram = [torch.stack(sample, dim=1) for sample in zip(*specgram)]
        else:
            specgram = torch.stack(specgram, dim=1)
        return specgram

    def _calculate_raw_features(
        self,
        waveform: Union[Tensor, list[Tensor]],
    ) -> Union[Tensor, list[Tensor]]:
        if isinstance(waveform, list):
            specgram_dict = {feature: [] for feature in self.features}
            for w in waveform:
                specgram_dict_temp = self._calculate_raw_features_batch(w)
                for feature in self.features:
                    specgram_dict[feature].append(
                        specgram_dict_temp.pop(feature))
                del specgram_dict_temp
        else:
            specgram_dict = self._calculate_raw_features_batch(waveform)
        return specgram_dict

    def _calculate_raw_features_batch(
        self,
        waveform: Tensor,
    ) -> dict[Literal["power", "ist_frq", "grp_dly"], Tensor]:
        if waveform.ndim != 3 or waveform.size(1) != 1:
            raise ValueError("waveform.shape must be (batch_size, 1, time).")
        if "power" in self.features:
            features = self.features
        else:
            features = list(self.features) + ["power"]
        cmplx_specgram_dict = {
            feature: AF.spectrogram(
                waveform=waveform,
                pad=0,
                window=getattr(self, f"window_{feature}"),
                n_fft=self.num_fft,
                hop_length=self.shift_length,
                win_length=self.window_length,
                power=None,
                normalized=False,
                center=True,
                pad_mode="constant",
                onesided=True,
            )[:, 0]
            for feature in features
        }
        mel_power_specgram = torch.clamp(self.mel_scale(
            cmplx_specgram_dict["power"].real**2 \
            + cmplx_specgram_dict["power"].imag**2
        ), min=self.power_threshold)

        specgram_dict = {}
        for feature in self.features:
            if feature == "power":
                specgram_dict["power"] = mel_power_specgram
            elif feature == "ist_frq":
                # Power spectral weighted smoothing via mel-filtering.
                # Mel filters plays the role of smoothing filters.
                # 1. Calculate the power.
                # 2. Calculate the mel-frequency power (A).
                # 3. Calculate the instantaneous frequency multiplied by power
                #    using Auger-Flandrin equation.
                # 4. Calculate the mel-frequency instantaneous frequency
                #    multiplied by power (B).
                # 5. Divide (B) by (A) to calculate the mel-frequency
                #    instantaneous frequency.
                specgram_dict["ist_frq"] = self.mel_scale(
                    cmplx_specgram_dict["ist_frq"].real \
                    * cmplx_specgram_dict["power"].imag \
                    - cmplx_specgram_dict["ist_frq"].imag \
                    * cmplx_specgram_dict["power"].real
                ) / mel_power_specgram
            elif feature == "grp_dly":
                # Power spectral weighted smoothing via mel-filtering.
                # Mel filters plays the role of smoothing filters.
                # 1. Calculate the power.
                # 2. Calculate the mel-frequency power (A).
                # 3. Calculate the group delay multiplied by power using
                #    Auger-Flandrin equation.
                # 4. Calculate the mel-frequency grouop delay multiplied by
                #    power (B).
                # 5. Divide (B) by (A) to calculate the mel-frequency group
                #    delay.
                specgram_dict["grp_dly"] = self.mel_scale(
                    cmplx_specgram_dict["grp_dly"].real \
                    * cmplx_specgram_dict["power"].real \
                    + cmplx_specgram_dict["grp_dly"].imag \
                    * cmplx_specgram_dict["power"].imag
                ) / mel_power_specgram
        cmplx_specgram_dict.clear()
        return specgram_dict

    def extra_repr(self) -> str:
        return f"fs={self.fs}, num_filters={self.num_filters}, " \
               f"window_length={self.window_length}, " \
               f"shift_length={self.shift_length}, f_min={self.f_min}, " \
               f"f_max={self.f_max}, num_fft={self.num_fft}, " \
               f"window_name={repr(self.window_name)}, " \
               f"power_threshold={self.power_threshold}, " \
               f"features={self.features}, " \


################################################################################
################################################################################
### Global variables
################################################################################
################################################################################
_FEATURES = frozenset({"power", "phase", "ist_frq", "grp_dly"})


################################################################################
################################################################################
### Helper classes and functions
################################################################################
################################################################################
class MelScale(nn.Module):
    r"""Turn a normal STFT into a mel frequency STFT, using a conversion
    matrix. This uses triangular filter banks.

    Note that this implementation is a modification of
    `torchaudio.transforms.MelScale` [1] to enable l^p-norm normalization.

    Args:
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`. (Default: ``201``)
        norm (str, float, or None, optional): If ``'slaney'``, divide the triangular mel weights by the width of the mel band
            (area normalization). If a positive scalar, l^{norm} normalization will be performed. (Default: ``1``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    References:
    [1] https://pytorch.org/audio/0.11.0/_modules/torchaudio/transforms.html#MelScale
        Copyright (c) 2017 Facebook Inc. (Soumith Chintala)
        BSD 2-Clause License
        https://opensource.org/licenses/BSD-2-Clause
    """
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[Union[Literal["slaney"], float]] = 1,
        mel_scale: str = "htk",
    ) -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        if f_min > self.f_max:
            raise ValueError(f"Require f_min: {f_min} < f_max: {self.f_max}")

        if self.norm is None or self.norm == "slaney":
            fb = AF.melscale_fbanks(
                n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate,
                self.norm, self.mel_scale)
        elif self.norm > 0:
            fb = AF.melscale_fbanks(
                n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate,
                None, self.mel_scale)
            fb /= fb.norm(p=self.norm, dim=0, keepdim=True)
        else:
            raise ValueError(
                'norm must be None, "slaney", or a positive value.')
        self.register_buffer("fb", fb)

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """

        # (..., time, freq) dot (freq, n_mels) -> (..., n_mels, time)
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)

        return mel_specgram


def _generate_general_cosine_window(
    window_length: int,
    window_name: Literal[
        "blackman", "blackmanharris", "boxcar", "hamming", "hann", "hanning",
        "nuttall", "rect",
    ] = "hann",
    mode: Optional[Literal["power", "ist_frq", "grp_dly"]] = None,
    dtype = None,
    device = None,
) -> Tensor:
    r"""The coefficients of the window function are referred from [1].

    References
    ----------
    [1] https://github.com/scipy/scipy/blob/v1.8.1/scipy/signal/windows/_windows.py
        Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
        BSD 3-Clause License
        https://opensource.org/licenses/BSD-3-Clause
    """
    factory_kwargs = dict(dtype=dtype, device=device)
    if not (mode is None or mode in {"power", "ist_frq", "grp_dly"}):
        raise ValueError(
            f'mode must be None, "ist_frq", or "grp_dly"; found {mode}')

    if isinstance(window_name, str):
        if window_name == "rect" or window_name == "boxcar":
            coefs = [1.0]
        elif window_name == "hann" or window_name == "hanning":
            coefs = [0.5, 0.5]
        elif window_name == "hamming":
            coefs = [0.54, 0.46]
        elif window_name == "blackman":
            coefs = [0.42, 0.5, 0.08]
        elif window_name == "blackmanharris":
            coefs = [0.35875, 0.48829, 0.14128, 0.01168]
        elif window_name == "nuttall":
            coefs = [0.3635819, 0.4891775, 0.1365995, 0.0106411]
        else:
            raise ValueError(f"Unknown window name; found {window_name}.")

    num_coefs = len(coefs)
    coefs = torch.tensor(coefs, **factory_kwargs).view(-1, 1)
    time_in_sample \
        = torch.arange(window_length, **factory_kwargs) - window_length/2
    time = time_in_sample / window_length  # [-0.5, 0.5)
    phase = time * (2*math.pi)  # [-pi, pi)
    phase = ((2*math.pi) * time.view(1, -1)) \
        * torch.arange(num_coefs, **factory_kwargs).view(-1, 1)
    window = torch.zeros(window_length, **factory_kwargs)
    window_norm = (coefs * torch.cos(phase)).sum(dim=0).pow(2.0).sum().sqrt()
    if mode is None or mode == "power":
        window = torch.sum(coefs * torch.cos(phase), dim=0)
    elif mode == "ist_frq":
        # The unit of the instantaneous frequency is [rad/sample].
        coefs *= (-2*math.pi / window_length) \
            * torch.arange(num_coefs, **factory_kwargs).view(-1, 1)
        window = torch.sum(coefs * torch.sin(phase), dim=0)
    elif mode == "grp_dly":
        # The unit of the group delay is [rad * sample].
        window = torch.sum(coefs * torch.cos(phase), dim=0)
        window *= (2*math.pi) * time_in_sample
    else:
        raise ValueError(
            f'mode must be None, "power", "ist_frq", or "grp_dly"; found '
            f'{mode}')

    return window / window_norm
