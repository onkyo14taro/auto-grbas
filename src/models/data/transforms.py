from __future__ import annotations
import random
from typing import Literal

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Fade
import torchaudio.functional as AF


class WaveProcessor(nn.Module):
    r"""Wave processor.

    This class performs the following series of processes (all optional):

    1. Amplitude Normalization (mean 0 and SD 1)
    2. Random speed perturbation
    3. Random crop
    4. Fade-in and Fade-out
    5. Padding to make waveform length no shorter than a certain minimum length

    Parameters
    ----------
    normalize : bool
        Whether to normalize the amplitude to mean 0 and SD 1.

        By default `True`.

    random_crop : bool
        Whether to crop waveforms randomly or not.

        By default, `True`.

    random_crop_min_percent : int
        Minimum percentage of cropped length relative to original length.

        By default, `80` percent.

    random_crop_max_percent : int
        Maximum percentage of cropped length relative to original length.

        By default, `100` percent.

    random_speed : bool
        Whether to apply random speed perturbation or not.

        By default, `True`

    random_speed_min_percent : int
        Minimum percentage of speed perturbation.

        By default, `85` percent.

    random_speed_max_percent : int
        Maximum percentage of speed perturbation.

        By default, `115` percent.

    fade : bool
        Whether to pergorm fading or not.

        By default, `True`.

    fade_in_len : int
        Fade-in length in sample.

        By default, `160` (if Fs is 16 kHz, 10 ms)

    fade_out_len : int
        Fade-out length in sample.

        By default, `160` (if Fs is 16 kHz, 10 ms)

    fade_shape : Literal["quarter_sine", "half_sine", "linear",
    "logarithmic", "exponential"]
        Shape of fade.

        Available shapes are as follows:

        - `"quarter_sine"`
        - `"half_sine"`
        - `"linear"`
        - `"logarithmic"`
        - `"exponential"`

        By default, `'linear'`

    minimum_pad : bool
        Whether to pad waveforms so that it is not shorter than a certain
        minimum length.

        By default, `False`.

    minimum_pad_len : int
        Minimum waveform length.

        By default, `8000` (if Fs is 16 kHz, 500 ms).
    """
    def __init__(
        self,
        normalize: bool = True,
        random_crop: bool = True,
        random_crop_min_percent: int = 80,
        random_crop_max_percent: int = 100,
        random_speed: bool = True,
        random_speed_min_percent: int = 85,
        random_speed_max_percent: int = 115,
        fade: bool = True,
        fade_in_len: int = 160,   # If Fs is 16 kHz, 10 ms
        fade_out_len: int = 160,  # If Fs is 16 kHz, 10 ms
        fade_shape: Literal[
            "quarter_sine",
            "half_sine",
            "linear",
            "logarithmic",
            "exponential",
        ] = 'linear',
        minimum_pad: bool = False,
        minimum_pad_len: int = 8000,  # If Fs is 16 kHz, 500 ms
    ) -> None:
        super().__init__()
        pre_processes = []
        augment_processes = []
        post_processes = []

        if normalize:
            pre_processes.append(Normalize())

        if random_speed:
            augment_processes.append(RandomSpeed(
                min_percent=random_speed_min_percent,
                max_percent=random_speed_max_percent))

        if random_crop:
            augment_processes.append(VariableLengthRandomCrop(
                min_percent=random_crop_min_percent,
                max_percent=random_crop_max_percent))

        if fade:
            post_processes.append(Fade(
                fade_in_len=fade_in_len,
                fade_out_len=fade_out_len,
                fade_shape=fade_shape))

        if minimum_pad:
            post_processes.append(MinimumPad(minimum_len=minimum_pad_len))

        self.pre_processes = nn.Sequential(*pre_processes) \
            if pre_processes else nn.Identity()
        self.augment_processes = nn.Sequential(*augment_processes) \
            if augment_processes else nn.Identity()
        self.post_processes = nn.Sequential(*post_processes) \
            if post_processes else nn.Identity()

    def forward(
        self,
        wave: Tensor,
        augment: bool = False,
    ) -> Tensor:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        wave : Tensor [shape=(*, time)]
            Waveform.

        augment : bool
            Whether to apply data augmentation or not.

            Specifically, data augmentation is the following process:

            - Random speed perturbation
            - Random crop

            By default, `False`.

        Returns
        -------
        wave_processed : Tensor [shape=(*, time_processed)]
            Processed waveform.
        """
        wave = self.pre_processes(wave)
        if augment:
            wave = self.augment_processes(wave)
        return self.post_processes(wave)


class Normalize(nn.Module):
    r"""Wave amplitude normalizer.

    This class normalizes the amplitude of each waveform to mean 0 and SD 1.
    """
    def forward(
        self,
        wave: Tensor,
    ) -> Tensor:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        wave : Tensor [shape=(*, time)]
            Waveform.

        Returns
        -------
        wave_normalized : Tensor [shape=(*, time)]
            Normalized waveform.
        """
        mean = wave.mean(dim=-1, keepdim=True)
        std = wave.std(dim=-1, keepdim=True)
        std[std == 0.0] = 1.0
        return (wave - mean) / std


class VariableLengthRandomCrop(nn.Module):
    r"""Wave random cropper.

    This class randomly cropped waveforms at random lengths.

    Parameters
    ----------
    min_percent : int
        Minimum percentage of cropped length relative to original length.

        By default, `80` percent.

    max_percent : int
        Maximum percentage of cropped length relative to original length.

        By default, `100` percent.
    """
    def __init__(
        self,
        min_percent: int = 80,
        max_percent: int = 100,
    ) -> None:
        super().__init__()
        if not (0 < min_percent <= max_percent <= 100):
            raise ValueError(
                'Must satisfy 0 < min_percent <= max_percent <= 100.')
        self.min_percent = min_percent
        self.max_percent = max_percent

    def forward(
        self,
        wave: Tensor,
    ) -> Tensor:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        wave : Tensor [shape=(*, time)]
            Waveform.

        Returns
        -------
        wave_cropped: Tensor [shape=(*, time_cropped)]
            Randomly cropped waveform.
        """
        orig_len = wave.size(-1)
        target_len = round(
            orig_len*random.randint(self.min_percent, self.max_percent)/100)
        start = random.randint(0, orig_len-target_len)
        return wave[..., start:start+target_len]

    def extra_repr(self) -> str:
        return f'min_percent={self.min_percent}, max_percent={self.max_percent}'


class RandomSpeed(nn.Module):
    r"""Wave random speed perturbator.

    This class randomly applies speed perturbation to waveforms.

    Parameters
    ----------
    min_percent : int
        Minimum percentage of cropped length relative to original length.

        By default, `80` percent.

    max_percent : int
        Maximum percentage of cropped length relative to original length.

        By default, `100` percent.
    """
    def __init__(
        self,
        min_percent: int = 85,
        max_percent: int = 115,
    ) -> None:
        super().__init__()
        if not (0 < min_percent <= max_percent):
            raise ValueError('Must satisfy 0 < min_percent <= max_percent.')
        self.min_percent = min_percent
        self.max_percent = max_percent

    def forward(
        self,
        wave: Tensor,
    ) -> Tensor:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        wave : Tensor [shape=(*, time)]
            Waveform.

        Returns
        -------
        wave_perturbed: Tensor [shape=(*, time_perturbed)]
            Waveform with random speed perturbation.
        """
        percent = random.randint(self.min_percent, self.max_percent)
        return self._change_speed(wave, percent)

    def _change_speed(
        self,
        wave: Tensor,
        percent: int,
    ) -> Tensor:
        # Resampling parameters are based on librosa’s Kaiser Window Best.
        # https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#comparison-against-librosa
        return AF.resample(
            wave,
            orig_freq=percent,
            new_freq=100,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="kaiser_window",
            beta=14.769656459379492
        )

    def extra_repr(self) -> str:
        return f'min_percent={self.min_percent}, max_percent={self.max_percent}'


class MinimumPad(nn.Module):
    r"""Wave padder.

    This class pads waveform to make the length no shorter than a certain
    minimum length.

    Parameters
    ----------
    minimum_len : int
        Minimum waveform length.

        By default, `8000` (if Fs is 16 kHz, 500 ms).
    """
    def __init__(
        self,
        minimum_len: int
    ) -> None:
        super().__init__()
        self.minimum_len = minimum_len

    def forward(
        self,
        wave: Tensor,
    ) -> Tensor:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        wave : Tensor [shape=(*, time)]
            Waveform.

        Returns
        -------
        wave_padded: Tensor [shape=(*, time_padded)]
            Padded waveform.

            If `wave.size(-1) < self.minimum_len`, `time_padded` will be
            `self.minimum_len`.
            If `wave.size(-1) >= self.minimum_len`, `time_padded` will be
            `wave.size(-1)` (no padding will be done).
        """
        wave_len = wave.size(-1)
        pad_len = self.minimum_len - wave_len
        if pad_len > 0:
            pre_pad = pad_len // 2
            post_pad = pad_len - pre_pad
            return F.pad(wave, (pre_pad, post_pad))
        else:
            return wave

    def extra_repr(self) -> str:
        return f'minimum_len={self.minimum_len}'
