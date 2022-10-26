from __future__ import annotations
from functools import lru_cache
import math
from pathlib import Path
from typing import Literal, Optional, Union

import librosa
import mosqito
import numba
import numpy as np
import numpy.typing as npt
import scipy.signal


################################################################################
################################################################################
### RALA
################################################################################
################################################################################
def calc_rala(
    path: Union[str, Path],
    fs: Optional[float] = 20000,
    window_second: float = 0.0125,
    window_fn: Optional[Union[
        str,
        Callable[[int], npt.NDArray[np.float_]]
    ]] = 'hamming',
    fft_second: Optional[float] = None,
    shift_second: float = 0.010,
    mod_window_second: float = 0.18,
    mod_window_fn: Optional[Union[
        str,
        Callable[[int], npt.NDArray[np.float_]]
    ]] = 'hamming',
    mod_num_bands: int = 1025,
    mod_max_hz: float = 240.0,
    with_padding: bool = False,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Calculate the ratio of points above linear average (RALA) [1].

    The procedure for calculation is as follows:

    1. Calculate the amplitude spectrogram with a shift width of one sample.
       `window_second`, `window_fn`, and `fft_second` are used for this step.
    2. Calculate the modulation spectrum with shifting the center time.
       1. Shift the center time by `shift_second`.
       2. Window the amplitude spectrogram in the time direction based on
          the center time. `mod_window_second` and `mod_window_fn` are used for
          this step.
       3. For each amplitude seuquence (i.e., envelope) of each acoustic
          frequency bin of the windowed spectrogram, perform DFT. At this time,
          the DFT is performed so that the number of modulation frequency bins
          in the frequency range from `-mod_max_hz` to `mod_max_hz` is
          `mod_num_bands`.
       4. The resulting two-dimensional representation with axes of acoustic
          frequency and modulation frequency is the modulation spectrum.
    3. Calculate the RALA for each modulation spectrum.
       1. Calculate the average value of the amplitude modulation spectrum.
       2. Calculate the area of the region with values above the average and the
          area of the rest region in the modulation spectrum.
       3. Divide the area of the region with values above the average by the
          area of the region with values below the average. This is the RALA.

    Parameters
    ----------
    path : path-like object
        Path of an audio file.

    fs : float
        Sampling frequency [Hz]. 

        By default, `20000`.

    window_second : float
        Length of the window function for STFT [s].

        By default, `0.0125`.

    window_fn : str, Callable[[int], npt.NDArray[np.float_]], or None
        Window function for STFT. Window name (e.g., `"hamming"`) or window
        function (e.g., `lambda window_sample: np.ones(window_sample))`.

        By default, `"hamming"`.

    fft_second : float or None
        Length of FFT [s]. If `None`, fft_second is set to `window_second`.

        By default, `None`.

    mod_window_second : float
        Length of a window function to calculate the modulation
        spectrum from the amplitude spectrogram [second].
        The default setting of 0.18 seconds is the value used in
        [Gómez-García2019a].

    shift_second : float
        Length of shifting [s].

        By default, `0.010`.

    mod_window_second : float
        Length of a window function to calculate the modulation spectrum from
        the amplitude spectrogram [s]. The default setting of 0.18 s is the
        value used in [2].

        By default, `0.18`.

    mod_window_fn : str, Callable[[int], npt.NDArray[np.float_]], or None
        Window to calculate the modulation spectrum from the amplitude
        spectrogram. Window name (e.g., `"hamming"`) or window function (e.g.,
        `lambda window_sample: np.ones(window_sample)`).

        By default, `"hamming"`.

    mod_num_bands : int
        Number of bands of modulation frequency including negative frequency
        range. In [2], `mod_num_bands` is set to 1024, but the default value is
        1025 because an even number makes the modulation spectrum asymmetric.

        By default, `1025`.

    mod_max_hz : float
        Maximal modulation frequency [Hz]. The default setting of 240 Hz is the
        value that gave the best results in [1].

        By default, `240`.

    Returns
    -------
    rala : NDArray [float, shape=(N, )]
        RALA.

    time : NDArray [float, shape=(N, )]
        Time [s].

    Note
    ----
    - There are several definitions and implementations of the "modulation
      spectrum." The Modulation Toolbox Version 2.1 for MATLAB [3] was used in
      [1]. The Toolbox implements several types of modulation spectra. In [1],
      the most effective definition was the simplest one; i.e., the modulation
      spectrum is defined by the Fourier transform of the Hilbert envelope.
    - In [2], for speech with a sampling frequency of 20 kHz, there are 128
      divisions from 0 Hz to the Nyquist frequency. If the number of DFT points
      is 255 and the window length matches the number of DFT points, the window
      length is 0.01275 seconds. As a close approximation, the default value for
      `window_second` is 0.0125 seconds.

    References
    ----------
    [1] L. Moro-Velázquez, J. A. Gómez-García, J. I. Godino-Llorente, and G.
        Andrade-Miranda, "Modulation Spectra Morphological Parameters: A New
        Method to Assess Voice Pathologies according to the GRBAS Scale," Biomed
        Res. Int., vol. 2015, 2015, doi: 10.1155/2015/259239.
    [2] J. A. Gómez-García, L. Moro-Velázquez, J. Mendes-Laureano, G.
        Castellanos-Dominguez, and J. I. Godino-Llorente, "Emulating the
        perceptual capabilities of a human evaluator to map the GRB scale for
        the assessment of voice disorders," Eng. Appl. Artif. Intell., vol. 82,
        pp. 236--251, 2019.
    [3] Atlas, L, P. Clark, and S. Schimmel, "Modulation Toolbox Version 2.1 for
        MATLAB." 2010. [Online].
        Available: http://isdl.ee.washington.edu/projects/modulationtoolbox/
    """
    if fft_second is None:
        fft_second = window_second

    if window_second <= 0:
        raise ValueError('window_second must be positive.')
    if shift_second <= 0:
        raise ValueError('shift_second must be positive.')
    if fft_second <= 0:
        raise ValueError('fft_second must be positive.')
    if mod_window_second <= 0:
        raise ValueError('mod_window_second must be positive.')
    if mod_num_bands <= 0:
        raise ValueError('mod_num_bands must be a positive integer.')
    if mod_max_hz <= 0:
        raise ValueError('mod_max_hz must be a positive number.')

    window_sample = round(window_second * fs)
    shift_sample = round(shift_second * fs)
    fft_sample = round(fft_second * fs)
    mod_window_sample = round(mod_window_second * fs)

    wave, fs = librosa.load(path, sr=fs)
    amp_spec = np.abs(librosa.stft(
        wave,
        n_fft=fft_sample,
        hop_length=1,
        win_length=window_sample,
        window=window_fn,
        center=True,
    )).T
    if with_padding:
        start_frame = 0
        indice_center = np.arange(
            0,
            amp_spec.shape[0]*shift_sample,
            shift_sample)
    else:
        pad_pre_sample = mod_window_sample//2 + window_sample//2
        pad_post_sample \
            = (mod_window_sample-1)//2 + (window_sample-1)//2
        start_frame = math.ceil(pad_pre_sample / shift_sample)
        indice_center = np.arange(
            start_frame * shift_sample,
            len(wave) - pad_post_sample,
            shift_sample)

    stop_frame = start_frame + len(indice_center)
    time = np.arange(
        shift_sample*start_frame,
        shift_sample*stop_frame,
        shift_sample) / fs

    rala = np.array([
        _calc_rala_frame(
            amp_spec=amp_spec,
            idx_center=idx_center,
            fs=fs,
            mod_window_second=mod_window_second,
            mod_window_fn=mod_window_fn,
            mod_num_bands=mod_num_bands,
            mod_max_hz=mod_max_hz,
        )
        for idx_center in indice_center
    ])

    return rala, time


def _calc_rala_frame(
    amp_spec: npt.NDArray[np.float_],
    idx_center: int,
    fs: float,
    mod_window_second: float,
    mod_window_fn: Optional[Union[
        str,
        Callable[[int], npt.NDArray[np.float_]]
    ]],
    mod_num_bands: int,
    mod_max_hz: float,
) -> float:
    cmplx_ms = _calc_modulation_spectrum(
        amp_spec,
        idx_center,
        fs=fs,
        amp_spec_shift_second=1/fs,
        mod_window_second=mod_window_second,
        mod_window_fn=mod_window_fn,
        mod_num_bands=mod_num_bands,
        mod_max_hz=mod_max_hz,
    )
    amp_ms = np.abs(cmplx_ms)
    NT = amp_ms.shape[0]*amp_ms.shape[1]
    NA = np.sum(amp_ms >= np.mean(amp_ms))
    NB = NT - NA
    return NA / NB


def _calc_modulation_spectrum(
    amp_spec: npt.NDArray[np.float_],
    idx_center: int,
    fs: float,
    amp_spec_shift_second: float,
    mod_window_second: float,
    mod_window_fn: Optional[Union[
        str,
        Callable[[int], npt.NDArray[np.float_]]
    ]],
    mod_num_bands: int,
    mod_max_hz: float,
) -> npt.NDArray[np.complex_]:
    """Calculate the modulation spectrum from a amplitude spectrogram.

    Parameters
    ----------
    amp_spec : NDArray [float, shape=(num_frames, num_ac_bands)]
        Amplitude spectrogram.

    idx_center : int
        Center index where the modulation spectrum is calculated.

    fs : float
        Sampling frequency.

    amp_spec_shift_second : float
        Length of shifting of `amp_spec` [s].

    mod_window_second : float
        Length of a window function to calculate the modulation spectrum from
        the amplitude spectrogram [s].

    mod_window_fn : str, Callable[[int], npt.NDArray[np.float_]], or None
        Window to calculate the modulation spectrum from the amplitude
        spectrogram. Window name (e.g., `"hamming"`) or window function (e.g.,
        `lambda window_sample: np.ones(window_sample)`).

    mod_num_bands : int
        Number of bands of modulation frequency including negative frequency
        range.

    mod_max_hz : float
        Maximal modulation frequency [Hz].
    """
    if amp_spec.ndim != 2:
        raise ValueError('amp_spec must be a 2D array.')

    amp_spec_shift_sample = round(fs * amp_spec_shift_second)
    fs_mod = fs / amp_spec_shift_sample
    mod_window_sample = round(fs_mod * mod_window_second)
    if amp_spec_shift_sample <= 0:
        raise ValueError('Must satisfy amp_spec_shift_second >= 1/fs.')
    if mod_window_sample <= 0:
        raise ValueError('Must satisfy mod_window_sample >= 1/fs.')

    # shape=(acoustic_num_bands, window_sample)
    framed_modulators = _frame_modulators(
        amp_spec, idx_center, mod_window_sample)
    acoustic_num_bands, _ = framed_modulators.shape
    # shape=(1+mod_num_bands//2, window_sample)
    cmplx_sinusoids = _complex_sinusoids(
        fs=fs,
        amp_spec_shift_sample=amp_spec_shift_sample,
        mod_window_sample=mod_window_sample,
        mod_window_fn=mod_window_fn,
        mod_num_bands=mod_num_bands,
        mod_max_hz=mod_max_hz)

    ms = np.empty(
        (acoustic_num_bands, mod_num_bands), dtype=np.complex64)
    idx_ms_0hz = (mod_num_bands-1)//2
    _dft(framed_modulators.astype(dtype=cmplx_sinusoids.dtype),
        cmplx_sinusoids,
        framed_modulators.shape[0],
        cmplx_sinusoids.shape[0],
        ms[:, idx_ms_0hz:])
    if mod_num_bands % 2 == 0:
        ms[:, :idx_ms_0hz] = np.conj(ms[:, -2:idx_ms_0hz:-1])
    else:
        ms[:, :idx_ms_0hz] = np.conj(ms[:, -1:idx_ms_0hz:-1])
    return ms


def _frame_modulators(
    amp_spec: npt.NDArray[np.float_],
    idx_center: int,
    mod_window_sample: int,
) -> npt.NDArray[np.float_]:
    """Frame the modulators (Hilbert envelopes)."""
    ### Frame and pad amp_spec centred on the idx_centre.
    ############################################################
    signal_len = amp_spec.shape[0]
    window_l_slope_len = mod_window_sample//2
    window_r_slope_len = mod_window_sample - window_l_slope_len - 1
    if not (-signal_len <= idx_center < signal_len):
        raise ValueError('idx_center must be in [-signal_len, signal_len)')
    if idx_center < 0:
        idx_center = signal_len + idx_center
    idx_start = max([idx_center-window_l_slope_len, 0])
    idx_stop = idx_center+1+window_r_slope_len
    pad_l = max([window_l_slope_len-idx_center, 0])
    pad_r = max([idx_center+1+window_r_slope_len-signal_len, 0])
    amp_spec = amp_spec[idx_start:idx_stop]
    if pad_l != 0 or pad_r != 0:
        amp_spec = np.pad(
            amp_spec, [[pad_l, pad_r], [0, 0]], mode='reflect')

    ### Change memory allocation for computational efficiency.
    ############################################################
    #    shape=(mod_window_sample, num_ac_bands)
    # -> shape=(num_ac_bands, mod_window_sample)
    modulators = np.ascontiguousarray(amp_spec.T)
    return modulators


@lru_cache
def _complex_sinusoids(
    fs: float,
    amp_spec_shift_sample: int,
    mod_window_sample: int,
    mod_window_fn: Optional[Union[
        str,
        Callable[[int], npt.NDArray[np.float_]]
    ]],
    mod_num_bands: int,
    mod_max_hz: float,
) -> npt.NDArray[np.complex_]:
    """Calculates the kernels for the modulation spectrum."""
    freq_mod = _modulation_frequency(
        mod_num_bands, mod_max_hz, onesided=True)
    fs_mod = fs / amp_spec_shift_sample
    angular_freq_mod_norm = (2*np.pi/fs_mod)*freq_mod
    time = np.linspace(
        -mod_window_sample/2, mod_window_sample/2-1, mod_window_sample)
    cmplx_sinusoids = np.exp(
        -1j*angular_freq_mod_norm.reshape(-1, 1) * time.reshape(1, -1))
    if mod_window_fn is not None:
        if isinstance(mod_window_fn, str):
            window = scipy.signal.get_window(mod_window_fn, mod_window_sample)
        else:
            window = mod_window_fn(mod_window_sample)
            if window.ndim != 1 or len(window) != mod_window_sample:
                raise ValueError(
                    'If window is a callable, must return a '
                    '1D array [shape=(window_sample, )].')
        cmplx_sinusoids *= window.reshape(1, -1)
    return cmplx_sinusoids.astype(np.complex64)  # shape=(1+mod_num_bands//2, mod_window_sample)


@lru_cache
def _modulation_frequency(
    mod_num_bands: int = 1024,
    mod_max_hz: float = 240.0,
    onesided: bool = False,
) -> npt.NDArray[np.float_]:
    """Calculate modulation frequencies."""
    if onesided:
        return np.linspace(0, mod_max_hz, 1+mod_num_bands//2)
    else:
        if mod_num_bands % 2 == 1:
            return np.linspace(-mod_max_hz, mod_max_hz, mod_num_bands)
        else:
            return np.linspace(-mod_max_hz, mod_max_hz, 1+mod_num_bands)[1:]


@numba.jit
def _dft(
    framed_modulators: npt.NDArray[np.float_],
    cmplx_sinusoids: npt.NDArray[np.complex_],
    num_ac_bands: int,
    num_sinusoids: int,
    out: npt.NDArray[np.complex_]
) -> None:
    for i in range(num_ac_bands):
        for j in range(num_sinusoids):
            out[i,j] = np.dot(
                framed_modulators[i], cmplx_sinusoids[j])


################################################################################
################################################################################
### Sharpness and loudness moments
################################################################################
################################################################################
def calc_sharpness_loudness_moments(
    path: Union[str, Path],
    wav_calib: Optional[float] = None,
    weighting: Literal["din", "aures", "bismarck", "fastl"] = "din",
    field_type: Literal["free", "diffuse"] = "free",
    skip: float = 0.2,
) -> dict[str, npt.NDArray[np.float_]]:
    """Calculate the acoustic sharpness and the moments of specific loudness.

    The Zwicker method for time-varying signals is used to calculate the
    acoustic loudness [1]. The moments (including skewness and kurtosis) of
    specific loudness were proposed in [2].

    Parameters:
    ----------
    path : path-like object
        Path of an audio file.

    wav_calib : float or None
        Wav file calibration factor [Pa/FS]. Level of the signal in Pa_peak
        corresponding to the full scale of the .wav file. If `None`, a
        calibration factor of 1 is considered.

        By default, `None`.

    weighting : "din", "aures", "bismarck", or "fastl"
        To specify the weighting function used for the sharpness computation.
        The available weightings are as follows:

        - `"din"`: DIN 45692 (2009) [3]
        - `"aures"`: Aures (1985) [4]
        - `"bismarck"`: Bismarck (1974) [5]
        - `"fastl"`: Fastl and Zwicker (2007) [6]

        sharpness computation.'din' by default,'aures', 'bismarck','fastl'

        By default, `"din"`.

    field_type : "free", "diffuse"
        Type of soundfield:

        - `"free"`: free field
        - `"diffuse"`: diffuse field

        By default, `"free"`.

    skip : float
        Seconds to be cut at the beginning of the analysis to skip the
        transient effect.

        By default, `0.2`.

    Returns
    -------
    results: dict
        A dictionary containing results:

        - `"bark_axis"` : NDArray[float, shape=(Nbark, )]
              Bark axis [Bark].
        - `"time_axis"` : NDArray[float, shape=(Ntime, )]
              Time axis [s].
        - `"loudness_specific"` : NDArray[float, shape=(Nbark, Ntime)]
              Specific loudness [sone/bark].
        - `"loudness"` : NDArray[float, shape=(Ntime, )]
              Total loudness [sone].
        - `"loudness_moment_1"` : NDArray[float, shape=(Ntime, )]
              First moment of the specific loudness.
        - `"loudness_moment_2"` : NDArray[float, shape=(Ntime, )]
              Second moment of the specific loudness.
        - `"loudness_moment_3"` : NDArray[float, shape=(Ntime, )]
              Third moment of the specific loudness.
        - `"loudness_moment_4"` : NDArray[float, shape=(Ntime, )]
              Fourth moment of the specific loudness.
        - `"loudness_skewness"` : NDArray[float, shape=(Ntime, )]
              Skewness of the specific loudness.
        - `"loudness_kurtosis"` : NDArray[float, shape=(Ntime, )]
              Kurtosis of the specific loudness.
        - `"sharpness"` : NDArray[float, shape=(N, )]
              Sharpness [acum].

    References
    ----------
    [1] International Organization for Standardization, "Acoustics — Methods for
        calculating loudness — Part 1: Zwicker method," ISO 532-1:2017, 2017.
    [2] S. Anand, L. M. Kopf, R. Shrivastav, and D. A. Eddins, "Objective
        Indices of Perceived Vocal Strain," J. Voice, vol. 33, no. 6,
        pp. 838--845, 2019.
    [3] Deutsches Institut für Normung, "Measurement technique for the
        simulation of the auditory sensation of sharpness," DIN 45692, 2009.
    [4] W. Aures, "Ein Berechnungsverfahren der Rauhigkeit," Acta Acustica
        united with Acustica, vol. 58, no. 5, pp. 268--281, 1985.
    [5] G. von Bismarck, "Sharpness as an Attribute of the Timbre of Steady
        Sounds," Acta Acustica united with Acustica, vol. 30, no. 3,
        pp. 159--172, 1974.
    [6] H. Fastl and E. Zwicker, "Sharpness and Sensory Pleasantness," in
        Psychoacoustics: Facts and Models, H. Fastl and E. Zwicker, Eds. Berlin,
        Heidelberg: Springer Berlin Heidelberg, 2007, pp. 239--246.
    """
    ############################################################
    ### Load
    ############################################################
    sig, fs = mosqito.load(path, wav_calib=wav_calib)

    ############################################################
    ### Loudness for time-varying signals
    ############################################################
    N, N_spec, bark_axis, time_axis = mosqito.loudness_zwtv(
        sig, fs, field_type=field_type)
    bark_axis = bark_axis  # shape=(Nbark, 1)

    ############################################################
    ### Loudness moments
    ############################################################
    moment_1, moment_2, moment_3, moment_4, skewness, kurtosis \
        = _spectral_moments(N_spec, bark_axis)

    ############################################################
    ### Sharpness
    ############################################################
    # Compute sharpness from loudness
    S = mosqito.sharpness_din_from_loudness(
        N, N_spec, weighting=weighting, skip=0)

    ############################################################
    ### Cut transient effect
    ############################################################
    cut_index = np.argmin(np.abs(time_axis - skip))

    return {
        "bark_axis": bark_axis,
        "time_axis": time_axis[cut_index:],
        "loudness_specific": N_spec[:, cut_index:],
        "loudness": N[cut_index:],
        "loudness_moment_1": moment_1[cut_index:],
        "loudness_moment_2": moment_2[cut_index:],
        "loudness_moment_3": moment_3[cut_index:],
        "loudness_moment_4": moment_4[cut_index:],
        "loudness_skewness": skewness[cut_index:],
        "loudness_kurtosis": kurtosis[cut_index:],
        "sharpness": S[cut_index:],
    }


def _spectral_moments(
    spec: npt.NDArray[np.float_],
    freq: npt.NDArray[np.float_],
) -> dict:
    r"""Calculate spectral moments including skewness and kurtosis.

    Parameters
    ----------
    spec : NDArray [float, shape=(num_bins, *)] 
        Spectrum-like objects.

    freq : NDArray [float, shape=(num_bins, )]
        Frequency-bins-like objects.
    """
    if not (isinstance(spec, np.ndarray) and freq.ndim > 0):
        raise ValueError("spec must be an array.")
    if not (isinstance(freq, np.ndarray) and freq.ndim == 1):
        raise ValueError("freq must be a 1D array.")
    if spec.shape[0] != freq.shape[0]:
        raise ValueError("Must satisfy: spec.shape[0] == freq.shape[0]")

    freq = freq.reshape(-1, 1)  # shape=(num_bins, 1)

    ############################################################
    ### Loudness moments
    ############################################################
    # Regard the specific loudness as the probility distribution
    prob = spec / spec.sum(axis=0, keepdims=True)

    # First moment (a.k.a. centroid); shape=(1, time)
    moment_1 = np.sum(freq * prob, axis=0, keepdims=True)

    # Second (a.k.a. spread), third, fourth moments; shape=(1, time)
    deviation = freq - moment_1  # shape=(num_bins, time)
    deviation_abs = np.abs(deviation)  # shape=(num_bins, time)
    moment_2 = np.sum((deviation_abs**2)*prob, axis=0, keepdims=True) ** (1/2)
    moment_3 = np.sum((deviation_abs**3)*prob, axis=0, keepdims=True) ** (1/3)
    moment_4 = np.sum((deviation_abs**4)*prob, axis=0, keepdims=True) ** (1/4)

    # Skewness; shape=(1, Ntime)
    skewness \
        = np.sum((deviation**3)*prob, axis=0, keepdims=True) / (moment_2**3)

    # Kurtosis; shape=(1, Ntime)
    kurtosis \
        = np.sum((deviation**4)*prob, axis=0, keepdims=True) / (moment_2**4)

    return (
        moment_1.reshape(*spec.shape[1:]),
        moment_2.reshape(*spec.shape[1:]),
        moment_3.reshape(*spec.shape[1:]),
        moment_4.reshape(*spec.shape[1:]),
        skewness.reshape(*spec.shape[1:]),
        kurtosis.reshape(*spec.shape[1:]),
    )
