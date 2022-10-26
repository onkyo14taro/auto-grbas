# Supplementary Programs

This directory contains programs for calculating acoustic features.  
*Note that some scripts are only available in "Supplementary Materials" of our paper.*


## Programs

- `acfeatures.praat`
  - *This file is a dummy.*
    - *The actual code is available in "Supplementary Materials" of our paper.*
  - Calculate acoustic features for voice waveforms contained in a specified directory using Praat.
    - The kinds of acoustic features calculated by this program are not included in `acfeatures.ipynb`.
  - This program can optionally perform preprocessing on the waveforms.
    - The preprocessing is the same as the preprocessing done in `preprocess.praat`.
    - If only the preprocessing is necessary, use `preprocess.praat`.
- `acfeatures.ipynb`
  - Calculate acoustic features for voice waveforms contained in a specified directory using Python.
    - **The waveforms should be preprocessed using `preprocess.praat` before the calculation of the acoustic features.**
    - The kinds of acoustic features calculated by this program are not included in `acfeatures.praat`.
- `acfeatures.py`
  - Define functions to calculate acoustic features that are not calculated by Praat.
- `preprocess.praat`
  - *This file is a dummy.*
    - *The actual code is available in "Supplementary Materials" of our paper.*
  - Perform preprocessing on the waveforms contained in a specified directory using Praat.
    - The preprocessed waveforms are output to another specified directory.
- `requirements.txt`
  - Packages required to run the Python scripts.


## Examples contents

- `sample_audio/`
  - Vowel /a/ samples.
- `sample_audio_preprocessed/`
  - Vowel /a/ samples preprocessed by `preprocess.praat`.
- `acfeatures_praat.csv`
  - The acoustic features calculated using `acfeatures.praat` for the audio contained in `sample_audio`
- `acfeatures_python.csv`
  - The acoustic features calculated using `acfeatures.ipynb` for the audio contained in `sample_audio`


## Acoustic features

The acoustic features that can be calculated are as follows:

- Acoustic features precomputed by using Praat:
  - Smoothed cepstral peak prominence (CPPS) [1, 2, 3]
  - Glottal-to-noise excitation (GNE) ratio (GNEmax-4500 Hz) [2, 4, 7]
  - High frequency noise (Hfno-6000 Hz) [2, 5]
  - Harmonics-to-noise ratio (HNR) [1, 7]
  - Harmonics-to-noise ratio from Dejonckere and Lebacq (HNR-D) [2, 6]
  - Differences between the amplitudes of the first and second harmonics in the
    spectrum (H1-H2) [2]
  - Jitter local (Jit) [2]
  - Standard deviation of period (PSD) [2]
  - Shimmer local (Shim) [1, 2]
  - Shimmer local dB (Shim-dB) [1, 2]
  - Slope of the long-term average spectrum (Slope) [1]
  - Tilt of of the trend line through the long-term average spectrum (Tilt) [1]
- Acoustic features precomputed by using Python:
  - Ratio of points above linear average (RALA) [7, 8]
  - Moments (first, second, third, fourth, skewness, kurtosis) of specific
    loudness [9]
  - Sharpness [9, 10]


## References

[1] Y. Maryn, P. Corthals, P. Van Cauwenberge, N. Roy, and M. D. Bodt, "Toward Improved Ecological Validity in the Acoustic Measurement of Overall Voice Quality: Combining Continuous Speech and Sustained Vowels," J. Voice, vol. 24, no. 5, pp. 540--555, 2010.  
[2] B. Barsties v. Latoszek, Y. Maryn, E. Gerrits, and M. De Bodt, "The Acoustic Breathiness Index (ABI): A Multivariate Acoustic Model for Breathiness," J. Voice, vol. 31, no. 4, pp. 511.e11--511.e27, Jul. 2017.  
[3] J. M. Hillenbrand and R. A. Houde, "Acoustic correlates of breathy vocal quality: Dysphonic voices and continuous speech," J. Speech Lang. Hear. Res., vol. 39, no. 2, pp. 311--321, 1996.  
[4] D. Michaelis, T. Gramss, and H. W. Strube, "Glottal-to-Noise Excitation Ratio - A New Measure for Describing Pathological Voices," Acustica, vol. 83, no. 4, pp. 700--706, 1997.  
[5] P. H. Dejonckere, "Recognition of hoarseness by means of LTAS," Int. J. Rehabil. Res., vol. 7, pp. 73--74, 1983.  
[6] P. H. Dejonckere and J. Lebacq, "Harmonic emergence in formant zone of a sustained [a] as a parameter for evaluating hoarseness," Acta Otorhinolaryngol. Belg., vol. 41, no. 6, pp. 988--996, 1987.  
[7] J. A. Gómez-García, L. Moro-Velázquez, J. Mendes-Laureano, G. Castellanos-Dominguez, and J. I. Godino-Llorente, "Emulating the perceptual capabilities of a human evaluator to map the GRB scale for the assessment of voice disorders," Eng. Appl. Artif. Intell., vol. 82, pp. 236--251, 2019.  
[8] L. Moro-Velázquez, J. A. Gómez-García, J. I. Godino-Llorente, and G. Andrade-Miranda, "Modulation Spectra Morphological Parameters: A New Method to Assess Voice Pathologies according to the GRBAS Scale," Biomed Res. Int., vol. 2015, 2015, doi: 10.1155/2015/259239.  
[9] S. Anand, L. M. Kopf, R. Shrivastav, and D. A. Eddins, "Objective Indices of Perceived Vocal Strain," J. Voice, vol. 33, no. 6, pp. 838--845, 2019.  
[10] H. Fastl and E. Zwicker, "Sharpness and Sensory Pleasantness," in Psychoacoustics: Facts and Models, H. Fastl and E. Zwicker, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2007, pp. 239--246
