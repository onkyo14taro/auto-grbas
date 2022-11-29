# Automatic GRBAS Scoring of Pathological Voices using Deep Learning and a Small Set of Labeled Voice Data

- Authors: Shunsuke Hidaka, Yogaku Lee, Moe Nakanishi, Kohei Wakamiya, Takashi Nakagawa, and Tokihiko, Kaburagi
- Journal: Journal of Voice
- Date: 2022
- Link: [https://www.sciencedirect.com/science/article/pii/S0892199722003472](https://www.sciencedirect.com/science/article/pii/S0892199722003472)

## About

This repository contains the implementation of "[Automatic GRBAS Scoring of Pathological Voices using Deep Learning and a Small Set of Labeled Voice Data](https://www.sciencedirect.com/science/article/pii/S0892199722003472)".

*Note that some scripts in `Supplementary_Programs/` are only available in ["Supplementary Materials" of our paper](https://www.sciencedirect.com/science/article/pii/S0892199722003472).*


## Contents

- `data/`: **Dummy** dataset (all audio data were identical and uttered by the author, Hidaka).
- `notebooks/`: Examples of time-frequency representations and data augmentation techniques.
- `sample_audio/`: Sample audio files used in the notebooks.
- `shell/`: Shell scripts.
- `src/`: Python scripts.
- `tests/`: Test scripts.
- `Supplementary_Programs/`: Programs for acoustic feature calculation.


## Dependencies

Install Poetry if it is not already installed.

```
pip install poetry
```

You can install the necessary packages with the following command:

```
poetry install
```


## Usage

First, activate the virtual environment created by Poetry.  
Then, you can run an experiment on the **dummy** dataset with the following code:

```
./shell/sample.sh
```

Note that the dataset is dummy, so the learning will not be successful.
