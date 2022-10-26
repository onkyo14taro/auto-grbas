# {{Title Place Holder}}


## About

This project is for a paper in submission.


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
