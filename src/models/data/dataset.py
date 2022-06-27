from __future__ import annotations
from pathlib import Path
import random
from typing import Any, Literal, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from .transforms import WaveProcessor


__all__ = [
    'GRBASDataset',
    'GRBASDataModule',
]


################################################################################
################################################################################
### Helper classes and functions
################################################################################
################################################################################
def seed_worker(
    worker_id: int,
) -> None:
    r"""Function for reproducibility of multi-process loading.

    This function is given in the DataLoader argument `worker_init_fn`. See [1]
    for details.

    Parameters
    ----------
    worker_id : int
        Worker ID.

    References
    ----------
    [1] https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _collate_one_wave(
    batch: list[tuple[Tensor, dict]]
) -> tuple[list[Tensor], dict[str, list]]:
    wave, properties = list(zip(*batch))
    properties = {k: [p[k] for p in properties]
                  for k in properties[0].keys()}
    return list(wave), properties


################################################################################
################################################################################
### Dataset
################################################################################
################################################################################
class GRBASDataset(Dataset):
    r"""GRBAS Dataset.

    Parameters
    ----------
    data_dir : str or Path
        Data directory path.

    mode : "supervised"
        Learning mode.

        By default, `"supervised"`.

    phase : "train", "valid", "test"
        Learning phase.

        By default, `"train"`.

    fold : 1, 2, 3, 4, or 5
        Partition number of five-fold cross-validation

        By default, `1`.

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
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        *,
        mode: Literal["supervised"] = "supervised",
        phase: Literal["train", "valid", "test"] = "train",
        fold: Literal[1, 2, 3, 4, 5] = 1,
        random_crop: bool = True,
        random_crop_min_percent: int = 80,
        random_crop_max_percent: int = 100,
        random_speed: bool = True,
        random_speed_min_percent: int = 85,
        random_speed_max_percent: int = 115,
    ) -> None:
        super().__init__()
        if mode not in {"supervised"}:
            raise ValueError(
                f'mode must be either "supervised"; found '
                f'{mode}.')
        if phase not in {"train", "valid", "test"}:
            raise ValueError(
                f'phase must be either "train", "valid", "test"; '
                f'found {phase}.')
        if fold not in {1, 2, 3, 4, 5}:
            raise ValueError(f"fold must be 1, 2, 3, 4, or 5; found {fold}.")
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.phase = phase
        self.fold = fold
        self._table = self._get_table()
        self._wave_processor = WaveProcessor(
            random_crop=random_crop,
            random_crop_min_percent=random_crop_min_percent,
            random_crop_max_percent=random_crop_max_percent,
            random_speed=random_speed,
            random_speed_min_percent=random_speed_min_percent,
            random_speed_max_percent=random_speed_max_percent,
        )

    def __len__(self):
        return len(self._table)

    def _get_table(
        self,
    ) -> pd.DataFrame:
        path = self.data_dir/'meta.csv'
        table = pd.read_csv(path).convert_dtypes().set_index('fileID')

        if self.phase == "test":
            mask = table["category"] == "test"

        else:
            other_folds = {i for i in range(1, 5+1) if i != self.fold}

            if self.phase == "train":
                mask = table["fold"].isin(other_folds)
            else:  # "valid"
                mask = table["fold"] == self.fold

            if self.mode == "supervised":
                mask &= table["category"] == "supervised"

        return table.loc[mask]

    def _load_audio(
        self,
        audio_filepath: Path,
    ) -> Tensor:
        wave, fs = sf.read(audio_filepath, dtype=np.float32)
        assert fs == 16000
        return torch.from_numpy(wave[None, None, :])

    def __getitem__(
        self,
        index: int
    ) -> Union[
        tuple[Tensor, dict[str, Any]],
        tuple[Tensor, Tensor, dict[str, Any]],
    ]:
        fileID = int(self._table.index[index])
        properties = {self._table.index.name: fileID}
        properties.update({
            k: None if pd.isna(v) else v
            for k, v in self._table.iloc[index].to_dict().items()
        })
        audio_filepath = self.data_dir/'audio'/f'{fileID}.wav'
        wave = self._load_audio(audio_filepath)
        wave = self._wave_processor(wave, augment=(self.phase=="train"))
        return wave, properties

    def __repr__(self) -> str:
        return f'GRBASDataset(data_dir="{self.data_dir}", mode="{self.mode}", '\
               f'phase="{self.phase}", fold={self.fold})'


################################################################################
################################################################################
### DataModule
################################################################################
################################################################################
class GRBASDataModule(LightningDataModule):
    r"""GRBAS Dataset.

    Parameters
    ----------
    data_dir : str or Path
        Data directory path.

    fold : 1, 2, 3, 4, or 5
        Partition number of five-fold cross-validation

        By default, `1`.

    mode : "supervised"
        Learning mode.

        By default, `"supervised"`.

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

    batch_size : int
        Batch size.

        By default, `50`.

    num_workers : int
        Number of additional workers for data loaders.

        By default, `1`.

    seed : int
        Random seed.

        By default, `0`.
seed
    """
    mode: Literal["supervised"]
    def __init__(
        self,
        data_dir: str | Path,
        fold: Literal[1, 2, 3, 4, 5],
        *,
        mode: Literal["supervised"] = "supervised",
        random_crop: bool = True,
        random_crop_min_percent: int = 80,
        random_crop_max_percent: int = 100,
        random_speed: bool = True,
        random_speed_min_percent: int = 85,
        random_speed_max_percent: int = 115,
        batch_size: int = 50,
        num_workers: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.fold = fold
        self.mode = mode
        self.random_crop=random_crop
        self.random_crop_min_percent=random_crop_min_percent
        self.random_crop_max_percent=random_crop_max_percent
        self.random_speed=random_speed
        self.random_speed_min_percent=random_speed_min_percent
        self.random_speed_max_percent=random_speed_max_percent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._collate_fn = _collate_one_wave

    def setup(self, stage=None):
        kwargs = dict(
            fold=self.fold, mode=self.mode,
            random_crop=self.random_crop,
            random_crop_min_percent=self.random_crop_min_percent,
            random_crop_max_percent=self.random_crop_max_percent,
            random_speed=self.random_speed,
            random_speed_min_percent=self.random_speed_min_percent,
            random_speed_max_percent=self.random_speed_max_percent
        )
        if stage == "fit" or stage is None:
            if not hasattr(self, "train_set"):
                self.train_set = GRBASDataset(
                    self.data_dir, phase="train", **kwargs)
            if not hasattr(self, "valid_set"):
                self.valid_set = GRBASDataset(
                    self.data_dir, phase="valid", **kwargs)
        if stage == "test" or stage is None:
            if not hasattr(self, "test_set"):
                self.test_set = GRBASDataset(
                    self.data_dir, phase="test", **kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set, shuffle=True, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
            batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=self._collate_fn, drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_set, shuffle=False, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(0),
            batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set, shuffle=False, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(0),
            batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
