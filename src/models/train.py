from __future__ import annotations
import argparse
from argparse import Namespace
import os
from pathlib import Path
from typing import Collection, Literal, Optional, Union

import pandas as pd
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from .data import GRBASDataModule
from .module import GRBASModule


################################################################################
################################################################################
### Run experiment
################################################################################
################################################################################
def run(hparams:argparse.Namespace) -> None:
    r"""Run the experiment based on `hparams`.

    Parameters
    ----------
    hparams : Namespace
        Hyperparameters.
    """
    version = _generate_version(hparams)
    logger = CSVLogger(hparams.results_dir,
                       name=hparams.exp_name, version=version)
    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            ckpt_dir, monitor="loss.valid", mode="min",
            filename="{step:07d}", save_top_k=1,
        )
    ]
    callbacks.extend([
        ModelCheckpoint(
            ckpt_dir, monitor=f"loss.{grbas}.valid", mode="min",
            filename=f"{grbas}."+"{step:07d}", save_top_k=1,
        )
        for grbas in hparams.grbas_item
    ])
    if hparams.earlystopping_patience is not None:
        monitor = ["loss.valid"]
        monitor += [f"loss.{grbas}.valid" for grbas in hparams.grbas_item]
        callbacks.append(_MultiEarlyStopping(
            monitor=monitor, mode="min",
            patience=hparams.earlystopping_patience,
            min_delta=0.01, verbose=True,
            strict=True, check_finite=True,
            stopping_threshold=None, divergence_threshold=None,
        ))

    # Construct the necessary model, datamodule, and trainer
    seed_everything(hparams.seed)
    model = _construct_model(hparams)
    datamodule = _construct_datamodule(hparams)
    trainer: Trainer = Trainer.from_argparse_args(
        hparams, logger=logger, callbacks=callbacks, weights_summary="full")

    # Save the model summary to a text file
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(os.path.join(logger.log_dir, "model.txt"), "w") as f:
        print(f"{ModelSummary(model, max_depth=-1)}\n\n{model}", file=f)

    # Fitting (training and validation)
    seed_everything(hparams.seed)
    trainer.fit(model, datamodule=datamodule)

    # Test
    ckpt_path = _find_best_checkpointpath(
        logger.log_dir)
    model = restore_from_version_dir(
        logger.log_dir, checkpoint_filename=ckpt_path.name,
        map_location=model.device)[0]
    trainer.test(model, datamodule=datamodule)
    for grbas in hparams.grbas_item:
        prefix = f"{grbas}."
        ckpt_path = _find_best_checkpointpath(
            logger.log_dir, prefix=prefix)
        model = restore_from_version_dir(
            logger.log_dir, checkpoint_filename=ckpt_path.name,
            map_location=model.device)[0]
        model._test_prefix = prefix
        trainer.test(model, datamodule=datamodule)

    # Display the instantaneous maximum GPU memory usage
    if torch.cuda.is_available():
        maximum_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"Maximum GPU memory {maximum_gpu_memory:.2f} GB")


def restore_from_version_dir(
    version_dir: Union[str, Path],
    checkpoint_filename: Union[str, Path] = None,
    map_location = torch.device("cpu"),
    **kwargs
) -> tuple[GRBASModule, GRBASDataModule, Trainer]:
    r"""Restore from version directory.

    Parameters
    ----------
    version_dir : str or Path
        Version directory.
    checkpoint_filename : str, Path, or None
        Checkpoint file name. If `None`, the checkpoint with minimum validation
        loss will be used.

        By default, `None`.
    map_location : optional
        A function, torch.device, string or a dict specifying
        how to remap storage locations.

        By default `torch.device("cpu")`.
    kwargs
        Any extra keyword args needed to init the model.
        Can also be used to override saved hyperparameter values.

    Returns
    -------
    model : GRBASModule
        Module restored with best checkpoint file.
    datamodule : GRBASDataModule
        DataModule instance corresponding to `model`.
    trainer : Trainer
        Trainer instance corresponding to `model`.
    """
    version_dir = Path(version_dir)
    hparams_file = version_dir/"hparams.yaml"

    if checkpoint_filename is None:
        checkpoint_path = _find_best_checkpointpath(version_dir)
    else:
        checkpoint_path = version_dir/"checkpoints"/checkpoint_filename
        if not Path(checkpoint_path).exists():
            raise FileExistsError(Path(checkpoint_path))

    with open(hparams_file, "r") as f:
        hparams = argparse.Namespace(**yaml.safe_load(f))

    model = GRBASModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        map_location=map_location,
        hparams_file=str(hparams_file),
        **kwargs
    )
    datamodule = _construct_datamodule(hparams)
    if str(map_location) == "cpu":
        trainer = Trainer.from_argparse_args(hparams, accelerator="cpu")
    else:
        trainer = Trainer.from_argparse_args(hparams)

    return model, datamodule, trainer


################################################################################
################################################################################
### Helper classes ans functions
################################################################################
################################################################################
class _MultiEarlyStopping(EarlyStopping):
    r"""This class can handle multiple monitor metrics. When all metrics have
    not improved by at least `patience`, the training is stopped.

    This implementation is based on [1].


    [1] https://github.com/PyTorchLightning/pytorch-lightning/blob/eb21135b2aad20aaad45aae44858c090f1e780e5/pytorch_lightning/callbacks/early_stopping.py#L35
        Copyright (c) 2018-2021 William Falcon
        Apache License 2.0
        https://opensource.org/licenses/Apache-2.0
    """
    mode_dict = {"min": torch.lt, "max": torch.gt}

    order_dict = {"min": "<", "max": ">"}

    def __init__(
        self,
        monitor: Union[str, Collection[str]],
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )
        if isinstance(monitor, str):
            monitor = [monitor]
        else:
            if not (isinstance(monitor, Collection) \
               and len(monitor) and all(isinstance(m, str) for m in monitor)):
                raise ValueError(
                    'monitor must be a str or a collection of strs.')
            monitor = sorted(set(monitor))
        self.monitor = monitor
        self.best_score = {m: self.best_score for m in monitor}
        self.wait_count = {m: self.wait_count for m in monitor}

    def _validate_condition_metric(self, logs: dict[str, float]) -> bool:
        monitor_val = [logs.get(m) for m in self.monitor]

        error_msg = (
            f"Early stopping conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `EarlyStopping` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if any(v is None for v in monitor_val):
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, category=RuntimeWarning)

            return False

        return True

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        should_stop, reason = list(zip(*[
            self._evaluate_stopping_criteria(logs[m].squeeze(), m)
            for m in self.monitor
        ]))
        min_wait_count = min(wc for wc in self.wait_count.values())
        should_stop = any(should_stop) or min_wait_count >= self.patience
        reason = "\n".join(r for r in reason if r is not None)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason)

    def _evaluate_stopping_criteria(self, current: torch.Tensor, monitor: str) -> tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {monitor} = {current} is not finite."
                f" Previous best value was {self.best_score[monitor]:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score[monitor].to(current.device)):
            reason = self._improvement_message(current, monitor)
            self.best_score[monitor] = current
            self.wait_count[monitor] = 0
        else:
            self.wait_count[monitor] += 1
            if self.wait_count[monitor] >= self.patience:
                reason = (
                    f"Monitored metric {monitor} did not improve in the last {self.wait_count[monitor]} records."
                    f" Best score: {self.best_score[monitor]:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason

    def _improvement_message(self, current: torch.Tensor, monitor: str) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score[monitor]):
            msg = (
                f"Metric {monitor} improved by {abs(self.best_score[monitor] - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {monitor} improved. New best score: {current:.3f}"
        return msg


def _generate_version(hparams: Namespace) -> str:
    return \
        f"item={hparams.grbas_item}__fold={hparams.fold}"


def _construct_model(
    hparams: Namespace,
) -> GRBASModule:
    return GRBASModule(hparams)


def _construct_datamodule(
    hparams: Namespace,
) -> GRBASDataModule:
    return GRBASDataModule(
        data_dir=hparams.data_dir,
        fold=hparams.fold,
        random_crop=hparams.random_crop,
        random_crop_min_percent=hparams.random_crop_min_percent,
        random_crop_max_percent=hparams.random_crop_max_percent,
        random_speed=hparams.random_speed,
        random_speed_min_percent=hparams.random_speed_min_percent,
        random_speed_max_percent=hparams.random_speed_max_percent,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        seed=hparams.seed,
    )


def _find_best_checkpointpath(
    version_dir: Union[str, Path],
    prefix: str = "",
) -> Path:
    version_dir = Path(version_dir)
    checkpoints_dir = version_dir/"checkpoints"
    metrics_file = version_dir/"metrics.csv"
    df = pd.read_csv(metrics_file)
    metric_name = f"loss.{prefix}valid"
    best_steps = df.loc[df[metric_name] == df[metric_name].min(), "step"]
    checkpoint_paths = [
        p for p in sorted(checkpoints_dir.glob(f"{prefix}step*.ckpt"))
        if int(p.stem.split("=")[1])-1 in best_steps.to_list()
    ]
    if checkpoint_paths:
        return checkpoint_paths[-1]
    raise FileExistsError(
        f"The checkpoint file corresponding to step={best_steps} could not be "
        f"found.")
