from __future__ import annotations
from argparse import ArgumentParser, Namespace
from collections import defaultdict
import os
from pathlib import Path
import re
import time
from typing import Any, Literal, Optional, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd

import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam, SGD
import pytorch_lightning as pl

from .nn import GRBASNet
from .metrics import cohen_kappa_score


################################################################################
################################################################################
### Base
################################################################################
################################################################################
class BaseModel(pl.LightningModule):
    r"""Base model.

    Parameters
    ----------
    hparams : Namespace
        Hyperparameters.
    net : GRBASNet or None
        If `None`, construct a new model, otherwise use `net` as it is.

        By default, `None`.
    """
    net: GRBASNet

    def __init__(
        self,
        hparams: Namespace,
        net: Optional[GRBASNet] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self._set_net(net)
        self._test_prefix = ""

    @property
    def _results_dir(self) -> Optional[str]:
        return None if self.trainer is None else self.logger.log_dir

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser
    ) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--version', type=int)
        parser.add_argument('--epoch_steps', type=int, default=10_000)

        ### neural network: frontend
        ########################################
        parser.add_argument('--fs', type=float, default=16000)
        parser.add_argument('--num_filters', type=int, default=128)
        parser.add_argument('--frontend_window_length', type=int, default=800)
        parser.add_argument('--frontend_shift_length', type=int, default=160)
        parser.add_argument('--frontend_window_name', type=str, default="hann")
        parser.add_argument('--frontend_features', type=str, nargs='+',
                            default=['power'])
        parser.add_argument('--frontend_freq_mask', action='store_true')
        parser.add_argument('--frontend_freq_mask_param', type=int, default=26)

        ### neural network: encoder (CRNN)
        ########################################
        parser.add_argument('--cnn_model_name', type=str,
                            default="tf_efficientnetv2_b0")
        parser.add_argument('--cnn_drop_path_rate', type=float, default=0.0)
        parser.add_argument('--rnn_hidden_size', type=int, default=256)
        parser.add_argument('--rnn_num_layers', type=int, default=2)
        parser.add_argument('--rnn_drop_rate', type=float, default=0.0)
        parser.add_argument('--rnn_bidirectional', action='store_true')

        ### neural network: head
        ########################################
        parser.add_argument('--head_hidden_features', type=int,
                            default=256)
        parser.add_argument('--head_num_layers', type=int, default=2)
        parser.add_argument('--head_drop_rate', type=float, default=0.0)

        ### neural network: DLDL
        ########################################
        parser.add_argument('--grbas_min', type=float, default=-3.0)
        parser.add_argument('--grbas_max', type=float, default=6.0)
        parser.add_argument('--grbas_num_bins', type=int, default=91)
        parser.add_argument('--grbas_item', type=str, default="G")

        ### Data
        ########################################
        parser.add_argument('--data_dir', type=str,
                            default=str(
                                Path(__file__).parent.parent.parent/"data"))
        parser.add_argument('--fold', type=int, default=1)
        parser.add_argument('--random_crop', action='store_true')
        parser.add_argument('--random_crop_min_percent', type=int, default=80)
        parser.add_argument('--random_crop_max_percent', type=int, default=100)
        parser.add_argument('--random_speed', action='store_true')
        parser.add_argument('--random_speed_min_percent', type=int, default=85)
        parser.add_argument('--random_speed_max_percent', type=int, default=115)

        ### learning
        ########################################
        parser.add_argument('--batch_size', type=int, default=50)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--warmup_steps', type=int, default=0)
        parser.add_argument('--optimizer', type=str, default="adam")

        ### learning: regularization
        ########################################
        parser.add_argument('--weight_decay', type=float, default=4e-5)

        ### experiment
        ########################################
        parser.add_argument('--exp_name', type=str, default="test")
        parser.add_argument('--earlystopping_patience', type=int)

        return parser

    def _set_net(
        self,
        net: Optional[GRBASNet] = None,
    ) -> None:
        self.net = GRBASNet(
            fs=self.hparams.fs,
            num_filters=self.hparams.num_filters,
            frontend_window_length=self.hparams.frontend_window_length,
            frontend_shift_length=self.hparams.frontend_shift_length,
            frontend_window_name=self.hparams.frontend_window_name,
            frontend_features=self.hparams.frontend_features,
            frontend_freq_mask=self.hparams.frontend_freq_mask,
            frontend_freq_mask_param=self.hparams.frontend_freq_mask_param,
            cnn_model_name=self.hparams.cnn_model_name,
            cnn_drop_path_rate=self.hparams.cnn_drop_path_rate,
            rnn_hidden_size=self.hparams.rnn_hidden_size,
            rnn_num_layers=self.hparams.rnn_num_layers,
            rnn_drop_rate=self.hparams.rnn_drop_rate,
            rnn_bidirectional=self.hparams.rnn_bidirectional,
            head_hidden_features=self.hparams.head_hidden_features,
            head_num_layers=self.hparams.head_num_layers,
            head_drop_rate=self.hparams.head_drop_rate,
            grbas_min=self.hparams.grbas_min,
            grbas_max=self.hparams.grbas_max,
            grbas_num_bins=self.hparams.grbas_num_bins,
            grbas_item=self.hparams.grbas_item,
        ) if net is None else net

    ############################################################
    ############################################################
    ### Optimizer-related
    ############################################################
    ############################################################
    def configure_optimizers(self):
        params_weight_decay = []
        params_no_weight_decay = []
        self._wakeup_params(self.net)
        self._extend_params(
            self.net,
            params_weight_decay=params_weight_decay,
            params_no_weight_decay=params_no_weight_decay,
            without_weight_decay=False,
        )
        if self.hparams.optimizer == "adam":
            optimizer_cls = Adam
        elif self.hparams.optimizer == "sgd":
            optimizer_cls = SGD
        optimizer = optimizer_cls(
            [{'params': params_weight_decay,
              'weight_decay': self.hparams.weight_decay},
             {'params': params_no_weight_decay}],
            lr=self.hparams.lr
        )
        return optimizer

    @staticmethod
    def _freeze_params(
        module_parent: nn.Module
    ) -> None:
        for param in module_parent.parameters(recurse=True):
            param.requires_grad = False

    @staticmethod
    def _wakeup_params(
        module_parent: nn.Module
    ) -> None:
        for param in module_parent.parameters(recurse=True):
            param.requires_grad = True

    @staticmethod
    def _extend_params(
        module_parent: nn.Module,
        params_weight_decay: list[nn.parameter.Parameter],
        params_no_weight_decay: list[nn.parameter.Parameter],
        without_weight_decay: bool = False
    ) -> None:
        if without_weight_decay:
            for module in module_parent.modules():
                params_no_weight_decay.extend([
                    param for param in module.parameters(recurse=False)])
        else:
            weight_decay_module_cls = (
                nn.modules.Linear,
                nn.modules.conv._ConvNd,
                nn.modules.rnn.RNNBase
            )
            for module in module_parent.modules():
                if isinstance(module, weight_decay_module_cls):
                    for name, param in module.named_parameters():
                        if name.startswith("weight"):
                            params_weight_decay.append(param)
                        else:  # bias
                            params_no_weight_decay.append(param)
                else:
                    params_no_weight_decay.extend([
                        param for param in module.parameters(recurse=False)])

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        for m in self.net.modules():
            if hasattr(m, "_clamp_constraint"):
                m._clamp_constraint()

    ############################################################
    ############################################################
    ### epoch_end helper
    ############################################################
    ############################################################
    def _arrange_outputs(
        self,
        outputs: list[dict[str, Union[NDArray[np.float32],
                                      Tensor,
                                      dict[str, list]]]]
    ) -> tuple[dict[str, NDArray[np.float32]], pd.DataFrame]:
        properties = defaultdict(list)
        for o in outputs:
            if "loss" in o:
                del o["loss"]
            p: dict[str, list] = o.pop("properties")
            for k in list(p.keys()):
                properties[k] += p.pop(k)
        properties = pd.DataFrame(properties).set_index("fileID")
        output = {k: np.concatenate([o[k] for o in outputs], axis=0)
                for k in outputs[0].keys()}
        return output, properties

    ############################################################
    ############################################################
    ### Save helper
    ############################################################
    ############################################################
    def _plot_metrics(
        self,
        metrics: Optional[dict[str, float]] = None
    ) -> None:
        """Plot the learing curve and the curves of other metrics."""
        path_csv = os.path.join(self._results_dir, "metrics.csv")
        if not os.path.exists(path_csv):
            return
        df = pd.read_csv(path_csv)

        # Add the current validation metrics.
        if metrics is not None:
            metrics["step"] = self.global_step - 1
            for key in metrics:
                if key not in df:
                    df[key] = pd.NA
            df = pd.concat([df, pd.Series(metrics).to_frame().T],
                           ignore_index=True)
        df["step"] += 1
        df = df.drop("epoch", axis="columns").set_index("step")

        # Loss; Learning curve
        metric_names: set[str] = {col.split(".")[0] for col in df.columns}
        cm = plt.get_cmap('tab10')
        for metric_name in metric_names:
            fig, ax = plt.subplots(figsize=(8, 6))
            cols = {col for col in df.columns
                    if col.split(".")[0] == metric_name}

            # Plot lines
            for col in sorted(cols):
                mask = ~df[col].isna()
                if not mask.any():
                    continue
                splits = col.split(".")
                color_index = "GRBAS".find(splits[1])
                color = cm(color_index) if color_index >= 0 else [0.0, 0.0, 0.0]
                if splits[-1] == "train":
                    color = [c + 0.75*(1-c) for c in color]
                ax.plot(df.loc[mask].index.to_numpy(),
                        df.loc[mask, col].to_numpy(),
                        label=col, color=color)

            # Save and clear
            yscale = "log" if metric_name == "loss" else "linear"
            ax.set(title=f'{metric_name}', xlabel="Step", ylabel="Value",
                   yscale=yscale)
            ax.grid()
            ax.legend(loc=(1.01, 0), frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self._results_dir, f"{metric_name}.png"))
            plt.clf()
            plt.close()

    def _save_array_dict(
        self,
        array_dict: dict[str, NDArray],
        group_name_base: str,
        dataset_name: str,
        exist: Literal["error", "overwrite", "skip"] = "error",
    ) -> None:
        for i in range(10):
            try:
                with h5py.File(
                    os.path.join(self._results_dir, "output.hdf5"), mode="a"
                ) as f:
                    group_base = f.require_group(group_name_base)
                    for k, array in array_dict.items():

                        # Crate groups
                        group_sub = group_base
                        for group_name_sub in k.split('.'):
                            group_sub = group_sub.require_group(group_name_sub)

                        # If already exists...
                        if dataset_name in group_sub:
                            if exist == "error":
                                raise ValueError(
                                    f"{group_sub}/{dataset_name} already exits.")
                            elif exist == "overwrite":
                                del group_sub[dataset_name]
                            elif exist == "skip":
                                continue
                            else:
                                raise ValueError(
                                    f'exist must be either "error", "overwrite", or ' \
                                    f'"skip"; found {exist}.')

                        group_sub.create_dataset(dataset_name, data=array)
                break
            except BlockingIOError:
                if i == 9:
                    raise BlockingIOError()
                time.sleep(5)

    def _save_output(
        self,
        output: dict[str, NDArray],
        group_name_base: str,
        dataset_name: str,
    ) -> None:
        self._save_array_dict(
            output, group_name_base, dataset_name=dataset_name)


################################################################################
################################################################################
### GRBAS
################################################################################
################################################################################
class GRBASModule(BaseModel):
    r"""GRBAS model.

    Parameters
    ----------
    hparams : Namespace
        Hyperparameters.
    net : GRBASNet or None
        If `None`, construct a new model, otherwise use `net` as it is.

        By default, `None`.
    """
    ############################################################
    ############################################################
    ### xxx_step
    ############################################################
    ############################################################
    def training_step(
        self,
        batch: tuple[Tensor, dict[str, list]],
        batch_idx: int,
    ) -> dict[str, Union[NDArray[np.float32],
                         Tensor,
                         dict[str, list]]]:
        return self._common_step(batch, training=True)

    def validation_step(
        self,
        batch: tuple[Tensor, list[int], dict[str, list]],
        batch_idx: int,
    ) -> dict[str, Union[NDArray[np.float32],
                         dict[str, list]]]:
        return self._common_step(batch, training=False)

    def test_step(
        self,
        batch: tuple[Tensor, list[int], dict[str, list]],
        batch_idx: int,
    ) -> dict[str, Union[NDArray[np.float32],
                         dict[str, list]]]:
        return self._common_step(batch, training=False)

    def _common_step(
        self,
        batch: tuple[list[Tensor], dict[str, list]],
        *,
        training: bool,
    ) -> dict[str, Union[NDArray[np.float32],
                         Tensor,
                         dict[str, list]]]:
        wave, properties = batch
        y: dict[Literal["G","R","B","A","S"], Tensor] = self.net(wave)
        output = {}

        ### Loss
        loss, loss_dict = self._calculate_loss(
            {grbas: t.detach() for grbas, t in y.items()},
            properties, reduction="none")
        output["loss_sample"] = loss.cpu().numpy()
        output.update({
            f"{grbas}.loss_sample": l.cpu().numpy()
            for grbas, l in loss_dict.items()
        })

        ### Mean and SD
        mean_sd = {
            grbas: self.net.dldl.calculate_mean_sd(t.detach())
            for grbas, t in y.items()
        }
        output.update({
            f"{grbas}.mean": mean_sd[0].cpu().numpy()
            for grbas, mean_sd in mean_sd.items()
        })
        output.update({
            f"{grbas}.sd": mean_sd[1].cpu().numpy()
            for grbas, mean_sd in mean_sd.items()
        })

        if training:
            # Required by training_step for training
            loss, _ = self._calculate_loss(y, properties, reduction="mean")
            output["loss"] = loss
        else:
            output.update({
                f"{grbas}.prob": \
                    self.net.dldl.probability(t.detach()).cpu().numpy()
                for grbas, t in y.items()
            })

        output["properties"] = properties
        return output

    def _calculate_loss(
        self,
        y: dict[Literal["G","R","B","A","S"], Tensor],
        properties: dict[str, Any],
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> tuple[Tensor, dict[Literal["G","R","B","A","S"], Tensor]]:
        loss_dict = {
            grbas: self.net.dldl.calculate_loss(
                tensor,
                torch.tensor(properties[f"{grbas}.mean"], device=tensor.device),
                torch.tensor(properties[f"{grbas}.sd"], device=tensor.device),
                reduction=reduction)
            for grbas, tensor in y.items()
        }
        loss = [loss for loss in loss_dict.values()][0]
        loss = torch.tensor(0.0, dtype=loss.dtype, device=loss.device)
        for loss_item in loss_dict.values():
            loss = loss + loss_item
        return loss, loss_dict

    ############################################################
    ############################################################
    ### xxx_epoch_end
    ############################################################
    ############################################################
    def training_epoch_end(
        self,
        outputs: list[dict[str, Union[NDArray[np.float32],
                                      Tensor,
                                      dict[str, list]]]]
    ) -> None:
        output, properties = self._arrange_outputs(outputs)
        print
        metrics = self._calculate_metrics(output, properties)
        metrics = {f"{k}.train": v for k, v in metrics.items()}
        self.log_dict(metrics)
        self.logger.save()
        self._plot_metrics(metrics=metrics)

    def validation_epoch_end(
        self,
        outputs: list[dict[str, Union[NDArray[np.float32],
                                      dict[str, list]]]]
    ) -> None:
        if self.trainer is None or self.trainer.sanity_checking:
            return
        self._validation_test_epoch_end(outputs, validation=True)

    def test_epoch_end(
        self,
        outputs: list[dict[str, Union[NDArray[np.float32],
                                      dict[str, list]]]]
    ) -> None:
        metrics = self._validation_test_epoch_end(outputs, validation=False)
        pd.Series(metrics).to_json(
            os.path.join(self._results_dir, f"{self._test_prefix}test.json"))

    def _validation_test_epoch_end(
        self,
        outputs: list[dict[str, Union[NDArray[np.float32],
                                      dict[str, list]]]],
        validation: bool,
    ) -> dict[str, float]:
        if validation:
            dataset_name = f"epoch_{self.current_epoch}"
            phase = "valid"
            group_name_base = phase
        else:
            dataset_name = "best"
            phase = "test"
            group_name_base = f"{self._test_prefix}test"
        output, properties = self._arrange_outputs(outputs)

        metrics = self._calculate_metrics(output, properties)
        metrics = {f"{k}.{phase}": v for k, v in metrics.items()}
        self._save_output(
            output, group_name_base=group_name_base, dataset_name=dataset_name)
        self._save_properties(
            properties, group_name_base=group_name_base)
        self.log_dict(metrics)
        self.logger.save()
        return metrics

    def _calculate_metrics(
        self,
        output: dict[str, NDArray],
        properties: pd.DataFrame,
    ) -> dict[str, float]:
        metrics = {}
        metrics["loss"] \
            = output["loss_sample"].mean()
        for grbas in self.hparams.grbas_item:
            metrics[f"loss.{grbas}"] \
                = output[f"{grbas}.loss_sample"].mean().item()
            metrics[f"kappa.{grbas}"] \
                = cohen_kappa_score(
                    output[f"{grbas}.mean"],
                    properties[f"{grbas}.mean"].values,
                    weights="quadratic")
        return metrics

    def _save_properties(
        self,
        properties: pd.DataFrame,
        group_name_base: str,
    ) -> None:
        self._save_array_dict(
            {"fileID": properties.index.to_numpy(np.int32)},
            group_name_base, dataset_name="fileID", exist="skip")
        self._save_array_dict(
            {"category": properties["category"].to_numpy(
                h5py.special_dtype(vlen=str))},
            group_name_base, dataset_name="category", exist="skip")
        target = {
            col: properties[col].to_numpy(np.float32)
            for col in properties
            if re.match(r"[GRBAS]\..+", col)
        }
        target.update({
            f"{grbas}.prob": self.net.dldl.generate_target(
                torch.from_numpy(target[f"{grbas}.mean"]),
                torch.from_numpy(target[f"{grbas}.sd"]),
            ).numpy()
            for grbas in self.hparams.grbas_item
        })
        self._save_array_dict(
            target, group_name_base, dataset_name="target", exist="skip")
