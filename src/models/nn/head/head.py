from __future__ import annotations

from torch import Tensor
import torch.nn as nn


class MultiOutMLPHead(nn.Module):
    r"""Head for multiple outputs.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features_dict : dict[str, int]
        Dictionary of numbers of output features.

        For example, `out_features_dict={"item_1": 4, "item_2": 10}`.

        By default `512`.
    hidden_features : int
        Number of hidden features.
    num_layers : int
        Number of fully connected layers, by default `2`.
    drop_rate : float
        Dropout rate, by default `0.0`.
    """
    def __init__(
        self,
        in_features: int,
        out_features_dict: dict[str, int],
        *,
        hidden_features: int = 512,
        num_layers: int = 2,
        drop_rate: float = 0.0,
        device = None,
        dtype = None,
    ) -> None:
        super().__init__()
        layers = []
        last_out_features = in_features
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(last_out_features, hidden_features,
                          bias=False, device=device, dtype=dtype),
                nn.BatchNorm1d(hidden_features),
                nn.SiLU()
            ])
            last_out_features = hidden_features
        layers.extend([nn.Dropout(drop_rate)])
        last_fc_dict = nn.ModuleDict({
            name: nn.Linear(last_out_features, out_features,
                            bias=True, device=device, dtype=dtype)
            for name, out_features in out_features_dict.items()
        })
        self.layers = nn.Sequential(*layers)
        self.last_fc_dict = last_fc_dict

    def forward(
        self,
        x: Tensor
    ) -> dict[str, Tensor]:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        x : Tensor [shape=(batch_size, in_features)]
            Input.

        Returns
        -------
        y : dict[str, Tensor [shape=(batch_size, out_features)]]
            Dictionary of output tensors. The keys and last dimensions are
            `out_features_dict.keys()` and `out_features_dict.values()`,
            respectively.
        """
        x = self.layers(x)
        return {k: fc(x) for k, fc in self.last_fc_dict.items()}
