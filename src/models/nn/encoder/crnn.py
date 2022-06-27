from __future__ import annotations
from typing import Union

from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence


class CRNN(nn.Module):
    r"""Convolutoinal-Recurrent Neural Network architecture.

    Parameters
    ----------
    cnn : Module
        Convolutional Neural Network architecture.
    rnn : Module
        Recurrent Neural Network architecture.
    inter_drop_rate : float
        Dropout rate between the CNN and RNN.

        By default, `0.0`.
    """
    def __init__(
        self,
        cnn: nn.Module,
        rnn: nn.Module,
        inter_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.cnn = cnn
        self.dropout = nn.Dropout(p=inter_drop_rate)
        self.rnn = rnn

    def forward(
        self,
        x: Union[Tensor, list[Tensor]]
    ) -> Union[Tensor, PackedSequence]:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        x : Tensor [shape=(batch_size, channels, num_bins, num_frames)] or
        list[Tensor [shape=(1, channels, num_bins, num_frames)]]
            Tensor of time-frequency representations of a list of them. If
            `x` is a `list[Tensor]`, each tensor can have variable-length
            `num_frames`, while must have `num_bins` of the same length.

        Returns
        -------
        h_n : Tensor [shape=(batch_size, D * self.rnn_hidden_size)]
            Final hidden state for each element in the sequence of the final
            layer. If `self.rnn_bidirectional == True`, `D` is 2, otherwise
            `D` is 1.
        """
        x = self.cnn(x)
        if isinstance(x, list):
            x = pack_sequence([
                self.dropout(e.flatten(start_dim=0, end_dim=2).T)  # shape=(time, channels*height)
                for e in x
            ], enforce_sorted=False)
        else:
            x = x.flatten(start_dim=1, end_dim=2)  # shape=(batch, channels*height, time)
            x = self.dropout(x.transpose(1, 2).transpose(0, 1))  # shape=(time, batch, channels*height)
        return self.rnn(x)
