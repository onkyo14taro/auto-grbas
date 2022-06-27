from __future__ import annotations
from typing import Union

from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class LastLSTM(nn.Module):
    """LSTM that outputs the final hidden states.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input x.
    hidden_size : int
        The number of features in the hidden state h.
    num_layers : int
        Number of recurrent layers. E.g., setting `num_layers=2` would mean
        stacking two LSTMs together to form a stacked LSTM, with the second LSTM
        taking in outputs of the first LSTM and computing the final results.

        By default, `1`.
    bias : bool
        If `False`, then the layer does not use bias weights b_ih and b_hh.

        By default, `True`.
    batch_first : bool
        If True, then the input tensors are provided as (batch, seq, feature)
        instead of (seq, batch, feature). Note that this does not apply to
        hidden or cell states.

        By default `False`.
    dropout : float
        If non-zero, introduces a Dropout layer on the outputs of each LSTM
        layer except the last layer, with dropout probability equal to dropout.

        By default `0.0`.
    bidirectional : bool
        If `True`, becomes a bidirectional LSTM.

        By default, `False`.
    proj_size : int
        If `> 0`, will use LSTM with projections of corresponding size.

        By default, `0`.
    device
        By default, `None`.
    dtype
        By default, `None`.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device = None,
        dtype = None,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
            device=device,
            dtype=dtype,
        )
        self.out_features = hidden_size*2 if bidirectional else hidden_size

    def forward(
        self,
        x: Union[Tensor, PackedSequence]
    ) -> Tensor:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        x : Tensor or PackedSequence
            Tensor of shape (sequence_length, batch_size, input_size) when
            `batch_first=False` or (batch_size, sequence_length, input_size)
            when `batch_first=True` containing the features of the input
            sequence. The input can also be a packed variable length sequence.

        Returns
        -------
        h_n : Tensor [shape=(batch_size, D * out_size)]
            Final hidden state for each element in the sequence of the final
            layer. If `self.bidirectional == True`, `D` is 2, otherwise `D` is
            1. If `self.proj_size > 0`, `out_size` is `self.proj_size`,
            otherwise `out_size` is `self.hidden_size`.
        """
        # Hidden states of all layers; shape=(D*num_layers, N, out_size)
        x = self.rnn(x)[1][0]

        # Hidden states of the last layer; shape=(D, N, out_size)
        # D = 2 if bidirectional else 1
        if self.rnn.bidirectional:
            # shape = (N, 2*hidden_size)
            x = x[-2:].transpose(0, 1).flatten(start_dim=1, end_dim=2)
        else:
            # shape = (N, 1*out_size)
            x = x[-1]
        return x
