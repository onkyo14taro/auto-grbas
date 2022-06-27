from __future__ import annotations
import math
from typing import Literal, Union

from torch import Tensor
import torch.nn as nn

from .crnn import CRNN
from .efficientnet import create_variable_length_efficientnet
from .rnn import LastLSTM


class Encoder(nn.Module):
    """CRNN encoder to compress variable length time-frequency representations.

    Parameters
    ----------
    cnn_in_channels : int
        Number of CNN input channels.
    cnn_model_name : str
        CNN model name.

        Available models are as follows:

        - `"tf_efficientnetv2_b0"`
        - `"tf_efficientnetv2_b1"`
        - `"tf_efficientnetv2_b2"`
        - `"tf_efficientnetv2_b3"`
        - `"tf_efficientnetv2_s"`
        - `"tf_efficientnetv2_m"`
        - `"tf_efficientnetv2_l"`

        By default, `"tf_efficientnetv2_b0"`.
    cnn_drop_path_rate : float
        Drop rate of the stochastic depth of the CNN.

        By default, `0.0`.
    rnn_hidden_size : int
        The number of features in the hidden state of the RNN.

        By default, `256`.
    rnn_num_layers : int
        The number of recurrent layers in the RNN.

        By default, `2`.
    rnn_drop_rate : float
        Dropout rate between recurrent layers in the RNN.

        By default, `0.0`.
    rnn_bidirectional : bool
        Whether to make the RNN bidirectional or not.

        By default, `True`.
    num_filters : int
        Number of filters of the input time-frequency representation.

        By default, `128`.
    """
    def __init__(
        self,
        *,
        cnn_in_channels: int,
        cnn_model_name: Literal[
            "tf_efficientnetv2_b0",
            "tf_efficientnetv2_b1",
            "tf_efficientnetv2_b2",
            "tf_efficientnetv2_b3",
            "tf_efficientnetv2_s",
            "tf_efficientnetv2_m",
            "tf_efficientnetv2_l",
        ] = "tf_efficientnetv2_b0",
        cnn_drop_path_rate: float = 0.0,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        rnn_drop_rate: float = 0.0,
        rnn_bidirectional: bool = True,
        num_filters: int = 128,
    ) -> None:
        super().__init__()
        cnn = create_variable_length_efficientnet(
            in_chans=cnn_in_channels,
            model_name=cnn_model_name,
            features_only=True,
            drop_path_rate=cnn_drop_path_rate,
        )
        input_size \
            = cnn.blocks[-1][-1].bn3.num_features * math.ceil(num_filters/32)
        rnn = LastLSTM(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_drop_rate,
            bidirectional=rnn_bidirectional,
        )
        self.crnn = CRNN(cnn=cnn, rnn=rnn, inter_drop_rate=rnn_drop_rate)

    def forward(
        self,
        input: Union[Tensor, list[Tensor]]
    ) -> Tensor:
        r"""Defines the computation performed at every call.

        Parameters
        ----------
        input : Tensor [shape=(batch_size, channels, num_bins, num_frames)] or
        list[Tensor [shape=(1, channels, num_bins, num_frames)]]
            Tensor of time-frequency representations of a list of them. If
            `input` is a `list[Tensor]`, each tensor can have variable-length
            `num_frames`, while must have `num_bins` of the same length.

        Returns
        -------
        h_n : Tensor [shape=(batch_size, D * self.rnn_hidden_size)]
            Final hidden state for each element in the sequence of the final
            layer. If `self.rnn_bidirectional == True`, `D` is 2, otherwise
            `D` is 1.
        """
        return self.crnn(input)
