from __future__ import annotations
from typing import Collection, Literal, Optional, Union

from torch import Tensor
import torch.nn as nn

from .encoder import Encoder
from .frontend import Frontend
from .head import MultiOutMLPHead
from ..dldl import DLDLRegression


class GRBASNet(nn.Module):
    r"""GRBAS neural network.

    Parameters
    ----------
    fs : float
        Sampling frequency, by default `16000` Hz.
    num_filters : int
        Number of filters, by default `128`.
    frontend_window_length : int
        Window length in samples, by default `800` (if `fs == 16000`, 50 ms).
    frontend_shift_length : int
        Shift length in samples, by default `160` (if `fs == 16000`, 10 ms).
    frontend_window_name : str
        Window name, by default "hann".
    frontend_features : "power", "ist_frq", "grp_dly", or Collection of them
        Feature or collection of features to be calculated.

        Available features are as follows:

        - `"power"` : log power
        - `"ist_frq"` : instantaneous frequency, which is the time-derivative of
        the phase of the complex spectrogram
        - `"grp_dly"` : group delay, which is the frequency-derivative of the
        phase of the complex spectrogram

        For example, if you want to calculate all features, set `features` to
        `["power", "ist_frq", "grp_dly"]`.

        By default, "power".
    frontend_freq_mask : bool
        Whether to use the frequency masking or not, by default `True`.
    frontend_freq_mask_param : int
        Maximum possible length of the frequency mask. Indices uniformly sampled
        from [0, freq_mask_param).

        By default, `26`.
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
    head_hidden_features : int
        Number of hidden features of the head for GRBAS, by default `256`.
    head_num_layers : int
        Number of fully connected layers of the head for GRABS, by default `2`.
    head_drop_rate : float
        Dropout rate of the head for GRABS, by default `0.0`.
    grbas_min : float
        Lower bound of the GRABS scores, by default `-3.0`.
    grbas_max : float
        Upper bound of the GRABS scores, by default `6.0`.
    grbas_num_bins : int
        Number of bins of each discrete distribution of GRBAS regression, by
        default `91`.
    grbas_item : "G", "R", "B", "A", "S", or "GRBAS"
        GRBAS items to be predicted, by default `"G"`.
    loss_reduction : "none", "mean", or "sum"
        Specifies the reduction to apply to the GRABS output.

        Available values are as follows:

        - `"none"` : no reduction will be applied.
        - `"mean"` : the sum of the output will be divided by the number of
          elements in the output.
        - `"sum"` : the output will be summed.

        By default, `"mean"`.
    """
    def __init__(
        self,
        *,
        fs: float = 16000,
        num_filters: int = 128,
        frontend_window_length: int = 800,
        frontend_shift_length: int = 160,
        frontend_window_name: str = "hann",
        frontend_features: Union[
            Literal["power", "ist_frq", "grp_dly"],
            Collection[Literal["power", "ist_frq", "grp_dly"]]
        ] = "power",
        frontend_freq_mask: bool = True,
        frontend_freq_mask_param: int = 26,
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
        head_hidden_features: int = 256,
        head_num_layers: int = 2,
        head_drop_rate: float = 0.0,
        grbas_min: float = -3.0,
        grbas_max: float = 6.0,
        grbas_num_bins: int = 91,
        grbas_item: Literal["G", "R", "B", "A", "S", "GRBAS"] = "G",
        loss_reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        if grbas_item not in {"G", "R", "B", "A", "S", "GRBAS"}:
            raise ValueError(
                f'grbas_item must be "G", "R", "B", "A", "S", or "GRBAS"; '
                f'found {grbas_item}.')
        super().__init__()
        self.frontend = Frontend(
            fs=fs,
            num_filters=num_filters,
            window_length=frontend_window_length,
            shift_length=frontend_shift_length,
            window_name=frontend_window_name,
            features=frontend_features,
            freq_mask=frontend_freq_mask,
            freq_mask_param=frontend_freq_mask_param,
        )
        self.encoder = Encoder(
            cnn_in_channels=self.frontend.out_channels,
            cnn_model_name=cnn_model_name,
            cnn_drop_path_rate=cnn_drop_path_rate,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_drop_rate=rnn_drop_rate,
            rnn_bidirectional=rnn_bidirectional,
            num_filters=num_filters,
        )
        head_in_features = self.encoder.crnn.rnn.out_features
        self.head = MultiOutMLPHead(
            in_features=head_in_features,
            out_features_dict={grbas: grbas_num_bins
                               for grbas in grbas_item},
            hidden_features=head_hidden_features,
            num_layers=head_num_layers,
            drop_rate=head_drop_rate,
        )
        self.grbas_num_bins = grbas_num_bins
        self.grbas_item = grbas_item
        self.dldl = DLDLRegression(
            min=grbas_min, max=grbas_max, num_bins=grbas_num_bins,
            loss_reduction=loss_reduction,
        )

    def encode(
        self,
        x: Union[Tensor, list[Tensor]],
    ) -> Tensor:
        return self.encoder(self.frontend(x))

    def forward(
        self,
        x: Union[Tensor, list[Tensor]],
    ) -> dict[Literal["G","R","B","A","S"], Tensor]:
        x = self.encode(x)
        return self.head(x)

    def predict(
        self,
        x: Union[Tensor, list[Tensor]],
    ) -> dict[Literal["G","R","B","A","S"], Tensor]:
        x: dict[Literal["G","R","B","A","S"], Tensor] = self(x)
        return {k: self.dldl.predict(v) for k, v in x.items()}

    def calculate_mean(
        self,
        x: Union[Tensor, list[Tensor]],
    ) -> dict[Literal["G","R","B","A","S"], Tensor]:
        x: dict[Literal["G","R","B","A","S"], Tensor] = self(x)
        return {k: self.dldl.calculate_mean(v) for k, v in x.items()}

    def calculate_mean_sd(
        self,
        x: Union[Tensor, list[Tensor]],
    ) -> dict[Literal["G","R","B","A","S"], tuple[Tensor, Tensor]]:
        x: dict[Literal["G","R","B","A","S"], Tensor] = self(x)
        return {k: self.dldl.calculate_mean_sd(v) for k, v in x.items()}

    def calculate_loss(
        self,
        x: Union[Tensor, list[Tensor]],
        target_mean: dict[Literal["G","R","B","A","S"], Tensor],
        target_sd: dict[Literal["G","R","B","A","S"], Tensor],
        *,
        reduction: Optional[Literal["none", "mean", "sum"]] = None,
    ) -> dict[Literal["G","R","B","A","S"], Tensor]:
        x: dict[Literal["G","R","B","A","S"], Tensor] = self(x)
        return {k: self.dldl.calculate_loss(
                    v, target_mean, target_sd, reduction=reduction)
                for k, v in x.items()}
