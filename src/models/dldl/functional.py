from __future__ import annotations
from typing import Literal, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


class DLDLRegression():
    r"""Class for deep label distribution learning (DLDL) [1].

    Parameters
    ----------
    min : float
        Lower bound of regression. It is recommended that the value be set
        smaller than it can actually take.

    max : float
        Upper bound of regression. It is recommended that the value be set
        greater than it can actually take.

    num_bins : int
        Number of bins of discrete distributions. The discrete distribution
        ranges from `min` to `max` and is equally divided by `num_bins`.

    loss_reduction : Literal["none", "mean", "sum"]
        Specifies the reduction to apply to the output.

        Available values are as follows:

        - `"none"` : no reduction will be applied.
        - `"mean"` : the sum of the output will be divided by the number of
          elements in the output.
        - `"sum"` : the output will be summed.

        By default, `"mean"`.

    References
    ----------
    [1] B.-B. Gao, C. Xing, C.-W. Xie, J. Wu, and X. Geng, "Deep Label
        Distribution Learning With Label Ambiguity,” IEEE Trans. Image
        Process., vol. 26, no. 6, pp. 2825--2838, Jun. 2017.
    """
    def __init__(
        self,
        min: float,
        max: float,
        num_bins: int,
        loss_reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        if loss_reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                'loss_reduction must be either "none", "mean", or "sum".')
        self.min = float(min)
        self.max = float(max)
        self.num_bins = int(num_bins)
        self.loss_reduction = loss_reduction
        self.labels = torch.linspace(min, max, num_bins)[None, :]

    def probability(
        self,
        input: Tensor,
    ) -> Tensor:
        r"""Calculate the probability from logit by using the softmax function.

        Parameters
        ----------
        input : Tensor [shape=(batch_size, self.num_bins)]
            Logits.

        Returns
        -------
        prob : Tensor [shape=(batch_size, self.num_bins)]
            Probability.
        """
        self._validate_input(input)
        return F.softmax(input, dim=-1)

    def predict(
        self,
        input: Tensor,
        *,
        from_logits: bool = True,
    ) -> Tensor:
        r"""Predict the regression values by taking the expectation values.

        Parameters
        ----------
        input : Tensor [shape=(batch_size, self.num_bins)]
            If `from_logits` is `True`, this parameter represents logits.
            If `from_logits` is `False`, this parameter represents probability.

        from_logits : bool
            Whether `input` is logits or not.

            By default, `True`.

        Returns
        -------
        pred : Tensor [shape=(batch_size, )]
            Regression values.
        """
        self._validate_input(input)
        prob = self.probability(input) if from_logits else input
        labels = self.labels.to(device=input.device)
        return torch.sum(prob*labels, dim=-1)

    def calculate_mean(
        self,
        input: Tensor,
        *,
        from_logits: bool = True,
    ) -> Tensor:
        r"""Calculate the mean values (expectation values).

        Note that this function is same as `predict()`.

        Parameters
        ----------
        input : Tensor [shape=(batch_size, self.num_bins)]
            If `from_logits` is `True`, this parameter represents logits.
            If `from_logits` is `False`, this parameter represents probability.

        from_logits : bool
            Whether `input` is logits or not.

            By default, `True`.

        Returns
        -------
        mean : Tensor [shape=(batch_size, )]
             Mean values.
        """
        return self.predict(input, from_logits=from_logits)

    def calculate_mean_sd(
        self,
        input: Tensor,
        *,
        from_logits: bool = True,
    ) -> tuple[Tensor, Tensor]:
        r"""Calculate the mean values (expectation values) and SDs.

        Parameters
        ----------
        input : Tensor [shape=(batch_size, self.num_bins)]
            If `from_logits` is `True`, this parameter represents logits.
            If `from_logits` is `False`, this parameter represents probability.

        from_logits : bool
            Whether `input` is logits or not.

            By default, `True`.

        Returns
        -------
        mean : Tensor [shape=(batch_size, )]
            Mean values.

        sd : Tensor [shape=(batch_size, )]
            Standard deviations.
        """
        self._validate_input(input)
        prob = self.probability(input) if from_logits else input
        labels = self.labels.to(device=input.device)
        mean = torch.sum(prob*labels, dim=-1)
        var = torch.sum(prob*(labels**2), dim=-1) - mean**2
        sd = torch.sqrt(var)
        return mean, sd

    def generate_target(
        self,
        mean: Tensor,
        sd: Tensor,
    ) -> Tensor:
        r"""Generate the target Gaussian distributions.

        Parameters
        ----------
        mean : Tensor [shape=(batch_size, )]
            Mean values of the targets.

        sd : Tensor [shape=(batch_size, )]
            Standard deviations of the targets.

        Returns
        -------
        target : Tensor [shape=(batch_size, self.num_bins)]
            Target gaussian distributions.
        """
        if not (mean.ndim == sd.ndim == 1 and len(mean) == len(sd)):
            raise ValueError(
                "mean and sd must be 1D tensors of the same length.")
        mean = mean.view(-1, 1)
        sd = sd.view(-1, 1)
        labels = self.labels.to(device=mean.device)
        if mean.min() < labels.min() or mean.max() > labels.max():
            raise ValueError(
                'Each element of mean must be in [labels.min(), labels.max()].')

        target = torch.exp(-0.5 * ((labels-mean)/sd)**2)

        # When the SD is 0 or extremely small, make the discrete distribution to
        # be a unit pulse.
        # Note that the device is temporarily changed to CPU to work around a
        # bug that prevents assignment operations from being performed on GPUs.
        target = target.to(device="cpu")
        target.nan_to_num_(nan=1.0)
        mask = torch.nonzero(target.sum(dim=1) == 0)
        for i in range(len(mask)):
            index = mask[i]
            target[index, (labels-mean[index]).abs().argmin()] = 1.0
        target = target.to(device=mean.device)

        # Normalize the areas of the discrete distributions to be 1.
        target /= target.sum(dim=1, keepdim=True)
        return target

    def calculate_loss(
        self,
        input: Tensor,
        target_mean: Tensor,
        target_sd: Tensor,
        *,
        from_logits: bool = True,
        reduction: Optional[Literal["none", "mean", "sum"]] = None,
    ) -> Tensor:
        r"""Generate the target Gaussian distributions.

        Parameters
        ----------
        input : Tensor [shape=(batch_size, self.num_bins)]
            If `from_logits` is `True`, this parameter represents logits.
            If `from_logits` is `False`, this parameter represents probability.

        target_mean : Tensor [shape=(batch_size, )]
            Mean values of the targets.

        target_sd : Tensor [shape=(batch_size, )]
            Standard deviations of the targets.

        from_logits : bool
            Whether `input` is logits or not.

            By default, `True`.

        loss_reduction : Literal["none", "mean", "sum"] or None
            Specifies the reduction to apply to the output.

            If `None`, `self.loss_reduction` will be used.
            Otherwise, `loss_reduction` (this parameter) will be used.

            Available values are as follows:

            - `"none"` : no reduction will be applied.
            - `"mean"` : the sum of the output will be divided by the number of
            elements in the output.
            - `"sum"` : the output will be summed.
            - `None` : `self.loss_reduction` will replace `loss_reduction`.

            By default, `None`.

        Returns
        -------
        target : Tensor [shape=(batch_size, self.num_bins)]
            Target gaussian distributions.
        """
        self._validate_input(input)
        pred = self.probability(input) if from_logits else input
        target = self.generate_target(target_mean, target_sd)
        loss = -torch.sum(torch.log(pred) * target, dim=-1)
        reduction = reduction if reduction is not None else self.loss_reduction
        return _reduce_loss(loss, reduction=reduction)

    def _validate_input(
        self,
        input: Tensor,
    ) -> None:
        if input.ndim != 2 or input.size(1) != self.labels.size(1):
            raise ValueError(
                "The input shape must be (batch_size, self.num_bins).")

    def __repr__(self) -> str:
        return f"DLDLRegression(min={self.min}, max={self.max}, num_bins=" \
               f"{self.num_bins}, loss_reduction={self.loss_reduction})"


################################################################################
################################################################################
### Helper functions
################################################################################
################################################################################
def _reduce_loss(
    loss: Tensor,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError('reduction must be "none", "mean", or "sum".')
