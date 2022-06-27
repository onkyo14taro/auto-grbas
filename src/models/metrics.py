from __future__ import annotations
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt


def cohen_kappa_score(
    y1: npt.NDArray[np.float_],
    y2: npt.NDArray[np.float_],
    weights: Optional[Literal['linear', 'quadratic']] = None,
) -> float:
    """Calculate Cohen's kappa [1-3].

    This implementation can handle continuous values unlike
    `sklearn.metrics.cohen_kappa_score`.

    If the scores are continuous, you should set `weights` to `"quadratic"` or
    `"linear"`.

    Parameters
    ----------
    y1 : NDArray [float, shape=(num_samples, )]
        Scores assigned by the first annotator.

    y2 : NDArray [float, shape=(num_samples, )]
        Scores assigned by the second annotator.

    weights : "linear", "quadratic", or None
        Weighting type to calculate the score. `None` means no weighted;
        `"linear"` means linear weighted; `"quadratic"` means quadratic
        weighted. By default, `None`.

    Returns
    -------
    kappa : float
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.

    References
    ----------
    [1] J. Cohen, "A Coefficient of Agreement for Nominal Scales," Educ.
        Psychol. Meas., vol. 20, no. 1, pp. 37-46, 1960.
    [2] J. Cohen, "Weighted kappa: Nominal scale agreement provision for scaled
        disagreement or partial credit," Psychol. Bull., vol. 70, no. 4,
        pp. 213-220, 1968.
    [3] K. L. Gwet, Handbook of Inter-Rater Reliability, 4th Edition: The
        Definitive Guide to Measuring The Extent of Agreement Among Raters.
        Advanced Analytics, LLC, 2014.
    """
    y1, y2 = _validate_inputs(y1, y2)
    observed = len(y1) * _kappa_score_helper(
        y1 - y2, weights=weights)
    expected = _kappa_score_helper(
        y1.reshape(1, -1) - y2.reshape(-1, 1), weights=weights)
    return (1 - observed/expected).item()


def conger_kappa_score(
    scores: npt.NDArray[np.float_],
    weights: Optional[Literal['linear', 'quadratic']] = None,
) -> float:
    """Calculate Conger's kappa [1,2].

    This implementation can handle continuous values.

    If the scores are continuous, you should set `weights` to `"quadratic"` or
    `"linear"`.

    Parameters
    ----------
    scores : NDArray [float, shape=(num_samples, num_raters > 1)]
        Scores assigned by the annotators.

    weights : "linear", "quadratic", or None
        Weighting type to calculate the score. `None` means no weighted;
        `"linear"` means linear weighted; `"quadratic"` means quadratic
        weighted. By default, `None`.

    Returns
    -------
    kappa : float
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.

    References
    ----------
    [1] A. J. Conger, "Integration and generalization of kappas for multiple
        raters," Psychol. Bull., vol. 88, no. 2, pp. 322-328, 1980.
    [2] K. L. Gwet, Handbook of Inter-Rater Reliability, 4th Edition: The
        Definitive Guide to Measuring The Extent of Agreement Among Raters.
        Advanced Analytics, LLC, 2014.
    """
    if not (
        isinstance(scores, np.ndarray) \
        and scores.ndim == 2 and scores.shape[1] > 1 \
    ):
        raise ValueError(
            "scores.shape must be (num_samples, num_raters > 1).")
    num_samples, num_raters = scores.shape
    observed = sum(
        _kappa_score_helper(
            scores[:, i] - scores[:, j],
            weights=weights,
        )
        for i in range(num_raters)
        for j in range(i+1, num_raters)
    ) / (num_samples * num_raters * (num_raters - 1) / 2)
    expected = _kappa_score_helper(
        scores.reshape(1, -1) - scores.reshape(-1, 1),
        weights=weights,
    ) / (num_samples**2 * num_raters * (num_raters-1))
    expected -= _kappa_score_helper(
        scores.reshape(1, num_samples, num_raters) - \
        scores.reshape(num_samples, 1, num_raters),
        weights=weights
    ) / (num_samples**2 * num_raters * (num_raters-1))
    return (1 - observed/expected).item()


def _validate_inputs(
    y1: npt.NDArray[np.float_],
    y2: npt.NDArray[np.float_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if y1.ndim != 1:
        raise ValueError('y1 and y2 must be 1-D arrays.')
    if y1.shape != y2.shape:
        raise ValueError('y1 and y2 must be the same length.')
    return y1.astype(np.float64), y2.astype(np.float64)


def _kappa_score_helper(
    dist: npt.NDArray[np.float_],
    weights: Optional[Literal['linear', 'quadratic']] = None,
) -> float:
    if weights == 'linear':
        return np.sum(np.abs(dist))
    elif weights == 'quadratic':
        return np.sum(dist**2)
    elif weights is None:
        return np.sum(dist != 0)
    else:
        raise ValueError('Unknown kappa weighting type.')
