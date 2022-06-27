import os
import sys
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.metrics

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.metrics import cohen_kappa_score, conger_kappa_score


def _conger_kappa_textbook(
    scores: npt.NDArray,
    weights: Optional[Literal['linear', 'quadratic']] = None,
) -> float:
    r"""Naive implementation of Conger's kappa in the textbook [1].

    [1] K. L. Gwet, Handbook of Inter-Rater Reliability, 4th Edition: The
        Definitive Guide to Measuring The Extent of Agreement Among Raters.
        Advanced Analytics, LLC, 2014.
    """
    df = pd.DataFrame(scores)
    min_val = np.nanmin(df.to_numpy())
    max_val = np.nanmax(df.to_numpy())
    interval = max_val - min_val

    p_a = 0
    num_rows_only_one_rater = 0
    for _, row in df.iterrows():
        row = row.dropna()
        num_raters = len(row)
        if num_raters <= 1:
            num_rows_only_one_rater += 1
            continue

        count_dict = row.value_counts()
        for val_k, count_k in count_dict.items():
            r = count_k
            r_star = 0
            for val_l, count_l in count_dict.items():
                if weights is None:
                    w = float(val_k == val_l)
                elif weights == "linear":
                    w = 1 - (abs(val_k - val_l) / interval)
                elif weights == "quadratic":
                    w = 1 - ((val_k - val_l) / interval)**2
                r_star += w * count_l
            p_a += (r * (r_star-1)) / (num_raters * (num_raters-1))
    p_a /= (len(df) - num_rows_only_one_rater)

    p_e = 0
    num_raters = scores.shape[1]
    prob_exp = []
    for rater, scores in df.items():
        prob_exp.append(scores.value_counts() / len(scores.dropna()))
    prob_exp = pd.DataFrame(prob_exp).fillna(0).mean()
    for val_k, p_k in prob_exp.items():
        for val_l, p_l in prob_exp.items():
            if weights is None:
                w = float(val_k == val_l)
            elif weights == "linear":
                w = 1 - (abs(val_k - val_l) / interval)
            elif weights == "quadratic":
                w = 1 - ((val_k - val_l) / interval)**2
            p_e_temp = (num_raters/(num_raters-1)) * (p_k*p_l)
            for rater, scores in df.items():
                prob_exp_rater = scores.value_counts() / len(scores.dropna())
                p_gk = prob_exp_rater[val_k] if val_k in prob_exp_rater else 0.0
                p_gl = prob_exp_rater[val_l] if val_l in prob_exp_rater else 0.0
                p_e_temp -= (p_gk * p_gl) \
                    / (num_raters * (num_raters-1))
            p_e += p_e_temp * w
    kappa = (p_a - p_e) / (1 - p_e)
    return kappa


def test_cohen_kappa_score():
    y1 = np.random.randint(0, 10, size=100)
    y2 = y1.copy()
    y2[::2] = np.random.randint(0, 10, size=50)

    for weights in (None, "linear", "quadratic"):
        kappa1 = cohen_kappa_score(y1, y2, weights=weights)

        # Comparison with sklearn implementation
        kappa2 = sklearn.metrics.cohen_kappa_score(y1, y2, weights=weights)        
        assert np.allclose(kappa1, kappa2)

        # Test for continuous values
        kappa3 = cohen_kappa_score(y1/2, y2/2, weights=weights)
        assert np.allclose(kappa1, kappa3)

        # Test for small noise 
        kappa4 = cohen_kappa_score(y1, y2+1e-10, weights=weights)
        if weights is None:
            assert kappa4 == 0
        else:
            np.allclose(kappa1, kappa4)

        print(f'weights: {weights}')
        print(f'kappa1: {kappa1}')
        print(f'kappa2: {kappa2}')
        print(f'kappa3: {kappa3}')
        print(f'kappa4: {kappa4}')


def test_conger_kappa_score():
    # Example 1.
    # Gwet (2014, p.55, 56)
    scores = np.array([
        [1.0,    1.0,    np.nan, 1.0],
        [2.0,    2.0,    3.0,    2.0],
        [3.0,    3.0,    3.0,    3.0],
        [3.0,    3.0,    3.0,    3.0],
        [2.0,    2.0,    2.0,    2.0],
        [1.0,    2.0,    3.0,    4.0],
        [4.0,    4.0,    4.0,    4.0],
        [1.0,    1.0,    2.0,    1.0],
        [2.0,    2.0,    2.0,    2.0],
        [np.nan, 5.0,    5.0,    5.0],
        [np.nan, np.nan, 1.0,    1.0],
        [np.nan, np.nan, 3.0,    np.nan],
    ])
    assert np.allclose(
        _conger_kappa_textbook(scores),
        0.7628, atol=0.00005,
    )

    # Example 2.
    # Gwet (2014, p.89, 90)
    scores = np.array([
        [1.0,    1.5, 1.0,    np.nan],
        [2.0,    2.0, 2.0,    2.0],
        [0.5,    1.0, 1.5,    1.5],
        [1.0,    1.0, 1.0,    1.0],
        [1.0,    1.0, 1.0,    1.5],
        [np.nan, 1.0, 2.5,    np.nan],
        [2.5,    2.5, 2.5,    2.5],
        [1.0,    1.0, np.nan, 1.0],
        [np.nan, 1.0, 2.0,    1.0],
        [1.0,    1.0, 0.5,    1.0],
        [1.5,    1.5, 1.5,    1.5],
        [1.0,    1.5, 1.0,    np.nan],
        [1.0,    1.0, 1.5,    np.nan],
        [1.0,    2.0, 2.5,    2.0],
        [np.nan, 1.0, 1.5,    1.0],
        [0.5,    0.5, 0.5,    0.5],
    ])
    assert np.allclose(
        _conger_kappa_textbook(scores, weights="quadratic"),
        0.5290, atol=0.00005,
    )

    # Comparison 1.
    scores = pd.DataFrame(scores).dropna().to_numpy()
    for weights in (None, "linear", "quadratic"):
        assert np.allclose(
            _conger_kappa_textbook(scores, weights=weights),
            conger_kappa_score(scores, weights=weights)
        )

    # Comparison 2.
    y1 = np.random.randint(0, 10, size=100)
    y2 = y1.copy()
    y2[::2] = np.random.randint(0, 10, size=50)

    for weights in (None, "linear", "quadratic"):
        kappa1 = cohen_kappa_score(y1, y2, weights=weights)
        kappa2 = conger_kappa_score(np.stack([y1, y2], axis=1), weights=weights)
        kappa3 = _conger_kappa_textbook(np.stack([y1, y2], axis=1), weights=weights)
        assert np.allclose(kappa1, kappa2)
        assert np.allclose(kappa1, kappa3)

        # Test for continuous values
        kappa1 = cohen_kappa_score(y1/2, y2/2, weights=weights)
        kappa2 = conger_kappa_score(np.stack([y1/2, y2/2], axis=1), weights=weights)
        kappa3 = _conger_kappa_textbook(np.stack([y1/2, y2/2], axis=1), weights=weights)
        assert np.allclose(kappa1, kappa2)
        assert np.allclose(kappa1, kappa3)

        # Test for small noise
        y2 = y2 + 1e-10
        kappa1 = cohen_kappa_score(y1, y2, weights=weights)
        kappa2 = conger_kappa_score(np.stack([y1, y2], axis=1), weights=weights)
        kappa3 = _conger_kappa_textbook(np.stack([y1, y2], axis=1), weights=weights)
        assert np.allclose(kappa1, kappa2)
        assert np.allclose(kappa1, kappa3)
