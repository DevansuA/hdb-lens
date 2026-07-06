"""Conformalized Quantile Regression (CQR).

Quantile models trained on historical data systematically under-cover on
*future* transactions: the market's variance drifts, and tree models cannot
extrapolate spread. CQR (Romano, Patterson & Candès, 2019) fixes this with a
distribution-free guarantee: compute conformity scores on a held-out
calibration window, then widen the interval by the appropriate empirical
quantile of those scores. Coverage on exchangeable data is then guaranteed
at the nominal level — and in practice transfers well even under mild drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hdblens.config import FEATURES, TARGET


def cqr_correction(bundle: dict, calib: pd.DataFrame, coverage: float = 0.80) -> float:
    """Compute the CQR correction (in log-price space) from a calibration set.

    Conformity score: E_i = max(lo_i - y_i, y_i - hi_i)   (log space)
    Correction:       q_hat = Quantile(E, ceil((n+1)*(coverage))/n)
    """
    y = np.log(calib[TARGET].to_numpy())
    lo = bundle["quantiles"][0.10].predict(calib[FEATURES])
    hi = bundle["quantiles"][0.90].predict(calib[FEATURES])
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(np.ceil((n + 1) * coverage) / n, 1.0)
    return float(np.quantile(scores, level))


def predict_interval(
    bundle: dict, df: pd.DataFrame, q_hat: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """P10/P90 interval in SGD, optionally widened by the CQR correction."""
    lo = np.exp(bundle["quantiles"][0.10].predict(df[FEATURES]) - q_hat)
    hi = np.exp(bundle["quantiles"][0.90].predict(df[FEATURES]) + q_hat)
    return lo, hi


def interval_report(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    covered = (y >= lo) & (y <= hi)
    return {
        "empirical_coverage_pct": float(covered.mean() * 100),
        "mean_width_sgd": float(np.mean(hi - lo)),
        "median_width_pct_of_price": float(np.median((hi - lo) / y) * 100),
    }
