"""Evaluation: honest, time-forward metrics plus interval calibration.

Reports MAE / MAPE / RMSE / R² for baseline vs. point model, and empirical
coverage + sharpness for the P10–P90 prediction interval (target: ~80%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from hdblens.config import FEATURES, TARGET


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_split(bundle: dict, df: pd.DataFrame) -> dict:
    """Evaluate all models on one split; returns a nested metrics dict."""
    y = df[TARGET].to_numpy()

    base_pred = bundle["baseline"].predict(df)
    point_pred = np.exp(bundle["point"].predict(df[FEATURES]))

    lo = np.exp(bundle["quantiles"][0.10].predict(df[FEATURES]))
    hi = np.exp(bundle["quantiles"][0.90].predict(df[FEATURES]))
    covered = (y >= lo) & (y <= hi)

    return {
        "n": int(len(df)),
        "baseline": point_metrics(y, base_pred),
        "lightgbm_point": point_metrics(y, point_pred),
        "interval_p10_p90": {
            "empirical_coverage_pct": float(covered.mean() * 100),
            "mean_width_sgd": float(np.mean(hi - lo)),
            "median_width_pct_of_price": float(np.median((hi - lo) / y) * 100),
        },
    }


def error_by_group(bundle: dict, df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Slice MAPE by a grouping column (e.g. town) to expose weak segments:
    aggregate metrics hide where a model quietly fails."""
    out = df[[group, TARGET]].copy()
    out["pred"] = np.exp(bundle["point"].predict(df[FEATURES]))
    out["ape"] = (out["pred"] - out[TARGET]).abs() / out[TARGET]
    return (
        out.groupby(group, observed=True)
        .agg(n=("ape", "size"), mape_pct=("ape", lambda s: s.mean() * 100))
        .sort_values("mape_pct", ascending=False)
    )
