import numpy as np
import pandas as pd

from hdblens.config import FEATURES, TARGET
from hdblens.conformal import adaptive_qhat, cqr_correction, interval_report, predict_interval
from hdblens.predict import predict_price


class _ConstModel:
    """Stub regressor returning a fixed log-price for every row."""

    def __init__(self, log_price: float):
        self.log_price = log_price

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.log_price)


def _feature_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({col: [0] * n for col in FEATURES})


def _stub_bundle(lo: float, hi: float, point: float | None = None) -> dict:
    bundle = {"quantiles": {0.10: _ConstModel(np.log(lo)), 0.90: _ConstModel(np.log(hi))}}
    if point is not None:
        bundle["point"] = _ConstModel(np.log(point))
        bundle["quantiles"][0.50] = _ConstModel(np.log(point))
    return bundle


def test_predict_interval_widens_with_q_hat():
    bundle = _stub_bundle(500_000, 600_000)
    df = _feature_frame(1)
    lo0, hi0 = predict_interval(bundle, df)
    assert np.isclose(lo0[0], 500_000) and np.isclose(hi0[0], 600_000)

    lo1, hi1 = predict_interval(bundle, df, q_hat=0.1)
    assert np.isclose(lo1[0], 500_000 * np.exp(-0.1))
    assert np.isclose(hi1[0], 600_000 * np.exp(0.1))


def test_cqr_correction_restores_coverage_on_calibration_set():
    rng = np.random.default_rng(0)
    y = 550_000 * np.exp(rng.normal(0, 0.3, 500))
    bundle = _stub_bundle(540_000, 560_000)  # far too narrow

    calib = _feature_frame(len(y))
    calib[TARGET] = y
    raw_lo, raw_hi = predict_interval(bundle, calib)
    assert ((y >= raw_lo) & (y <= raw_hi)).mean() < 0.5

    q_hat = cqr_correction(bundle, calib, coverage=0.80)
    assert q_hat > 0
    lo, hi = predict_interval(bundle, calib, q_hat)
    assert ((y >= lo) & (y <= hi)).mean() >= 0.80


def test_adaptive_qhat_tracks_drift():
    rng = np.random.default_rng(1)
    bundle = _stub_bundle(520_000, 580_000)

    def _month_frame(period: str, drift: float, n: int = 200) -> pd.DataFrame:
        df = _feature_frame(n)
        df["month"] = pd.Period(period, freq="M")
        df[TARGET] = 550_000 * np.exp(rng.normal(drift, 0.05, n))
        return df

    calib = pd.concat([_month_frame(f"2025-{m:02d}", 0.0) for m in (10, 11, 12)])
    test = pd.concat(
        [_month_frame(f"2026-0{i}", drift) for i, drift in [(1, 0.05), (2, 0.10), (3, 0.15)]]
    )

    monthly, overall = adaptive_qhat(bundle, calib, test, coverage=0.80, window=3)
    assert monthly["month"].tolist() == ["2026-01", "2026-02", "2026-03"]
    assert monthly["q_hat"].iloc[-1] > monthly["q_hat"].iloc[0]  # widens as drift grows

    frozen = cqr_correction(bundle, calib, coverage=0.80)
    lo, hi = predict_interval(bundle, test, frozen)
    y = test[TARGET].to_numpy()
    frozen_coverage = ((y >= lo) & (y <= hi)).mean() * 100
    assert overall["empirical_coverage_pct"] > frozen_coverage


def test_cqr_correction_clamps_finite_sample_level_on_tiny_calibration():
    # With n=2 and coverage=0.8, ceil((n+1)*coverage)/n = 1.2 must clamp to
    # the max score rather than crash np.quantile.
    bundle = _stub_bundle(500_000, 600_000)
    calib = _feature_frame(2)
    calib[TARGET] = [700_000.0, 800_000.0]  # both above hi -> positive scores
    q_hat = cqr_correction(bundle, calib, coverage=0.80)
    assert np.isclose(q_hat, np.log(800_000) - np.log(600_000))


def test_interval_report_known_values():
    y = np.array([100.0, 200.0, 300.0, 400.0])
    lo = np.array([90.0, 190.0, 310.0, 390.0])
    hi = np.array([110.0, 210.0, 330.0, 410.0])  # third interval misses low
    report = interval_report(y, lo, hi)
    assert report["empirical_coverage_pct"] == 75.0
    assert report["mean_width_sgd"] == 20.0
    assert report["median_width_pct_of_price"] == np.median([20 / 100, 20 / 200, 20 / 300, 20 / 400]) * 100


def test_predict_price_applies_cqr_correction():
    bundle = _stub_bundle(500_000, 600_000, point=550_000)
    row = _feature_frame(1)
    raw = predict_price(bundle, row)
    calibrated = predict_price(bundle, row, q_hat=0.05)
    assert calibrated["p10"] < raw["p10"]
    assert calibrated["p90"] > raw["p90"]
    assert calibrated["p50"] == raw["p50"]
