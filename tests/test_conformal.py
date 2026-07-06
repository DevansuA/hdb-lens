import numpy as np
import pandas as pd

from hdblens.config import FEATURES, TARGET
from hdblens.conformal import cqr_correction, predict_interval
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


def test_predict_price_applies_cqr_correction():
    bundle = _stub_bundle(500_000, 600_000, point=550_000)
    row = _feature_frame(1)
    raw = predict_price(bundle, row)
    calibrated = predict_price(bundle, row, q_hat=0.05)
    assert calibrated["p10"] < raw["p10"]
    assert calibrated["p90"] > raw["p90"]
    assert calibrated["p50"] == raw["p50"]
