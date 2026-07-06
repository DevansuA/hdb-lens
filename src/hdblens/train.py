"""Model training.

Three-tier modelling strategy:
1. Naive baseline  : median price per (town, flat_type). Every serious project
   needs a baseline; beating "a lookup table" is the bar that matters.
2. Point model     : LightGBM regression on log-price (L1 objective, robust
   to the heavy right tail of the price distribution).
3. Quantile models : LightGBM pinball-loss models at P10 / P50 / P90, giving
   calibrated price *ranges* instead of a single number. A buyer cares far
   more about "likely between $612k and $694k" than "$651k".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from hdblens.config import FEATURES, MODEL_DIR, QUANTILES, TARGET

logger = logging.getLogger(__name__)

LGB_COMMON = dict(
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=40,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


@dataclass
class BaselineModel:
    """Median price per (town, flat_type), with global-median fallback."""

    table: pd.Series = field(default_factory=pd.Series)
    global_median: float = 0.0

    def fit(self, df: pd.DataFrame) -> "BaselineModel":
        self.table = df.groupby(["town", "flat_type"], observed=True)[TARGET].median()
        self.global_median = float(df[TARGET].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_frame(df[["town", "flat_type"]])
        preds = self.table.reindex(keys).to_numpy()
        return np.where(np.isnan(preds), self.global_median, preds)


def _fit_lgb(params: dict, train: pd.DataFrame, val: pd.DataFrame, y_transform) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train[FEATURES],
        y_transform(train[TARGET]),
        eval_set=[(val[FEATURES], y_transform(val[TARGET]))],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        categorical_feature=[c for c in FEATURES if str(train[c].dtype) == "category"],
    )
    return model


def train_all(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """Fit baseline, point, and quantile models. Returns a model bundle."""
    logger.info("Training baseline on %d rows", len(train))
    baseline = BaselineModel().fit(train)

    logger.info("Training point model (log-price, L1)")
    point = _fit_lgb({**LGB_COMMON, "objective": "regression_l1"}, train, val, np.log)

    quantile_models: dict[float, lgb.LGBMRegressor] = {}
    for q in QUANTILES:
        logger.info("Training quantile model q=%.2f", q)
        params = {**LGB_COMMON, "objective": "quantile", "alpha": q}
        quantile_models[q] = _fit_lgb(params, train, val, np.log)

    return {"baseline": baseline, "point": point, "quantiles": quantile_models}


def save_bundle(bundle: dict, metadata: dict | None = None) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / "hdblens_bundle.joblib")
    if metadata:
        (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Saved model bundle to %s", MODEL_DIR)


def load_bundle() -> dict:
    return joblib.load(MODEL_DIR / "hdblens_bundle.joblib")
