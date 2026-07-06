"""Single-flat inference: build a feature row from user inputs and return
a point estimate plus a calibrated P10–P90 price range."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hdblens.config import CATEGORICAL_FEATURES, FEATURES
from hdblens.features import add_geo_features


def make_feature_row(
    *,
    town: str,
    flat_type: str,
    flat_model: str,
    floor_area_sqm: float,
    storey_mid: float,
    remaining_lease_years: float,
    month_index: int,
    reference_categories: dict[str, pd.CategoricalDtype],
) -> pd.DataFrame:
    """Assemble one model-ready row from raw user inputs."""
    row = pd.DataFrame(
        {
            "town": [town],
            "flat_type": [flat_type],
            "flat_model": [flat_model],
            "floor_area_sqm": [float(floor_area_sqm)],
            "storey_mid": [float(storey_mid)],
            "remaining_lease_months": [float(remaining_lease_years) * 12.0],
            "flat_age_years": [99.0 - float(remaining_lease_years)],  # 99-yr leases
            "month_index": [int(month_index)],
        }
    )
    row = add_geo_features(row)
    for col in CATEGORICAL_FEATURES:
        row[col] = row[col].astype(reference_categories[col])
    return row[FEATURES]


def predict_price(bundle: dict, row: pd.DataFrame) -> dict[str, float]:
    """Return point estimate and P10/P50/P90 in SGD."""
    return {
        "point": float(np.exp(bundle["point"].predict(row))[0]),
        "p10": float(np.exp(bundle["quantiles"][0.10].predict(row))[0]),
        "p50": float(np.exp(bundle["quantiles"][0.50].predict(row))[0]),
        "p90": float(np.exp(bundle["quantiles"][0.90].predict(row))[0]),
    }
