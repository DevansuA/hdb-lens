import numpy as np
import pandas as pd
import pytest

from hdblens.features import (
    build_features,
    haversine_km,
    parse_remaining_lease,
    parse_storey_mid,
    temporal_split,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("61 years 04 months", 61 * 12 + 4),
        ("99 years", 99 * 12),
        ("5 years 0 months", 60),
        ("garbage", np.nan),
        (None, np.nan),
    ],
)
def test_parse_remaining_lease(text, expected):
    result = parse_remaining_lease(text)
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "text,expected",
    [("10 TO 12", 11.0), ("01 TO 03", 2.0), ("40 TO 42", 41.0), ("", np.nan)],
)
def test_parse_storey_mid(text, expected):
    result = parse_storey_mid(text)
    if isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected


def test_haversine_known_distance():
    # Raffles Place -> Woodlands is roughly 17-18 km as the crow flies
    d = haversine_km(1.2840, 103.8515, np.array([1.4382]), np.array([103.7890]))
    assert 16 < d[0] < 19


def _toy_raw():
    return pd.DataFrame(
        {
            "month": ["2019-05", "2024-11", "2026-02"],
            "town": ["ANG MO KIO", "PUNGGOL", "QUEENSTOWN"],
            "flat_type": ["3 ROOM", "4 ROOM", "5 ROOM"],
            "block": ["1", "2", "3"],
            "street_name": ["A", "B", "C"],
            "storey_range": ["01 TO 03", "10 TO 12", "13 TO 15"],
            "floor_area_sqm": [67, 93, 112],
            "flat_model": ["New Generation", "Model A", "Improved"],
            "lease_commence_date": [1978, 2015, 2000],
            "remaining_lease": ["58 years", "89 years 06 months", "73 years"],
            "resale_price": [300000, 550000, 900000],
        }
    )


def test_build_features_shapes_and_types():
    df = build_features(_toy_raw())
    assert len(df) == 3
    assert df["month_index"].tolist() == [28, 94, 109]
    assert df["flat_age_years"].tolist() == [41, 9, 26]
    assert (df["dist_cbd_km"] > 0).all()
    assert str(df["town"].dtype) == "category"


def test_temporal_split_is_disjoint_and_ordered():
    df = build_features(_toy_raw())
    train, val, test = temporal_split(df, "2024-12", "2025-12")
    assert len(train) == 2 and len(val) == 0 and len(test) == 1
    assert train["month"].max() < test["month"].min()


def test_temporal_split_boundary_months_are_inclusive():
    months = ["2024-12", "2025-01", "2025-12", "2026-01"]
    df = pd.DataFrame({"month": pd.PeriodIndex(months, freq="M")})
    train, val, test = temporal_split(df, "2024-12", "2025-12")
    assert train["month"].astype(str).tolist() == ["2024-12"]
    assert val["month"].astype(str).tolist() == ["2025-01", "2025-12"]
    assert test["month"].astype(str).tolist() == ["2026-01"]
