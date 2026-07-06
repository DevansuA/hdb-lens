"""Cleaning and feature engineering.

Turns raw HDB transaction records into a leakage-safe model matrix with:
- parsed storey / remaining-lease fields,
- flat age at time of sale,
- haversine distances to the CBD and the nearest URA regional centre,
- a monotone month index that lets tree models learn the market trend.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from hdblens.config import (
    CATEGORICAL_FEATURES,
    CBD,
    REGIONAL_CENTRES,
    TOWN_COORDS,
)

_LEASE_RE = re.compile(r"(?P<years>\d+)\s*years?(?:\s*(?P<months>\d+)\s*months?)?")

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) between one point and arrays of points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def parse_remaining_lease(text: str) -> float:
    """'61 years 04 months' -> 736.0 (months). Returns NaN if unparseable."""
    if not isinstance(text, str):
        return np.nan
    m = _LEASE_RE.search(text)
    if not m:
        return np.nan
    years = int(m.group("years"))
    months = int(m.group("months") or 0)
    return float(years * 12 + months)


def parse_storey_mid(text: str) -> float:
    """'10 TO 12' -> 11.0. Returns NaN if unparseable."""
    if not isinstance(text, str):
        return np.nan
    nums = re.findall(r"\d+", text)
    if not nums:
        return np.nan
    vals = [int(n) for n in nums]
    return (min(vals) + max(vals)) / 2


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach town centroid coordinates and distances to employment hubs."""
    coords = df["town"].map(TOWN_COORDS)
    df["lat"] = coords.map(lambda c: c[0] if isinstance(c, tuple) else np.nan)
    df["lon"] = coords.map(lambda c: c[1] if isinstance(c, tuple) else np.nan)

    df["dist_cbd_km"] = haversine_km(CBD[0], CBD[1], df["lat"].values, df["lon"].values)

    centre_dists = np.column_stack(
        [
            haversine_km(lat, lon, df["lat"].values, df["lon"].values)
            for (lat, lon) in REGIONAL_CENTRES.values()
        ]
    )
    df["dist_regional_centre_km"] = centre_dists.min(axis=1)
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Full raw -> model-ready transformation. Pure function of one row's data
    plus static geography, safe against temporal leakage by construction."""
    df = raw.copy()

    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df["txn_year"] = df["month"].dt.year
    origin = pd.Period("2017-01", freq="M")
    df["month_index"] = (df["month"] - origin).map(lambda p: p.n).astype(int)

    df["storey_mid"] = df["storey_range"].map(parse_storey_mid)
    df["remaining_lease_months"] = df["remaining_lease"].map(parse_remaining_lease)
    df["flat_age_years"] = df["txn_year"] - df["lease_commence_date"].astype(int)

    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")

    df = add_geo_features(df)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    # Drop the handful of rows with unparseable core fields.
    core = ["resale_price", "floor_area_sqm", "storey_mid", "remaining_lease_months", "lat"]
    df = df.dropna(subset=core).reset_index(drop=True)
    return df


def temporal_split(
    df: pd.DataFrame, train_end: str, val_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split, the only honest way to evaluate
    a model that will be used to price *future* transactions."""
    train_cut = pd.Period(train_end, freq="M")
    val_cut = pd.Period(val_end, freq="M")
    train = df[df["month"] <= train_cut]
    val = df[(df["month"] > train_cut) & (df["month"] <= val_cut)]
    test = df[df["month"] > val_cut]
    return train, val, test
