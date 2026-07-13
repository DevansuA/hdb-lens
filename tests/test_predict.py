import pandas as pd

from hdblens.config import FEATURES
from hdblens.predict import make_feature_row

REF_CATS = {
    "town": pd.CategoricalDtype(categories=["BEDOK", "PUNGGOL"]),
    "flat_type": pd.CategoricalDtype(categories=["3 ROOM", "4 ROOM"]),
    "flat_model": pd.CategoricalDtype(categories=["Model A", "Improved"]),
}


def _row(**overrides):
    kwargs = dict(
        town="BEDOK",
        flat_type="4 ROOM",
        flat_model="Model A",
        floor_area_sqm=93,
        storey_mid=8,
        remaining_lease_years=75,
        month_index=100,
        reference_categories=REF_CATS,
    )
    kwargs.update(overrides)
    return make_feature_row(**kwargs)


def test_make_feature_row_matches_training_schema():
    row = _row()
    assert list(row.columns) == FEATURES
    assert row["remaining_lease_months"].iloc[0] == 75 * 12
    assert row["flat_age_years"].iloc[0] == 99 - 75
    assert row["dist_cbd_km"].iloc[0] > 0
    assert row["dist_regional_centre_km"].iloc[0] > 0
    for col, dtype in REF_CATS.items():
        assert row[col].dtype == dtype


def test_make_feature_row_unknown_category_becomes_missing():
    # A category the model never saw must map to NaN (LightGBM treats it as
    # missing), never to a silently wrong known category.
    row = _row(flat_model="Brand New Model")
    assert row["flat_model"].isna().all()
