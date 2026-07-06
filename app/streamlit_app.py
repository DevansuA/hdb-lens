"""HDB-Lens: interactive resale price estimator.

Run:  streamlit run app/streamlit_app.py
Requires a trained bundle (python scripts/run_pipeline.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from hdblens.config import CATEGORICAL_FEATURES, MODEL_DIR, TOWN_COORDS
from hdblens.predict import make_feature_row, predict_price
from hdblens.train import load_bundle

st.set_page_config(page_title="HDB-Lens", page_icon="🏠", layout="centered")
st.title("🏠 HDB-Lens")
st.caption(
    "Calibrated resale price ranges for Singapore public housing, "
    "trained on 230k+ real transactions from data.gov.sg."
)


@st.cache_resource
def _load():
    bundle = load_bundle()
    booster = bundle["point"]
    # Recover the category dtypes the model was trained with
    cats = {
        col: pd.CategoricalDtype(categories=cat)
        for col, cat in zip(
            CATEGORICAL_FEATURES,
            [booster.booster_.pandas_categorical[i] for i in range(len(CATEGORICAL_FEATURES))],
        )
    }
    q_hat = float(np.load(MODEL_DIR / "q_hat.npy"))
    return bundle, cats, q_hat


bundle, ref_cats, q_hat = _load()

col1, col2 = st.columns(2)
with col1:
    town = st.selectbox("Town", sorted(TOWN_COORDS))
    flat_type = st.selectbox(
        "Flat type", ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"], index=3
    )
    flat_model = st.selectbox(
        "Flat model", ref_cats["flat_model"].categories.tolist(), index=0
    )
with col2:
    floor_area = st.slider("Floor area (sqm)", 30, 180, 93)
    storey = st.slider("Storey", 1, 50, 8)
    lease_left = st.slider("Remaining lease (years)", 40, 99, 75)

if st.button("Estimate price", type="primary"):
    meta_month_index = 114  # ~mid-2026; latest month in training data
    row = make_feature_row(
        town=town,
        flat_type=flat_type,
        flat_model=flat_model,
        floor_area_sqm=floor_area,
        storey_mid=storey,
        remaining_lease_years=lease_left,
        month_index=meta_month_index,
        reference_categories=ref_cats,
    )
    est = predict_price(bundle, row, q_hat=q_hat)
    st.metric("Estimated price (P50)", f"S${est['p50']:,.0f}")
    st.write(
        f"**Likely range (P10–P90):** S${est['p10']:,.0f} - S${est['p90']:,.0f}"
    )
    st.progress(min(est["p50"] / 1_500_000, 1.0))
    st.caption(
        "The 80% interval is calibrated on out-of-time data: roughly 8 in 10 "
        "actual transactions fall inside the quoted range."
    )
