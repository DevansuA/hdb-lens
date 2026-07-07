"""HDB-Lens: interactive resale price estimator.

Run:  streamlit run app/streamlit_app.py
Requires a trained bundle (python scripts/run_pipeline.py).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from matplotlib.ticker import FuncFormatter

from hdblens.config import CATEGORICAL_FEATURES, CBD, MODEL_DIR, REGIONAL_CENTRES, TOWN_COORDS
from hdblens.features import haversine_km
from hdblens.predict import make_feature_row, predict_price
from hdblens.train import load_bundle

st.set_page_config(page_title="HDB-Lens", page_icon="🏠", layout="wide")

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"
AQUA = "#1baf7a"
RED = "#e34948"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"

FEATURE_LABELS = {
    "town": "Town",
    "flat_type": "Flat type",
    "flat_model": "Flat model",
    "floor_area_sqm": "Floor area",
    "storey_mid": "Storey",
    "remaining_lease_months": "Remaining lease",
    "flat_age_years": "Flat age",
    "dist_cbd_km": "Distance to CBD",
    "dist_regional_centre_km": "Nearest regional centre",
    "month_index": "Market month (trend)",
}


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


@st.cache_data
def _load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    trends = pd.read_csv(MODEL_DIR / "town_trends.csv")
    recent = pd.read_csv(MODEL_DIR / "recent_sales.csv")
    errors = pd.read_csv(MODEL_DIR / "town_errors.csv")
    return meta, metrics, trends, recent, errors


def _sgd(x: float) -> str:
    return f"S${x:,.0f}"


def _style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_axisbelow(True)


def town_map(town: str) -> pdk.Deck:
    towns = pd.DataFrame(
        [{"name": t, "lat": lat, "lon": lon} for t, (lat, lon) in TOWN_COORDS.items()]
    )
    selected = towns[towns["name"] == town]
    anchors = pd.DataFrame(
        [{"name": "CBD (Raffles Place)", "lat": CBD[0], "lon": CBD[1], "color": [235, 104, 52]}]
        + [
            {"name": n, "lat": lat, "lon": lon, "color": [74, 58, 167]}
            for n, (lat, lon) in REGIONAL_CENTRES.items()
        ]
    )
    s_lat, s_lon = TOWN_COORDS[town]
    rc_name, _ = nearest_regional_centre(s_lat, s_lon)
    rc = REGIONAL_CENTRES[rc_name]
    lines = pd.DataFrame(
        [
            {"from": [s_lon, s_lat], "to": [CBD[1], CBD[0]], "color": [235, 104, 52]},
            {"from": [s_lon, s_lat], "to": [rc[1], rc[0]], "color": [74, 58, 167]},
        ]
    )
    layers = [
        pdk.Layer(
            "ScatterplotLayer", towns, get_position="[lon, lat]", get_radius=450,
            get_fill_color=[137, 135, 129, 150], pickable=True,
        ),
        pdk.Layer(
            "LineLayer", lines, get_source_position="from", get_target_position="to",
            get_color="color", get_width=2.5,
        ),
        pdk.Layer(
            "ScatterplotLayer", anchors, get_position="[lon, lat]", get_radius=550,
            get_fill_color="color", pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer", selected, get_position="[lon, lat]", get_radius=800,
            get_fill_color=[42, 120, 214, 230], pickable=True,
        ),
        pdk.Layer(
            "TextLayer", pd.concat([anchors, selected.assign(name=town.title())]),
            get_position="[lon, lat]", get_text="name", get_size=11,
            get_color=[82, 81, 78], get_pixel_offset=[0, -14],
        ),
    ]
    view = pdk.ViewState(latitude=1.352, longitude=103.82, zoom=10)
    return pdk.Deck(layers=layers, initial_view_state=view, map_style=None,
                    tooltip={"text": "{name}"})


def nearest_regional_centre(lat: float, lon: float) -> tuple[str, float]:
    dists = {
        name: float(haversine_km(lat, lon, np.array([c[0]]), np.array([c[1]]))[0])
        for name, c in REGIONAL_CENTRES.items()
    }
    return min(dists.items(), key=lambda kv: kv[1])


def interval_figure(raw: dict, cal: dict, cov_raw: float, cov_cal: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    rows = [
        (1, raw, BASELINE, f"Raw model quantiles\n{cov_raw:.0f}% actual coverage"),
        (0, cal, BLUE, f"CQR-calibrated (served)\n{cov_cal:.0f}% actual coverage"),
    ]
    for y, est, color, _ in rows:
        ax.plot([est["p10"], est["p90"]], [y, y], color=color, lw=7, solid_capstyle="round")
        ax.text(est["p10"], y + 0.22, _sgd(est["p10"]), ha="center", fontsize=8, color=INK_2)
        ax.text(est["p90"], y + 0.22, _sgd(est["p90"]), ha="center", fontsize=8, color=INK_2)
    ax.plot([cal["p50"]], [0], "o", ms=7, color=INK, zorder=5)
    ax.text(cal["p50"], -0.42, f"P50 {_sgd(cal['p50'])}", ha="center", fontsize=8, color=INK)
    ax.set_yticks([r[0] for r in rows], [r[3] for r in rows], fontsize=8.5, color=INK)
    ax.set_ylim(-0.8, 1.7)
    span = cal["p90"] - cal["p10"]
    ax.set_xlim(cal["p10"] - 0.25 * span, cal["p90"] + 0.25 * span)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e3:,.0f}k"))
    _style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def shap_figure(bundle: dict, row: pd.DataFrame) -> tuple[plt.Figure, float]:
    contribs = bundle["point"].predict(row, pred_contrib=True)[0]
    base_price = float(np.exp(contribs[-1]))
    effects = pd.Series(contribs[:-1], index=row.columns)
    effects = effects.reindex(effects.abs().sort_values().index)
    pct = (np.exp(effects) - 1) * 100

    r = row.iloc[0]
    values = {
        "town": str(r["town"]).title(),
        "flat_type": str(r["flat_type"]).title(),
        "flat_model": str(r["flat_model"]),
        "floor_area_sqm": f"{r['floor_area_sqm']:.0f} sqm",
        "storey_mid": f"level {r['storey_mid']:.0f}",
        "remaining_lease_months": f"{r['remaining_lease_months'] / 12:.0f} yrs",
        "flat_age_years": f"{r['flat_age_years']:.0f} yrs",
        "dist_cbd_km": f"{r['dist_cbd_km']:.1f} km",
        "dist_regional_centre_km": f"{r['dist_regional_centre_km']:.1f} km",
        "month_index": "mid-2026",
    }
    labels = [f"{FEATURE_LABELS[f]} · {values[f]}" for f in pct.index]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = [BLUE if v >= 0 else RED for v in pct]
    ax.barh(labels, pct, color=colors, height=0.62)
    for i, v in enumerate(pct):
        ax.text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK_2)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_xlabel("Impact on estimated price (%)", fontsize=8.5, color=INK_2)
    lim = max(abs(pct.min()), abs(pct.max())) * 1.35 + 2
    ax.set_xlim(-lim, lim)
    ax.grid(axis="x", color=GRID, lw=0.7)
    _style_axis(ax)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    fig.tight_layout()
    return fig, base_price


def trend_figure(trends: pd.DataFrame, town: str, flat_type: str, est: dict) -> plt.Figure | None:
    t = trends[(trends["town"] == town) & (trends["flat_type"] == flat_type)]
    if len(t) < 6:
        return None
    x = pd.PeriodIndex(t["month"], freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.plot(x, t["median_price"] / 1e3, color=BLUE, lw=1.7,
            label=f"Median actual sale · {flat_type.title()} in {town.title()}")
    mx = x.max()
    ax.errorbar([mx], [est["p50"] / 1e3],
                yerr=[[(est["p50"] - est["p10"]) / 1e3], [(est["p90"] - est["p50"]) / 1e3]],
                fmt="o", ms=6, color=INK, ecolor=BLUE_LIGHT, elinewidth=5, capsize=0,
                label="This flat · model P50 and calibrated range")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}k"))
    ax.set_ylabel("Price (S$ '000)", fontsize=8.5, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    return fig


bundle, ref_cats, q_hat = _load()
meta, metrics, trends, recent, errors = _load_artifacts()
test_metrics = metrics["test_2026"]
cov_raw = test_metrics["interval_p10_p90"]["empirical_coverage_pct"]
cov_cal = test_metrics["interval_p10_p90_conformal"]["empirical_coverage_pct"]
cov_adaptive = test_metrics.get("interval_p10_p90_adaptive", {}).get("empirical_coverage_pct")

with st.sidebar:
    st.header("Model card")
    st.markdown(
        f"""
- **Data** · {meta["n_train"]:,} resale transactions, live from data.gov.sg
- **Trained through** · {meta["trained_through"]}
- **Tested on** · {meta["tested_on"]} (never seen in training)
"""
    )
    c1, c2 = st.columns(2)
    c1.metric("2026 test MAPE", f"{test_metrics['lightgbm_point']['mape_pct']:.1f}%")
    c2.metric("2026 test MAE", _sgd(test_metrics["lightgbm_point"]["mae"]))
    st.markdown("**Interval coverage** (80% nominal)")
    rows = [("Raw P10–P90 quantiles", cov_raw), ("+ CQR, frozen q̂ (served here)", cov_cal)]
    if cov_adaptive:
        rows.append(("+ monthly adaptive recalibration", cov_adaptive))
    st.markdown("\n".join(f"- {label}: **{v:.0f}%**" for label, v in rows))
    st.caption(
        "Every number above is out-of-time: the model prices 2026 flats knowing "
        "only 2017–2024. The gap to the nominal 80% is honest, measured price drift."
    )
    st.markdown("[Source & methodology on GitHub](https://github.com/DevansuA/hdb-lens)")

st.title("🏠 HDB-Lens")
st.caption(
    "Calibrated resale price ranges for Singapore public housing, trained on 230k+ "
    "real transactions from data.gov.sg — with the uncertainty made honest by "
    "conformal calibration."
)

left, right = st.columns([5, 7], gap="large")
with left:
    st.subheader("Your flat")
    c1, c2 = st.columns(2)
    with c1:
        town = st.selectbox("Town", sorted(TOWN_COORDS))
        flat_type = st.selectbox(
            "Flat type", ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"], index=2
        )
        flat_model = st.selectbox(
            "Flat model", ref_cats["flat_model"].categories.tolist(), index=0
        )
    with c2:
        floor_area = st.slider("Floor area (sqm)", 30, 180, 93)
        storey = st.slider("Storey", 1, 50, 8)
        lease_left = st.slider("Remaining lease (years)", 40, 99, 75)

    latest_month = pd.Period(trends["month"].max(), freq="M")
    month_index = (latest_month - pd.Period("2017-01", freq="M")).n
    row = make_feature_row(
        town=town,
        flat_type=flat_type,
        flat_model=flat_model,
        floor_area_sqm=floor_area,
        storey_mid=storey,
        remaining_lease_years=lease_left,
        month_index=month_index,
        reference_categories=ref_cats,
    )
    est = predict_price(bundle, row, q_hat=q_hat)
    est_raw = predict_price(bundle, row)

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimate (P50)", _sgd(est["p50"]))
    m2.metric("Calibrated range", f"{est['p10'] / 1e3:,.0f}k – {est['p90'] / 1e3:,.0f}k")
    m3.metric("Per sqm", _sgd(est["p50"] / floor_area))

with right:
    st.pydeck_chart(town_map(town), height=430)
    rc_name, rc_km = nearest_regional_centre(*TOWN_COORDS[town])
    cbd_km = float(row["dist_cbd_km"].iloc[0])
    st.markdown(
        f"<span style='color:{BLUE}'>●</span> {town.title()} &nbsp; "
        f"<span style='color:{ORANGE}'>●</span> CBD · {cbd_km:.1f} km &nbsp; "
        f"<span style='color:{VIOLET}'>●</span> nearest regional centre "
        f"({rc_name}) · {rc_km:.1f} km &nbsp; "
        f"<span style='color:{MUTED}'>●</span> other towns — distances are model features",
        unsafe_allow_html=True,
    )

st.divider()
chart_l, chart_r = st.columns(2, gap="large")
with chart_l:
    st.subheader("The range is the product")
    st.caption(
        "Raw quantile models under-cover on future sales, so the served interval is "
        "widened by conformal calibration (CQR) fitted on the most recent window."
    )
    st.pyplot(interval_figure(est_raw, est, cov_raw, cov_cal), width="stretch")
with chart_r:
    st.subheader("Why this price")
    fig, base_price = shap_figure(bundle, row)
    st.caption(
        f"Per-prediction SHAP: starting from the market baseline of {_sgd(base_price)}, "
        f"each feature pushes the estimate up or down to land at {_sgd(est['point'])}."
    )
    st.pyplot(fig, width="stretch")

st.divider()
real_l, real_r = st.columns(2, gap="large")
with real_l:
    st.subheader("Against the actual market")
    fig = trend_figure(trends, town, flat_type, est)
    if fig is None:
        st.info(f"Too few recorded {flat_type.title()} sales in {town.title()} to plot a trend.")
    else:
        st.pyplot(fig, width="stretch")
with real_r:
    st.subheader(f"Recent {flat_type.title()} sales in {town.title()}")
    sales = recent[(recent["town"] == town) & (recent["flat_type"] == flat_type)].head(8)
    if sales.empty:
        st.info("No transactions in the last three months for this combination.")
    else:
        st.dataframe(
            sales[["month", "flat_model", "storey_range", "floor_area_sqm",
                   "remaining_lease", "resale_price"]],
            hide_index=True,
            column_config={
                "month": "Month",
                "flat_model": "Model",
                "storey_range": "Storey",
                "floor_area_sqm": st.column_config.NumberColumn("Area (sqm)"),
                "remaining_lease": "Lease left",
                "resale_price": st.column_config.NumberColumn("Sold for", format="S$%,d"),
            },
        )
    town_err = errors[errors["town"] == town]
    if not town_err.empty:
        rank = int((errors["mape_pct"] < town_err["mape_pct"].iloc[0]).sum()) + 1
        st.caption(
            f"Honesty check: on unseen 2026 sales in {town.title()}, the point model "
            f"was off by {town_err['mape_pct'].iloc[0]:.1f}% on average "
            f"(rank {rank} of {len(errors)} towns, 1 = most accurate)."
        )
