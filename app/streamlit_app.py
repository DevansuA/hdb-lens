"""HDB-Lens: what is your HDB flat worth?

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

HERO_CSS = """
<style>
.hero-card {
    border: 1px solid #e1e0d9;
    border-radius: 16px;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(11, 11, 11, 0.04), 0 8px 24px rgba(11, 11, 11, 0.05);
    padding: 2.2rem 2.4rem 2rem;
    margin: 0.4rem 0 0.6rem;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #898781;
    margin-bottom: 0.35rem;
}
.hero-price {
    font-size: 3.6rem;
    font-weight: 750;
    letter-spacing: -0.025em;
    line-height: 1.05;
    color: #0b0b0b;
}
.hero-persqm {
    font-size: 0.9rem;
    color: #52514e;
    margin-top: 0.3rem;
}
.hero-range-track {
    position: relative;
    height: 10px;
    border-radius: 999px;
    background: linear-gradient(90deg, #dce9fa, #9ec5f4 50%, #dce9fa);
    margin: 1.4rem 0 0.45rem;
}
.hero-range-marker {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #2a78d6;
    border: 3px solid #ffffff;
    box-shadow: 0 0 0 1px #2a78d6;
}
.hero-range-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #52514e;
    margin-bottom: 1.1rem;
}
.hero-sentence {
    font-size: 1.08rem;
    line-height: 1.55;
    color: #0b0b0b;
    max-width: 46rem;
    margin: 0;
}
</style>
"""


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


def _sgd_k(x: float) -> str:
    """Round to the nearest thousand; an estimate should not pretend to dollar precision."""
    return f"S${round(x, -3):,.0f}"


def _floor_band(storey: int) -> str:
    if storey <= 3:
        return "low"
    if storey <= 9:
        return "mid"
    return "high"


def _flat_phrase(flat_type: str) -> str:
    """'4 ROOM' -> 'A 4-room flat', 'EXECUTIVE' -> 'An executive flat'."""
    label = flat_type.lower().replace(" ", "-")
    article = "An" if label[0] in "aeiou" else "A"
    return f"{article} {label} flat"


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
        (1, raw, BASELINE, f"Model's raw range\ncaught {cov_raw:.0f}% of real 2026 prices"),
        (0, cal, BLUE, f"Calibrated range (what you see)\ncaught {cov_cal:.0f}%"),
    ]
    for y, est, color, _ in rows:
        ax.plot([est["p10"], est["p90"]], [y, y], color=color, lw=7, solid_capstyle="round")
        ax.text(est["p10"], y + 0.22, _sgd(est["p10"]), ha="center", fontsize=8, color=INK_2)
        ax.text(est["p90"], y + 0.22, _sgd(est["p90"]), ha="center", fontsize=8, color=INK_2)
    ax.plot([cal["p50"]], [0], "o", ms=7, color=INK, zorder=5)
    ax.text(cal["p50"], -0.42, f"midpoint {_sgd(cal['p50'])}", ha="center", fontsize=8, color=INK)
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


def contributions(bundle: dict, row: pd.DataFrame) -> tuple[pd.Series, float]:
    """Per-feature SHAP contributions (log-price space) and the baseline price."""
    contribs = bundle["point"].predict(row, pred_contrib=True)[0]
    base_price = float(np.exp(contribs[-1]))
    return pd.Series(contribs[:-1], index=row.columns), base_price


def driver_sentences(effects: pd.Series, row: pd.DataFrame, point_price: float) -> list[str]:
    """Translate the largest SHAP contributions into plain-language dollar effects.

    A feature's dollar effect is what the estimate would lose if its contribution
    were removed: point * (1 - exp(-c)).
    """
    r = row.iloc[0]
    phrases = {
        "town": f"Being in {str(r['town']).title()}",
        "flat_type": f"The {str(r['flat_type']).lower().replace(' ', '-')} flat type",
        "flat_model": f"The “{r['flat_model']}” build",
        "floor_area_sqm": f"A floor area of {r['floor_area_sqm']:.0f} sqm",
        "storey_mid": f"Sitting on level {r['storey_mid']:.0f}",
        "remaining_lease_months": (
            f"Having {r['remaining_lease_months'] / 12:.0f} years left on the lease"
        ),
        "flat_age_years": f"The flat's age ({r['flat_age_years']:.0f} years)",
        "dist_cbd_km": f"Being {r['dist_cbd_km']:.1f} km from the city centre",
        "dist_regional_centre_km": (
            f"Being {r['dist_regional_centre_km']:.1f} km from the nearest regional centre"
        ),
        "month_index": "Today's market level",
    }
    dollars = point_price * (1 - np.exp(-effects))
    sentences = []
    for feat, d in dollars.reindex(dollars.abs().sort_values(ascending=False).index).items():
        amount = round(float(d), -3)
        if abs(amount) < 1000:
            continue
        verb = "adds about" if amount > 0 else "takes off about"
        sentences.append(f"{phrases[feat]} {verb} **S${abs(amount):,.0f}**.")
        if len(sentences) == 5:
            break
    return sentences


def shap_figure(effects: pd.Series, row: pd.DataFrame) -> plt.Figure:
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
    return fig


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
                label="This flat · estimate with 80% range")
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
n_sales = meta["n_train"] + metrics["val_2025"]["n"] + test_metrics["n"]
n_sales_str = f"{round(n_sales, -4):,}+"

st.markdown(HERO_CSS, unsafe_allow_html=True)

st.title("🏠 What's your HDB flat worth?")
st.caption(
    f"Priced from {n_sales_str} real resale transactions published by data.gov.sg."
)

with st.container(border=True):
    st.markdown("**Describe your flat**")
    c1, c2, c3 = st.columns(3)
    with c1:
        town = st.selectbox("Town", sorted(TOWN_COORDS))
        floor_area = st.slider("Floor area (sqm)", 30, 180, 93)
    with c2:
        flat_type = st.selectbox(
            "Flat type", ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"], index=2
        )
        storey = st.slider("Storey", 1, 50, 8)
    with c3:
        flat_model = st.selectbox(
            "Flat model", ref_cats["flat_model"].categories.tolist(), index=0
        )
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
# Rare quantile crossing: the P50 model can land outside the P10/P90 pair for
# sparse combinations. Clamp for display so ranges always contain the midpoint.
for e in (est, est_raw):
    e["p50"] = float(np.clip(e["p50"], e["p10"], e["p90"]))

marker_pct = 100 * (est["p50"] - est["p10"]) / (est["p90"] - est["p10"])
sentence = (
    f"{_flat_phrase(flat_type)} in {town.title()} on a {_floor_band(storey)} floor is worth "
    f"about <b>{_sgd_k(est['p50'])}</b>. Based on {n_sales_str} real transactions, we're 80% "
    f"confident it would sell between <b>{_sgd_k(est['p10'])}</b> and "
    f"<b>{_sgd_k(est['p90'])}</b>."
)
st.markdown(
    f"""<div class="hero-card">
  <div class="hero-eyebrow">Estimated resale value</div>
  <div class="hero-price">{_sgd_k(est["p50"])}</div>
  <div class="hero-persqm">{_sgd(est["p50"] / floor_area)} per square metre</div>
  <div class="hero-range-track"><div class="hero-range-marker" style="left:{marker_pct:.1f}%"></div></div>
  <div class="hero-range-labels"><span>low · {_sgd_k(est["p10"])}</span><span>high · {_sgd_k(est["p90"])}</span></div>
  <p class="hero-sentence">{sentence}</p>
</div>""",
    unsafe_allow_html=True,
)

map_col, market_col = st.columns(2, gap="medium")
with map_col, st.container(border=True):
    st.markdown("**Where it sits**")
    st.pydeck_chart(town_map(town), height=340)
    rc_name, rc_km = nearest_regional_centre(*TOWN_COORDS[town])
    cbd_km = float(row["dist_cbd_km"].iloc[0])
    st.markdown(
        f"<span style='color:{BLUE}'>●</span> {town.title()} &nbsp; "
        f"<span style='color:{ORANGE}'>●</span> city centre · {cbd_km:.1f} km &nbsp; "
        f"<span style='color:{VIOLET}'>●</span> nearest regional centre "
        f"({rc_name}) · {rc_km:.1f} km",
        unsafe_allow_html=True,
    )
with market_col, st.container(border=True):
    st.markdown("**Against the actual market**")
    fig = trend_figure(trends, town, flat_type, est)
    if fig is None:
        st.info(f"Too few recorded {flat_type.title()} sales in {town.title()} to plot a trend.")
    else:
        st.pyplot(fig, width="stretch")
        st.caption(
            "The line is what buyers actually paid, month by month. The dot is this "
            "flat's estimate, with its 80% range."
        )

tab_why, tab_conf, tab_sales, tab_method = st.tabs(
    ["Why this price?", "How confident are we, really?", "Recent sales nearby",
     "Model & methodology"]
)

with tab_why:
    effects, base_price = contributions(bundle, row)
    st.markdown("**The biggest things moving this price:**")
    st.markdown("\n".join(f"- {s}" for s in driver_sentences(effects, row, est["point"])))
    st.caption(
        f"Each effect is measured against a typical resale flat at today's market level "
        f"({_sgd(base_price)}). All effects together land at {_sgd(est['point'])}. "
        "The full picture, feature by feature:"
    )
    st.pyplot(shap_figure(effects, row), width="stretch")

with tab_conf:
    st.markdown(
        "The price above is not a single guess. It is a range built to catch the real "
        "sale price **8 times out of 10**, and we checked that claim on "
        f"{test_metrics['n']:,} sales from 2026 that the model never saw."
    )
    st.markdown(
        f"- The model's raw range caught only **{cov_raw:.0f}%** of those real prices. "
        "It ran too narrow because prices kept climbing after training.\n"
        f"- So we widen it using a held-out recent window (conformal calibration), which "
        f"brings coverage to **{cov_cal:.0f}%**. That is the range you see above."
        + (
            f"\n- Recalibrating every month as new sales arrive reaches "
            f"**{cov_adaptive:.0f}%**. The pipeline tracks this as the "
            "deployment-ready variant."
            if cov_adaptive
            else ""
        )
    )
    st.pyplot(interval_figure(est_raw, est, cov_raw, cov_cal), width="stretch")
    town_err = errors[errors["town"] == town]
    if not town_err.empty:
        rank = int((errors["mape_pct"] < town_err["mape_pct"].iloc[0]).sum()) + 1
        st.caption(
            f"In {town.title()} specifically, the estimate missed real 2026 sale prices by "
            f"{town_err['mape_pct'].iloc[0]:.1f}% on average. That ranks {rank} of "
            f"{len(errors)} towns, where rank 1 is the most accurate."
        )

with tab_sales:
    st.markdown(f"**Recent {flat_type.title()} sales in {town.title()}**")
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

with tab_method:
    st.markdown(
        f"""
- **Data** · {meta["n_train"]:,} training transactions, live from data.gov.sg
- **Trained through** · {meta["trained_through"]}
- **Tested on** · {meta["tested_on"]} (never seen in training)
- **Models** · LightGBM point + P10/P50/P90 quantile models on log price,
  with conformalized quantile regression (CQR) for the served interval
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
