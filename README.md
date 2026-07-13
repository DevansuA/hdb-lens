# 🏠 HDB-Lens

**Production-grade price intelligence for Singapore's HDB resale market, with calibrated uncertainty, not just point guesses.**

![CI](https://github.com/DevansuA/hdb-lens/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Most price predictors answer *"what will this flat sell for?"* with a single number that is silently wrong. HDB-Lens answers with a **statistically calibrated price range**, trained on **234,000+ real transactions** pulled live from the [data.gov.sg](https://data.gov.sg) API, and evaluated the only honest way: **on the future.** Models are trained on 2017–2024, tuned on 2025, and tested on 2026 transactions they have never seen.

| | Naive baseline¹ | **HDB-Lens (LightGBM)** |
|---|---|---|
| MAE (2026 test, n=12,605) | S$173,880 | **S$39,942** |
| MAPE | 24.5% | **5.7%** |
| R² | 0.02 | **0.92** |

¹ median price per (town, flat type); every model must beat a lookup table before it deserves attention.

<p align="center">
  <img src="reports/figures/app_screenshot.png" width="720" alt="HDB-Lens app: a hero price estimate with a plain-English explanation, a town map, and a market-trend chart"/>
</p>

The [live app](app/streamlit_app.py) leads with one number and one sentence anyone can read, not a wall of model diagnostics. The technical case for trusting that number, raw vs. calibrated coverage, the SHAP breakdown, per-town accuracy, sits one click away in tabs, not competing for attention with the answer.

**Uncertainty that means something:** raw P10–P90 quantile models covered only **61%** of future prices (classic distribution shift, 2026's market moved). Applying **Conformalized Quantile Regression** frozen at deployment lifted empirical coverage to **76%**; recalibrating monthly against the trailing 6 months of observed sales (adaptive CQR) closes the gap further to **77.9%** of the nominal 80%, at a median interval width of 16.5% of price. The Streamlit app serves the frozen-CQR interval (what a real deployment would ship); the adaptive result quantifies how much of the remaining gap is recoverable with an online update loop.

<p align="center">
  <img src="reports/figures/prediction_intervals.png" width="640" alt="Conformal prediction intervals on unseen 2026 transactions"/>
</p>
<p align="center">
  <img src="reports/figures/adaptive_coverage.png" width="640" alt="Monthly interval coverage: raw quantiles vs frozen CQR vs adaptive CQR"/>
</p>

---

## Why this project is built the way it is

1. **Live data, not a Kaggle CSV.** `hdblens.ingest` hits data.gov.sg's initiate/poll download API, so every retrain reflects the latest monthly release. CI includes a schema-drift smoke test against the live API.
2. **Time-forward evaluation.** Random splits leak the future into training and flatter your metrics. All numbers here are out-of-time: the model prices 2026 flats knowing only 2017–2024.
   <p align="center"><img src="reports/figures/market_trend.png" width="640"/></p>
3. **Ranges beat points.** Buyers and policymakers act on risk, not point estimates. Three LightGBM pinball-loss models (P10/P50/P90) + CQR ([Romano et al., 2019](https://arxiv.org/abs/1905.03222)) produce intervals with a distribution-free coverage guarantee under exchangeability.
4. **Geography as features.** Haversine distances from town centroids to the CBD (Raffles Place) and the nearest URA regional centre (Tampines, Jurong Lake District, Woodlands, Punggol Digital District) encode Singapore's polycentric planning directly into the model.
5. **Explainability.** SHAP confirms the model has learned real economics: floor area, remaining lease, market epoch, storey, and centrality dominate, in that order.
   <p align="center"><img src="reports/figures/shap_summary.png" width="600"/></p>
6. **Designed like a product.** A custom wordmark, a two-color palette, and one font pairing (Fraunces for numbers, Inter for everything else, including the charts). The result leads with a plain-English estimate, not a dashboard of MAPE and coverage percentages.
7. **Engineered like software.** Typed, documented `src/` package · 24 unit tests · ruff linting · GitHub Actions CI across Python 3.10–3.12 · Dockerized Streamlit app · Makefile workflow.

<p align="center">
  <img src="reports/figures/pred_vs_actual.png" width="480" alt="Predicted vs actual on 2026 test set"/>
</p>

## Architecture

```mermaid
flowchart LR
    A[data.gov.sg API] -->|ingest.py| B[Raw transactions CSV]
    B -->|features.py| C[Feature matrix<br/>parsing + geospatial]
    C -->|temporal_split| D[Train ≤2024 / Val 2025 / Test 2026]
    D -->|train.py| E[Baseline + LightGBM point<br/>+ P10/P50/P90 quantiles]
    E -->|conformal.py| F[CQR-calibrated intervals]
    F --> G[metrics.json + SHAP + figures]
    F --> H[Streamlit app / predict.py API]
```

## Quickstart

```bash
git clone https://github.com/DevansuA/hdb-lens && cd hdb-lens
make install          # pip install -e ".[app,dev]"
make train            # downloads live data, trains, calibrates, evaluates
make app              # launches the Streamlit price estimator
make test             # pytest + ruff
make docker           # containerized app on :8501
```

`make train` writes `models/metrics.json`, the serialized model bundle, and all figures; the numbers in this README are its direct output.

## Repository layout

```
src/hdblens/          the package
  ingest.py           live data.gov.sg download client
  features.py         parsing, flat age, haversine geo-features
  train.py            baseline + LightGBM point & quantile models
  conformal.py        CQR interval calibration
  evaluate.py         time-forward metrics, per-town error slicing
  predict.py          single-flat inference API
app/streamlit_app.py  hero estimate, town map, market chart, and tabbed model detail
app/assets/           wordmark, vendored Inter font files (charts match the app's type)
scripts/run_pipeline.py  one-command end-to-end reproduction
tests/                unit tests (parsers, geo, splits, baseline)
reports/figures/      generated diagnostics
```

## Honest limitations & roadmap

- **Coverage gap under drift.** Even adaptive CQR (`conformal.adaptive_qhat`, monthly recalibration on trailing observed sales) only closes 76% → 77.9% vs nominal 80% on 2026: tree models cannot extrapolate the `month_index` trend, so upper quantiles still undershoot in a rising market. Next step: a linear trend prior on log-price, or shorter recalibration windows.
- **Town-centroid geography.** Distances use town centroids, not block-level geocoding. OneMap geocoding of the ~9,600 unique blocks (plus distance-to-nearest-MRT) is the highest-value feature upgrade.
- **No macro covariates.** Interest rates and cooling measures (e.g., 2025 loan-to-value changes) enter only implicitly through `month_index`.
- Per-town error slicing (`evaluate.error_by_group`) shows where the model is weakest (small-volume central towns), a candidate for hierarchical pooling.

## Data

[HDB Resale Flat Prices (from Jan 2017)](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view), Housing & Development Board, via data.gov.sg. Made available under the Singapore Open Data Licence.

---

*Built by [Devansu Agarwal](https://www.linkedin.com/in/devansua9/): Data Science @ NUS, Data Analytics @ A\*STAR, Student Researcher @ Lee Kuan Yew School of Public Policy.*
