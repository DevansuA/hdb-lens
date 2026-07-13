# Model Card: HDB-Lens v1.0

## Model details
- **Type:** LightGBM gradient-boosted trees. One L1-objective point model and three pinball-loss quantile models (α = 0.10, 0.50, 0.90), all on log-price. Intervals post-calibrated with Conformalized Quantile Regression (CQR).
- **Features (10):** town, flat type, flat model, floor area, storey midpoint, remaining lease (months), flat age at sale, distance to CBD, distance to nearest URA regional centre, month index.
- **Training data:** 196,982 HDB resale transactions, Jan 2017 – Dec 2024 (data.gov.sg).
- **Calibration data:** Jul – Dec 2025 (most recent validation window).

## Intended use
Indicative pricing for Singapore HDB resale flats: buyer/seller anchoring, affordability analysis, and market research. **Not** a substitute for a professional valuation; not intended for lending decisions.

## Evaluation (strictly out-of-time)
| Split | n | MAE | MAPE | R² |
|---|---|---|---|---|
| Validation (2025) | 25,085 | S$33,929 | 4.92% | 0.941 |
| Test (2026 H1) | 12,605 | S$39,942 | 5.69% | 0.925 |

P10–P90 interval on 2026 test: 75.7% empirical coverage after CQR frozen at deployment (nominal 80%), median width 15.5% of price. Recalibrating monthly against the trailing six months of observed sales (adaptive CQR) recovers 77.9% coverage at a median width of 16.5%.

## Known limitations
- Coverage degrades under sustained price drift; the model cannot extrapolate the time trend.
- Geography is town-level, not block-level; within-town location premiums are averaged away.
- Segments with sparse volume (e.g., Central Area, rare flat models) carry higher error.
- Trained only on completed transactions; no listing/negotiation dynamics.

## Ethical considerations
Predictions could anchor negotiations; the app therefore always displays the calibrated range and coverage caveat alongside any point estimate. Data is fully public and contains no personal information.
