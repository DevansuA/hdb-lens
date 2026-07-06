"""End-to-end pipeline: ingest -> features -> train -> evaluate -> figures.

Run:  python scripts/run_pipeline.py [--refresh]
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from hdblens.config import FEATURES, FIGURE_DIR, MODEL_DIR, TARGET, TRAIN_END, VAL_END
from hdblens.conformal import cqr_correction, interval_report, predict_interval
from hdblens.evaluate import error_by_group, evaluate_split
from hdblens.features import build_features, temporal_split
from hdblens.ingest import download_resale_data
from hdblens.train import save_bundle, train_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("pipeline")


def make_figures(bundle: dict, test: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    # 1. Predicted vs actual on the held-out (future) test set
    pred = np.exp(bundle["point"].predict(test[FEATURES]))
    y = test[TARGET].to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.hexbin(y / 1e3, pred / 1e3, gridsize=60, cmap="viridis", mincnt=1, bins="log")
    lims = [y.min() / 1e3, np.percentile(y, 99.5) / 1e3]
    ax.plot(lims, lims, "r--", lw=1, label="Perfect prediction")
    ax.set(xlabel="Actual price (S$ '000)", ylabel="Predicted price (S$ '000)",
           title="Out-of-time test set: predicted vs actual", xlim=lims, ylim=lims)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pred_vs_actual.png")
    plt.close(fig)

    # 2. SHAP global importance (sampled for speed)
    sample = test.sample(min(3000, len(test)), random_state=42)
    explainer = shap.TreeExplainer(bundle["point"])
    sv = explainer.shap_values(sample[FEATURES])
    fig = plt.figure(figsize=(6.5, 4.5))
    shap.summary_plot(sv, sample[FEATURES], show=False, max_display=10)
    plt.title("What drives resale prices (SHAP, log-price model)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_summary.png", dpi=130)
    plt.close("all")

    # 3. Prediction intervals: 60 random test flats sorted by price
    idx = test.sample(60, random_state=7).sort_values(TARGET).reset_index(drop=True)
    lo = np.exp(bundle["quantiles"][0.10].predict(idx[FEATURES]))
    hi = np.exp(bundle["quantiles"][0.90].predict(idx[FEATURES]))
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(idx))
    inside = (idx[TARGET] >= lo) & (idx[TARGET] <= hi)
    ax.vlines(xs, lo / 1e3, hi / 1e3, color="#9ecae1", lw=3, label="P10–P90 interval")
    ax.scatter(xs[inside], idx.loc[inside, TARGET] / 1e3, s=14, color="#08519c", label="Actual (covered)")
    ax.scatter(xs[~inside], idx.loc[~inside, TARGET] / 1e3, s=14, color="#de2d26", label="Actual (missed)")
    ax.set(xlabel="60 random unseen 2026 transactions (sorted)", ylabel="Price (S$ '000)",
           title="Calibrated price ranges on future transactions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "prediction_intervals.png")
    plt.close(fig)


def main(refresh: bool) -> None:
    path = download_resale_data(force=refresh)
    raw = pd.read_csv(path)
    df = build_features(raw)
    logger.info("Feature matrix: %s rows, span %s .. %s", len(df), df["month"].min(), df["month"].max())

    train, val, test = temporal_split(df, TRAIN_END, VAL_END)
    logger.info("Split sizes — train %d | val %d | test %d", len(train), len(val), len(test))

    bundle = train_all(train, val)

    metrics = {
        "val_2025": evaluate_split(bundle, val),
        "test_2026": evaluate_split(bundle, test),
    }

    # Conformal calibration on the most recent 6 months of validation data
    recent = val[val["month"] >= (val["month"].max() - 5)]
    q_hat = cqr_correction(bundle, recent, coverage=0.80)
    np.save(MODEL_DIR / "q_hat.npy", q_hat)
    lo, hi = predict_interval(bundle, test, q_hat)
    metrics["test_2026"]["interval_p10_p90_conformal"] = interval_report(
        test[TARGET].to_numpy(), lo, hi
    )
    metrics["cqr_q_hat"] = q_hat
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    worst = error_by_group(bundle, test, "town").head(5)
    print("\nWeakest towns on 2026 test data:\n", worst.round(2))

    save_bundle(
        bundle,
        metadata={
            "trained_through": TRAIN_END,
            "validated_on": f"2025-01..{VAL_END}",
            "tested_on": f"2026-01..{df['month'].max()}",
            "n_train": len(train),
            "features": FEATURES,
        },
    )
    make_figures(bundle, test)
    logger.info("Figures written to %s", FIGURE_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="Re-download data from data.gov.sg")
    main(ap.parse_args().refresh)
