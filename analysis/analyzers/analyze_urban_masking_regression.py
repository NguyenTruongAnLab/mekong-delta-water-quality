#!/usr/bin/env python3
"""analyze_urban_masking_regression.py

Evaluates upstream-downstream COD gradients while controlling for confounders
such as salinity/downstream position and aquaculture extent.

This script runs a within-branch multiple regression for the My Tho tributary
(Ben Tre MTxx stations), using annual means matched to LULC metrics:

    COD ~ Urban_PD + Aquaculture_% + Salinity + Year

Predictors are standardized to allow coefficient comparison.

Outputs (analysis/analysis_outputs):
- urban_masking_regression_mytho_latest.csv

Notes
-----
- Coefficients represent conditional associations in the observational panel.
- Salinity is treated as a proxy for downstream position/tidal influence.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = str(ANALYSIS_DIR / "analysis_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WQ_FILE = str(REPO_ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv")
LULC_FILE = str(ANALYSIS_DIR / "cache" / "lulc_metrics_corrected_16km.csv")

TARGET = "COD (mg/L)"


def zscore(s: pd.Series) -> pd.Series:
    s2 = s.astype(float)
    mu = s2.mean()
    sd = float(s2.std(ddof=0))
    if sd == 0.0 or not np.isfinite(sd):
        return s2 * np.nan
    return (s2 - mu) / sd


def main() -> None:
    master_file = str(ANALYSIS_DIR / "cache" / "station_year_master_latest.csv")
    if os.path.exists(WQ_FILE) and os.path.exists(LULC_FILE):
        df_wq = pd.read_csv(WQ_FILE)
        df_wq["year"] = pd.to_datetime(df_wq["datetime"], errors="coerce").dt.year
        wq_annual = (
            df_wq.groupby(["stationName", "River", "year"], as_index=False)
            .mean(numeric_only=True)
        )
        if TARGET not in wq_annual.columns or "Salinity" not in wq_annual.columns:
            raise ValueError("Required columns missing in annual WQ table (need COD and Salinity).")
        df_lulc = pd.read_csv(LULC_FILE)
        merged = pd.merge(wq_annual, df_lulc, on=["stationName", "year"], how="inner")
    elif os.path.exists(master_file):
        merged = pd.read_csv(master_file)
        merged[TARGET] = merged["COD (mg/L)_annual"]
        merged["Salinity"] = merged["Salinity_annual"]
    else:
        raise FileNotFoundError(WQ_FILE)

    # My Tho branch, Ben Tre stations only
    merged = merged[(merged["stationName"].astype(str).str.match(r"^MT\d{2}$"))].copy()

    # Use the full overlap window where LULC metrics exist (typically through 2020)
    merged = merged[(merged["year"] >= 2005) & (merged["year"] <= 2020)].copy()

    # Required predictors
    pred_cols = [
        "Residential_Urban_config_PD",
        "Aquaculture_composition_pct",
        "Salinity",
    ]

    keep = ["stationName", "year", TARGET] + pred_cols
    merged = merged[keep].dropna().copy()

    if merged.shape[0] < 20:
        raise RuntimeError(f"Insufficient complete cases for regression (N={merged.shape[0]}).")

    # Standardize predictors to compare effect sizes
    X = pd.DataFrame({
        "Urban_PD_z": zscore(merged["Residential_Urban_config_PD"]),
        "Aquaculture_pct_z": zscore(merged["Aquaculture_composition_pct"]),
        "Salinity_z": zscore(merged["Salinity"]),
        "Year_z": zscore(merged["year"]),
    })
    y = merged[TARGET].astype(float)

    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")

    # Build compact coefficient table
    out = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "se_HC3": model.bse.values,
        "t": model.tvalues.values,
        "p": model.pvalues.values,
    })

    meta = {
        "N": int(model.nobs),
        "R2": float(model.rsquared),
        "R2_adj": float(model.rsquared_adj),
    }

    # Attach model summary stats on every row for manuscript printing convenience
    for k, v in meta.items():
        out[k] = v

    out_path = os.path.join(OUTPUT_DIR, "urban_masking_regression_mytho_latest.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved regression output: {out_path}")


if __name__ == "__main__":
    main()
