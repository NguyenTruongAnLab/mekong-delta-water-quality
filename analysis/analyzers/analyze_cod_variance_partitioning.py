#!/usr/bin/env python3
"""analyze_cod_variance_partitioning.py

Reviewer4 request: quantify how much of COD variability is explained by
hydrodynamic covariates vs LULC predictors, and by their combination.

This script computes a *station-held-out* diagnostic (Leave-One-Station-Out CV)
for COD in the Ben Tre network (2005–2020, using year-level hydro covariates).

We compare three predictor sets:
  1) LULC-only (baseline attribution predictors)
  2) Hydro-only (My Thuan discharge + tidal range + rainfall + year)
  3) Combined (LULC + hydro)

Outputs (analysis/analysis_outputs):
- cod_variance_partitioning_loso_latest.csv

Notes
-----
- This is not causal variance decomposition.
- The LOSO R^2 here is a strict generalization diagnostic and can be negative.
  The relative ordering (hydro vs LULC vs combined) is intended to support
  reviewer-facing statements like: "hydrodynamics are necessary but not
  sufficient; LULC adds incremental explanatory power".
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
OUTPUT_DIR = os.path.join(ROOT_DIR, "analysis", "analysis_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WQ_FILE = os.path.join(ROOT_DIR, "data", "mekong_delta_water_quality_1985_2023.csv")
LULC_FILE = os.path.join(ROOT_DIR, "analysis", "cache", "lulc_metrics_corrected_16km.csv")
TIDAL_FILE = os.path.join(OUTPUT_DIR, "Table_Flushing_Capacity.csv")

TARGET = "COD (mg/L)"
YEAR_MIN = 2005
YEAR_MAX = 2020

try:
    from analysis.lib.hydro_ingest import HydroClimateColumns, join_branch_hydroclimate
except ModuleNotFoundError:
    from lib.hydro_ingest import HydroClimateColumns, join_branch_hydroclimate

def _load_tidal_range_proxy() -> Dict[str, float]:
    """Return mapping from branch prefix to a tidal range proxy (m)."""

    if not os.path.exists(TIDAL_FILE):
        # Allow running without this file; downstream will drop NaNs.
        return {}

    tr = pd.read_csv(TIDAL_FILE)
    tr_map = dict(zip(tr["Station"], tr["Mean_Tidal_Range_m"]))

    # These proxies follow the manuscript convention used in figure_rda_shap.py
    return {
        "HL": float(tr_map.get("Vam_Kenh", np.nan)),
        "CC": float(tr_map.get("Vam_Kenh", np.nan)),
        "MT": float(tr_map.get("My_Tho", np.nan)),
        "BL": float(tr_map.get("Vam_Kenh", np.nan)),
    }


def _prepare_dataset() -> pd.DataFrame:
    for fp in [WQ_FILE, LULC_FILE]:
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)

    df_wq = pd.read_csv(WQ_FILE)
    df_wq["year"] = pd.to_datetime(df_wq["datetime"], errors="coerce").dt.year

    wq_annual = (
        df_wq.groupby(["stationName", "River", "year"], as_index=False)
        .mean(numeric_only=True)
    )

    if TARGET not in wq_annual.columns:
        raise ValueError(f"Missing {TARGET} in annual WQ table")

    df_lulc = pd.read_csv(LULC_FILE)
    df = pd.merge(wq_annual, df_lulc, on=["stationName", "year"], how="inner")

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    df = df[df["stationName"].astype(str).str.match(r"^(HL|CC|MT|BL)\d{2}$")].copy()

    df["branch"] = df["stationName"].astype(str).str[:2]

    # Hydro-climate covariates from Gold Standard
    # Using My Thuan mainstem Q (year-level, same for all stations) instead of
    # branch Q (which excludes Ba Lai) — see audit Part 12B.
    h = join_branch_hydroclimate(df[["stationName", "year"]].copy())
    hcols = HydroClimateColumns()
    for c in [
        hcols.q_mythuan_annual_z,
        hcols.q_mythuan_dry_z,
        hcols.rain_annual_mm,
        hcols.rain_dry_mm,
    ]:
        if c in h.columns:
            df[c] = h[c].values

    # Tidal range at My Thuan (year-level, from tidal decomposition)
    tr_path = os.path.join(OUTPUT_DIR, 'Table_Tidal_Range_By_Year.csv')
    if os.path.exists(tr_path):
        tr = pd.read_csv(tr_path)
        tr = tr[(tr['Station'] == 'MyThuan') & (tr['Metric'] == 'AnnualMeanDailyTidalRange_m')]
        tr_map = dict(zip(tr['year'].astype(int), tr['Value'].astype(float)))
        df['TR_MyThuan_m'] = df['year'].astype(int).map(tr_map)

    # Center year (hydro/climate trend proxy)
    df["Year_centered"] = df["year"].astype(float) - float(df["year"].astype(float).mean())

    return df


def _loso_metrics(X: pd.DataFrame, y: np.ndarray, groups: pd.Series) -> Tuple[float, float, float]:
    logo = LeaveOneGroupOut()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])

    yhat = cross_val_predict(model, X, y, groups=groups, cv=logo)

    r2 = float(r2_score(y, yhat))
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))

    # Spearman (robust to outliers)
    rho = float(pd.Series(y).corr(pd.Series(yhat), method="spearman"))
    rho_f = rho if np.isfinite(rho) else float("nan")

    return r2, rmse, rho_f


def main() -> None:
    df = _prepare_dataset()

    lulc_cols = [
        "Residential_Urban_composition_pct",
        "Residential_Urban_config_PD",
        "Aquaculture_composition_pct",
        "Aquaculture_config_ED",
        "Rice_Cultivation_composition_pct",
        "Rice_Cultivation_config_PD",
    ]

    hcols = HydroClimateColumns()
    hydro_cols = [
        hcols.q_mythuan_annual_z,
        hcols.q_mythuan_dry_z,
        hcols.rain_annual_mm,
        hcols.rain_dry_mm,
        "TR_MyThuan_m",
        "Year_centered",
    ]

    needed = [TARGET, "stationName"] + lulc_cols + hydro_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for variance partitioning: {missing}")

    df2 = df[needed].dropna().copy()

    if df2.shape[0] < 40:
        raise RuntimeError(f"Insufficient complete cases after hydro proxy alignment (N={df2.shape[0]}).")

    y_np = df2[TARGET].astype(float).to_numpy(dtype=float)
    groups = df2["stationName"].astype(str)

    rows = []
    for name, cols in [
        ("LULC_only", lulc_cols),
        ("Hydro_only", hydro_cols),
        ("Combined", lulc_cols + hydro_cols),
    ]:
        X = df2[cols].astype(float)
        r2, rmse, rho = _loso_metrics(X, y_np, groups)

        # In-sample OLS (variance explained, not a generalization metric)
        X_ols_np = sm.add_constant(X.to_numpy(dtype=float), has_constant="add")
        ols = sm.OLS(y_np, X_ols_np, missing="drop").fit()

        rows.append(
            {
                "Predictor_Set": name,
                "LOSO_R2": r2,
                "LOSO_RMSE": rmse,
                "LOSO_Spearman_rho": rho,
                "OLS_R2": float(ols.rsquared),
                "OLS_R2_adj": float(ols.rsquared_adj),
                "N": int(len(df2)),
                "N_stations": int(groups.nunique()),
                "N_features": int(len(cols)),
                "Years": f"{YEAR_MIN}-{YEAR_MAX}",
            }
        )

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "cod_variance_partitioning_loso_latest.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved COD variance partitioning diagnostic: {out_path}")

    # Phase 3 requested deterministic comparison table (alias of the above).
    out_path2 = os.path.join(OUTPUT_DIR, "attribution_with_hydroclimate_comparison_latest.csv")
    out.to_csv(out_path2, index=False)
    print(f"Saved hydro-climate comparison table: {out_path2}")


if __name__ == "__main__":
    main()
