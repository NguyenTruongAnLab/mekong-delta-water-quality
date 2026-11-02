#!/usr/bin/env python3
"""analyze_cod_shap_robustness.py

Tests whether the ranking of key LULC predictors is stable across
alternative COD response formulations:
  1) COD concentration (mg/L)
  2) COD load (tons/day) using branch discharge proxies
  3) Flow-normalized COD: residuals from a simple linear C~Q model (annual scale)

Outputs (analysis/analysis_outputs):
- cod_shap_robustness_top_features_latest.csv

Notes
-----
- Discharge for Ben Tre stations is proxy-based at branch/year resolution.
- Discharge + rainfall covariates are derived from the Gold Standard hydrology master
  via `analysis.lib.hydro_ingest` to keep aggregation/join rules consistent.
- Flow-normalization is a diagnostic (not a calibrated hydrological correction).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from analysis.lib.hydro_ingest import HydroClimateColumns, join_branch_hydroclimate
except ModuleNotFoundError:
    from lib.hydro_ingest import HydroClimateColumns, join_branch_hydroclimate

try:
    import shap
except ImportError as e:
    raise ImportError(
        "Missing dependency 'shap'. Install via requirements.txt (shap>=0.42)."
    ) from e


OUTPUT_DIR = str(ANALYSIS_DIR / "analysis_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WQ_FILE = str(REPO_ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv")
LULC_FILE = str(ANALYSIS_DIR / "cache" / "lulc_metrics_corrected_16km.csv")

TARGET = "COD (mg/L)"


def _load_tons_day(c_mg_l: float, q_m3_s: float) -> float:
    return float(c_mg_l * q_m3_s * 0.0864)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master_file = str(ANALYSIS_DIR / "cache" / "station_year_master_latest.csv")
    if os.path.exists(WQ_FILE) and os.path.exists(LULC_FILE):
        df_wq = pd.read_csv(WQ_FILE)
        df_wq["year"] = pd.to_datetime(df_wq["datetime"], errors="coerce").dt.year
        df_lulc = pd.read_csv(LULC_FILE)
        return df_wq, df_lulc, pd.DataFrame()
    if os.path.exists(master_file):
        master = pd.read_csv(master_file)
        return pd.DataFrame(), pd.DataFrame(), master
    raise FileNotFoundError(WQ_FILE)


def build_bentre_station_year(df_wq: pd.DataFrame, df_lulc: pd.DataFrame, master: pd.DataFrame | None = None) -> pd.DataFrame:
    if master is not None and not master.empty:
        merged = master[(master["year"] >= 2013) & (master["year"] <= 2020)].copy()
        merged = merged[merged["stationName"].astype(str).str.match(r"^(HL|CC|MT|BL)\d{2}$")].copy()
        merged[TARGET] = merged["COD (mg/L)_annual"]
        merged["Salinity"] = merged["Salinity_annual"]
        merged["Q_MyThuan_m3s"] = merged["Q_MyThuan_AnnualMean_m3s"]
    else:
        # Annual mean concentrations
        wq_annual = (
            df_wq.groupby(["stationName", "River", "year"], as_index=False)
            .mean(numeric_only=True)
        )

        if TARGET not in wq_annual.columns:
            raise ValueError(f"Missing {TARGET} in water quality annual table")

        # Merge with LULC
        merged = pd.merge(wq_annual, df_lulc, on=["stationName", "year"], how="inner")

        # Match the main attribution window
        merged = merged[(merged["year"] >= 2013) & (merged["year"] <= 2020)].copy()

        # Ben Tre station IDs
        merged = merged[merged["stationName"].astype(str).str.match(r"^(HL|CC|MT|BL)\d{2}$")].copy()

        # Join My Thuan mainstem annual discharge from Gold Standard.
        h = join_branch_hydroclimate(merged[["stationName", "year"]].copy())
        hcols = HydroClimateColumns()
        merged["Q_MyThuan_m3s"] = h[hcols.q_mythuan_annual].values

    merged = merged.dropna(subset=[TARGET, "Q_MyThuan_m3s"]).copy()

    # Response variants
    merged["COD_load_tons_day"] = merged.apply(lambda r: _load_tons_day(r[TARGET], r["Q_MyThuan_m3s"]), axis=1)

    # Flow-normalization (annual): residuals from a simple linear fit C ~ Q
    x = merged["Q_MyThuan_m3s"].astype(float).to_numpy(dtype=float)
    y = merged[TARGET].astype(float).to_numpy(dtype=float)
    # y = a + b x
    x_var = float(np.var(x, ddof=0))
    if x_var > 0:
        b = float(np.cov(x, y, ddof=0)[0, 1] / x_var)
    else:
        b = 0.0
    a = float(np.mean(y) - b * float(np.mean(x)))
    merged["COD_flow_norm_resid"] = y - (a + b * x)

    return merged


def shap_rankings(df: pd.DataFrame, y_col: str, feature_cols: List[str], random_state: int = 42) -> pd.DataFrame:
    X = df[feature_cols].copy()
    y = df[y_col].astype(float).values

    # Drop rows with missing
    good = np.isfinite(y)
    X_np = X.to_numpy(dtype=float)
    good &= np.isfinite(X_np).all(axis=1)
    X = X.loc[good]
    y = y[good]

    # RF (same general settings as manuscript)
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # mean absolute SHAP per feature
    mean_abs = np.abs(shap_values).mean(axis=0)

    out = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    out["response"] = y_col
    return out


def main() -> None:
    df_wq, df_lulc, master = load_data()
    df = build_bentre_station_year(df_wq, df_lulc, master)

    # Match the manuscript baseline predictor set used in figure_rda_shap.py
    candidate_cols = [
        "Residential_Urban_composition_pct",
        "Residential_Urban_config_PD",
        "Aquaculture_composition_pct",
        "Aquaculture_config_ED",
        "Rice_Cultivation_composition_pct",
        "Rice_Cultivation_config_PD",
    ]

    feature_cols = [c for c in candidate_cols if c in df.columns]
    if len(feature_cols) != len(candidate_cols):
        missing = [c for c in candidate_cols if c not in feature_cols]
        raise RuntimeError(f"Missing baseline LULC predictors required for robustness check: {missing}")

    results = []
    for y_col in [TARGET, "COD_load_tons_day", "COD_flow_norm_resid"]:
        r = shap_rankings(df, y_col=y_col, feature_cols=feature_cols)
        # keep top 6 for compact table
        r = r.head(6).copy()
        r["rank"] = np.arange(1, len(r) + 1)
        results.append(r)

    out = pd.concat(results, ignore_index=True)

    # Wide format (ranked features for each response)
    wide_rows = []
    for y_col, grp in out.groupby("response"):
        for _, row in grp.iterrows():
            wide_rows.append({
                "Response": y_col,
                "Rank": int(row["rank"]),
                "Feature": row["feature"],
                "MeanAbsSHAP": float(row["mean_abs_shap"]),
            })

    wide = pd.DataFrame(wide_rows)
    out_path = os.path.join(OUTPUT_DIR, "cod_shap_robustness_top_features_latest.csv")
    wide.to_csv(out_path, index=False)
    print(f"Saved SHAP robustness table: {out_path}")


if __name__ == "__main__":
    main()
