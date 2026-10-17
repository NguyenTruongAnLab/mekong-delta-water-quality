"""Outcome decomposition: evaluate COD, DO, and NO3-N transformations
to find which response formulation yields the most robust spatial generalization.

Transformations tested:
1. Raw annual mean
2. log1p(annual mean)
3. Within-station anomaly (station-year mean - station long-term mean)
4. Flow-normalized residual (residual from C ~ Q_MyThuan annual regression)

Outputs
-------
- analysis/analysis_outputs/outcome_decomposition_skill_latest.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.lib.cv_utils import evaluate_model_cv
from analysis.lib.hydro_ingest import HydroClimateColumns

OUT_DIR = ROOT / "analysis" / "analysis_outputs"
CACHE_DIR = ROOT / "analysis" / "cache"

LULC_FEATURES = [
    "Residential_Urban_composition_pct", "Residential_Urban_config_PD",
    "Rice_Cultivation_composition_pct", "Rice_Cultivation_config_PD",
    "Aquaculture_composition_pct", "Aquaculture_config_ED",
]
HYDRO_FEATURES = [
    HydroClimateColumns().q_mythuan_annual_z,
    HydroClimateColumns().q_mythuan_dry_z,
    HydroClimateColumns().rain_annual_z,
    HydroClimateColumns().rain_dry_z,
]
ALL_FEATURES = LULC_FEATURES + HYDRO_FEATURES + ["pop_density"]

WQ_OUTCOMES = {
    "COD": "COD (mg/L)_annual",
    "DO": "DO (mg/L)_annual",
    "NO3": "NO3-N (mg/L)_annual",
}


def _scaler_preprocess(X_train, X_test):
    sc = StandardScaler()
    Xtr = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)
    return Xtr, Xte


def _compute_station_anomaly(df: pd.DataFrame, col: str) -> pd.Series:
    """Subtract station long-term mean from each station-year value."""
    means = df.groupby("stationName")[col].transform("mean")
    return df[col] - means


def _compute_flow_residual(df: pd.DataFrame, col: str) -> pd.Series:
    """Residual from simple OLS: C ~ Q_MyThuan_annual (within complete cases)."""
    q_col = HydroClimateColumns().q_mythuan_annual
    mask = df[[col, q_col]].notna().all(axis=1)
    resid = pd.Series(np.nan, index=df.index)
    if mask.sum() < 10:
        return resid
    from numpy.polynomial.polynomial import polyfit
    x = df.loc[mask, q_col].values.astype(float)
    y = df.loc[mask, col].values.astype(float)
    coefs = polyfit(x, y, 1)
    predicted = coefs[0] + coefs[1] * x
    resid.loc[mask] = y - predicted
    return resid


def run(master_path: Path | None = None) -> pd.DataFrame:
    if master_path is None:
        master_path = CACHE_DIR / "station_year_master_latest.csv"
    df = pd.read_csv(master_path)
    df_bt = df[df["network"] == "Ben Tre"].copy()

    all_rows = []

    for outcome_label, raw_col in WQ_OUTCOMES.items():
        if raw_col not in df_bt.columns:
            continue

        # Build outcome variants
        variants = {}
        variants["raw"] = df_bt[raw_col].copy()
        variants["log1p"] = np.log1p(df_bt[raw_col].clip(lower=0))
        variants["station_anomaly"] = _compute_station_anomaly(df_bt, raw_col)
        variants["flow_residual"] = _compute_flow_residual(df_bt, raw_col)

        for var_name, y_series in variants.items():
            sub = df_bt.copy()
            sub["_target"] = y_series

            # Feature availability
            avail = [c for c in ALL_FEATURES if c in sub.columns]
            sub_clean = sub.dropna(subset=avail + ["_target"])
            if len(sub_clean) < 20:
                continue

            X = sub_clean[avail].astype(float)
            y = sub_clean["_target"].values.astype(float)
            stations = sub_clean["stationName"].values
            branches = sub_clean["branch_code"].values
            years = sub_clean["year"].values

            for model_cls, model_name in [
                (RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1), "RF"),
                (Ridge(alpha=1.0), "Ridge"),
            ]:
                rows = evaluate_model_cv(
                    model_cls, X, y, stations, branches, years,
                    preprocess_fn=_scaler_preprocess,
                    label=f"{model_name}_{var_name}",
                )
                for r in rows:
                    r.update({
                        "outcome": outcome_label,
                        "transformation": var_name,
                        "model": model_name,
                        "feature_set": "LULC+Hydro+Pop",
                    })
                all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def main():
    result = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "outcome_decomposition_skill_latest.csv"
    result.to_csv(out_path, index=False)
    print(f"[Phase 3] Wrote {out_path}  ({result.shape[0]} rows)")
    # Summary
    if not result.empty:
        summary = result[result["CV_scheme"] == "LOSO"].pivot_table(
            index=["outcome", "transformation"], columns="model",
            values="R2", aggfunc="first"
        )
        print("\n=== LOSO R² by outcome × transformation ===")
        print(summary.to_string())


if __name__ == "__main__":
    main()
