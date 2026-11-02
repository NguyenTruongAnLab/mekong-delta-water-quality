"""Phase 6: Cross-network harmonization via within-network standardized anomalies.

Implements within-network log-scale standardized anomalies for COD/DO/NO3/NH4,
then fits models on the pooled anomaly dataset with network as a factor.
This allows using both MRC + Ben Tre without pretending raw mg/L values are comparable.

Outputs
-------
- analysis/analysis_outputs/cross_network_anomaly_models_latest.csv
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
    HydroClimateColumns().rain_annual_z,
    HydroClimateColumns().rain_dry_z,
    HydroClimateColumns().q_mythuan_annual_z,
    HydroClimateColumns().q_mythuan_dry_z,
]
POP_FEATURES = ["pop_density"]

WQ_PARAMS = {
    "COD": "COD (mg/L)_annual",
    "DO": "DO (mg/L)_annual",
    "NO3": "NO3-N (mg/L)_annual",
    "NH4": "NH4-N (mg/L)_annual",
}


def _scaler_preprocess(X_train, X_test):
    sc = StandardScaler()
    Xtr = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)
    return Xtr, Xte


def compute_within_network_anomalies(
    df: pd.DataFrame,
    param_col: str,
    network_col: str = "network",
) -> pd.Series:
    """Standardize within each network on log scale."""
    anomaly = pd.Series(np.nan, index=df.index, dtype=float)
    for net in df[network_col].unique():
        mask = df[network_col] == net
        vals = df.loc[mask, param_col]
        # log1p to handle zeros
        log_vals = np.log1p(vals.clip(lower=0))
        mu = log_vals.mean()
        sd = log_vals.std()
        if sd > 0:
            anomaly.loc[mask] = (log_vals - mu) / sd
    return anomaly


def run(master_path: Path | None = None) -> pd.DataFrame:
    if master_path is None:
        master_path = CACHE_DIR / "station_year_master_latest.csv"
    df = pd.read_csv(master_path)

    all_rows = []

    for param_label, param_col in WQ_PARAMS.items():
        if param_col not in df.columns:
            continue

        # Compute within-network anomaly
        df[f"{param_label}_anomaly"] = compute_within_network_anomalies(df, param_col)
        target_col = f"{param_label}_anomaly"

        # Features: LULC + hydro + pop + network indicator
        avail_features = [c for c in LULC_FEATURES + HYDRO_FEATURES + POP_FEATURES
                          if c in df.columns]
        # Add network as binary indicator
        df["_is_mrc"] = (df["network"] == "MRC").astype(float)

        sub = df.dropna(subset=avail_features + [target_col]).copy()
        if len(sub) < 20:
            continue

        X_without_net = sub[avail_features].astype(float)
        X_with_net = sub[avail_features + ["_is_mrc"]].astype(float)

        y = sub[target_col].values.astype(float)
        stations = sub["stationName"].values
        branches = sub["branch_code"].values
        years = sub["year"].values

        for X_df, fs_label in [(X_without_net, "no_network"), (X_with_net, "with_network")]:
            for model_cls, model_name in [
                (RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1), "RF"),
                (Ridge(alpha=1.0), "Ridge"),
            ]:
                rows = evaluate_model_cv(
                    model_cls, X_df, y, stations, branches, years,
                    preprocess_fn=_scaler_preprocess,
                    label=f"{model_name}_{fs_label}",
                )
                for r in rows:
                    r.update({
                        "outcome": param_label,
                        "model": model_name,
                        "feature_set": fs_label,
                        "scope": "Both_Networks_Anomaly",
                    })
                all_rows.extend(rows)

        # Also run Ben Tre-only on anomalies for comparison
        sub_bt = sub[sub["network"] == "Ben Tre"].copy()
        if len(sub_bt) >= 15:
            X_bt = sub_bt[avail_features].astype(float)
            y_bt = sub_bt[target_col].values.astype(float)
            st_bt = sub_bt["stationName"].values
            br_bt = sub_bt["branch_code"].values
            yr_bt = sub_bt["year"].values

            for model_cls, model_name in [
                (RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1), "RF"),
            ]:
                rows = evaluate_model_cv(
                    model_cls, X_bt, y_bt, st_bt, br_bt, yr_bt,
                    preprocess_fn=_scaler_preprocess,
                    label=f"{model_name}_BenTre_anomaly",
                )
                for r in rows:
                    r.update({
                        "outcome": param_label,
                        "model": model_name,
                        "feature_set": "LULC+Hydro+Pop",
                        "scope": "BenTre_Only_Anomaly",
                    })
                all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def main():
    result = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "cross_network_anomaly_models_latest.csv"
    result.to_csv(out_path, index=False)
    print(f"[Phase 6] Wrote {out_path}  ({result.shape[0]} rows)")
    if not result.empty:
        summary = result[result["CV_scheme"] == "LOSO"].pivot_table(
            index=["outcome", "scope"], columns=["model", "feature_set"],
            values="R2", aggfunc="first"
        )
        print("\n=== Cross-Network Anomaly LOSO R² ===")
        print(summary.to_string())


if __name__ == "__main__":
    main()
