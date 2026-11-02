"""Phase 2: Model skill against naive baselines.

Evaluates RF (LULC-only, Hydro-only, Combined, Augmented) under LOSO/LOBO/LOYO
and compares against naive baselines (global mean, station mean, branch mean).

Outputs
-------
- analysis/analysis_outputs/model_skill_against_naive_latest.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.lib.cv_utils import evaluate_model_cv, naive_baselines, cv_metrics
from analysis.lib.hydro_ingest import HydroClimateColumns

OUT_DIR = ROOT / "analysis" / "analysis_outputs"
CACHE_DIR = ROOT / "analysis" / "cache"

# Feature sets
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

WQ_OUTCOMES = {
    "COD": "COD (mg/L)_annual",
    "DO": "DO (mg/L)_annual",
    "NO3": "NO3-N (mg/L)_annual",
}


def _make_scaler_preprocess(feature_cols):
    """Return a preprocess_fn that fits StandardScaler on train and transforms both."""
    def fn(X_train, X_test):
        sc = StandardScaler()
        Xtr = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        Xte = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)
        return Xtr, Xte
    return fn


def run(master_path: Path | None = None) -> pd.DataFrame:
    if master_path is None:
        master_path = CACHE_DIR / "station_year_master_latest.csv"
    df = pd.read_csv(master_path)

    # Filter to Ben Tre only (attribution scope)
    df_bt = df[df["network"] == "Ben Tre"].copy()

    all_rows = []

    for outcome_label, outcome_col in WQ_OUTCOMES.items():
        if outcome_col not in df_bt.columns:
            continue

        # Drop missing outcome
        sub = df_bt.dropna(subset=[outcome_col]).copy()
        y = sub[outcome_col].values.astype(float)
        stations = sub["stationName"].values
        branches = sub["branch_code"].values
        years = sub["year"].values

        # Naive baselines
        baselines = naive_baselines(y, stations, branches)
        for bname, bm in baselines.items():
            bm.update({"outcome": outcome_label, "model_label": f"naive_{bname}",
                        "CV_scheme": "naive", "feature_set": "none"})
            all_rows.append(bm)

        # Feature sets to evaluate
        feature_sets = {
            "LULC_only": LULC_FEATURES,
            "Hydro_only": HYDRO_FEATURES,
            "LULC+Hydro": LULC_FEATURES + HYDRO_FEATURES,
            "LULC+Hydro+Pop": LULC_FEATURES + HYDRO_FEATURES + POP_FEATURES,
        }

        for fs_name, fs_cols in feature_sets.items():
            avail_cols = [c for c in fs_cols if c in sub.columns]
            if len(avail_cols) < 2:
                continue

            sub_fs = sub.dropna(subset=avail_cols + [outcome_col])
            if len(sub_fs) < 20:
                continue

            X = sub_fs[avail_cols].astype(float)
            y_fs = sub_fs[outcome_col].values.astype(float)
            st_fs = sub_fs["stationName"].values
            br_fs = sub_fs["branch_code"].values
            yr_fs = sub_fs["year"].values

            rf = RandomForestRegressor(n_estimators=500, random_state=42,
                                        oob_score=False, n_jobs=-1)
            preproc = _make_scaler_preprocess(avail_cols)

            rows = evaluate_model_cv(
                rf, X, y_fs, st_fs, br_fs, yr_fs,
                preprocess_fn=preproc, label=f"RF_{fs_name}",
            )
            for r in rows:
                r.update({"outcome": outcome_label, "feature_set": fs_name})
            all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    return result


def main():
    result = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "model_skill_against_naive_latest.csv"
    result.to_csv(out_path, index=False)
    print(f"[Phase 2] Wrote {out_path}  ({result.shape[0]} rows)")
    print(result.to_string())


if __name__ == "__main__":
    main()
