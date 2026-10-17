"""Phase 4: Mixed-effects models for annual station-year data.

Fits statsmodels MixedLM with random intercept by station and fixed effects
for LULC + hydroclimate + population, then compares against RF and Ridge
under leakage-safe LOSO/LOBO.

Outputs
-------
- analysis/analysis_outputs/mixedlm_effects_latest.csv
- analysis/analysis_outputs/mixedlm_model_comparison_latest.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.lib.cv_utils import evaluate_model_cv, grouped_cv_predict, cv_metrics
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


def fit_mixedlm(df: pd.DataFrame, outcome_col: str,
                fixed_features: list[str]) -> pd.DataFrame | None:
    """Fit MixedLM with random intercept by station. Return fixed-effects table."""
    avail = [c for c in fixed_features if c in df.columns]
    sub = df.dropna(subset=avail + [outcome_col, "stationName"]).copy()
    if len(sub) < 20:
        return None

    # Standardize predictors for comparable coefficients
    scaler = StandardScaler()
    for c in avail:
        sub[f"z_{c}"] = scaler.fit_transform(sub[[c]])

    z_cols = [f"z_{c}" for c in avail]
    formula = f"Q('{outcome_col}') ~ " + " + ".join(z_cols)

    import warnings
    try:
        md = smf.mixedlm(formula, sub, groups=sub["stationName"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = md.fit(reml=True, method="lbfgs")
            except np.linalg.LinAlgError:
                result = md.fit(reml=True, method="powell")
            fe = result.summary().tables[1]
        # Parse
        coef_df = pd.DataFrame({
            "term": [t.replace("z_", "") if t != "Intercept" else t
                     for t in result.fe_params.index],
            "coef": result.fe_params.values,
            "se": result.bse_fe.values if hasattr(result, "bse_fe") else [np.nan]*len(result.fe_params),
            "pvalue": result.pvalues.reindex(result.fe_params.index).values if hasattr(result, "pvalues") else [np.nan]*len(result.fe_params),
        })
        coef_df["outcome"] = outcome_col

        # Variance components
        var_resid = result.scale
        var_station = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, "iloc") else float(result.cov_re)
        var_total = var_station + var_resid
        coef_df["var_station_pct"] = 100 * var_station / var_total if var_total > 0 else np.nan
        coef_df["var_resid_pct"] = 100 * var_resid / var_total if var_total > 0 else np.nan
        coef_df["N"] = len(sub)
        coef_df["N_stations"] = sub["stationName"].nunique()

        return coef_df
    except Exception as e:
        print(f"  MixedLM fit failed for {outcome_col}: {e}")
        return None


def run(master_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if master_path is None:
        master_path = CACHE_DIR / "station_year_master_latest.csv"
    df = pd.read_csv(master_path)
    df_bt = df[df["network"] == "Ben Tre"].copy()

    effect_tables = []
    comparison_rows = []

    for outcome_label, outcome_col in WQ_OUTCOMES.items():
        if outcome_col not in df_bt.columns:
            continue

        all_features = LULC_FEATURES + HYDRO_FEATURES + POP_FEATURES
        avail = [c for c in all_features if c in df_bt.columns]
        sub = df_bt.dropna(subset=avail + [outcome_col]).copy()
        if len(sub) < 20:
            continue

        # --- Mixed-effects model ---
        fe_table = fit_mixedlm(sub, outcome_col, avail)
        if fe_table is not None:
            fe_table["outcome_label"] = outcome_label
            effect_tables.append(fe_table)

        # --- Compare RF vs Ridge under LOSO/LOBO ---
        X = sub[avail].astype(float)
        y = sub[outcome_col].values.astype(float)
        stations = sub["stationName"].values
        branches = sub["branch_code"].values
        years = sub["year"].values

        for model_cls, model_name in [
            (RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1), "RF"),
            (Ridge(alpha=1.0), "Ridge"),
        ]:
            rows = evaluate_model_cv(
                model_cls, X, y, stations, branches, years,
                preprocess_fn=_scaler_preprocess,
                label=f"{model_name}_LULC+Hydro+Pop",
            )
            for r in rows:
                r.update({"outcome": outcome_label, "model": model_name})
            comparison_rows.extend(rows)

    effects_df = pd.concat(effect_tables, ignore_index=True) if effect_tables else pd.DataFrame()
    comparison_df = pd.DataFrame(comparison_rows)
    return effects_df, comparison_df


def main():
    effects_df, comparison_df = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not effects_df.empty:
        p1 = OUT_DIR / "mixedlm_effects_latest.csv"
        effects_df.to_csv(p1, index=False)
        print(f"[Phase 4] Wrote {p1}  ({effects_df.shape[0]} rows)")
        print(effects_df.to_string())

    if not comparison_df.empty:
        p2 = OUT_DIR / "mixedlm_model_comparison_latest.csv"
        comparison_df.to_csv(p2, index=False)
        print(f"[Phase 4] Wrote {p2}  ({comparison_df.shape[0]} rows)")
        print(comparison_df.to_string())


if __name__ == "__main__":
    main()
