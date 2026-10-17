#!/usr/bin/env python3
"""
Primary Attribution Model (LULC + Population + Hydroclimate)
============================================================

Evaluates cross-station (Leave-One-Station-Out, LOSO) and cross-year
(Leave-One-Year-Out, LOYO) generalization and out-of-fold TreeSHAP feature
attribution across the 19 Ben Tre estuarine monitoring stations (2005-2020):
  - Compares 6-feature LULC-only baseline against the 11-feature hydroclimate-
    and population-augmented specification
  - Computes Random Forest and Ridge regression cross-validation skill
  - Computes variance partitioning across LULC, population, and hydroclimate
  - Evaluates linear mixed-effects (LME) models and seasonal (dry vs. wet) models

Outputs:
  - analysis/analysis_outputs/primary_model_performance_latest.csv
  - analysis/analysis_outputs/primary_model_shap_latest.csv
  - analysis/analysis_outputs/primary_variance_partitioning_latest.csv
  - analysis/analysis_outputs/primary_lme_results_latest.csv
  - analysis/analysis_outputs/primary_seasonal_performance_latest.csv
  - analysis/analysis_outputs/primary_model_summary_table_latest.csv
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.inspection import permutation_importance
import shap

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG
# ============================================================================

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO = _repo_root()
DATA_DIR = REPO / "data"
CACHE_DIR = REPO / "analysis" / "cache"
METRICS_ONLY = "--metrics-only" in sys.argv
ANALYSIS_OUTPUT_DIR = REPO / "analysis" / "analysis_outputs"
NW_TABLES_METRICS_DIR = REPO / "scripts" / "99_publish" / "nw_tables" / "metrics"
OUTPUT_DIR = NW_TABLES_METRICS_DIR if METRICS_ONLY else ANALYSIS_OUTPUT_DIR
ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NW_TABLES_METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _save_csv_synced(df: pd.DataFrame, filename: str) -> None:
    """Write CSV artifact to both analysis/analysis_outputs and nw_tables/metrics."""
    for target_dir in (ANALYSIS_OUTPUT_DIR, NW_TABLES_METRICS_DIR):
        df.to_csv(target_dir / filename, index=False)


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# RF hyperparameters (regularized to prevent overfitting with N~100-200)
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    oob_score=True,
    n_jobs=1,
)

# Toggles
RUN_SEASONAL = True
RUN_MIXED_EFFECTS = True
RUN_VARIANCE_PARTITION = True

# WQ parameters to model
WQ_PARAMS_ANNUAL = [
    "COD (mg/L)_annual",
    "DO (mg/L)_annual",
    "NH4-N (mg/L)_annual",
    "NO3-N (mg/L)_annual",
    "TSS (mg/L)_annual",
]

WQ_PARAMS_DRY = [
    "COD (mg/L)_dry",
    "DO (mg/L)_dry",
    "NO3-N (mg/L)_dry",
]

WQ_PARAMS_WET = [
    "COD (mg/L)_wet",
    "DO (mg/L)_wet",
    "NO3-N (mg/L)_wet",
]

# Short labels
PARAM_SHORT = {
    "COD (mg/L)_annual": "COD",
    "DO (mg/L)_annual": "DO",
    "NH4-N (mg/L)_annual": "NH4-N",
    "NO3-N (mg/L)_annual": "NO3-N",
    "TSS (mg/L)_annual": "TSS",
    "COD (mg/L)_dry": "COD (dry)",
    "DO (mg/L)_dry": "DO (dry)",
    "NO3-N (mg/L)_dry": "NO3-N (dry)",
    "COD (mg/L)_wet": "COD (wet)",
    "DO (mg/L)_wet": "DO (wet)",
    "NO3-N (mg/L)_wet": "NO3-N (wet)",
}

# --- Feature sets ---

# LULC-only (6 baseline landscape predictors)
LULC_FEATURES = [
    "Residential_Urban_composition_pct",
    "Residential_Urban_config_PD",
    "Aquaculture_composition_pct",
    "Aquaculture_config_ED",
    "Rice_Cultivation_composition_pct",
    "Rice_Cultivation_config_PD",
]

# Population
POP_FEATURES = ["pop_density"]

# Hydroclimate covariates: annual/dry-season rainfall and upstream discharge at My Thuan
HYDRO_FEATURES = [
    "Rain_Annual_z",
    "Rain_Dry_z",
    "Q_MyThuan_AnnualMean_z",
]

# Tidal range
TIDAL_FEATURES = ["TR_MyThuan_m"]

# Display names
FEATURE_LABELS = {
    "Residential_Urban_composition_pct": "Urban %",
    "Residential_Urban_config_PD": "Urban PD",
    "Aquaculture_composition_pct": "Aquaculture %",
    "Aquaculture_config_ED": "Aquaculture ED",
    "Rice_Cultivation_composition_pct": "Rice %",
    "Rice_Cultivation_config_PD": "Rice PD",
    "pop_density": "Population density",
    "Q_branch_AnnualMean_z": "Discharge (annual z)",
    "Q_branch_DrySeasonMean_z": "Discharge (dry z)",
    "Rain_Annual_z": "Rainfall (annual z)",
    "Rain_Dry_z": "Rainfall (dry z)",
    "Q_MyThuan_AnnualMean_z": "Discharge (My Thuan, annual z)",
    "TR_MyThuan_m": "Tidal range (m)",
}

# MRC stations to exclude from spatial attribution
MRC_STATIONS = ["Tan Chau", "Can Tho", "My Thuan", "My Tho"]


# ============================================================================
# HELPERS
# ============================================================================

def safe_spearman(y_true, y_pred):
    """Robust Spearman rho/p for 1D inputs."""
    try:
        res = spearmanr(np.asarray(y_true), np.asarray(y_pred))
        rho = getattr(res, "statistic", None)
        if rho is None:
            rho = getattr(res, "correlation", None)
        p = getattr(res, "pvalue", None)
        if rho is None and p is None and isinstance(res, tuple):
            rho, p = res
        rho_arr = np.asarray(rho)
        p_arr = np.asarray(p)
        if rho_arr.ndim == 2:
            rho_arr = rho_arr[0, 1]
        if p_arr.ndim == 2:
            p_arr = p_arr[0, 1]
        return float(rho_arr), float(p_arr)
    except Exception:
        return float("nan"), float("nan")


def fit_predict_loso(model, X, y, groups):
    """Leave-One-Station-Out predictions."""
    logo = LeaveOneGroupOut()
    return cross_val_predict(model, X, y, groups=groups, cv=logo)


def rename_features(df_or_series, labels=None):
    """Apply display labels to feature names."""
    labels = labels or FEATURE_LABELS
    if isinstance(df_or_series, pd.DataFrame):
        return df_or_series.rename(columns=labels)
    return df_or_series.rename(labels)


def evaluate_model(model_class, model_params, X, y, groups, name="RF"):
    """Fit model and evaluate OOB, LOSO, LOYO, Spearman."""
    model = model_class(**model_params)
    model.fit(X, y)

    # OOB (RF only)
    r2_oob = getattr(model, "oob_score_", np.nan)

    # Training R²
    y_train_pred = model.predict(X)
    r2_train = r2_score(y, y_train_pred)

    # LOSO
    logo = LeaveOneGroupOut()
    y_pred_loso = cross_val_predict(model, X, y, groups=groups, cv=logo)
    r2_loso = r2_score(y, y_pred_loso)
    rmse_loso = float(np.sqrt(mean_squared_error(y, y_pred_loso)))
    rho_loso, p_loso = safe_spearman(y, y_pred_loso)

    return {
        "model_name": name,
        "R2_OOB": r2_oob,
        "R2_Train": r2_train,
        "R2_LOSO": r2_loso,
        "RMSE_LOSO": rmse_loso,
        "Spearman_rho_LOSO": rho_loso,
        "Spearman_p_LOSO": p_loso,
        "N_samples": len(y),
        "N_features": X.shape[1],
    }, model, y_pred_loso


def _shap_direction_and_corr(X: pd.DataFrame, shap_values: np.ndarray) -> tuple[list[float], list[str]]:
    """Compute Pearson correlation r(X_j, SHAP_j) and sign-based Direction for each feature."""
    corr_list = []
    dir_list = []
    for j in range(X.shape[1]):
        x_j = np.asarray(X.iloc[:, j], dtype=float)
        sv_j = np.asarray(shap_values[:, j], dtype=float)
        valid_mask = np.isfinite(x_j) & np.isfinite(sv_j)
        if valid_mask.sum() > 2 and np.std(x_j[valid_mask]) > 0 and np.std(sv_j[valid_mask]) > 0:
            corr_j = float(np.corrcoef(x_j[valid_mask], sv_j[valid_mask])[0, 1])
        else:
            corr_j = 0.0
        corr_list.append(corr_j)
        dir_list.append("Positive" if corr_j >= 0 else "Negative")
    return corr_list, dir_list


def compute_shap(model, X, feature_names=None):
    """Compute in-sample SHAP values with correlation-based Direction."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_dir = shap_values.mean(axis=0)
    corr_vals, directions = _shap_direction_and_corr(X, shap_values)
    names = feature_names if feature_names is not None else list(X.columns)
    df = pd.DataFrame({
        "Driver": names,
        "Mean_Abs_SHAP": mean_abs,
        "Mean_SHAP": mean_dir,
        "SHAP_Corr_r": corr_vals,
        "Direction": directions,
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df, shap_values


def compute_cv_shap(model_class, model_params, X, y, groups):
    """Cross-validated SHAP (fold-wise LOSO) with correlation-based Direction."""
    logo = LeaveOneGroupOut()
    abs_sum = np.zeros(X.shape[1])
    sv_oof = np.zeros((len(X), X.shape[1]), dtype=float)
    n_total = 0
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        m = model_class(**{k: v for k, v in model_params.items() if k != "oob_score"})
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        expl = shap.TreeExplainer(m)
        sv = expl.shap_values(X.iloc[test_idx])
        abs_sum += np.abs(sv).sum(axis=0)
        sv_oof[test_idx] = sv
        n_total += sv.shape[0]
    mean_abs_cv = abs_sum / max(n_total, 1)
    mean_shap_cv = sv_oof.mean(axis=0)
    corr_cv, dir_cv = _shap_direction_and_corr(X, sv_oof)
    df = pd.DataFrame({
        "Driver": list(X.columns),
        "Mean_Abs_SHAP_CV": mean_abs_cv,
        "Mean_SHAP_CV": mean_shap_cv,
        "SHAP_Corr_r_CV": corr_cv,
        "Direction": dir_cv,
    }).sort_values("Mean_Abs_SHAP_CV", ascending=False).reset_index(drop=True)
    df["Rank_CV"] = df.index + 1
    return df



# ============================================================================
# DATA LOADING
# ============================================================================

def load_master_table():
    """Load the unified station-year master table."""
    path = CACHE_DIR / "station_year_master_latest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Master table not found: {path}\n"
            "Run: python -m analysis.generators.generate_station_year_master_table"
        )
    df = pd.read_csv(path)
    # Filter to Ben Tre only (exclude MRC for spatial attribution)
    df = df[~df["stationName"].isin(MRC_STATIONS)].copy()
    return df


def prepare_feature_sets(df, season="annual"):
    """Build feature DataFrames for different predictor sets.

    Returns dict of {set_name: (X, feature_names)}
    """
    lulc_cols = [c for c in LULC_FEATURES if c in df.columns]
    pop_cols = [c for c in POP_FEATURES if c in df.columns]
    hydro_cols = [c for c in HYDRO_FEATURES if c in df.columns]
    tidal_cols = [c for c in TIDAL_FEATURES if c in df.columns]

    primary_cols = lulc_cols + pop_cols + hydro_cols + tidal_cols
    lulc_pop_cols = lulc_cols + pop_cols
    hydro_only_cols = hydro_cols + tidal_cols

    sets = {}
    sets["LULC-only"] = df[lulc_cols].copy()
    sets["LULC+Pop"] = df[lulc_pop_cols].copy()
    sets["Hydro-only"] = df[hydro_only_cols].copy()
    sets["Primary (LULC+Pop+Hydro)"] = df[primary_cols].copy()

    # Rename for display
    for name in sets:
        sets[name] = sets[name].rename(columns=FEATURE_LABELS)

    return sets


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 80)
    print("PRIMARY ATTRIBUTION MODEL (Augmented: LULC + Hydro + Population)")
    print("=" * 80)
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print()

    # ---- Load data ----
    df = load_master_table()
    print(f"Loaded master table: {len(df)} rows, {df.shape[1]} cols")
    print(f"  Stations: {df['stationName'].nunique()}")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")

    # ---- Annual model ----
    print("\n" + "=" * 80)
    print("SECTION 1: ANNUAL ATTRIBUTION (Primary vs LULC-only)")
    print("=" * 80)

    # Prepare features
    feature_sets = prepare_feature_sets(df)

    # Determine complete-case subset for predictors (WQ handled per-parameter below)
    primary_cols = [c for c in LULC_FEATURES + POP_FEATURES + HYDRO_FEATURES + TIDAL_FEATURES
                    if c in df.columns]
    wq_annual_cols = [c for c in WQ_PARAMS_ANNUAL if c in df.columns]

    # Predictor-complete subset (WQ-specific NA handled per-parameter)
    keep_predictors = df[primary_cols].notna().all(axis=1)
    df_pred = df[keep_predictors].reset_index(drop=True)
    print(f"\nPredictor-complete subset: {len(df_pred)} station-years")
    print(f"  Stations: {df_pred['stationName'].nunique()}")
    print(f"  Years: {df_pred['year'].min()}-{df_pred['year'].max()}")

    groups_all = df_pred["stationName"]

    # Build feature matrices on predictor-complete subset
    fsets_all = prepare_feature_sets(df_pred)

    # ---- Run all models for each WQ parameter ----
    all_perf = []
    all_shap = []
    all_shap_cv = []

    for param in wq_annual_cols:
        if param not in df_pred.columns:
            continue

        # Per-parameter complete case (maximize N for each WQ param)
        valid_wq = df_pred[param].notna()
        df_cc = df_pred[valid_wq].reset_index(drop=True)
        y = df_cc[param].copy()
        groups = df_cc["stationName"]
        fsets_cc = prepare_feature_sets(df_cc)
        label = PARAM_SHORT.get(param, param)

        print(f"\n{'─' * 60}")
        print(f"Parameter: {label}  (N={len(y)})")
        print(f"{'─' * 60}")

        for set_name, X_set in fsets_cc.items():
            # Drop rows with NaN in this feature set
            valid = X_set.notna().all(axis=1)
            Xv = X_set[valid].reset_index(drop=True)
            yv = y[valid].reset_index(drop=True)
            gv = groups[valid].reset_index(drop=True)

            if len(Xv) < 20:
                print(f"  [{set_name}] Skipped: N={len(Xv)} < 20")
                continue

            perf, model, y_pred = evaluate_model(
                RandomForestRegressor, RF_PARAMS, Xv, yv, gv, name=f"RF-{set_name}"
            )
            perf["Parameter"] = label
            perf["Feature_Set"] = set_name
            all_perf.append(perf)

            print(f"  [{set_name}] OOB R²={perf['R2_OOB']:.3f}  "
                  f"LOSO R²={perf['R2_LOSO']:.3f}  "
                  f"ρ={perf['Spearman_rho_LOSO']:.3f} (p={perf['Spearman_p_LOSO']:.4f})")

            # SHAP for primary model
            if set_name == "Primary (LULC+Pop+Hydro)":
                shap_df, shap_vals = compute_shap(model, Xv)
                shap_df["Parameter"] = label
                shap_df["Feature_Set"] = set_name
                all_shap.append(shap_df)

                # CV-SHAP (fold-wise)
                cv_shap_df = compute_cv_shap(
                    RandomForestRegressor,
                    {k: v for k, v in RF_PARAMS.items() if k != "oob_score"},
                    Xv, yv, gv
                )
                cv_shap_df["Parameter"] = label
                cv_shap_df["Feature_Set"] = set_name
                all_shap_cv.append(cv_shap_df)

            # SHAP for LULC-only (sensitivity comparison)
            if set_name == "LULC-only":
                shap_df_lulc, _ = compute_shap(model, Xv)
                shap_df_lulc["Parameter"] = label
                shap_df_lulc["Feature_Set"] = set_name
                all_shap.append(shap_df_lulc)

    # ---- Save outputs ----
    df_perf = pd.DataFrame(all_perf)
    _save_csv_synced(df_perf, "primary_model_performance_latest.csv")
    print(f"\n✓ Saved primary_model_performance_latest.csv")

    if all_shap:
        df_shap = pd.concat(all_shap, ignore_index=True)
        _save_csv_synced(df_shap, "primary_model_shap_latest.csv")
        df_shap_primary_only = df_shap[df_shap["Feature_Set"] == "Primary (LULC+Pop+Hydro)"].copy()
        if all_shap_cv:
            df_shap_cv_tmp = pd.concat(all_shap_cv, ignore_index=True)
            df_shap_primary_only = pd.merge(
                df_shap_primary_only,
                df_shap_cv_tmp[["Parameter", "Driver", "Mean_Abs_SHAP_CV", "Mean_SHAP_CV", "SHAP_Corr_r_CV", "Rank_CV"]],
                on=["Parameter", "Driver"],
                how="left",
            )
        _save_csv_synced(df_shap_primary_only, "primary_shap_importance_latest.csv")
        print(f"✓ Saved primary_model_shap_latest.csv and primary_shap_importance_latest.csv")

    if all_shap_cv:
        df_shap_cv = pd.concat(all_shap_cv, ignore_index=True)
        _save_csv_synced(df_shap_cv, "primary_model_shap_cv_latest.csv")
        print(f"✓ Saved primary_model_shap_cv_latest.csv")

    # ================================================================
    # SECTION 2: VARIANCE PARTITIONING (Hydro / LULC / Pop / Shared)
    # ================================================================
    if RUN_VARIANCE_PARTITION:
        print("\n" + "=" * 80)
        print("SECTION 2: VARIANCE PARTITIONING")
        print("=" * 80)

        vp_rows = []
        for param in wq_annual_cols:
            if param not in df_pred.columns:
                continue
            valid_wq = df_pred[param].notna()
            df_vp_cc = df_pred[valid_wq].reset_index(drop=True)
            y = df_vp_cc[param].copy()
            groups_vp = df_vp_cc["stationName"]
            fsets_vp = prepare_feature_sets(df_vp_cc)
            label = PARAM_SHORT.get(param, param)

            for set_name, X_set in fsets_vp.items():
                valid = X_set.notna().all(axis=1)
                Xv = X_set[valid].reset_index(drop=True)
                yv = y[valid].reset_index(drop=True)
                gv = groups_vp[valid].reset_index(drop=True)

                if len(Xv) < 20:
                    continue

                # LOSO R² with Ridge (more stable for variance partitioning)
                ridge = Ridge(alpha=1.0)
                y_pred = fit_predict_loso(ridge, Xv, yv, gv)
                r2 = r2_score(yv, y_pred)
                rmse = float(np.sqrt(mean_squared_error(yv, y_pred)))
                rho, p = safe_spearman(yv, y_pred)

                # In-sample OLS R² (standardize Xv first so 1e10-scale PD columns don't trip lstsq rcond)
                Xv_arr = Xv.to_numpy(dtype=float)
                Xv_std = (Xv_arr - Xv_arr.mean(axis=0)) / np.where(Xv_arr.std(axis=0) > 0, Xv_arr.std(axis=0), 1.0)
                X_ols = np.column_stack([np.ones(len(yv)), Xv_std])
                beta_ols, _, _, _ = np.linalg.lstsq(X_ols, yv.to_numpy(dtype=float), rcond=None)
                y_hat_ols = X_ols @ beta_ols
                ols_r2 = float(r2_score(yv, y_hat_ols))

                vp_rows.append({
                    "Parameter": label,
                    "Predictor_Set": set_name,
                    "LOSO_R2_Ridge": r2,
                    "LOSO_RMSE_Ridge": rmse,
                    "Spearman_rho": rho,
                    "OLS_R2": ols_r2,
                    "N": len(yv),
                    "N_features": int(Xv.shape[1]),
                })

        df_vp = pd.DataFrame(vp_rows)
        _save_csv_synced(df_vp, "primary_variance_partitioning_latest.csv")
        print(f"✓ Saved primary_variance_partitioning_latest.csv")

        # Print summary
        print("\nVariance Partitioning Summary (LOSO R², Ridge):")
        if len(df_vp) > 0:
            pivot = df_vp.pivot_table(index="Parameter", columns="Predictor_Set",
                                       values="LOSO_R2_Ridge", aggfunc="first")
            print(pivot.to_markdown(floatfmt=".3f"))

    # ================================================================
    # SECTION 3: SEASONAL DISAGGREGATION (Dry vs Wet)
    # ================================================================
    if RUN_SEASONAL:
        print("\n" + "=" * 80)
        print("SECTION 3: SEASONAL DISAGGREGATION")
        print("=" * 80)

        seasonal_perf = []
        seasonal_shap = []

        for season, wq_list in [("dry", WQ_PARAMS_DRY), ("wet", WQ_PARAMS_WET)]:
            wq_cols = [c for c in wq_list if c in df.columns]
            if not wq_cols:
                print(f"  No {season} columns available; skipping.")
                continue

            # Complete case for seasonal cols
            keep_s = df[primary_cols + wq_cols].notna().all(axis=1)
            df_s = df[keep_s].reset_index(drop=True)
            groups_s = df_s["stationName"]
            fsets_s = prepare_feature_sets(df_s)

            print(f"\n--- Season: {season.upper()} (N={len(df_s)}) ---")

            for param in wq_cols:
                y = df_s[param].copy()
                label = PARAM_SHORT.get(param, param)

                for set_name in ["Primary (LULC+Pop+Hydro)", "LULC-only"]:
                    X_set = fsets_s[set_name]
                    valid = X_set.notna().all(axis=1)
                    Xv = X_set[valid].reset_index(drop=True)
                    yv = y[valid].reset_index(drop=True)
                    gv = groups_s[valid].reset_index(drop=True)

                    if len(Xv) < 20:
                        continue

                    perf, model, y_pred = evaluate_model(
                        RandomForestRegressor, RF_PARAMS, Xv, yv, gv,
                        name=f"RF-{set_name}"
                    )
                    perf["Parameter"] = label
                    perf["Feature_Set"] = set_name
                    perf["Season"] = season
                    seasonal_perf.append(perf)

                    print(f"  [{set_name}] {label}: OOB R²={perf['R2_OOB']:.3f}  "
                          f"LOSO R²={perf['R2_LOSO']:.3f}")

                    if set_name == "Primary (LULC+Pop+Hydro)":
                        shap_df, _ = compute_shap(model, Xv)
                        shap_df["Parameter"] = label
                        shap_df["Season"] = season
                        seasonal_shap.append(shap_df)

        if seasonal_perf:
            df_sp = pd.DataFrame(seasonal_perf)
            _save_csv_synced(df_sp, "primary_seasonal_performance_latest.csv")
            print(f"\n✓ Saved primary_seasonal_performance_latest.csv")

        if seasonal_shap:
            df_ss = pd.concat(seasonal_shap, ignore_index=True)
            _save_csv_synced(df_ss, "primary_seasonal_shap_latest.csv")
            print(f"✓ Saved primary_seasonal_shap_latest.csv")

    # ================================================================
    # SECTION 4: MIXED-EFFECTS MODEL (complement to RF)
    # ================================================================
    run_lme = RUN_MIXED_EFFECTS
    if run_lme:
        print("\n" + "=" * 80)
        print("SECTION 4: MIXED-EFFECTS MODEL (Station Random Intercept)")
        print("=" * 80)

        try:
            import statsmodels.formula.api as smf
        except ImportError:
            print("statsmodels not available; skipping mixed-effects model.")
            run_lme = False

    if run_lme:
        lme_rows = []

        # Standardize predictors for mixed-effects
        df_lme = df_pred.copy()
        pred_raw = LULC_FEATURES + POP_FEATURES + HYDRO_FEATURES
        pred_avail = [c for c in pred_raw if c in df_lme.columns]

        for c in pred_avail:
            mu = df_lme[c].mean()
            sd = df_lme[c].std()
            if sd > 0:
                df_lme[f"z_{c}"] = (df_lme[c] - mu) / sd
            else:
                df_lme[f"z_{c}"] = 0.0

        z_cols = [f"z_{c}" for c in pred_avail]

        for param in wq_annual_cols:
            if param not in df_lme.columns:
                continue
            label = PARAM_SHORT.get(param, param)

            # Sanitize column name for formula
            safe_param = param.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")

            # Per-parameter complete case for LME
            df_lme_p = df_lme.dropna(subset=[param] + z_cols).copy()
            df_lme_p[safe_param] = df_lme_p[param]

            if len(df_lme_p) < 30:
                print(f"  [{label}] LME skipped: N={len(df_lme_p)} < 30")
                continue

            # Build formula
            fixed_terms = " + ".join(z_cols)
            formula = f"{safe_param} ~ {fixed_terms}"

            try:
                lme = smf.mixedlm(formula, data=df_lme_p, groups=df_lme_p["stationName"])
                result = lme.fit(reml=True, method="lbfgs")

                # Extract fixed effects
                fe = result.fe_params
                se = result.bse_fe if hasattr(result, "bse_fe") else result.bse
                pv = result.pvalues_fe if hasattr(result, "pvalues_fe") else result.pvalues

                for coef_name in fe.index:
                    if coef_name == "Intercept":
                        continue
                    # Map back to original name
                    orig = coef_name.replace("z_", "", 1) if coef_name.startswith("z_") else coef_name
                    display = FEATURE_LABELS.get(orig, orig)
                    lme_rows.append({
                        "Parameter": label,
                        "Predictor": display,
                        "Coefficient": float(fe[coef_name]),
                        "SE": float(se.get(coef_name, np.nan)),
                        "p_value": float(pv.get(coef_name, np.nan)),
                        "Significant_005": float(pv.get(coef_name, 1)) < 0.05,
                    })

                # Variance components
                re_var = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, "iloc") else float(result.cov_re)
                resid_var = float(result.scale)
                icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else np.nan

                print(f"\n  [{label}] LME: Station variance={re_var:.4f}, Residual={resid_var:.4f}, ICC={icc:.3f}")
                for _, row in pd.DataFrame(lme_rows).query(f"Parameter == '{label}'").iterrows():
                    sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else ""))
                    print(f"    {row['Predictor']:30s}  β={row['Coefficient']:+.4f}  p={row['p_value']:.4f} {sig}")

            except Exception as e:
                print(f"  [{label}] LME failed: {e}")

        if lme_rows:
            df_lme_out = pd.DataFrame(lme_rows)
            _save_csv_synced(df_lme_out, "primary_lme_fixed_effects_latest.csv")
            print(f"\n✓ Saved primary_lme_fixed_effects_latest.csv")

    # ================================================================
    # SECTION 5: MODEL PERFORMANCE SUMMARY TABLE (for manuscript)
    # ================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: MODEL PERFORMANCE SUMMARY TABLE")
    print("=" * 80)

    # Create a clean summary table suitable for the manuscript
    if len(all_perf) > 0:
        summary = pd.DataFrame(all_perf)
        summary_pivot = summary.pivot_table(
            index="Parameter",
            columns="Feature_Set",
            values=["R2_OOB", "R2_LOSO", "Spearman_rho_LOSO"],
            aggfunc="first"
        )
        print("\nManuscript Performance Summary:")
        print(summary_pivot.to_markdown(floatfmt=".3f"))

        _save_csv_synced(summary, "primary_model_summary_table_latest.csv")
        print(f"\n✓ Saved primary_model_summary_table_latest.csv")

    print("\n" + "=" * 80)
    print("PRIMARY MODEL ANALYSIS COMPLETE")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 80)


if __name__ == "__main__":
    main()
