#!/usr/bin/env python3
"""analyze_confirmatory_cod_fixed_effects.py

Confirmatory fixed-effects regression check alongside RF+SHAP attribution.

Model (19 Ben Tre stations, complete-case, 2005-2020):
    COD ~ Urban_PD + Aquaculture_ED + Rice_PD + Salinity + Year + station fixed effects

- MRC stations excluded (systematic permanganate/dichromate COD offset).
- Station fixed effects (C(stationName)) control for time-invariant site baselines.
- Robust (HC3) standard errors are used.

Outputs (analysis/analysis_outputs):
- cod_confirmatory_fixed_effects_latest.csv

Interpretation:
- Coefficients reflect conditional associations in the observational panel.
- This is a confirmatory robustness check, not causal proof.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(ANALYSIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from analysis.lib.hydro_ingest import HydroClimateColumns, join_branch_hydroclimate
except ModuleNotFoundError:
    from lib.hydro_ingest import HydroClimateColumns, join_branch_hydroclimate


OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "analysis_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WQ_FILE = os.path.join(REPO_ROOT, "data", "mekong_delta_water_quality_1985_2023.csv")
LULC_FILE = os.path.join(ANALYSIS_DIR, "cache", "lulc_metrics_corrected_16km.csv")
MASTER_FILE = os.path.join(ANALYSIS_DIR, "cache", "station_year_master_latest.csv")

TARGET = "COD (mg/L)"


def zscore(s: pd.Series) -> pd.Series:
    s2 = s.astype(float)
    mu = float(s2.mean())
    sd = float(s2.std(ddof=0))
    if sd == 0.0 or not np.isfinite(sd):
        return s2 * np.nan
    return (s2 - mu) / sd


def main() -> None:
    MRC = {"Tan Chau", "Can Tho", "My Thuan", "My Tho"}
    hcols = HydroClimateColumns()

    if os.path.exists(MASTER_FILE):
        df = pd.read_csv(MASTER_FILE)
        df = df[~df["stationName"].isin(MRC)].copy()
        df = df[(df["year"] >= 2005) & (df["year"] <= 2020)].copy()
        df[TARGET] = df["COD (mg/L)_annual"]
        df["Salinity"] = df["Salinity_annual"]
    elif os.path.exists(WQ_FILE) and os.path.exists(LULC_FILE):
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
        df = df[~df["stationName"].isin(MRC)].copy()
        h = join_branch_hydroclimate(df[["stationName", "year"]].copy())
        for c in [
            hcols.q_mythuan_annual_z,
            hcols.q_mythuan_dry_z,
            hcols.rain_annual_z,
            hcols.rain_dry_z,
        ]:
            if c in h.columns:
                df[c] = h[c].values
    else:
        raise FileNotFoundError(MASTER_FILE)

    # Variables (intentionally small set)
    needed = [
        TARGET,
        "Residential_Urban_config_PD",
        "Aquaculture_config_ED",
        "Rice_Cultivation_config_PD",
        "Salinity",
        "year",
        "stationName",
    ]
    # Hydro-climate columns are kept but NOT required for the Baseline model.
    hydro_cols = [c for c in [hcols.q_mythuan_annual_z, hcols.q_mythuan_dry_z,
                              hcols.rain_annual_z, hcols.rain_dry_z]
                  if c in df.columns]
    all_cols = needed + hydro_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for confirmatory model: {missing}")

    df = df[[c for c in all_cols if c in df.columns]].copy()
    # Drop only on core predictors (not hydro) so BL stations are retained for Baseline.
    df = df.dropna(subset=needed).copy()

    if df.shape[0] < 40:
        raise RuntimeError(f"Insufficient complete cases for fixed-effects regression (N={df.shape[0]}).")

    # Standardize continuous predictors for interpretability
    df["Urban_PD_z"] = zscore(df["Residential_Urban_config_PD"])
    df["Aqua_ED_z"] = zscore(df["Aquaculture_config_ED"])
    df["Rice_PD_z"] = zscore(df["Rice_Cultivation_config_PD"])
    df["Salinity_z"] = zscore(df["Salinity"])
    df["Year_z"] = zscore(df["year"])

    df = df.dropna(subset=["Urban_PD_z", "Aqua_ED_z", "Rice_PD_z", "Salinity_z", "Year_z"]).copy()

    # Baseline + hydro-climate controlled variants
    baseline_formula = "Q('COD (mg/L)') ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z + C(stationName)"

    # Add hydro-climate controls if present. We use anomalies (z-score) to represent interannual variability.
    hc_terms = []
    for c in [hcols.q_mythuan_annual_z, hcols.q_mythuan_dry_z, hcols.rain_annual_z, hcols.rain_dry_z]:
        if c in df.columns:
            hc_terms.append(f"Q('{c}')")

    controlled_formula = baseline_formula
    if hc_terms:
        controlled_formula = baseline_formula.replace(" + C(stationName)", "") + " + " + " + ".join(hc_terms) + " + C(stationName)"

    formulas = [("Baseline", baseline_formula)]
    if hc_terms:
        formulas.append(("HydroClimate", controlled_formula))

    out_parts = []
    robust_rows = []
    stability_rows = []
    for model_name, formula in formulas:
        # Ensure complete-case for any hydro-climate terms used
        if model_name == "HydroClimate" and hc_terms:
            needed_cols = [t.replace("Q('", "").replace("')", "") for t in hc_terms]
            df_fit = df.dropna(subset=needed_cols).copy()
        else:
            df_fit = df.copy()

        if df_fit.shape[0] < 40:
            raise RuntimeError(
                f"Insufficient complete cases for {model_name} fixed-effects regression (N={df_fit.shape[0]})."
            )

        # Fit once, then evaluate multiple covariance estimators on same coefficients
        ols_model = smf.ols(formula, data=df_fit).fit()

        # ----- Covariance robustness table -----
        terms = list(ols_model.params.index)
        params = pd.Series(ols_model.params.values, index=terms)

        # HC3
        hc3 = ols_model.get_robustcov_results(cov_type="HC3")
        bse_hc3 = pd.Series(hc3.bse, index=terms)
        t_hc3 = pd.Series(hc3.tvalues, index=terms)
        p_hc3 = pd.Series(hc3.pvalues, index=terms)

        # One-way clustered SE (station / year)
        station_codes = pd.Categorical(df_fit["stationName"]).codes.astype(int)
        year_codes = pd.Categorical(df_fit["year"]).codes.astype(int)

        cl_station = ols_model.get_robustcov_results(
            cov_type="cluster", groups=station_codes
        )
        bse_station = pd.Series(cl_station.bse, index=terms)
        p_station = pd.Series(cl_station.pvalues, index=terms)

        cl_year = ols_model.get_robustcov_results(
            cov_type="cluster", groups=year_codes
        )
        bse_year = pd.Series(cl_year.bse, index=terms)
        p_year = pd.Series(cl_year.pvalues, index=terms)

        # Two-way clustered covariance (station x year)
        cov_tw, _, _ = cov_cluster_2groups(
            ols_model,
            group=station_codes,
            group2=year_codes,
            use_correction=True,
        )
        bse_tw = pd.Series(np.sqrt(np.diag(cov_tw)), index=terms)
        t_tw = params / bse_tw
        p_tw = pd.Series(2 * stats.norm.sf(np.abs(t_tw)), index=terms)

        for term in terms:
            if term == "Intercept" or term.startswith("C("):
                continue
            clean_term = term.replace("Q('", "").replace("')", "")
            out_parts.append(
                pd.DataFrame(
                    [
                        {
                            "Model": model_name,
                            "term": clean_term,
                            "coef": round(float(params[term]), 3),
                            "se_HC3": round(float(bse_hc3[term]), 3),
                            "t": round(float(t_hc3[term]), 3),
                            "p": round(float(p_hc3[term]), 3),
                            "se_Cluster_Stn": round(float(bse_station[term]), 3),
                            "p_Cluster_Stn": round(float(p_station[term]), 3),
                            "se_Cluster_Yr": round(float(bse_year[term]), 3),
                            "p_Cluster_Yr": round(float(p_year[term]), 3),
                            "se_TwoWay": round(float(bse_tw[term]), 3),
                            "p_TwoWay": round(float(p_tw[term]), 3),
                            "N": int(ols_model.nobs),
                            "R2": round(float(ols_model.rsquared), 3),
                            "R2_adj": round(float(ols_model.rsquared_adj), 3),
                        }
                    ]
                )
            )

        # Long-format rows for direct SI table rendering
        cov_sets = [
            ("HC3", bse_hc3, p_hc3),
            ("Cluster_Station", bse_station, p_station),
            ("Cluster_Year", bse_year, p_year),
            ("Cluster_TwoWay", bse_tw, p_tw),
        ]
        for cov_name, bse_s, p_s in cov_sets:
            for term in terms:
                robust_rows.append(
                    {
                        "Model": model_name,
                        "term": term,
                        "cov_estimator": cov_name,
                        "coef": float(params[term]),
                        "se": float(bse_s[term]),
                        "t": float(params[term] / bse_s[term]) if np.isfinite(bse_s[term]) and bse_s[term] != 0 else np.nan,
                        "p": float(p_s[term]),
                        "significant_0_05": bool(float(p_s[term]) < 0.05),
                        "N": int(ols_model.nobs),
                        "R2": float(ols_model.rsquared),
                        "R2_adj": float(ols_model.rsquared_adj),
                    }
                )

        # Wide stability summary across covariance estimators
        for term in terms:
            se_vals = {
                "HC3": float(bse_hc3[term]),
                "Cluster_Station": float(bse_station[term]),
                "Cluster_Year": float(bse_year[term]),
                "Cluster_TwoWay": float(bse_tw[term]),
            }
            p_vals = {
                "HC3": float(p_hc3[term]),
                "Cluster_Station": float(p_station[term]),
                "Cluster_Year": float(p_year[term]),
                "Cluster_TwoWay": float(p_tw[term]),
            }
            sig_count = sum(v < 0.05 for v in p_vals.values())
            hc3_se = se_vals["HC3"]
            se_ratio = np.nan
            if np.isfinite(hc3_se) and hc3_se > 0:
                se_ratio = max(se_vals.values()) / hc3_se

            stability_rows.append(
                {
                    "Model": model_name,
                    "term": term,
                    "coef": float(params[term]),
                    "coef_sign": int(np.sign(params[term])) if np.isfinite(params[term]) else np.nan,
                    "se_HC3": se_vals["HC3"],
                    "se_Cluster_Station": se_vals["Cluster_Station"],
                    "se_Cluster_Year": se_vals["Cluster_Year"],
                    "se_Cluster_TwoWay": se_vals["Cluster_TwoWay"],
                    "p_HC3": p_vals["HC3"],
                    "p_Cluster_Station": p_vals["Cluster_Station"],
                    "p_Cluster_Year": p_vals["Cluster_Year"],
                    "p_Cluster_TwoWay": p_vals["Cluster_TwoWay"],
                    "significant_estimators_count": int(sig_count),
                    "se_max_to_hc3_ratio": float(se_ratio) if np.isfinite(se_ratio) else np.nan,
                    "N": int(ols_model.nobs),
                    "R2": float(ols_model.rsquared),
                    "R2_adj": float(ols_model.rsquared_adj),
                }
            )

    out = pd.concat(out_parts, ignore_index=True)

    out_path = os.path.join(OUTPUT_DIR, "cod_confirmatory_fixed_effects_latest.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved confirmatory fixed-effects regression: {out_path}")

    # Export covariance robustness artifacts for SI transparency
    if robust_rows:
        robust_df = pd.DataFrame(robust_rows)
        robust_path = os.path.join(
            OUTPUT_DIR,
            "cod_confirmatory_fixed_effects_covariance_robustness_latest.csv",
        )
        robust_df.to_csv(robust_path, index=False)
        print(f"Saved FE covariance robustness (long): {robust_path}")

    if stability_rows:
        stability_df = pd.DataFrame(stability_rows)
        stability_path = os.path.join(
            OUTPUT_DIR,
            "cod_confirmatory_fixed_effects_covariance_stability_latest.csv",
        )
        stability_df.to_csv(stability_path, index=False)
        print(f"Saved FE covariance stability (wide): {stability_path}")


if __name__ == "__main__":
    main()
