#!/usr/bin/env python3
"""
Supplementary Table S5: Panel Fixed-Effects Regression for COD & Sensitivity Suite
==================================================================================

Computes from RAW DATA / Station-Year Master (2005–2020, 19 Ben Tre stations):
  1. Primary Confirmatory Station Fixed-Effects Models (N = 213):
     - Baseline: COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z + C(stationName)
     - HydroClimate: adds Q_MyThuan_AnnualMean_z + Q_MyThuan_DrySeasonMean_z + Rain_Annual_z + Rain_Dry_z
     Evaluated across 4 covariance estimators:
       (a) HC3 heteroskedasticity-robust SE
       (b) One-way station-clustered SE (G = 19 stations)
       (c) One-way year-clustered SE (T = 16 years)
       (d) Two-way station + year clustered SE (Cameron–Gelbach–Miller, 2011)

  2. Two-Way Fixed-Effects & Station-Varying Hydrodynamic Exposure Sensitivity Suite:
     - Pure Two-Way FE vs. Two-Way FE + LULC (N = 229 and N = 213; 3 config LULC and 6 full LULC),
       reporting classical nested F-test and joint Wald tests across all 4 covariance estimators.
     - Station-varying hydrodynamic exposure interactions (Salinity_z × Q_MyThuan and
       time-invariant station mean salinity StnSalMean_z × Q_MyThuan) in Station FE and Two-Way FE,
       plus parsimonious single-discharge specifications and joint hydro Wald tests.

Outputs:
  manuscript/tables/Supplementary_Table_S5_COD_FixedEffects.md
  scripts/99_publish/nw_tables/metrics/S5_cod_fixed_effects_metrics.csv
  scripts/99_publish/nw_tables/metrics/fe_hydro_interaction_sensitivity.csv
  scripts/99_publish/nw_tables/metrics/fe_twoway_lulc_joint_tests.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups

from _nw_common_tables import (
    LULC_VARS, load_master_bentre,
    ROOT, MANU_TABLES, METRICS_OUT, write_table_md, write_metrics_csv,
)

ANALYSIS_OUT = ROOT / "analysis" / "analysis_outputs"


def zscore(s: pd.Series) -> pd.Series:
    """Standardize to zero mean and unit variance (population SD, ddof=0)."""
    s2 = s.astype(float)
    mu = float(s2.mean())
    sd = float(s2.std(ddof=0))
    if sd == 0.0 or not np.isfinite(sd):
        return s2 * np.nan
    return (s2 - mu) / sd


def compute_4way_covs(
    ols_uncorr,
    df_fit: pd.DataFrame,
    station_col: str = "stationName",
    year_col: str = "year",
) -> dict[str, tuple[np.ndarray, float]]:
    """Return dictionary mapping SE estimator name -> (cov_matrix, df_resid_or_clusters).

    Estimators:
      - HC3: MacKinnon-White HC3 covariance, df = df_resid
      - Cluster_Stn: one-way station-clustered covariance, df = G_stn - 1
      - Cluster_Yr: one-way year-clustered covariance, df = G_yr - 1
      - TwoWay: Cameron-Gelbach-Miller (2011) two-way clustered covariance via
                statsmodels cov_cluster_2groups(use_correction=True),
                V_2way = V_stn + V_yr - V_{stn ∩ yr}
    """
    ols_hc3 = ols_uncorr.get_robustcov_results(cov_type="HC3")
    cov_hc3 = np.asarray(ols_hc3.cov_params(), dtype=float)

    stn_groups = pd.Categorical(df_fit[station_col]).codes.astype(int)
    yr_groups = pd.Categorical(df_fit[year_col]).codes.astype(int)
    n_stn = int(df_fit[station_col].nunique())
    n_yr = int(df_fit[year_col].nunique())

    cov_stn = np.asarray(cov_cluster(ols_uncorr, stn_groups), dtype=float)
    cov_yr = np.asarray(cov_cluster(ols_uncorr, yr_groups), dtype=float)
    cov_2way_raw, _, _ = cov_cluster_2groups(
        ols_uncorr,
        group=stn_groups,
        group2=yr_groups,
        use_correction=True,
    )
    cov_2way = np.asarray(cov_2way_raw, dtype=float)

    return {
        "HC3": (cov_hc3, float(ols_uncorr.df_resid)),
        "Cluster_Stn": (cov_stn, float(max(n_stn - 1, 1))),
        "Cluster_Yr": (cov_yr, float(max(n_yr - 1, 1))),
        "TwoWay": (cov_2way, float("inf")),
    }


def extract_term_stats(
    ols_uncorr,
    cov_dict: dict[str, tuple[np.ndarray, float]],
    term: str,
) -> dict[str, float]:
    """Extract beta, SE, t, and two-sided p across all 4 covariance estimators for one term."""
    idx = list(ols_uncorr.params.index).index(term)
    beta = float(ols_uncorr.params.iloc[idx])
    out: dict[str, float] = {"coef": beta}

    for cov_name, (cov_mat, df_dof) in cov_dict.items():
        var_ii = float(cov_mat[idx, idx])
        se = float(np.sqrt(max(var_ii, 1e-16)))
        t_val = beta / se
        if np.isinf(df_dof):
            p_val = float(2.0 * sp_stats.norm.sf(abs(t_val)))
        else:
            p_val = float(2.0 * sp_stats.t.sf(abs(t_val), df=df_dof))
        out[f"se_{cov_name}"] = se
        out[f"t_{cov_name}"] = t_val
        out[f"p_{cov_name}"] = p_val

    return out


def joint_wald_test(
    ols_uncorr,
    cov_mat: np.ndarray,
    df_dof: float,
    terms: list[str],
) -> tuple[float, float, int]:
    """Compute joint Wald F-statistic and p-value (or chi2 p-value when df_dof is inf) for a subset of terms."""
    param_names = list(ols_uncorr.params.index)
    idxs = [param_names.index(t) for t in terms if t in param_names]
    k = len(idxs)
    if k == 0:
        return float("nan"), float("nan"), 0

    b = np.asarray(ols_uncorr.params.iloc[idxs], dtype=float)
    # Apply Cameron-Gelbach-Miller (2011, Eq. 2.10) spectral PSD projection to full covariance matrix if indefinite
    full_eigvals, full_eigvecs = np.linalg.eigh(cov_mat)
    if np.any(full_eigvals < 0.0):
        cov_psd = full_eigvecs @ np.diag(np.maximum(full_eigvals, 0.0)) @ full_eigvecs.T
    else:
        cov_psd = cov_mat
    V = np.asarray(cov_psd[np.ix_(idxs, idxs)], dtype=float)
    try:
        V_inv = np.linalg.pinv(V, rcond=1e-4)
        wald_chi2 = float(b.T @ V_inv @ b)
        wald_f = wald_chi2 / k
        if np.isinf(df_dof):
            p_f = float(sp_stats.chi2.sf(max(wald_chi2, 0.0), df=k))
        else:
            p_f = float(sp_stats.f.sf(max(wald_f, 0.0), dfn=k, dfd=df_dof))
        return wald_f, p_f, k
    except Exception:
        return float("nan"), float("nan"), k


def prepare_fe_dataframe(require_salinity: bool = True) -> pd.DataFrame:
    """Load 2005-2020 Ben Tre station-year master and compute z-scored predictors."""
    master = load_master_bentre(year_min=2005, year_max=2020)
    target = "COD (mg/L)_annual"

    hc_cols = [
        "Q_MyThuan_AnnualMean_z",
        "Q_MyThuan_DrySeasonMean_z",
        "Rain_Annual_z",
        "Rain_Dry_z",
    ]
    base_needed = [target, "stationName", "year"] + LULC_VARS + hc_cols
    if require_salinity:
        base_needed.append("Salinity_annual")

    df = master.dropna(subset=base_needed).copy()
    df["COD"] = df[target].astype(float)

    # Standardize predictors within the analysis sample
    df["Urban_PD_z"] = zscore(df["Residential_Urban_config_PD"])
    df["Aqua_ED_z"] = zscore(df["Aquaculture_config_ED"])
    df["Rice_PD_z"] = zscore(df["Rice_Cultivation_config_PD"])
    df["Urban_pct_z"] = zscore(df["Residential_Urban_composition_pct"])
    df["Aqua_pct_z"] = zscore(df["Aquaculture_composition_pct"])
    df["Rice_pct_z"] = zscore(df["Rice_Cultivation_composition_pct"])
    df["Year_z"] = zscore(df["year"])

    if "Salinity_annual" in df.columns and df["Salinity_annual"].notna().any():
        if require_salinity:
            df["Salinity_z"] = zscore(df["Salinity_annual"])
        # Time-invariant station mean salinity (estuarine exposure index)
        stn_sal_mean = master.groupby("stationName")["Salinity_annual"].mean()
        df["Stn_Mean_Salinity"] = df["stationName"].map(stn_sal_mean)
        df["StnSalMean_z"] = zscore(df["Stn_Mean_Salinity"])

        if require_salinity:
            df["Sal_x_Q_Ann"] = df["Salinity_z"] * df["Q_MyThuan_AnnualMean_z"]
            df["Sal_x_Q_Dry"] = df["Salinity_z"] * df["Q_MyThuan_DrySeasonMean_z"]
        df["StnSal_x_Q_Ann"] = df["StnSalMean_z"] * df["Q_MyThuan_AnnualMean_z"]
        df["StnSal_x_Q_Dry"] = df["StnSalMean_z"] * df["Q_MyThuan_DrySeasonMean_z"]

    return df


def build_s5() -> pd.DataFrame:
    """
    Fit primary confirmatory OLS with station fixed effects for COD (N = 213, 2005-2020),
    reporting HC3, station-clustered, year-clustered, and two-way clustered SEs.
    """
    df = prepare_fe_dataframe(require_salinity=True)
    print(f"  Primary FE Data (2005-2020 with Salinity): N={len(df)} station-years "
          f"({df['stationName'].nunique()} stations, {df['year'].nunique()} years)")

    baseline_formula = (
        "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z "
        "+ Salinity_z + Year_z + C(stationName)"
    )
    controlled_formula = (
        "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z "
        "+ Salinity_z + Year_z "
        "+ Q_MyThuan_AnnualMean_z + Q_MyThuan_DrySeasonMean_z "
        "+ Rain_Annual_z + Rain_Dry_z + C(stationName)"
    )

    out_rows = []
    for model_name, formula in [
        ("Baseline", baseline_formula),
        ("HydroClimate", controlled_formula),
    ]:
        ols_uncorr = smf.ols(formula, data=df).fit()
        cov_dict = compute_4way_covs(ols_uncorr, df)

        for term in ols_uncorr.params.index:
            if term == "Intercept" or term.startswith("C("):
                continue
            tstats = extract_term_stats(ols_uncorr, cov_dict, term)
            out_rows.append({
                "Model": model_name,
                "term": term,
                "coef": round(tstats["coef"], 3),
                "se_HC3": round(tstats["se_HC3"], 3),
                "t": round(tstats["t_HC3"], 3),
                "p": round(tstats["p_HC3"], 3),
                "se_Cluster_Stn": round(tstats["se_Cluster_Stn"], 3),
                "p_Cluster_Stn": round(tstats["p_Cluster_Stn"], 3),
                "se_Cluster_Yr": round(tstats["se_Cluster_Yr"], 3),
                "p_Cluster_Yr": round(tstats["p_Cluster_Yr"], 3),
                "se_TwoWay": round(tstats["se_TwoWay"], 3),
                "p_TwoWay": round(tstats["p_TwoWay"], 3),
                "N": int(ols_uncorr.nobs),
                "R2": round(float(ols_uncorr.rsquared), 3),
                "R2_adj": round(float(ols_uncorr.rsquared_adj), 3),
            })

        print(f"    {model_name}: N={int(ols_uncorr.nobs)}, R²={ols_uncorr.rsquared:.3f}, "
              f"R²_adj={ols_uncorr.rsquared_adj:.3f}")

    return pd.DataFrame(out_rows)


def run_fe_hydro_exposure_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run comprehensive Fixed-Effects sensitivity suite:
      1) Pure Two-Way FE vs. Two-Way FE + LULC (N=229 without Salinity, N=213 with Salinity,
         15 open-estuary stations excluding dammed Ba Lai BL01-BL04, and low-salinity subsets),
         testing incremental R² and joint LULC significance.
      2) Station-varying hydrodynamic exposure specifications in Station FE and Two-Way FE
         (parsimonious single-Q, Salinity_z × Q interactions, StnSalMean_z × Q interactions,
         open-estuary no-Ba-Lai subset, and low-salinity < 2 PSU / < 1 PSU subsets).
    """
    print("\n  Running Two-Way FE & Station-Varying Hydrodynamic Exposure Sensitivity Suite...")
    df_229 = prepare_fe_dataframe(require_salinity=False)
    df_213 = prepare_fe_dataframe(require_salinity=True)

    # Subset 1: Exclude 4 impounded Ba Lai stations (BL01-BL04; gated by Ba Lai dam since 2002)
    df_229_no_bl = df_229[~df_229["stationName"].astype(str).str.startswith("BL")].copy()
    df_213_no_bl = df_213[~df_213["stationName"].astype(str).str.startswith("BL")].copy()

    # Subset 2: Low-salinity freshwater observations (< 2.0 PSU and < 1.0 PSU) to rule out COD_Cr Cl- interference
    df_low_sal_2 = df_213[df_213["Salinity_annual"] < 2.0].copy()
    df_low_sal_1 = df_213[df_213["Salinity_annual"] < 1.0].copy()

    lulc_3 = ["Urban_PD_z", "Aqua_ED_z", "Rice_PD_z"]
    lulc_6 = ["Urban_pct_z", "Urban_PD_z", "Aqua_pct_z", "Aqua_ED_z", "Rice_pct_z", "Rice_PD_z"]

    # ── Part 1: Two-Way FE Joint LULC Tests ──────────────────────────
    twoway_specs = [
        (
            "TWFE_N229_3LULC",
            df_229,
            "COD ~ C(stationName) + C(year)",
            "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + C(stationName) + C(year)",
            lulc_3,
        ),
        (
            "TWFE_N229_6LULC",
            df_229,
            "COD ~ C(stationName) + C(year)",
            "COD ~ Urban_pct_z + Urban_PD_z + Aqua_pct_z + Aqua_ED_z + Rice_pct_z + Rice_PD_z + C(stationName) + C(year)",
            lulc_6,
        ),
        (
            "TWFE_N213_3LULC_NoSal",
            df_213,
            "COD ~ C(stationName) + C(year)",
            "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + C(stationName) + C(year)",
            lulc_3,
        ),
        (
            "TWFE_N213_3LULC_WithSal",
            df_213,
            "COD ~ Salinity_z + C(stationName) + C(year)",
            "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + C(stationName) + C(year)",
            lulc_3,
        ),
        (
            "TWFE_NoBaLai_15Stn_6LULC",
            df_229_no_bl,
            "COD ~ C(stationName) + C(year)",
            "COD ~ Urban_pct_z + Urban_PD_z + Aqua_pct_z + Aqua_ED_z + Rice_pct_z + Rice_PD_z + C(stationName) + C(year)",
            lulc_6,
        ),
        (
            "TWFE_LowSal_lt2PSU_3LULC",
            df_low_sal_2,
            "COD ~ Salinity_z + C(stationName) + C(year)",
            "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + C(stationName) + C(year)",
            lulc_3,
        ),
        (
            "TWFE_LowSal_lt1PSU_3LULC",
            df_low_sal_1,
            "COD ~ Salinity_z + C(stationName) + C(year)",
            "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + C(stationName) + C(year)",
            lulc_3,
        ),
    ]

    joint_rows = []
    coef_rows = []

    for spec_id, df_sub, restricted_f, full_f, test_terms in twoway_specs:
        m_rest = smf.ols(restricted_f, data=df_sub).fit()
        m_full = smf.ols(full_f, data=df_sub).fit()
        cov_dict = compute_4way_covs(m_full, df_sub)

        # Classical nested F-test
        f_nested, p_nested, df_diff = m_full.compare_f_test(m_rest)

        w_hc3_f, w_hc3_p, k_terms = joint_wald_test(m_full, *cov_dict["HC3"], test_terms)
        w_stn_f, w_stn_p, _ = joint_wald_test(m_full, *cov_dict["Cluster_Stn"], test_terms)
        w_yr_f, w_yr_p, _ = joint_wald_test(m_full, *cov_dict["Cluster_Yr"], test_terms)
        w_2w_f, w_2w_p, _ = joint_wald_test(m_full, *cov_dict["TwoWay"], test_terms)

        joint_rows.append({
            "Specification": spec_id,
            "N": int(m_full.nobs),
            "N_Stations": int(df_sub["stationName"].nunique()),
            "N_Years": int(df_sub["year"].nunique()),
            "K_LULC_Terms": int(k_terms),
            "R2_Restricted_TWFE": round(float(m_rest.rsquared), 4),
            "R2_adj_Restricted_TWFE": round(float(m_rest.rsquared_adj), 4),
            "R2_Full_TWFE_LULC": round(float(m_full.rsquared), 4),
            "R2_adj_Full_TWFE_LULC": round(float(m_full.rsquared_adj), 4),
            "Delta_R2": round(float(m_full.rsquared - m_rest.rsquared), 4),
            "Delta_R2_adj": round(float(m_full.rsquared_adj - m_rest.rsquared_adj), 4),
            "Classical_F": round(float(f_nested), 3),
            "Classical_F_p": round(float(p_nested), 4),
            "Wald_F_HC3": round(w_hc3_f, 3),
            "Wald_p_HC3": round(w_hc3_p, 4),
            "Wald_F_Cluster_Stn": round(w_stn_f, 3),
            "Wald_p_Cluster_Stn": round(w_stn_p, 4),
            "Wald_F_Cluster_Yr": round(w_yr_f, 3),
            "Wald_p_Cluster_Yr": round(w_yr_p, 4),
            "Wald_F_TwoWay": round(w_2w_f, 3),
            "Wald_p_TwoWay": round(w_2w_p, 4),
        })

        for term in m_full.params.index:
            if term == "Intercept" or term.startswith("C("):
                continue
            tstats = extract_term_stats(m_full, cov_dict, term)
            coef_rows.append({
                "Spec_Group": "Part1_TwoWay_FE",
                "Specification": spec_id,
                "FE_Structure": "Station + Year FE",
                "N": int(m_full.nobs),
                "R2": round(float(m_full.rsquared), 4),
                "R2_adj": round(float(m_full.rsquared_adj), 4),
                "term": term,
                "coef": round(tstats["coef"], 4),
                "se_HC3": round(tstats["se_HC3"], 4),
                "p_HC3": round(tstats["p_HC3"], 4),
                "se_Cluster_Stn": round(tstats["se_Cluster_Stn"], 4),
                "p_Cluster_Stn": round(tstats["p_Cluster_Stn"], 4),
                "se_Cluster_Yr": round(tstats["se_Cluster_Yr"], 4),
                "p_Cluster_Yr": round(tstats["p_Cluster_Yr"], 4),
                "se_TwoWay": round(tstats["se_TwoWay"], 4),
                "p_TwoWay": round(tstats["p_TwoWay"], 4),
                "Joint_Hydro_Wald_F_HC3": np.nan,
                "Joint_Hydro_Wald_p_HC3": np.nan,
                "Joint_Hydro_Wald_p_Stn": np.nan,
                "Joint_Hydro_Wald_p_TwoWay": np.nan,
                "Joint_LULC_Wald_F_HC3": round(w_hc3_f, 3),
                "Joint_LULC_Wald_p_HC3": round(w_hc3_p, 4),
                "Joint_LULC_Wald_p_Stn": round(w_stn_p, 4),
                "Joint_LULC_Wald_p_TwoWay": round(w_2w_p, 4),
            })

    # ── Part 2: Station-Varying Hydrodynamic Exposure & Subsample Specifications ──
    hydro_specs = [
        (
            "H0_StnFE_UniformHydro_CollinearQ",
            "Station FE",
            df_213,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z "
                "+ Q_MyThuan_AnnualMean_z + Q_MyThuan_DrySeasonMean_z "
                "+ Rain_Annual_z + Rain_Dry_z + C(stationName)"
            ),
            ["Q_MyThuan_AnnualMean_z", "Q_MyThuan_DrySeasonMean_z", "Rain_Annual_z", "Rain_Dry_z"],
        ),
        (
            "H1_StnFE_Parsimonious_Q_Annual",
            "Station FE",
            df_213,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z "
                "+ Q_MyThuan_AnnualMean_z + Rain_Annual_z + Rain_Dry_z + C(stationName)"
            ),
            ["Q_MyThuan_AnnualMean_z", "Rain_Annual_z", "Rain_Dry_z"],
        ),
        (
            "H2_StnFE_Parsimonious_Q_Dry",
            "Station FE",
            df_213,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z "
                "+ Q_MyThuan_DrySeasonMean_z + Rain_Annual_z + Rain_Dry_z + C(stationName)"
            ),
            ["Q_MyThuan_DrySeasonMean_z", "Rain_Annual_z", "Rain_Dry_z"],
        ),
        (
            "H3_StnFE_Salinity_x_Q_Interaction",
            "Station FE",
            df_213,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z "
                "+ Q_MyThuan_AnnualMean_z + Q_MyThuan_DrySeasonMean_z "
                "+ Sal_x_Q_Ann + Sal_x_Q_Dry "
                "+ Rain_Annual_z + Rain_Dry_z + C(stationName)"
            ),
            ["Q_MyThuan_AnnualMean_z", "Q_MyThuan_DrySeasonMean_z", "Sal_x_Q_Ann", "Sal_x_Q_Dry", "Rain_Annual_z", "Rain_Dry_z"],
        ),
        (
            "H4_StnFE_StnSalExposure_x_Q_Interaction",
            "Station FE",
            df_213,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z "
                "+ Q_MyThuan_AnnualMean_z + Q_MyThuan_DrySeasonMean_z "
                "+ StnSal_x_Q_Ann + StnSal_x_Q_Dry "
                "+ Rain_Annual_z + Rain_Dry_z + C(stationName)"
            ),
            ["Q_MyThuan_AnnualMean_z", "Q_MyThuan_DrySeasonMean_z", "StnSal_x_Q_Ann", "StnSal_x_Q_Dry", "Rain_Annual_z", "Rain_Dry_z"],
        ),
        (
            "H5_TwoWayFE_StnSalExposure_x_Q_Interaction",
            "Station + Year FE",
            df_213,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z "
                "+ StnSal_x_Q_Ann + StnSal_x_Q_Dry + C(stationName) + C(year)"
            ),
            ["Salinity_z", "StnSal_x_Q_Ann", "StnSal_x_Q_Dry"],
        ),
        (
            "H6_TwoWayFE_ContempSalinity_x_Q_Interaction",
            "Station + Year FE",
            df_213,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z "
                "+ Sal_x_Q_Ann + Sal_x_Q_Dry + C(stationName) + C(year)"
            ),
            ["Salinity_z", "Sal_x_Q_Ann", "Sal_x_Q_Dry"],
        ),
        (
            "H7_TwoWayFE_N229_StnSalExposure_x_Q_Interaction",
            "Station + Year FE",
            df_229,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z "
                "+ StnSal_x_Q_Ann + StnSal_x_Q_Dry + C(stationName) + C(year)"
            ),
            ["StnSal_x_Q_Ann", "StnSal_x_Q_Dry"],
        ),
        (
            "H8_StnFE_NoBaLai_15Stn_HydroClimate",
            "Station FE (No Ba Lai)",
            df_213_no_bl,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z "
                "+ Q_MyThuan_AnnualMean_z + Q_MyThuan_DrySeasonMean_z "
                "+ Rain_Annual_z + Rain_Dry_z + C(stationName)"
            ),
            ["Q_MyThuan_AnnualMean_z", "Q_MyThuan_DrySeasonMean_z", "Rain_Annual_z", "Rain_Dry_z"],
        ),
        (
            "H9_StnFE_NoBaLai_15Stn_Parsimonious_Q_Ann",
            "Station FE (No Ba Lai)",
            df_213_no_bl,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z "
                "+ Q_MyThuan_AnnualMean_z + Rain_Annual_z + Rain_Dry_z + C(stationName)"
            ),
            ["Q_MyThuan_AnnualMean_z", "Rain_Annual_z", "Rain_Dry_z"],
        ),
        (
            "H10_StnFE_LowSal_lt2PSU_HydroClimate",
            "Station FE (Salinity < 2 PSU)",
            df_low_sal_2,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z "
                "+ Q_MyThuan_AnnualMean_z + Q_MyThuan_DrySeasonMean_z "
                "+ Rain_Annual_z + Rain_Dry_z + C(stationName)"
            ),
            ["Q_MyThuan_AnnualMean_z", "Q_MyThuan_DrySeasonMean_z", "Rain_Annual_z", "Rain_Dry_z"],
        ),
        (
            "H11_StnFE_LowSal_lt1PSU_HydroClimate",
            "Station FE (Salinity < 1 PSU)",
            df_low_sal_1,
            (
                "COD ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z "
                "+ Q_MyThuan_AnnualMean_z + Q_MyThuan_DrySeasonMean_z "
                "+ Rain_Annual_z + Rain_Dry_z + C(stationName)"
            ),
            ["Q_MyThuan_AnnualMean_z", "Q_MyThuan_DrySeasonMean_z", "Rain_Annual_z", "Rain_Dry_z"],
        ),
    ]

    for spec_id, fe_struct, df_sub, formula, hydro_test_terms in hydro_specs:
        m_fit = smf.ols(formula, data=df_sub).fit()
        cov_dict = compute_4way_covs(m_fit, df_sub)

        h_hc3_f, h_hc3_p, _ = joint_wald_test(m_fit, *cov_dict["HC3"], hydro_test_terms)
        _, h_stn_p, _ = joint_wald_test(m_fit, *cov_dict["Cluster_Stn"], hydro_test_terms)
        _, h_2w_p, _ = joint_wald_test(m_fit, *cov_dict["TwoWay"], hydro_test_terms)

        l_hc3_f, l_hc3_p, _ = joint_wald_test(m_fit, *cov_dict["HC3"], lulc_3)
        _, l_stn_p, _ = joint_wald_test(m_fit, *cov_dict["Cluster_Stn"], lulc_3)
        _, l_2w_p, _ = joint_wald_test(m_fit, *cov_dict["TwoWay"], lulc_3)

        for term in m_fit.params.index:
            if term == "Intercept" or term.startswith("C("):
                continue
            tstats = extract_term_stats(m_fit, cov_dict, term)
            coef_rows.append({
                "Spec_Group": "Part2_Hydro_Exposure",
                "Specification": spec_id,
                "FE_Structure": fe_struct,
                "N": int(m_fit.nobs),
                "R2": round(float(m_fit.rsquared), 4),
                "R2_adj": round(float(m_fit.rsquared_adj), 4),
                "term": term,
                "coef": round(tstats["coef"], 4),
                "se_HC3": round(tstats["se_HC3"], 4),
                "p_HC3": round(tstats["p_HC3"], 4),
                "se_Cluster_Stn": round(tstats["se_Cluster_Stn"], 4),
                "p_Cluster_Stn": round(tstats["p_Cluster_Stn"], 4),
                "se_Cluster_Yr": round(tstats["se_Cluster_Yr"], 4),
                "p_Cluster_Yr": round(tstats["p_Cluster_Yr"], 4),
                "se_TwoWay": round(tstats["se_TwoWay"], 4),
                "p_TwoWay": round(tstats["p_TwoWay"], 4),
                "Joint_Hydro_Wald_F_HC3": round(h_hc3_f, 3),
                "Joint_Hydro_Wald_p_HC3": round(h_hc3_p, 4),
                "Joint_Hydro_Wald_p_Stn": round(h_stn_p, 4),
                "Joint_Hydro_Wald_p_TwoWay": round(h_2w_p, 4),
                "Joint_LULC_Wald_F_HC3": round(l_hc3_f, 3),
                "Joint_LULC_Wald_p_HC3": round(l_hc3_p, 4),
                "Joint_LULC_Wald_p_Stn": round(l_stn_p, 4),
                "Joint_LULC_Wald_p_TwoWay": round(l_2w_p, 4),
            })

    joint_df = pd.DataFrame(joint_rows)
    sens_df = pd.DataFrame(coef_rows)

    out_sens = METRICS_OUT / "fe_hydro_interaction_sensitivity.csv"
    out_joint = METRICS_OUT / "fe_twoway_lulc_joint_tests.csv"
    sens_df.to_csv(out_sens, index=False)
    joint_df.to_csv(out_joint, index=False)
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    sens_df.to_csv(ANALYSIS_OUT / "fe_hydro_interaction_sensitivity.csv", index=False)
    joint_df.to_csv(ANALYSIS_OUT / "fe_twoway_lulc_joint_tests.csv", index=False)
    print(f"  ✓ {out_sens.name}")
    print(f"  ✓ {out_joint.name}")

    return joint_df, sens_df


def main():
    print("=" * 60)
    print("Table S5: Panel Fixed-Effects Regression for COD (2005-2020)")
    print("=" * 60)

    df_s5 = build_s5()
    joint_df, sens_df = run_fe_hydro_exposure_sensitivity()

    sections = []
    sections.append(
        "**Panel A.** Station fixed-effects OLS regression for COD "
        "(19 Ben Tre stations, 2005–2020, $N = 213$) across four covariance estimators.\n\n"
        + df_s5.to_markdown(index=False, floatfmt=".3f")
    )
    sections.append(
        "**Panel B.** Two-way fixed-effects (station and year fixed effects, $\\alpha_i + \\gamma_t$) nested $F$-tests "
        "and joint Wald tests for within-station LULC configuration and composition ($N = 229$ and $N = 213$).\n\n"
        + joint_df.to_markdown(index=False, floatfmt=".4f")
    )

    body = "\n\n".join(sections)
    write_table_md(
        content=body,
        out_path=MANU_TABLES / "Supplementary_Table_S5_COD_FixedEffects.md",
        caption=(
            "Supplementary Table S5. Station and two-way fixed-effects OLS regressions for COD "
            "(19 Ben Tre stations, 2005--2020, $N = 213$ with salinity and $N = 229$ full panel). "
            "Panel A reports standardized coefficients ($\\beta$, in mg L$^{-1}$ SD$^{-1}$) and "
            "inference across four covariance estimators: HC3 heteroskedasticity-robust, "
            "station-clustered ($G = 19$), year-clustered ($T = 16$), and Cameron--Gelbach--Miller "
            "two-way clustered standard errors. Panel B reports nested $F$-tests and joint Wald "
            "tests comparing pure two-way fixed effects ($\\alpha_i + \\gamma_t$) against "
            "two-way fixed effects augmented with LULC metrics, demonstrating that once "
            "time-invariant station baselines and delta-wide annual shocks are absorbed, "
            "residual within-station interannual LULC variation adds $\\Delta R^2 \\le 0.032$ "
            "($p > 0.08$ across all covariance estimators and open-estuary/low-salinity subsamples)."
        ),
        label="#tbl-s5-cod-fe",
    )

    write_metrics_csv(df_s5, "S5_cod_fixed_effects")
    df_s5.to_csv(ANALYSIS_OUT / "cod_confirmatory_fixed_effects_latest.csv", index=False)
    print("\n  Table S5 written successfully.\n")


if __name__ == "__main__":
    main()

