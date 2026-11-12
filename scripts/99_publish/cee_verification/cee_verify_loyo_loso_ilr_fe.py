#!/usr/bin/env python3
"""
CEE Phase 1 Verification Script 3:
Spatiotemporal Driver Attribution ("When vs. Where" Resolved), Compositional ILR Check,
Multi-Buffer (2 km, 5 km, 10 km, 16 km) Fixed-Effects Stress Test, Halide-Safe Subsamples,
and Chloride-Immune BOD5 Panel Fixed Effects across 4 Covariance Estimators.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups, cov_nw_groupsum

ROOT = Path(__file__).resolve().parents[3]
WQ_PATH = ROOT / "data" / "mekong_delta_water_quality_1985_2023.csv"
MASTER_PATH = ROOT / "analysis" / "cache" / "station_year_master_latest.csv"
TIDAL_CSV = ROOT / "analysis" / "analysis_outputs" / "Table_Tidal_Range_By_Year.csv"
CACHE_DIR = ROOT / "analysis" / "cache"
OUT_DIR = Path(__file__).resolve().parent / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LULC_6 = [
    "Residential_Urban_composition_pct",
    "Residential_Urban_config_PD",
    "Aquaculture_composition_pct",
    "Aquaculture_config_ED",
    "Rice_Cultivation_composition_pct",
    "Rice_Cultivation_config_PD",
]
POP_1 = ["pop_density"]
HYDRO_4 = [
    "Rain_Annual_z",
    "Rain_Dry_z",
    "Q_MyThuan_AnnualMean_z",
    "TR_MyThuan_m",
]
FULL_11 = LULC_6 + POP_1 + HYDRO_4


def zscore(s: pd.Series) -> pd.Series:
    s2 = s.astype(float)
    sd = float(s2.std(ddof=0))
    if sd == 0.0 or not np.isfinite(sd):
        return s2 * np.nan
    return (s2 - float(s2.mean())) / sd


def compute_ilr_basis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Egozcue et al. (2003) / Sigdel & Husic (2026) Isometric Log-Ratio (ILR) transformation
    for compositional land-cover percentages (Urban, Aquaculture, Rice, Other/Remainder),
    with multiplicative replacement (delta = 0.01%) for structural zeros.
    """
    comp_cols = [
        "Residential_Urban_composition_pct",
        "Aquaculture_composition_pct",
        "Rice_Cultivation_composition_pct",
    ]
    mat = df[comp_cols].to_numpy(dtype=float).copy()
    rem = np.clip(100.0 - np.sum(mat, axis=1, keepdims=True), 0.01, 100.0)
    parts = np.hstack([np.clip(mat, 0.01, 100.0), rem])
    parts = parts / np.sum(parts, axis=1, keepdims=True)
    log_p = np.log(parts)

    # Helmert orthonormal contrast basis for D = 4 parts -> D-1 = 3 ILR coordinates
    ilr1 = (log_p[:, 0] - log_p[:, 1]) / np.sqrt(2.0)  # Urban vs. Aquaculture contrast
    ilr2 = (log_p[:, 0] + log_p[:, 1] - 2.0 * log_p[:, 2]) / np.sqrt(6.0)  # Anthropogenic-intensive vs. Rice
    ilr3 = (log_p[:, 0] + log_p[:, 1] + log_p[:, 2] - 3.0 * log_p[:, 3]) / np.sqrt(12.0)  # Managed vs. Remainder
    out = pd.DataFrame({
        "ILR1_Urban_vs_Aqua": ilr1,
        "ILR2_Intensive_vs_Rice": ilr2,
        "ILR3_Managed_vs_Other": ilr3,
    }, index=df.index)
    return out


def eval_cv(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    group_col: str,
    demean_by_year: bool = False,
) -> dict[str, float]:
    """Evaluate exact out-of-group CV (LOSO when group_col='stationName'; LOYO when group_col='year')."""
    cols = list(dict.fromkeys([group_col, "stationName", "year", y_col] + x_cols))
    sub = df[cols].dropna().copy()
    if len(sub) < 30:
        return {"N": len(sub), "R2_Ridge": np.nan, "R2_GBR": np.nan, "RMSE_Ridge": np.nan}

    groups = sorted(sub[group_col].unique())
    y_true_all, y_pred_ridge, y_pred_gbr = [], [], []

    for g in groups:
        tr = sub[sub[group_col] != g].copy()
        te = sub[sub[group_col] == g].copy()
        if tr.empty or te.empty:
            continue

        y_tr = tr[y_col].to_numpy(dtype=float)
        y_te = te[y_col].to_numpy(dtype=float)

        if demean_by_year:
            # Remove shared year mean computed strictly from training stations to test pure spatial transfer
            yr_means = tr.groupby("year")[y_col].mean()
            global_tr_mean = float(np.mean(y_tr))
            y_tr_fit = y_tr - tr["year"].map(yr_means).fillna(global_tr_mean).to_numpy(dtype=float)
            te_yr_offset = te["year"].map(yr_means).fillna(global_tr_mean).to_numpy(dtype=float)
        else:
            y_tr_fit = y_tr
            te_yr_offset = np.zeros_like(y_te)

        X_tr = tr[x_cols].to_numpy(dtype=float)
        X_te = te[x_cols].to_numpy(dtype=float)

        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
        ridge.fit(X_tr, y_tr_fit)
        pred_r = ridge.predict(X_te) + te_yr_offset

        gbr = GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=42)
        gbr.fit(X_tr, y_tr_fit)
        pred_g = gbr.predict(X_te) + te_yr_offset

        y_true_all.extend(y_te.tolist())
        y_pred_ridge.extend(pred_r.tolist())
        y_pred_gbr.extend(pred_g.tolist())

    yt = np.asarray(y_true_all, dtype=float)
    pr = np.asarray(y_pred_ridge, dtype=float)
    pg = np.asarray(y_pred_gbr, dtype=float)
    return {
        "N": int(len(yt)),
        "n_groups": int(len(groups)),
        "R2_Ridge": round(float(r2_score(yt, pr)), 4),
        "R2_GBR": round(float(r2_score(yt, pg)), 4),
        "RMSE_Ridge": round(float(np.sqrt(mean_squared_error(yt, pr))), 4),
    }


def compute_4way_covs(ols_uncorr, df_fit: pd.DataFrame) -> dict[str, tuple[np.ndarray, float]]:
    ols_hc3 = ols_uncorr.get_robustcov_results(cov_type="HC3")
    cov_hc3 = np.asarray(ols_hc3.cov_params(), dtype=float)
    stn_groups = pd.Categorical(df_fit["stationName"]).codes.astype(int)
    yr_groups = pd.Categorical(df_fit["year"]).codes.astype(int)
    n_stn = int(df_fit["stationName"].nunique())
    n_yr = int(df_fit["year"].nunique())
    cov_stn = np.asarray(cov_cluster(ols_uncorr, stn_groups), dtype=float)
    cov_yr = np.asarray(cov_cluster(ols_uncorr, yr_groups), dtype=float)
    cov_2way_raw, _, _ = cov_cluster_2groups(ols_uncorr, group=stn_groups, group2=yr_groups, use_correction=True)
    cov_2way = np.asarray(cov_2way_raw, dtype=float)
    return {
        "HC3": (cov_hc3, float(ols_uncorr.df_resid)),
        "Cluster_Stn": (cov_stn, float(max(n_stn - 1, 1))),
        "Cluster_Yr": (cov_yr, float(max(n_yr - 1, 1))),
        "TwoWay": (cov_2way, float("inf")),
    }


def extract_term_row(model_label: str, ols_uncorr, cov_dict: dict, term: str) -> dict:
    idx = list(ols_uncorr.params.index).index(term)
    beta = float(ols_uncorr.params.iloc[idx])
    row: dict = {
        "Model": model_label,
        "Term": term,
        "N": int(ols_uncorr.nobs),
        "R2": round(float(ols_uncorr.rsquared), 4),
        "Beta": round(beta, 4),
    }
    for cname, (cmat, dof) in cov_dict.items():
        # Match canonical Cameron-Gelbach-Miller (2011) diagonal extraction in nw_table_s5_cod_fixed_effects.py;
        # fall back to spectral PSD projection only if finite-sample diagonal entry is non-positive.
        var_ii = float(cmat[idx, idx])
        if var_ii <= 0.0:
            eigvals, eigvecs = np.linalg.eigh(cmat)
            cmat_psd = eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.T
            var_ii = float(cmat_psd[idx, idx])
        se = float(np.sqrt(max(var_ii, 1e-16)))
        tval = beta / se
        pval = float(2.0 * sp_stats.norm.sf(abs(tval))) if np.isinf(dof) else float(2.0 * sp_stats.t.sf(abs(tval), df=dof))
        crit = 1.959964 if np.isinf(dof) else float(sp_stats.t.ppf(0.975, df=dof))
        row[f"SE_{cname}"] = round(se, 4)
        row[f"CI95_lo_{cname}"] = round(beta - crit * se, 4)
        row[f"CI95_hi_{cname}"] = round(beta + crit * se, 4)
        row[f"p_{cname}"] = round(pval, 5)
    return row


def main() -> None:
    master = pd.read_csv(MASTER_PATH)
    bt = master[(master["network"] == "Ben Tre") & (master["year"] >= 2005) & (master["year"] <= 2020)].copy()

    # Merge annual BOD5 and halide-masked COD_Cr into Ben Tre station-year master
    bod_supp_cache = CACHE_DIR / "bentre_station_year_bod5_halide_supplement.csv"
    if WQ_PATH.exists():
        wq = pd.read_csv(WQ_PATH)
        wq["year"] = pd.to_datetime(wq["datetime"], errors="coerce").dt.year
        bt_raw = wq[(wq["Data source"] == "Ben Tre") & (wq["year"] >= 2005) & (wq["year"] <= 2022)].copy()
        bod_ann = bt_raw.groupby(["stationName", "year"], as_index=False).agg(
            BOD5_annual=("BOD5 (mg/L)", "mean"),
            COD_Sal_le_3p6_annual=("COD (mg/L)", lambda s: s[bt_raw.loc[s.index, "Salinity"] <= 3.6].mean()),
        )
        bod_ann.to_csv(bod_supp_cache, index=False)
    elif bod_supp_cache.exists():
        bod_ann = pd.read_csv(bod_supp_cache)
    else:
        raise FileNotFoundError(f"Missing both {WQ_PATH} and derived cache {bod_supp_cache}")
    bt = bt.merge(bod_ann, on=["stationName", "year"], how="left")

    # Add station mean salinity & ILR coordinates
    stn_sal = bt.groupby("stationName")["Salinity_annual"].mean()
    bt["Stn_Mean_Salinity"] = bt["stationName"].map(stn_sal)
    ilr_df = compute_ilr_basis(bt)
    bt = pd.concat([bt, ilr_df], axis=1)
    ILR_6 = [
        "ILR1_Urban_vs_Aqua", "ILR2_Intensive_vs_Rice", "ILR3_Managed_vs_Other",
        "Residential_Urban_config_PD", "Aquaculture_config_ED", "Rice_Cultivation_config_PD",
    ]

    print(f"[Conservation] Ben Tre station-year master (2005-2020): {len(bt)} rows across {bt['stationName'].nunique()} stations, {bt['year'].nunique()} years")

    # 1. Two-way ANOVA Variance Partitioning including BOD5!
    vp_rows = []
    for col, lbl in [
        ("COD (mg/L)_annual", "COD_Cr"),
        ("BOD5_annual", "BOD5"),
        ("DO (mg/L)_annual", "DO"),
        ("NH4-N (mg/L)_annual", "NH4-N"),
        ("NO3-N (mg/L)_annual", "NO3-N"),
        ("TSS (mg/L)_annual", "TSS"),
    ]:
        sub = bt[["stationName", "year", col]].dropna()
        gm = float(sub[col].mean())
        ss_tot = float(((sub[col] - gm) ** 2).sum())
        ss_stn = float(sum(n * (m - gm) ** 2 for m, n in zip(sub.groupby("stationName")[col].mean(), sub.groupby("stationName")[col].count())))
        ss_yr = float(sum(n * (m - gm) ** 2 for m, n in zip(sub.groupby("year")[col].mean(), sub.groupby("year")[col].count())))
        ss_res = max(0.0, ss_tot - ss_stn - ss_yr)
        vp_rows.append({
            "Parameter": lbl,
            "N": int(len(sub)),
            "Spatial_Station_Share": round(ss_stn / ss_tot, 4),
            "Temporal_Year_Share": round(ss_yr / ss_tot, 4),
            "Residual_Interaction_Share": round(ss_res / ss_tot, 4),
        })
    df_vp = pd.DataFrame(vp_rows)
    df_vp.to_csv(OUT_DIR / "cee_variance_partitioning_with_bod5.csv", index=False)
    print("\n--- 1. Spatiotemporal Variance Partitioning (with BOD5) ---")
    print(df_vp.to_string(index=False))

    # 2. LOYO (When) vs. LOSO (Where) Cross-Validation Matrix
    cv_rows = []
    for y_col, y_lbl in [
        ("COD (mg/L)_annual", "COD_Cr"),
        ("BOD5_annual", "BOD5"),
        ("DO (mg/L)_annual", "DO"),
        ("NH4-N (mg/L)_annual", "NH4-N"),
        ("NO3-N (mg/L)_annual", "NO3-N"),
    ]:
        feature_sets = [
            ("LULC_Only_6", LULC_6),
            ("ILR_LULC_6", ILR_6),
            ("LULC_Pop_Salinity_8", LULC_6 + POP_1 + ["Stn_Mean_Salinity"]),
            ("HydroClimate_Only_4", HYDRO_4),
            ("Primary_Augmented_11", FULL_11),
            ("ILR_Augmented_11", ILR_6 + POP_1 + HYDRO_4),
        ]
        for fs_name, x_cols in feature_sets:
            loso_raw = eval_cv(bt, y_col, x_cols, group_col="stationName", demean_by_year=False)
            loso_pure_spatial = eval_cv(bt, y_col, x_cols, group_col="stationName", demean_by_year=True)
            loyo_temporal = eval_cv(bt, y_col, x_cols, group_col="year", demean_by_year=False)
            cv_rows.append({
                "Parameter": y_lbl,
                "Feature_Set": fs_name,
                "N": loso_raw["N"],
                "LOSO_R2_Ridge": loso_raw["R2_Ridge"],
                "LOSO_R2_GBR": loso_raw["R2_GBR"],
                "LOSO_YearDemeaned_Spatial_R2_Ridge": loso_pure_spatial["R2_Ridge"],
                "LOYO_Temporal_R2_Ridge": loyo_temporal["R2_Ridge"],
                "LOYO_Temporal_R2_GBR": loyo_temporal["R2_GBR"],
            })
    df_cv = pd.DataFrame(cv_rows)
    df_cv.to_csv(OUT_DIR / "cee_loyo_vs_loso_cv_matrix.csv", index=False)
    print("\n--- 2. LOYO (Temporal) vs. LOSO (Spatial) CV Matrix (COD_Cr, BOD5, DO) ---")
    print(df_cv[df_cv["Parameter"].isin(["COD_Cr", "BOD5", "DO"])].to_string(index=False))

    # 3. TreeSHAP / RandomForest / GBR Rank Concordance: Raw LULC % vs. ILR Compositional Basis
    rf_cols_raw = FULL_11
    rf_cols_ilr = ILR_6 + POP_1 + HYDRO_4
    sub_cod = bt[["COD (mg/L)_annual"] + rf_cols_raw + ["ILR1_Urban_vs_Aqua", "ILR2_Intensive_vs_Rice", "ILR3_Managed_vs_Other"]].dropna()
    rf_raw = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42).fit(sub_cod[rf_cols_raw], sub_cod["COD (mg/L)_annual"])
    rf_ilr = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42).fit(sub_cod[rf_cols_ilr], sub_cod["COD (mg/L)_annual"])

    imp_rows = []
    for f, imp in sorted(zip(rf_cols_raw, rf_raw.feature_importances_), key=lambda t: -t[1]):
        imp_rows.append({"Basis": "Raw_Pct", "Feature": f, "RF_Importance": round(float(imp), 4)})
    for f, imp in sorted(zip(rf_cols_ilr, rf_ilr.feature_importances_), key=lambda t: -t[1]):
        imp_rows.append({"Basis": "ILR_Compositional", "Feature": f, "RF_Importance": round(float(imp), 4)})
    df_imp = pd.DataFrame(imp_rows)
    df_imp.to_csv(OUT_DIR / "cee_ilr_vs_raw_importance.csv", index=False)
    print("\n--- 3. Top Feature Importances: Raw LULC % vs. ILR Compositional Basis (COD_Cr) ---")
    print(df_imp.groupby("Basis").head(5).to_string(index=False))

    # 4. Comprehensive Fixed-Effects Stress Test Suite:
    #    - H1 Parsimonious Single-Discharge Station FE (N = 213)
    #    - H9 Parsimonious Single-Discharge Station FE Excluding Impounded Ba Lai Branch (N = 165)
    #    - Halide-safe COD_Cr subsamples (Salinity <= 3.6, Salinity <= 2.0, Salinity <= 1.0)
    #    - Chloride-immune BOD5 Station FE
    #    - ILR Compositional LULC Station FE & Two-Way FE
    #    - Multi-buffer (2 km, 5 km, 10 km, 16 km) LULC Station FE & Two-Way FE
    fe_rows = []

    def prep_sub(d_in: pd.DataFrame, y_target: str) -> pd.DataFrame:
        need = [y_target, "stationName", "year", "Salinity_annual", "Q_MyThuan_AnnualMean_z", "Rain_Annual_z", "Rain_Dry_z"] + LULC_6
        d = d_in.dropna(subset=need).copy()
        d["Y"] = d[y_target].astype(float)
        d["Urban_PD_z"] = zscore(d["Residential_Urban_config_PD"])
        d["Aqua_ED_z"] = zscore(d["Aquaculture_config_ED"])
        d["Rice_PD_z"] = zscore(d["Rice_Cultivation_config_PD"])
        d["Urban_pct_z"] = zscore(d["Residential_Urban_composition_pct"])
        d["Aqua_pct_z"] = zscore(d["Aquaculture_composition_pct"])
        d["Rice_pct_z"] = zscore(d["Rice_Cultivation_composition_pct"])
        d["Salinity_z"] = zscore(d["Salinity_annual"])
        d["Year_z"] = zscore(d["year"])
        # Preserve canonical master hydroclimate z-scores (matching nw_table_s5_cod_fixed_effects.py)
        d["Q_Ann_z"] = d["Q_MyThuan_AnnualMean_z"].astype(float)
        d["Rain_Ann_z"] = d["Rain_Annual_z"].astype(float)
        d["Rain_Dry_z2"] = d["Rain_Dry_z"].astype(float)
        ilr_sub = compute_ilr_basis(d)
        d["ILR1_z"] = zscore(ilr_sub["ILR1_Urban_vs_Aqua"])
        d["ILR2_z"] = zscore(ilr_sub["ILR2_Intensive_vs_Rice"])
        d["ILR3_z"] = zscore(ilr_sub["ILR3_Managed_vs_Other"])
        return d

    # 4a. Primary H1 COD_Cr (N = 213)
    d_h1 = prep_sub(bt, "COD (mg/L)_annual")
    f_h1 = "Y ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + Year_z + Q_Ann_z + Rain_Ann_z + Rain_Dry_z2 + C(stationName)"
    m_h1 = smf.ols(f_h1, data=d_h1).fit()
    c_h1 = compute_4way_covs(m_h1, d_h1)
    for t in ["Q_Ann_z", "Year_z", "Salinity_z", "Urban_PD_z", "Aqua_ED_z", "Rice_PD_z", "Rain_Ann_z", "Rain_Dry_z2"]:
        fe_rows.append(extract_term_row("H1_COD_StationFE_All19Stns", m_h1, c_h1, t))

    # 4b. H9 No-Ba-Lai (excluding sluice-impounded Ba Lai river stations)
    no_balai_mask = ~bt["River"].astype(str).str.contains("Ba Lai", case=False, na=False)
    d_h9 = prep_sub(bt[no_balai_mask], "COD (mg/L)_annual")
    m_h9 = smf.ols(f_h1, data=d_h9).fit()
    c_h9 = compute_4way_covs(m_h9, d_h9)
    for t in ["Q_Ann_z", "Year_z", "Salinity_z", "Urban_PD_z", "Aqua_ED_z", "Rice_PD_z"]:
        fe_rows.append(extract_term_row("H9_COD_StationFE_NoBaLai", m_h9, c_h9, t))

    # 4c. Halide-safe COD_Cr subsamples: Annual Salinity <= 3.6, <= 2.0, <= 1.0, plus raw-filtered COD_Sal_le_3p6_annual
    for sal_cap, lbl in [(3.6, "COD_Sal_le_3.6"), (2.0, "COD_Sal_le_2.0"), (1.0, "COD_Sal_le_1.0")]:
        d_sub = prep_sub(bt[bt["Salinity_annual"] <= sal_cap], "COD (mg/L)_annual")
        if len(d_sub) >= 60:
            m_sub = smf.ols(f_h1, data=d_sub).fit()
            c_sub = compute_4way_covs(m_sub, d_sub)
            for t in ["Q_Ann_z", "Year_z", "Urban_PD_z", "Aqua_ED_z", "Rice_PD_z"]:
                fe_rows.append(extract_term_row(lbl, m_sub, c_sub, t))

    d_raw_safe = prep_sub(bt, "COD_Sal_le_3p6_annual")
    if len(d_raw_safe) >= 80:
        m_rs = smf.ols(f_h1, data=d_raw_safe).fit()
        c_rs = compute_4way_covs(m_rs, d_raw_safe)
        for t in ["Q_Ann_z", "Year_z", "Urban_PD_z", "Aqua_ED_z", "Rice_PD_z"]:
            fe_rows.append(extract_term_row("COD_ObsFiltered_Sal_le_3.6", m_rs, c_rs, t))

    # 4d. Chloride-immune BOD5 Station FE
    d_bod = prep_sub(bt, "BOD5_annual")
    m_bod = smf.ols(f_h1, data=d_bod).fit()
    c_bod = compute_4way_covs(m_bod, d_bod)
    for t in ["Q_Ann_z", "Year_z", "Salinity_z", "Urban_PD_z", "Aqua_ED_z", "Rice_PD_z"]:
        fe_rows.append(extract_term_row("BOD5_ChlorideImmune_StationFE", m_bod, c_bod, t))

    # 4e. ILR Compositional LULC Station FE & Two-Way FE
    f_ilr = "Y ~ ILR1_z + ILR2_z + ILR3_z + Salinity_z + Year_z + Q_Ann_z + Rain_Ann_z + Rain_Dry_z2 + C(stationName)"
    m_ilr = smf.ols(f_ilr, data=d_h1).fit()
    c_ilr = compute_4way_covs(m_ilr, d_h1)
    for t in ["Q_Ann_z", "Year_z", "ILR1_z", "ILR2_z", "ILR3_z"]:
        fe_rows.append(extract_term_row("COD_ILR_Compositional_StationFE", m_ilr, c_ilr, t))

    # 4f. Multi-buffer (2 km, 5 km, 10 km, 16 km) LULC sensitivity
    base_cols_keep = [
        "stationName", "year", "COD (mg/L)_annual", "BOD5_annual", "Salinity_annual",
        "Q_MyThuan_AnnualMean_z", "Rain_Annual_z", "Rain_Dry_z",
    ]
    bt_core = bt[base_cols_keep].copy()
    for buf_km in [2, 5, 10, 15, 16]:
        buf_file = CACHE_DIR / f"lulc_metrics_corrected_{buf_km}km.csv"
        if not buf_file.exists():
            continue
        ldf = pd.read_csv(buf_file)
        merged = bt_core.merge(ldf[["stationName", "year"] + LULC_6], on=["stationName", "year"], how="inner")
        d_buf = prep_sub(merged, "COD (mg/L)_annual")
        m_buf = smf.ols(f_h1, data=d_buf).fit()
        c_buf = compute_4way_covs(m_buf, d_buf)
        for t in ["Q_Ann_z", "Year_z", "Urban_PD_z", "Aqua_ED_z", "Rice_PD_z"]:
            fe_rows.append(extract_term_row(f"Buffer_{buf_km}km_StationFE", m_buf, c_buf, t))

        # Also Two-Way FE at each buffer radius
        f_2w = "Y ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Urban_pct_z + Aqua_pct_z + Rice_pct_z + Salinity_z + C(stationName) + C(year)"
        m_2w = smf.ols(f_2w, data=d_buf).fit()
        c_2w = compute_4way_covs(m_2w, d_buf)
        for t in ["Urban_PD_z", "Aqua_ED_z", "Rice_PD_z", "Urban_pct_z", "Aqua_pct_z", "Rice_pct_z"]:
            fe_rows.append(extract_term_row(f"Buffer_{buf_km}km_TwoWayFE", m_2w, c_2w, t))

    df_fe = pd.DataFrame(fe_rows)
    df_fe.to_csv(OUT_DIR / "cee_fixed_effects_comprehensive_suite.csv", index=False)
    print("\n--- 4. Comprehensive Fixed-Effects Stress Test Summary ---")
    key_models = [
        "H1_COD_StationFE_All19Stns",
        "H9_COD_StationFE_NoBaLai",
        "COD_ObsFiltered_Sal_le_3.6",
        "COD_Sal_le_1.0",
        "BOD5_ChlorideImmune_StationFE",
        "COD_ILR_Compositional_StationFE",
        "Buffer_2km_StationFE",
        "Buffer_5km_StationFE",
    ]
    sub_show = df_fe[df_fe["Model"].isin(key_models) & df_fe["Term"].isin(["Q_Ann_z", "Year_z", "Urban_PD_z", "Aqua_ED_z", "Rice_PD_z", "ILR1_z", "ILR2_z"])]
    print(sub_show[["Model", "Term", "N", "Beta", "p_HC3", "p_Cluster_Stn", "p_TwoWay", "CI95_lo_Cluster_Stn", "CI95_hi_Cluster_Stn"]].to_string(index=False))

    # 5. Tidal-Range Station FE Verification (Ben Tre 2005-2020 & MRC Mainstem 1985-2022)
    tidal_rows = []
    if TIDAL_CSV.exists():
        tdf = pd.read_csv(TIDAL_CSV)
        mt_dry = tdf[(tdf["Station"] == "MyThuan") & (tdf["Metric"] == "DrySeasonMeanDailyTidalRange_m")][["year", "Value"]].rename(columns={"Value": "TR_Dry_MT_m"})
        ct_dry = tdf[(tdf["Station"] == "CanTho") & (tdf["Metric"] == "DrySeasonMeanDailyTidalRange_m")][["year", "Value"]].rename(columns={"Value": "TR_Dry_CT_m"})

        # 5a. Ben Tre Station FE replacing linear Year_z with z-scored dry-season tidal range (dual-Q matching SNote 7 & single-Q H1)
        d_bt_tr = d_h1.merge(mt_dry, on="year", how="left").merge(ct_dry, on="year", how="left")
        d_bt_tr["Q_Dry_z"] = d_bt_tr["Q_MyThuan_DrySeasonMean_z"].astype(float)
        d_bt_tr["TR_Dry_MT_z"] = zscore(d_bt_tr["TR_Dry_MT_m"])
        d_bt_tr["TR_Dry_CT_z"] = zscore(d_bt_tr["TR_Dry_CT_m"])

        for m_lbl, tr_col, extra_q in [
            ("BenTre_DualQ_TR_Dry_MyThuan", "TR_Dry_MT_z", " + Q_Dry_z"),
            ("BenTre_DualQ_TR_Dry_CanTho", "TR_Dry_CT_z", " + Q_Dry_z"),
            ("BenTre_SingleQ_TR_Dry_MyThuan", "TR_Dry_MT_z", ""),
            ("BenTre_SingleQ_TR_Dry_CanTho", "TR_Dry_CT_z", ""),
        ]:
            f_tr = f"Y ~ Urban_PD_z + Aqua_ED_z + Rice_PD_z + Salinity_z + {tr_col} + Q_Ann_z{extra_q} + Rain_Ann_z + Rain_Dry_z2 + C(stationName)"
            m_tr = smf.ols(f_tr, data=d_bt_tr).fit()
            c_tr = compute_4way_covs(m_tr, d_bt_tr)
            for t in [tr_col, "Q_Ann_z", "Urban_PD_z"]:
                tidal_rows.append(extract_term_row(m_lbl, m_tr, c_tr, t))

        # 5b. Independent 4-station x 38-year MRC Mainstem Station FE (1985-2022, N = 151)
        mrc = master[(master["network"] == "MRC") & (master["year"] >= 1985) & (master["year"] <= 2022)].copy()
        mrc = mrc.merge(mt_dry, on="year", how="left")
        mrc_need = ["COD (mg/L)_annual", "stationName", "year", "TR_Dry_MT_m", "Q_MyThuan_AnnualMean_z", "Q_MyThuan_DrySeasonMean_z", "Rain_Annual_z", "Rain_Dry_z"]
        mrc_sub = mrc.dropna(subset=mrc_need).sort_values(["year", "stationName"]).copy()
        mrc_sub["Y"] = mrc_sub["COD (mg/L)_annual"].astype(float)
        mrc_sub["TR_Dry_MT_z"] = zscore(mrc_sub["TR_Dry_MT_m"])
        mrc_sub["Q_Ann_z"] = mrc_sub["Q_MyThuan_AnnualMean_z"].astype(float)
        mrc_sub["Q_Dry_z"] = mrc_sub["Q_MyThuan_DrySeasonMean_z"].astype(float)
        mrc_sub["Rain_Ann_z"] = mrc_sub["Rain_Annual_z"].astype(float)
        mrc_sub["Rain_Dry_z2"] = mrc_sub["Rain_Dry_z"].astype(float)

        for mrc_lbl, mrc_f in [
            ("MRC_1985_2022_DualQ_TR_Dry_MT", "Y ~ TR_Dry_MT_z + Q_Ann_z + Q_Dry_z + Rain_Ann_z + Rain_Dry_z2 + C(stationName)"),
            ("MRC_1985_2022_SingleQ_TR_Dry_MT", "Y ~ TR_Dry_MT_z + Q_Ann_z + Rain_Ann_z + Rain_Dry_z2 + C(stationName)"),
        ]:
            m_mrc = smf.ols(mrc_f, data=mrc_sub).fit()
            c_mrc = compute_4way_covs(m_mrc, mrc_sub)
            # Driscoll-Kraay (1998) cross-sectionally and serially robust covariance across years (nlags=3)
            cov_dk = np.asarray(cov_nw_groupsum(m_mrc, nlags=3, time=mrc_sub["year"].to_numpy(dtype=int)), dtype=float)
            c_mrc["DriscollKraay"] = (cov_dk, float(max(mrc_sub["year"].nunique() - 1, 1)))
            for t in ["TR_Dry_MT_z", "Q_Ann_z"]:
                tidal_rows.append(extract_term_row(mrc_lbl, m_mrc, c_mrc, t))

        df_tidal = pd.DataFrame(tidal_rows)
        df_tidal.to_csv(OUT_DIR / "cee_tidal_range_fe_verification.csv", index=False)
        print("\n--- 5. Tidal-Range Station FE Verification (Ben Tre & MRC Mainstem) ---")
        cols_t = [c for c in ["Model", "Term", "N", "R2", "Beta", "p_HC3", "p_Cluster_Stn", "p_Cluster_Yr", "p_TwoWay", "p_DriscollKraay"] if c in df_tidal.columns]
        print(df_tidal[cols_t].to_string(index=False))


if __name__ == "__main__":
    main()
