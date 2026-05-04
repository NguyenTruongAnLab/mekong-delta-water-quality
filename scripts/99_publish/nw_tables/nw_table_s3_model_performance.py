#!/usr/bin/env python3
"""
Supplementary Table S3 — Model Performance & Generalization
============================================================

Computes from RAW DATA (4 panels):

  Panel A: Random Forest LOSO cross-validation (baseline LULC-only model)
           OOB R², LOSO R², Spearman rho, RMSE, N permutation-significant features

  Panel B: Generalization comparison: RF + Ridge × Baseline (LULC) vs Augmented
           (LULC + branch discharge + rainfall + year)

  Panel C: COD variance partition (LULC-only vs Hydro-only vs Combined)
           Using Ridge LOSO + OLS R²

  Panel D: RDA cross-network variance explained (RDA1+RDA2)
           For 4 configurations: Ben Tre LULC-only, Ben Tre hydro-proxy,
           MRC LULC-only, MRC hydro

Data sources: raw WQ CSV + LULC CSV + Gold Standard hydro + branch discharge

Output:
  manuscript/tables/Supplementary_Table_S3_Model_Performance.md
"""
from __future__ import annotations

import sys
import os
import warnings
import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from skbio.stats.ordination import rda
import statsmodels.api as sm

from _nw_common_tables import (
    ROOT, WQ_PATH, LULC_PATH, Q_FILE, TIDAL_FILE,
    MRC_STATIONS, WQ_VARS, LULC_VARS, LABEL_MAP_Y,
    merge_wq_lulc, prepare_xy, load_wq_raw, load_lulc,
    MANU_TABLES, write_table_md, write_metrics_csv,
    ensure_hydro_ingest_importable,
)

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
# RF CONFIG (consistent with 03_05 baseline script)
# ═══════════════════════════════════════════════════════════════════════
RF_PARAMS = dict(
    n_estimators=500, random_state=42, max_depth=10,
    min_samples_split=5, min_samples_leaf=2,
)


def safe_spearman(y_true, y_pred) -> float:
    try:
        rho, _ = spearmanr(y_true, y_pred)
        return float(rho) if np.isfinite(rho) else float("nan")
    except Exception:
        return float("nan")


# ═══════════════════════════════════════════════════════════════════════
# PANEL A: Baseline RF LOSO
# ═══════════════════════════════════════════════════════════════════════

def panel_a(Y, X, groups) -> pd.DataFrame:
    """RF OOB + LOSO + LOYO + permutation importance (baseline LULC-only)."""
    print("\n  Panel A: Baseline RF LOSO...")
    logo = LeaveOneGroupOut()
    rows = []

    for param in Y.columns:
        y = Y[param]

        rf = RandomForestRegressor(oob_score=True, **RF_PARAMS)
        rf.fit(X, y)
        r2_oob = rf.oob_score_

        # LOSO
        y_pred_loso = cross_val_predict(rf, X, y, groups=groups, cv=logo)
        r2_loso = r2_score(y, y_pred_loso)
        rmse_loso = float(np.sqrt(mean_squared_error(y, y_pred_loso)))
        rho_loso = safe_spearman(y, y_pred_loso)

        # LOYO (by year)
        years = pd.Series(
            [int(x) for x in groups.index],
            index=groups.index,
        )
        # We can't do LOYO directly from groups, so use merged df's year.
        # For simplicity, compute LOSO only (the primary metric)
        r2_loyo = np.nan  # Placeholder - not available without year groups

        # Permutation importance
        perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
        n_sig = int((perm.importances_mean > 2 * perm.importances_std).sum())

        rows.append({
            "Parameter": param,
            "OOB_R2_score": round(r2_oob, 3),
            "CV_R2_LOSO": round(r2_loso, 3),
            "Spearman_rho_LOSO": round(rho_loso, 3),
            "RMSE_LOSO": round(rmse_loso, 3),
            "N_Perm_Significant": n_sig,
        })
        print(f"    {param}: OOB R²={r2_oob:.3f}, LOSO R²={r2_loso:.3f}, ρ={rho_loso:.3f}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# PANEL B: Generalization (RF + Ridge × Baseline vs Augmented)
# ═══════════════════════════════════════════════════════════════════════

def panel_b(merged_full: pd.DataFrame) -> pd.DataFrame:
    """RF + Ridge × Baseline vs Augmented feature sets → LOSO metrics."""
    print("\n  Panel B: Generalization comparison...")

    # Add hydro covariates via Gold Standard
    ensure_hydro_ingest_importable()
    from lib.hydro_ingest import join_branch_hydroclimate, hydroclimate_feature_columns

    merged = merged_full.copy()
    hdf = join_branch_hydroclimate(merged[["stationName", "year"]].copy())
    hydro_cols = hydroclimate_feature_columns()
    for c in hydro_cols:
        if c in hdf.columns:
            merged[c] = hdf[c].values

    # Add year as numeric
    merged["Year_num"] = merged["year"].astype(float)

    # Build Branch dummy
    merged["branch"] = merged["stationName"].astype(str).str[:2]
    for br in ["HL", "CC", "MT", "BL"]:
        merged[f"branch_{br}"] = (merged["branch"] == br).astype(float)

    wq_vars = [c for c in WQ_VARS if c in merged.columns]
    lulc_vars = [c for c in LULC_VARS if c in merged.columns]

    # Augmented feature set = LULC + hydro covariates + year + branch dummies
    augmented_extra = [c for c in hydro_cols if c in merged.columns]
    augmented_extra += ["Year_num"]
    augmented_extra += [f"branch_{br}" for br in ["HL", "CC", "MT", "BL"]
                        if f"branch_{br}" in merged.columns]

    logo = LeaveOneGroupOut()
    rows = []

    for param in wq_vars:
        param_label = LABEL_MAP_Y.get(param, param)
        df_clean = merged.dropna(subset=[param] + lulc_vars + augmented_extra).copy()
        y = df_clean[param]
        groups = df_clean["stationName"]

        for feature_set_name, feat_cols in [
            ("Baseline", lulc_vars),
            ("Augmented", lulc_vars + augmented_extra),
        ]:
            X = df_clean[feat_cols].copy()
            for model_name, model_cls in [("RF", RandomForestRegressor), ("Ridge", Ridge)]:
                if model_name == "RF":
                    model = RandomForestRegressor(**RF_PARAMS)
                else:
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge(alpha=1.0)),
                    ])
                y_pred = cross_val_predict(model, X, y, groups=groups, cv=logo)

                rows.append({
                    "Parameter": param_label,
                    "Feature_Set": feature_set_name,
                    "Model": model_name,
                    "LOSO_R2": round(r2_score(y, y_pred), 3),
                    "LOSO_RMSE": round(float(np.sqrt(mean_squared_error(y, y_pred))), 3),
                    "LOSO_Spearman_rho": round(safe_spearman(y, y_pred), 3),
                    "N_samples": len(X),
                    "N_features": X.shape[1],
                })
        print(f"    {param_label}: done")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# PANEL C: COD Variance Partition
# ═══════════════════════════════════════════════════════════════════════

def panel_c(merged_full: pd.DataFrame) -> pd.DataFrame:
    """COD variance partition: LULC-only vs Hydro-only vs Combined."""
    print("\n  Panel C: COD variance partition...")

    ensure_hydro_ingest_importable()
    from lib.hydro_ingest import HydroClimateColumns, join_branch_hydroclimate

    merged = merged_full.copy()
    hdf = join_branch_hydroclimate(merged[["stationName", "year"]].copy())
    hcols = HydroClimateColumns()
    for c in [hcols.q_branch_annual, hcols.q_branch_dry,
              hcols.rain_annual_mm, hcols.rain_dry_mm]:
        if c in hdf.columns:
            merged[c] = hdf[c].values

    # Tidal range proxy
    merged["branch"] = merged["stationName"].astype(str).str[:2]
    if TIDAL_FILE.exists():
        tr = pd.read_csv(TIDAL_FILE)
        tr_map = dict(zip(tr["Station"], tr["Mean_Tidal_Range_m"]))
        branch_tidal = {
            "HL": tr_map.get("Vam_Kenh", np.nan),
            "CC": tr_map.get("Vam_Kenh", np.nan),
            "MT": tr_map.get("My_Tho", np.nan),
            "BL": tr_map.get("Vam_Kenh", np.nan),
        }
        merged["Tidal_Range_Proxy_m"] = merged["branch"].map(branch_tidal)

    merged["Year_centered"] = merged["year"].astype(float) - merged["year"].astype(float).mean()

    target = "COD (mg/L)"
    lulc_cols = [c for c in LULC_VARS if c in merged.columns]
    hydro_cols = [c for c in ["Salinity", hcols.q_branch_annual, hcols.q_branch_dry,
                              hcols.rain_annual_mm, hcols.rain_dry_mm,
                              "Tidal_Range_Proxy_m", "Year_centered"]
                  if c in merged.columns]

    df2 = merged[[target, "stationName"] + lulc_cols + hydro_cols].dropna().copy()
    y = df2[target].astype(float).values
    groups = df2["stationName"]

    rows = []
    for name, cols in [
        ("LULC_only", lulc_cols),
        ("Hydro_only", hydro_cols),
        ("Combined", lulc_cols + hydro_cols),
    ]:
        X = df2[cols].astype(float)
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        logo = LeaveOneGroupOut()
        yhat = cross_val_predict(model, X, y, groups=groups, cv=logo)

        r2 = float(r2_score(y, yhat))
        rmse = float(np.sqrt(mean_squared_error(y, yhat)))
        rho = safe_spearman(y, yhat)

        # In-sample OLS
        X_ols = sm.add_constant(X.values.astype(float), has_constant="add")
        ols = sm.OLS(y, X_ols, missing="drop").fit()

        rows.append({
            "Predictor_Set": name,
            "LOSO_R2": round(r2, 3),
            "LOSO_RMSE": round(rmse, 3),
            "LOSO_Spearman_rho": round(rho, 3),
            "OLS_R2": round(float(ols.rsquared), 3),
            "N": len(df2),
            "N_features": len(cols),
        })
        print(f"    {name}: LOSO R²={r2:.3f}, OLS R²={ols.rsquared:.3f}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# PANEL D: RDA Cross-Network (4 configurations, all from raw data)
# ═══════════════════════════════════════════════════════════════════════

def _run_rda_variance(Y: pd.DataFrame, X: pd.DataFrame) -> float:
    """Run RDA and return total variance explained (RDA1+RDA2)."""
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X), columns=X.columns, index=X.index
    )
    ord_res = rda(Y, X_scaled, scale_Y=True, scaling=2)
    prop = ord_res.proportion_explained
    return float(prop.iloc[0] + prop.iloc[1])


def panel_d() -> pd.DataFrame:
    """RDA variance explained for Ben Tre and MRC, LULC-only and hydro-augmented."""
    print("\n  Panel D: RDA cross-network variance explained...")
    rows = []

    # ----- Ben Tre LULC-only -----
    merged_bt = merge_wq_lulc(bentre_only=True, include_mrc=False)
    wq_vars = [c for c in WQ_VARS if c in merged_bt.columns]
    lulc_vars = [c for c in LULC_VARS if c in merged_bt.columns]
    df_bt = merged_bt.dropna(subset=wq_vars + lulc_vars).copy()
    Y_bt = df_bt[wq_vars]
    X_bt = df_bt[lulc_vars]
    var_bt = _run_rda_variance(Y_bt, X_bt)
    rows.append({"Dataset": "Ben Tre (LULC only)", "RDA1+RDA2": round(var_bt, 3)})
    print(f"    Ben Tre LULC-only: {var_bt:.3f}")

    # ----- Ben Tre hydro-proxy -----
    ensure_hydro_ingest_importable()
    merged_bt2 = merged_bt.copy()
    merged_bt2["branch"] = merged_bt2["stationName"].astype(str).str[:2]

    # Branch discharge proxy
    if Q_FILE.exists():
        q = pd.read_csv(Q_FILE)
        q["year"] = pd.to_datetime(q["Date"], errors="coerce").dt.year
        annual_q = q.groupby("year").mean(numeric_only=True).reset_index()
        q_rows = []
        for _, row in annual_q.iterrows():
            yr = int(row["year"])
            for branch_prefix, col_name in [("HL", "Q_HamLuong_Est"), ("CC", "Q_CoChien_Est"), ("MT", "Q_MyTho_Est")]:
                if col_name in row.index:
                    q_rows.append({"branch": branch_prefix, "year": yr, "Q_branch": row[col_name]})
        if q_rows:
            q_df = pd.DataFrame(q_rows)
            merged_bt2 = merged_bt2.merge(q_df, on=["branch", "year"], how="left")

    # Tidal range proxy
    if TIDAL_FILE.exists():
        tr = pd.read_csv(TIDAL_FILE)
        tr_map = dict(zip(tr["Station"], tr["Mean_Tidal_Range_m"]))
        branch_tidal = {"HL": tr_map.get("Vam_Kenh", np.nan),
                        "CC": tr_map.get("Vam_Kenh", np.nan),
                        "MT": tr_map.get("My_Tho", np.nan)}
        merged_bt2["Tidal_Range_Proxy"] = merged_bt2["branch"].map(branch_tidal)

    # Coastal zone proxy
    if "Salinity" in merged_bt2.columns:
        merged_bt2["Coastal_Zone"] = (merged_bt2["Salinity"] > 2).astype(int)
    else:
        merged_bt2["Coastal_Zone"] = 0

    hydro_extra = [c for c in ["Q_branch", "Tidal_Range_Proxy", "Coastal_Zone"]
                   if c in merged_bt2.columns]
    bt_hydro_x_cols = lulc_vars + hydro_extra
    df_bt2 = merged_bt2.dropna(subset=wq_vars + bt_hydro_x_cols).copy()
    if len(df_bt2) >= 10:
        var_bt2 = _run_rda_variance(df_bt2[wq_vars], df_bt2[bt_hydro_x_cols])
        rows.append({"Dataset": "Ben Tre (hydro-proxy)", "RDA1+RDA2": round(var_bt2, 3)})
        print(f"    Ben Tre hydro-proxy: {var_bt2:.3f}")

    # ----- MRC LULC-only -----
    wq_raw = load_wq_raw()
    wq_mrc = wq_raw[wq_raw["stationName"].isin(MRC_STATIONS)]
    wq_mrc_annual = wq_mrc.groupby(["stationName", "year"]).mean(numeric_only=True).reset_index()
    lulc = load_lulc()
    merged_mrc = pd.merge(wq_mrc_annual, lulc, on=["stationName", "year"], how="inner")
    merged_mrc = merged_mrc[(merged_mrc["year"] >= 2013) & (merged_mrc["year"] <= 2020)].copy()

    mrc_wq = [c for c in WQ_VARS if c in merged_mrc.columns]
    mrc_lulc = [c for c in LULC_VARS if c in merged_mrc.columns]
    df_mrc = merged_mrc.dropna(subset=mrc_wq + mrc_lulc).copy()
    if len(df_mrc) >= 5:
        var_mrc = _run_rda_variance(df_mrc[mrc_wq], df_mrc[mrc_lulc])
        rows.append({"Dataset": "MRC (LULC only)", "RDA1+RDA2": round(var_mrc, 3)})
        print(f"    MRC LULC-only: {var_mrc:.3f}")

    # ----- MRC hydro (LULC + discharge + water level) -----
    if Q_FILE.exists():
        q = pd.read_csv(Q_FILE)
        q["year"] = pd.to_datetime(q["Date"], errors="coerce").dt.year
        q_annual = q.groupby("year").mean(numeric_only=True).reset_index()

        # Map station to discharge column
        station_q_map = {
            "Tan Chau": "Q_TanChau_Est",
            "Can Tho": "Q_Hau_Est",
            "My Tho": "Q_MyTho_Est",
            "My Thuan": ["Q_CoChien_Est", "Q_HamLuong_Est", "Q_MyTho_Est"],  # Sum of Tien branches
        }
        q_rows_mrc = []
        for _, row in q_annual.iterrows():
            yr = int(row["year"])
            for st, col_or_cols in station_q_map.items():
                if isinstance(col_or_cols, list):
                    val = sum(row.get(c, 0) for c in col_or_cols if c in row.index)
                else:
                    val = row.get(col_or_cols, np.nan)
                q_rows_mrc.append({"stationName": st, "year": yr, "Q_mean": val})
        q_mrc_df = pd.DataFrame(q_rows_mrc)
        merged_mrc2 = df_mrc.merge(q_mrc_df, on=["stationName", "year"], how="left")

        # Water level files (optional)
        wl_dir = ROOT / "analysis" / "analysis_outputs"
        wl_patterns = {
            "Tan Chau": "*Tan_Chau*points.csv",
            "My Thuan": "*My_Thuan*points.csv",
            "My Tho": "*My_Tho*points.csv",
            "Can Tho": "*Can_Tho*points.csv",
        }
        wl_records = []
        for station, pattern in wl_patterns.items():
            matches = glob.glob(str(wl_dir / pattern))
            if not matches:
                continue
            wl = pd.read_csv(matches[0])
            wl["timestamp"] = pd.to_datetime(wl["timestamp"], errors="coerce")
            wl["year"] = wl["timestamp"].dt.year
            annual_wl = wl.groupby("year")["value"].agg(
                WL_mean="mean",
                WL_p95=lambda s: np.nanpercentile(s, 95),
                WL_p05=lambda s: np.nanpercentile(s, 5),
            ).reset_index()
            annual_wl["WL_range"] = annual_wl["WL_p95"] - annual_wl["WL_p05"]
            annual_wl["stationName"] = station
            wl_records.append(annual_wl[["stationName", "year", "WL_mean", "WL_range"]])

        if wl_records:
            wl_df = pd.concat(wl_records, ignore_index=True)
            merged_mrc2 = merged_mrc2.merge(wl_df, on=["stationName", "year"], how="left")

        hydro_extra_mrc = [c for c in ["Q_mean", "WL_mean", "WL_range"]
                           if c in merged_mrc2.columns]
        mrc_x_hydro = mrc_lulc + hydro_extra_mrc
        df_mrc2 = merged_mrc2.dropna(subset=mrc_wq + mrc_x_hydro).copy()
        if len(df_mrc2) >= 5:
            var_mrc2 = _run_rda_variance(df_mrc2[mrc_wq], df_mrc2[mrc_x_hydro])
            rows.append({"Dataset": "MRC (hydro covariates)", "RDA1+RDA2": round(var_mrc2, 3)})
            print(f"    MRC hydro: {var_mrc2:.3f}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Table S3: Model Performance & Generalization (4 panels)")
    print("=" * 60)

    # Load base data
    merged = merge_wq_lulc(bentre_only=True)
    Y, X, groups = prepare_xy(merged, use_display_labels=True)
    print(f"  Data: {len(Y)} station-years, {Y.shape[1]} params, {X.shape[1]} predictors")

    # Build panels
    df_a = panel_a(Y, X, groups)
    df_b = panel_b(merged)
    df_c = panel_c(merged)
    df_d = panel_d()

    # Write multi-panel Markdown
    sections = []
    sections.append(
        "**Panel A.** Random Forest leave-one-station-out cross-validation "
        "(baseline LULC-only model).\n\n"
        + df_a.to_markdown(index=False, floatfmt=".3f")
    )
    sections.append(
        "**Panel B.** Generalization comparison "
        "(RF + Ridge, baseline vs. augmented feature sets, LOSO metrics).\n\n"
        + df_b.to_markdown(index=False, floatfmt=".3f")
    )
    sections.append(
        "**Panel C.** COD variance partition "
        "(LULC-only vs. hydro-only vs. combined).\n\n"
        + df_c.to_markdown(index=False, floatfmt=".3f")
    )
    sections.append(
        "**Panel D.** RDA cross-network variance explained (RDA1+RDA2).\n\n"
        + df_d.to_markdown(index=False, floatfmt=".3f")
    )

    body = "\n\n".join(sections)
    write_table_md(
        content=body,
        out_path=MANU_TABLES / "Supplementary_Table_S3_Model_Performance.md",
        caption=(
            "Supplementary Table S3. Model performance and generalization diagnostics."
        ),
        label="#tbl-s3-performance",
    )

    # Metrics
    all_metrics = pd.concat([
        df_a.assign(Panel="A"),
        df_b.assign(Panel="B"),
        df_c.assign(Panel="C"),
        df_d.assign(Panel="D"),
    ], ignore_index=True)
    write_metrics_csv(all_metrics, "S3_model_performance")

    print("\n  Table S3 written successfully.\n")


if __name__ == "__main__":
    main()
