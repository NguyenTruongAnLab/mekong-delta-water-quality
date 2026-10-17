#!/usr/bin/env python3
"""export_quarto_tables.py

Generate *static* Markdown tables for Quarto from deterministic CSV outputs.

Rationale
---------
The manuscript .qmd files should remain clean (no embedded Python for figure/table
creation). All tables and figures are generated in analysis/ and then included
as static artifacts in manuscript/.

This script:
- Reads canonical CSVs (from analysis/analysis_outputs/ and manuscript/tables/)
- Applies any required post-processing (e.g., FDR correction + pivots)
- Writes Markdown tables (*.md) alongside the CSVs (or into manuscript/tables/)
  with captions + labels for Quarto cross-referencing.

Usage
-----
python -m analysis.export_quarto_tables
python -m analysis.export_quarto_tables --core-only

Notes
-----
If an input CSV is missing, the script writes a placeholder Markdown note so the
manuscript can still render, and prints a warning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_OUT = REPO_ROOT / "analysis" / "analysis_outputs"
MANU_TABLES = REPO_ROOT / "manuscript" / "tables"


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _placeholder_md(out_md: Path, *, label: str, caption: str, missing: Path) -> None:
    _ensure_parent(out_md)
    msg = (
        f"**Table not available (missing source artifact).**\n\n"
        f"Expected source file: `{missing.as_posix()}`\n\n"
        f": {caption} {{{label}}}\n"
    )
    out_md.write_text(msg, encoding="utf-8")


def _write_df_as_md(
    df: pd.DataFrame,
    out_md: Path,
    *,
    label: str,
    caption: str,
    floatfmt: str = ".3f",
) -> None:
    _ensure_parent(out_md)
    table = df.to_markdown(index=False, floatfmt=floatfmt)
    out_md.write_text(f"{table}\n\n: {caption} {{{label}}}\n", encoding="utf-8")


def _read_csv_or_placeholder(src_csv: Path, out_md: Path, *, label: str, caption: str) -> pd.DataFrame | None:
    if not src_csv.exists():
        print(f"⚠ Missing CSV: {src_csv.relative_to(REPO_ROOT)}")
        _placeholder_md(out_md, label=label, caption=caption, missing=src_csv)
        return None
    return pd.read_csv(src_csv)


def export_core_tables() -> None:
    """Tables referenced in main Results + Appendix D that were previously created in-qmd."""

    # --- Results: Phase stats (already exists as CSV in manuscript/tables)
    src = MANU_TABLES / "Table_3_1_Phase_Statistics.csv"
    out = MANU_TABLES / "Table_3_1_Phase_Statistics.md"
    cap = (
        "Phase 1 vs Phase 2 basin-scale statistics (MRC stations). Mean concentrations (±SD) and "
        "Theil–Sen slopes summarized for the pre-break (Phase 1) and post-break (Phase 2) segments."
    )
    df = _read_csv_or_placeholder(src, out, label="#tbl-phase-stats", caption=cap)
    if df is not None:
        keep = [
            "Parameter",
            "Station",
            "Split_Year",
            "Phase1_Mean_SD",
            "Phase2_Mean_SD",
            "Phase1_Slope",
            "Phase2_Slope",
        ]
        keep = [c for c in keep if c in df.columns]
        show = df[keep].copy()
        ren = {
            "Split_Year": "Split Year",
            "Phase1_Mean_SD": "Phase 1 Mean (±SD)",
            "Phase2_Mean_SD": "Phase 2 Mean (±SD)",
            "Phase1_Slope": "Phase 1 Slope",
            "Phase2_Slope": "Phase 2 Slope",
        }
        show = show.rename(columns=ren)
        _write_df_as_md(show, out, label="#tbl-phase-stats", caption=cap, floatfmt=".3f")

    # --- Results: Spearman correlations (needs FDR + pivot formatting)
    src = MANU_TABLES / "Figure_3_6_LULC_Correlations.csv"
    out = MANU_TABLES / "Table_3_6_LULC_Correlations.md"
    cap = (
        "Spearman rank correlation coefficients between water quality parameters and LULC metrics "
        "(16 km buffer, Phase 2: 2005–2022). Significance marks indicate Benjamini–Hochberg FDR q-values "
        "(** q < 0.01; * q < 0.05)."
    )
    df = _read_csv_or_placeholder(src, out, label="#tbl-correlations", caption=cap)
    if df is not None:
        try:
            from statsmodels.stats.multitest import multipletests

            qvals = multipletests(df["P_Value"].astype(float).to_numpy(), method="fdr_bh")[1]
            df = df.copy()
            df["Q_Value"] = qvals

            def fmt_r(row: pd.Series) -> str:
                sig = "**" if row["Q_Value"] < 0.01 else "*" if row["Q_Value"] < 0.05 else ""
                return f"{float(row['Spearman_r']):.2f}{sig}"

            df["Formatted"] = df.apply(fmt_r, axis=1)

            pivot = df.pivot(index="LULC_Metric", columns="WQ_Parameter", values="Formatted")
            # Stable column order when present
            cols = [c for c in ["COD", "NO3", "NH4"] if c in pivot.columns]
            pivot = pivot[cols] if cols else pivot
            pivot = pivot.reset_index().rename(columns={"LULC_Metric": "LULC metric"})
            _write_df_as_md(pivot, out, label="#tbl-correlations", caption=cap, floatfmt=".3f")
        except Exception as e:
            print(f"⚠ Failed to post-process correlations table; writing raw CSV view. Reason: {e}")
            _write_df_as_md(df, out, label="#tbl-correlations", caption=cap, floatfmt=".3f")

    # --- Results: Kruskal-Wallis cluster differences
    src = MANU_TABLES / "Clustering_KruskalWallis.csv"
    out = MANU_TABLES / "Clustering_KruskalWallis.md"
    cap = (
        "Kruskal–Wallis test statistics comparing water quality distributions between LULC clusters "
        "(2005–2022; Ben Tre observations only)."
    )
    df = _read_csv_or_placeholder(src, out, label="#tbl-kw", caption=cap)
    if df is not None:
        show = df.copy()
        if "Significant" in show.columns:
            show = show[show["Significant"].astype(bool)].copy()
        cols = [c for c in ["Parameter", "H_Statistic", "P_Value"] if c in show.columns]
        show = show[cols].copy()
        if "H_Statistic" in show.columns:
            show["H Statistic"] = pd.to_numeric(show["H_Statistic"], errors="coerce").round(1)
            show = show.drop(columns=["H_Statistic"])
        if "P_Value" in show.columns:
            # Most are extremely small; keep a readable bound
            show["p-value"] = "<0.001"
            show = show.drop(columns=["P_Value"])
        _write_df_as_md(show, out, label="#tbl-kw", caption=cap, floatfmt=".3f")

    # --- Appendix D: LULC product consistency (summary)
    src = MANU_TABLES / "Table_A7_LULC_Product_Consistency_2017_2020.csv"
    out = MANU_TABLES / "Table_A7_LULC_Product_Consistency_2017_2020_Summary_OneRowPerYear.md"
    cap = (
        "Summary of the LULC cross-product consistency check (2017–2020). Values report the maximum "
        "absolute difference in harmonized class proportion (percentage points) across the 8 classes for each year, "
        "and the class responsible for the maximum."
    )
    df = _read_csv_or_placeholder(src, out, label="#tbl-a7-lulc-consistency-summary", caption=cap)
    if df is not None:
        d = df.copy()
        if "abs_diff_prop" in d.columns:
            d["abs_pp"] = pd.to_numeric(d["abs_diff_prop"], errors="coerce") * 100
        summary = (
            d.sort_values(["year", "abs_pp"], ascending=[True, False])
            .groupby("year", as_index=False)
            .head(1)
            [["year", "category", "abs_pp"]]
            .rename(columns={"year": "Year", "category": "Max-diff class", "abs_pp": "Max |Δ| (pp)"})
        )
        _write_df_as_md(summary, out, label="#tbl-a7-lulc-consistency-summary", caption=cap, floatfmt=".1f")

    # --- Appendix D: LULC product consistency (validation table)
    src = MANU_TABLES / "Table_A7_LULC_Product_Consistency_2017_2020_Summary.csv"
    out = MANU_TABLES / "Table_A7_LULC_Product_Consistency_2017_2020_Validation.md"
    cap = (
        "Category-level validation summary for cross-product consistency (2017–2020). A category is marked as passing "
        "only if, for all overlap years, both (i) max absolute proportion difference is ≤ 5 percentage points and "
        "(ii) max relative area difference is ≤ 10%."
    )
    df = _read_csv_or_placeholder(src, out, label="#tbl-a7-lulc-consistency-validation", caption=cap)
    if df is not None:
        show = df[[
            c
            for c in ["category", "max_abs_diff_pp", "max_rel_area_diff_pct", "pass_all_years"]
            if c in df.columns
        ]].copy()
        show = show.rename(
            columns={
                "category": "Category",
                "max_abs_diff_pp": "Max |Δ| (pp)",
                "max_rel_area_diff_pct": "Max rel area diff (%)",
                "pass_all_years": "Pass (all years)",
            }
        )
        _write_df_as_md(show, out, label="#tbl-a7-lulc-consistency-validation", caption=cap, floatfmt=".1f")

    # --- Appendix D: LULC evolution snapshots (single combined table)
    src_v = MANU_TABLES / "Table_A7_LULC_StudyArea_Timeseries_VLUCD_1990_2020.csv"
    src_l = MANU_TABLES / "Table_A7_LULC_StudyArea_Timeseries_LULC_2017_2022.csv"
    out = MANU_TABLES / "Table_A7_LULC_Evolution_Snapshots.md"
    cap = "Study-area LULC composition snapshots (selected years). Values are proportions (%) under the harmonized 8-class scheme, shown separately for each product."

    if (not src_v.exists()) or (not src_l.exists()):
        missing = src_v if not src_v.exists() else src_l
        _placeholder_md(out, label="#tbl-a7-lulc-evolution-snapshot", caption=cap, missing=missing)
    else:
        v = pd.read_csv(src_v)
        l = pd.read_csv(src_l)

        def _snap(df_in: pd.DataFrame, years: list[int], product: str) -> pd.DataFrame:
            snap = df_in[df_in["year"].isin(years)].copy()
            wide = (
                snap.pivot_table(index=["year"], columns="category", values="proportion", aggfunc="mean")
                .fillna(0.0)
                * 100
            )
            wide = wide.reset_index().rename(columns={"year": "Year"})
            wide.insert(0, "Product", product)
            return wide

        show = pd.concat(
            [
                _snap(v, [1990, 2005, 2020], "VLUCD 1990–2020"),
                _snap(l, [2017, 2020, 2022], "LULC 2017–2022"),
            ],
            ignore_index=True,
        )
        # keep stable column order
        base_cols = ["Product", "Year"]
        other_cols = [c for c in show.columns if c not in base_cols]
        show = show[base_cols + other_cols]
        _write_df_as_md(show, out, label="#tbl-a7-lulc-evolution-snapshot", caption=cap, floatfmt=".1f")

    # --- Appendix D: population snapshots
    src = MANU_TABLES / "Table_A7_Population_StudyArea_Timeseries.csv"
    out = MANU_TABLES / "Table_A7_Population_StudyArea_Snapshots.md"
    cap = "Population snapshots within the study boundary (selected years)."
    df = _read_csv_or_placeholder(src, out, label="#tbl-a7-pop-snapshot", caption=cap)
    if df is not None:
        sel = df[df["year"].isin([1990, 2005, 2022])].copy()
        if "pop_total" in sel.columns:
            sel["Population (millions)"] = pd.to_numeric(sel["pop_total"], errors="coerce") / 1e6
        if "pop_density" in sel.columns:
            sel["Density (persons/km²)"] = pd.to_numeric(sel["pop_density"], errors="coerce")
        show = sel[[c for c in ["year", "Population (millions)", "Density (persons/km²)"] if c in sel.columns]].copy()
        show = show.rename(columns={"year": "Year"})
        _write_df_as_md(show, out, label="#tbl-a7-pop-snapshot", caption=cap, floatfmt=".2f")

    # --- Appendix D: population vs fragmentation metrics
    src = MANU_TABLES / "Table_A7_Population_vs_Fragmentation_Metrics.csv"
    out = MANU_TABLES / "Table_A7_Population_vs_Fragmentation_Metrics.md"
    cap = "Spearman diagnostics for the population–fragmentation validation (Ben Tre-only station set)."
    df = _read_csv_or_placeholder(src, out, label="#tbl-a7-pop-frag-metrics", caption=cap)
    if df is not None:
        d = df.copy()
        if "station_set" in d.columns:
            d = d[d["station_set"] == "ben_tre_only"].copy()
        keep = [c for c in ["station_set", "metric", "spearman_rho", "p_value", "n"] if c in d.columns]
        d = d[keep].rename(
            columns={
                "station_set": "Station set",
                "metric": "Test",
                "spearman_rho": "Spearman ρ",
                "p_value": "p",
                "n": "n",
            }
        )
        _write_df_as_md(d, out, label="#tbl-a7-pop-frag-metrics", caption=cap, floatfmt=".3f")

    # --- Appendix C: seasonal rainfall decomposition
    src = ANALYSIS_OUT / "rainfall_seasonal_trends_latest.csv"
    out = MANU_TABLES / "Table_C_Rainfall_Seasonal_Trends.md"
    cap = "Seasonal rainfall trends (1985–2022) from independent precipitation products. Trends are linear slopes (mm/yr) for seasonal total rainfall."
    df = _read_csv_or_placeholder(src, out, label="#tbl-rain-seasonal", caption=cap)
    if df is not None:
        keep = [c for c in ["Dataset", "Season", "Mean_mm", "Trend_mm_per_year", "Trend_pvalue"] if c in df.columns]
        show = df[keep].copy()
        show = show.rename(
            columns={
                "Mean_mm": "Mean seasonal rainfall (mm)",
                "Trend_mm_per_year": "Trend (mm/yr)",
                "Trend_pvalue": "p",
            }
        )
        _write_df_as_md(show, out, label="#tbl-rain-seasonal", caption=cap, floatfmt=".2f")

    # --- Appendix C: endpoint load–dilution decomposition
    src = ANALYSIS_OUT / "cod_mass_balance_decomposition_latest.csv"
    out = MANU_TABLES / "Table_C_COD_Mass_Balance_Decomposition.md"
    cap = "Endpoint decomposition of COD concentration change into load-change vs dilution-change components (2005–2022). Discharge values are reconstructed/proxy-based; this table is intended as supporting evidence rather than a calibrated budget."
    df = _read_csv_or_placeholder(src, out, label="#tbl-cod-mass-balance", caption=cap)
    if df is not None:
        keep = [
            c
            for c in [
                "Station",
                "Year0",
                "Year1",
                "dC_total_mgL",
                "dC_load_mgL",
                "dC_dilution_mgL",
                "pct_load",
                "pct_dilution",
            ]
            if c in df.columns
        ]
        show = df[keep].copy()
        show = show.rename(
            columns={
                "Year0": "Start year",
                "Year1": "End year",
                "dC_total_mgL": "ΔC total (mg/L)",
                "dC_load_mgL": "ΔC from load (mg/L)",
                "dC_dilution_mgL": "ΔC from dilution (mg/L)",
                "pct_load": "Load share (%)",
                "pct_dilution": "Dilution share (%)",
            }
        )
        _write_df_as_md(show, out, label="#tbl-cod-mass-balance", caption=cap, floatfmt=".2f")

    # --- Appendix C: load intensity diagnostic vs urban fragmentation (two labels used in text)
    src = ANALYSIS_OUT / "cod_load_intensity_by_urban_fragmentation_latest.csv"

    out1 = MANU_TABLES / "Table_C_COD_Load_Intensity_By_Urban_PD_Summary.md"
    cap1 = "COD load intensity (kg/ha/day) summarized across Urban Patch Density quantiles (Ben Tre branches, 2005–2020 where LULC metrics are available)."
    df = _read_csv_or_placeholder(src, out1, label="#tbl-load-frag", caption=cap1)
    if df is not None:
        keep = [
            c
            for c in [
                "Urban_PD_quantile",
                "N",
                "Urban_pct_median",
                "COD_Load_tons_day_mean",
                "COD_Load_tons_day_sd",
                "COD_kg_ha_buffer_day_mean",
                "COD_kg_ha_buffer_day_sd",
            ]
            if c in df.columns
        ]
        show = df[keep].copy().rename(
            columns={
                "Urban_PD_quantile": "Urban PD quantile",
                "N": "N (station-years)",
                "Urban_pct_median": "Median Urban %",
                "COD_Load_tons_day_mean": "Mean load (tons/day)",
                "COD_Load_tons_day_sd": "SD load (tons/day)",
                "COD_kg_ha_buffer_day_mean": "Mean intensity (kg/ha/day)",
                "COD_kg_ha_buffer_day_sd": "SD intensity (kg/ha/day)",
            }
        )
        _write_df_as_md(show, out1, label="#tbl-load-frag", caption=cap1, floatfmt=".2f")

    out2 = MANU_TABLES / "Table_C_COD_Load_Intensity_By_Urban_PD_Compact.md"
    cap2 = "Proxy-based COD load intensity by Urban Patch Density (PD) quantile (Ben Tre branches). Load intensities are expressed per buffer area (kg/ha/day) using branch discharge proxies; values summarize station-years and should be interpreted as diagnostic evidence rather than calibrated mass balance."
    df = _read_csv_or_placeholder(src, out2, label="#tbl-cod-load-intensity-urban-frag", caption=cap2)
    if df is not None:
        keep = [
            c
            for c in [
                "Urban_PD_quantile",
                "N",
                "Urban_PD_median",
                "Urban_pct_median",
                "COD_kg_ha_buffer_day_mean",
                "COD_kg_ha_buffer_day_sd",
            ]
            if c in df.columns
        ]
        show = df[keep].copy().rename(
            columns={
                "Urban_PD_quantile": "Urban PD quantile",
                "N": "N",
                "Urban_PD_median": "Median Urban PD",
                "Urban_pct_median": "Median Urban %",
                "COD_kg_ha_buffer_day_mean": "COD (kg/ha/day) mean",
                "COD_kg_ha_buffer_day_sd": "SD",
            }
        )
        _write_df_as_md(show, out2, label="#tbl-cod-load-intensity-urban-frag", caption=cap2, floatfmt=".3f")

    # --- Appendix C: tidal influence indices
    src = ANALYSIS_OUT / "Table_Tidal_Influence_Indices.csv"
    out = MANU_TABLES / "Table_C_Tidal_Influence_Indices.md"
    cap = "Tidal influence diagnostics for the two main branches. Daily tidal range is computed from hourly water levels as (daily max – daily min). The optional dry-season index normalizes tidal range by dry-season mean discharge to provide a simple residence-time/flushing proxy."
    df = _read_csv_or_placeholder(src, out, label="#tbl-tidal-influence", caption=cap)
    if df is not None:
        _write_df_as_md(df, out, label="#tbl-tidal-influence", caption=cap, floatfmt=".3g")


def export_supplementary_tables() -> None:
    """Static Markdown versions of Supplementary tables (S1–S22)."""

    nw_metrics_dir = REPO_ROOT / "scripts" / "99_publish" / "nw_tables" / "metrics"
    s2_metrics_path = nw_metrics_dir / "S2_shap_importance_metrics.csv"
    s3_metrics_path = nw_metrics_dir / "S3_model_performance_metrics.csv"
    s4_metrics_path = nw_metrics_dir / "S4_cod_robustness_metrics.csv"
    s5_metrics_path = nw_metrics_dir / "S5_cod_fixed_effects_metrics.csv"

    s2_df = pd.read_csv(s2_metrics_path) if s2_metrics_path.exists() else None
    s3_df = pd.read_csv(s3_metrics_path) if s3_metrics_path.exists() else None
    s4_df = pd.read_csv(s4_metrics_path) if s4_metrics_path.exists() else None

    mappings: list[dict] = [
        {
            "label": "#tbl-rda-scores",
            "caption": "Supplementary Table S1. RDA biplot scores and vector lengths.",
            "src": ANALYSIS_OUT / "rda_biplot_scores_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S1_RDA_Biplot_Scores.md",
            "cols": ["Driver", "RDA1", "RDA2", "Length"],
            "rename": {"Driver": "LULC Driver", "RDA1": "RDA1 Score", "RDA2": "RDA2 Score", "Length": "Vector Length"},
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-vif",
            "caption": "Supplementary Table S2. Variance inflation factors (VIF) for LULC predictors.",
            "src": ANALYSIS_OUT / "vif_analysis_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S2_VIF.md",
            "floatfmt": ".1f",
        },
        {
            "label": "#tbl-shap-global",
            "caption": "Supplementary Table S3. Global SHAP feature-importance rankings (cross-validated).",
            "src": ANALYSIS_OUT / "shap_global_importance_cv_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S3_SHAP_Global.md",
            "cols": ["Rank", "Driver", "Mean_Abs_SHAP", "SHAP_Corr_r", "Direction"],
            "rename": {"Mean_Abs_SHAP": "Mean |SHAP|"},
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-shap-param",
            "caption": "Supplementary Table S4. Parameter-specific dominant drivers (cross-validated SHAP; rank 1 only).",
            "src": ANALYSIS_OUT / "shap_by_parameter_cv_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S4_SHAP_ByParameter_Top1.md",
            "floatfmt": ".3f",
            "top1": True,
        },
        {
            "label": "#tbl-rf-cv",
            "caption": "Supplementary Table S5. Random Forest model performance using leave-one-station-out cross-validation.",
            "src": ANALYSIS_OUT / "rf_robustness_metrics_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S5_RF_Performance.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-shap-global-orth",
            "caption": "Supplementary Table S6. Sensitivity: global SHAP rankings using orthogonalized configuration metrics (cross-validated).",
            "src": ANALYSIS_OUT / "shap_global_importance_cv_orth_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S6_SHAP_Global_Orth.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-shap-param-orth",
            "caption": "Supplementary Table S7. Sensitivity: parameter-specific dominant drivers using orthogonalized configuration metrics (rank 1 only).",
            "src": ANALYSIS_OUT / "shap_by_parameter_cv_orth_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S7_SHAP_ByParameter_Orth_Top1.md",
            "floatfmt": ".3f",
            "top1": True,
        },
        {
            "label": "#tbl-shap-salinity",
            "caption": "Supplementary Table S8. Sensitivity: COD model with salinity covariate (top drivers).",
            "src": ANALYSIS_OUT / "shap_cod_with_salinity_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S8_SHAP_COD_Salinity.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-shap-cod-response-robustness",
            "caption": "Supplementary Table S9. Robustness: top SHAP driver rankings for COD across alternative response formulations.",
            "src": ANALYSIS_OUT / "cod_shap_robustness_top_features_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S9_COD_Response_Robustness.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-cod-fixed-effects",
            "caption": "Supplementary Table S10. Confirmatory check: fixed-effects regression for COD (19 Ben Tre stations, 2005–2020).",
            "src": s5_metrics_path if s5_metrics_path.exists() else (ANALYSIS_OUT / "cod_confirmatory_fixed_effects_latest.csv"),
            "out": MANU_TABLES / "Supplementary_Table_S10_COD_FixedEffects.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-lulc-uncertainty",
            "caption": "Supplementary Table S11. Sensitivity: COD driver-rank stability under Monte Carlo perturbation of LULC predictors.",
            "src": ANALYSIS_OUT / "lulc_uncertainty_cod_shap_stability_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S11_LULC_Uncertainty.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-generalization",
            "caption": "Supplementary Table S12. Enhanced spatial generalization diagnostic (Ben Tre, 2005–2020 station-year master, 6 Baseline vs. 11 Primary Augmented features).",
            "src": ANALYSIS_OUT / "generalization_model_comparison_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S12_Generalization.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-generalization-focus",
            "caption": "Supplementary Table S13. Focused view of the enhanced generalization diagnostic for COD and NO3-N (2005–2020).",
            "src": ANALYSIS_OUT / "generalization_model_comparison_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S13_Generalization_Focus.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-cod-variance-partition",
            "caption": "Supplementary Table S14. Variance partitioning diagnostic for COD (2005–2020 Ben Tre station-year master, N = 229).",
            "src": ANALYSIS_OUT / "cod_variance_partitioning_loso_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S14_COD_VariancePartition.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-rda-mrc",
            "caption": "Supplementary Table S15. MRC-only sensitivity (1990–2020): RDA constrained variance explained.",
            "src": ANALYSIS_OUT / "rda_variance_mrc_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S15_RDA_MRC.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-rf-mrc",
            "caption": "Supplementary Table S16. MRC-only sensitivity (1990–2020): grouped CV performance by parameter.",
            "src": ANALYSIS_OUT / "rf_model_performance_mrc_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S16_RF_MRC.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-rda-mrc-hydro",
            "caption": "Supplementary Table S17. MRC-only sensitivity with hydrodynamic covariates: RDA constrained variance explained.",
            "src": ANALYSIS_OUT / "rda_variance_mrc_hydro_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S17_RDA_MRC_Hydro.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-rf-mrc-hydro",
            "caption": "Supplementary Table S18. MRC-only sensitivity with hydrodynamic covariates: grouped CV performance by parameter.",
            "src": ANALYSIS_OUT / "rf_model_performance_mrc_hydro_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S18_RF_MRC_Hydro.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-rda-bentre",
            "caption": "Supplementary Table S19. Ben Tre-only sensitivity (2005–2022): RDA constrained variance explained.",
            "src": ANALYSIS_OUT / "rda_variance_bentre_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S19_RDA_BenTre.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-rda-bentre-hydro",
            "caption": "Supplementary Table S20. Ben Tre hydro-proxy sensitivity: RDA constrained variance explained.",
            "src": ANALYSIS_OUT / "rda_variance_bentre_hydro_proxy_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S20_RDA_BenTre_HydroProxy.md",
            "floatfmt": ".3f",
        },
        {
            "label": "#tbl-rf-bentre-hydro",
            "caption": "Supplementary Table S21. Ben Tre hydro-proxy sensitivity: grouped CV performance by parameter.",
            "src": ANALYSIS_OUT / "rf_model_performance_bentre_hydro_proxy_latest.csv",
            "out": MANU_TABLES / "Supplementary_Table_S21_RF_BenTre_HydroProxy.md",
            "floatfmt": ".3f",
        },
    ]

    for m in mappings:
        # Override SHAP and generalization/variance-partition tables with harmonized nw_tables metrics when available
        if m["label"] == "#tbl-shap-global" and s2_df is not None:
            show = s2_df[(s2_df["Panel"] == "A: Global") & (s2_df["Specification"] == "Baseline")].copy()
            show = show.rename(columns={"Mean SHAP": "Mean_Abs_SHAP"})
        elif m["label"] == "#tbl-shap-param" and s2_df is not None:
            show = s2_df[(s2_df["Panel"] == "B: Per-parameter") & (s2_df["Specification"] == "Baseline")].copy()
            show = show.rename(columns={"Mean SHAP": "Mean_Abs_SHAP"})
        elif m["label"] == "#tbl-shap-global-orth" and s2_df is not None:
            show = s2_df[(s2_df["Panel"] == "A: Global") & (s2_df["Specification"] == "Orthogonalized")].copy()
            show = show.rename(columns={"Mean SHAP": "Mean_Abs_SHAP"})
        elif m["label"] == "#tbl-shap-param-orth" and s2_df is not None:
            show = s2_df[(s2_df["Panel"] == "B: Per-parameter") & (s2_df["Specification"] == "Orthogonalized")].copy()
            show = show.rename(columns={"Mean SHAP": "Mean_Abs_SHAP"})
        elif m["label"] == "#tbl-shap-salinity" and s4_df is not None:
            show = s4_df[s4_df["Panel"] == "A"].drop(columns=["Panel"], errors="ignore").copy()
        elif m["label"] == "#tbl-shap-cod-response-robustness" and s4_df is not None:
            show = s4_df[s4_df["Panel"] == "B"].drop(columns=["Panel"], errors="ignore").copy()
        elif m["label"] in ("#tbl-generalization", "#tbl-generalization-focus") and s3_df is not None:
            show = s3_df[s3_df["Panel"] == "B"][
                ["Parameter", "Feature_Set", "Model", "LOSO_R2", "LOSO_RMSE", "LOSO_Spearman_rho", "N_samples", "N_features"]
            ].copy()
            if m["label"] == "#tbl-generalization-focus":
                show = show[show["Parameter"].isin(["COD", "NO3-N"])].copy()
        elif m["label"] == "#tbl-cod-variance-partition" and s3_df is not None:
            show = s3_df[s3_df["Panel"] == "C"][
                ["Predictor_Set", "LOSO_R2", "LOSO_R2_RF", "LOSO_RMSE", "LOSO_Spearman_rho", "OLS_R2", "N", "N_features"]
            ].copy()
        else:
            df = _read_csv_or_placeholder(m["src"], m["out"], label=m["label"], caption=m["caption"])
            if df is None:
                continue
            show = df.copy()

        if m.get("top1") and "Rank" in show.columns:
            show = show[show["Rank"] == 1].copy()

        cols = m.get("cols")
        if cols:
            cols = [c for c in cols if c in show.columns]
            show = show[cols].copy()

        rename = m.get("rename")
        if rename:
            show = show.rename(columns=rename)

        # VIF: normalize schema and sort
        if m["label"] == "#tbl-vif":
            cols = list(show.columns)
            if "feature" in cols and "VIF" in cols:
                show = show[["feature", "VIF"]].rename(columns={"feature": "Predictor"})
            elif "Predictor" not in cols and "VIF" in cols:
                pass
            if "VIF" in show.columns:
                show["VIF"] = pd.to_numeric(show["VIF"], errors="coerce")
                show = show.sort_values("VIF", ascending=False)

        # COD FE: keep key terms when present
        if m["label"] == "#tbl-cod-fixed-effects" and "term" in show.columns:
            keep_terms = [
                "Urban_PD_z",
                "Aqua_ED_z",
                "Rice_PD_z",
                "Salinity_z",
                "Year_z",
                "Q_MyThuan_AnnualMean_z",
                "Q_MyThuan_DrySeasonMean_z",
                "Rain_Annual_z",
                "Rain_Dry_z",
                "Q_branch_AnnualMean_z",
            ]
            show = show[show["term"].isin(keep_terms)].copy()

        _write_df_as_md(
            show,
            m["out"],
            label=m["label"],
            caption=m["caption"],
            floatfmt=m.get("floatfmt", ".3f"),
        )

    # Table S22: computed comparison from up to four RDA variance files
    out = MANU_TABLES / "Supplementary_Table_S22_RDA_Compare.md"
    cap = "Supplementary Table S22. Cross-network comparison of RDA variance explained (RDA1+RDA2)."

    def _load_rda_total(path: Path, label: str) -> dict | None:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "RDA_total_RDA1_RDA2" in df.columns:
            val = float(df["RDA_total_RDA1_RDA2"].iloc[0])
        else:
            # fallback: first row, last column
            val = float(df.iloc[0, -1])
        return {"Dataset": label, "RDA1+RDA2": val}

    rows = []
    for p, lbl in [
        (ANALYSIS_OUT / "rda_variance_bentre_latest.csv", "Ben Tre (LULC only)"),
        (ANALYSIS_OUT / "rda_variance_bentre_hydro_proxy_latest.csv", "Ben Tre (hydro-proxy)"),
        (ANALYSIS_OUT / "rda_variance_mrc_latest.csv", "MRC (LULC only)"),
        (ANALYSIS_OUT / "rda_variance_mrc_hydro_latest.csv", "MRC (hydro covariates)"),
    ]:
        r = _load_rda_total(p, lbl)
        if r is not None:
            rows.append(r)

    if not rows:
        _placeholder_md(out, label="#tbl-rda-compare", caption=cap, missing=ANALYSIS_OUT / "rda_variance_*.csv")
    else:
        show = pd.DataFrame(rows)
        _write_df_as_md(show, out, label="#tbl-rda-compare", caption=cap, floatfmt=".3f")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export static Markdown tables for Quarto.")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only export core manuscript tables (skip Supplementary tables).",
    )
    args = parser.parse_args(argv)

    export_core_tables()
    if not args.core_only:
        export_supplementary_tables()


if __name__ == "__main__":
    main()
