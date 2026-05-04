#!/usr/bin/env python3
"""
Figure 5 — Driver Attribution
================================
(a) Variance decomposition (station / year / residual)
(b) LOSO R² comparison (LULC-only vs Primary/Hydro-augmented)
(c) SHAP feature importance (COD vs DO)

ALL values are computed or read from analysis CSVs — nothing hardcoded.

Exported metrics CSV:
  - Variance fractions per parameter
  - LOSO R² per parameter per feature set
  - SHAP rankings for COD and DO
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from _nw_common import (
    apply_style, load_master, load_bt,
    MM, SHAP_CSV, VP_CSV, PARAM_CLR,
    CLR_SPATIAL, CLR_TEMPORAL, CLR_RESIDUAL,
    panel_label, save_fig, require_file,
)

apply_style()


def figure_5() -> None:
    master = load_master()
    # Variance decomposition uses Ben Tre only (19 stations)
    # to avoid inflating between-station variance with the MRC permanganate/dichromate COD offset.
    all_stations = load_bt(master)

    metrics_rows = []

    fig = plt.figure(figsize=(180 * MM, 68 * MM))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45,
                           width_ratios=[1, 0.9, 1.1])

    # ═══════════════════════════════════════════════════════════════════
    # (a) Variance decomposition — computed from data
    # ═══════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0])

    # Core parameters: COD, DO, NH4, TN, TSS (matching manuscript)
    params_vd = {
        "COD (mg/L)_annual": "COD",
        "DO (mg/L)_annual":  "DO",
        "NH4-N (mg/L)_annual": r"NH$_4^+$",
        "TN (mg/L)_annual": "TN",
        "TSS (mg/L)_annual": "TSS",
    }
    results_vd = []
    for col, label in params_vd.items():
        if col not in all_stations.columns:
            continue
        sub = all_stations[["stationName", "year", col]].dropna()
        if len(sub) < 20:
            continue
        gm = sub[col].mean()
        ss_total = ((sub[col] - gm) ** 2).sum()
        ss_stn = sum(
            n * (m - gm) ** 2
            for m, n in zip(sub.groupby("stationName")[col].mean(),
                            sub.groupby("stationName")[col].count()))
        ss_yr = sum(
            n * (m - gm) ** 2
            for m, n in zip(sub.groupby("year")[col].mean(),
                            sub.groupby("year")[col].count()))
        ss_res = max(0, ss_total - ss_stn - ss_yr)
        frac_stn = ss_stn / ss_total if ss_total else 0
        frac_yr  = ss_yr / ss_total if ss_total else 0
        frac_res = ss_res / ss_total if ss_total else 0
        results_vd.append({
            "param": label, "stn": frac_stn, "yr": frac_yr, "res": frac_res,
        })
        metrics_rows.append({
            "Panel": "a_variance_decomp", "Parameter": label,
            "Station_frac": round(frac_stn, 4),
            "Year_frac": round(frac_yr, 4),
            "Residual_frac": round(frac_res, 4),
            "N": int(len(sub)),
        })

    df_vd = pd.DataFrame(results_vd)
    param_order = ["COD", "DO", r"NH$_4^+$", "TN", "TSS"]
    df_vd["sort_key"] = df_vd["param"].map({p: i for i, p in enumerate(param_order)})
    df_vd = df_vd.sort_values("sort_key")

    x_vd = np.arange(len(df_vd))
    w_vd = 0.65
    stn_vals = df_vd["stn"].to_numpy(dtype=float)
    yr_vals = df_vd["yr"].to_numpy(dtype=float)
    res_vals = df_vd["res"].to_numpy(dtype=float)

    ax.barh(x_vd, stn_vals, w_vd, color=CLR_SPATIAL, alpha=0.85,
            edgecolor="white", linewidth=0.3, label="Spatial", zorder=3)
    ax.barh(x_vd, yr_vals, w_vd, left=stn_vals,
            color=CLR_TEMPORAL, alpha=0.85,
            edgecolor="white", linewidth=0.3, label="Temporal", zorder=3)
    ax.barh(x_vd, res_vals, w_vd,
            left=(stn_vals + yr_vals),
            color=CLR_RESIDUAL, alpha=0.7,
            edgecolor="white", linewidth=0.3, label="Residual", zorder=3)

    for i, (_, row) in enumerate(df_vd.iterrows()):
        if row["stn"] > 0.08:
                ax.text(row["stn"] / 2, i, f"{row['stn']:.0%}",
                    ha="center", va="center", fontsize=6.2,
                    fontweight="bold", color="white")
        if row["yr"] > 0.08:
            ax.text(row["stn"] + row["yr"] / 2, i, f"{row['yr']:.0%}",
                    ha="center", va="center", fontsize=6.2,
                    fontweight="bold", color="white")

    ax.set_yticks(x_vd)
    ax.set_yticklabels(df_vd["param"])
    ax.set_xlabel("Fraction of variance")
    ax.set_xlim(0, 1.02)
    # Single-row legend above plot
    ax.legend(fontsize=6.2,
              loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0,
              ncol=3, frameon=True, framealpha=0.9,
              edgecolor="none", handlelength=1.0)
    ax.invert_yaxis()
    panel_label(ax, "(a) Variance decomposition", fontsize=8, y=1.18)

    # ═══════════════════════════════════════════════════════════════════
    # (b) LOSO R² — READ from analysis CSV, not hardcoded
    # ═══════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1])

    require_file(VP_CSV, label="primary_variance_partitioning_latest.csv (VP_CSV)")
    if VP_CSV.exists():
        vp = pd.read_csv(VP_CSV)
        # Extract LULC-only and Primary LOSO R² per parameter
        # Core parameters: COD, DO, NH4, TN, TSS
        loso_data = {"param": [], "LULC": [], "Aug": []}
        param_name_map = {
            "COD": "COD", "DO": "DO", "NH4-N": r"NH$_4^+$",
            "TN": "TN", "TSS": "TSS",
        }
        for csv_name, display_name in param_name_map.items():
            lulc_row = vp[(vp["Parameter"] == csv_name) &
                          (vp["Predictor_Set"] == "LULC-only")]
            aug_row = vp[(vp["Parameter"] == csv_name) &
                         (vp["Predictor_Set"] == "Primary (LULC+Pop+Hydro)")]
            if not lulc_row.empty and not aug_row.empty:
                loso_data["param"].append(display_name)
                lulc_r2 = float(lulc_row["LOSO_R2_Ridge"].iloc[0])
                aug_r2 = float(aug_row["LOSO_R2_Ridge"].iloc[0])
                loso_data["LULC"].append(lulc_r2)
                loso_data["Aug"].append(aug_r2)
                metrics_rows.append({
                    "Panel": "b_LOSO_R2", "Parameter": display_name,
                    "LULC_only_LOSO_R2": round(lulc_r2, 4),
                    "Augmented_LOSO_R2": round(aug_r2, 4),
                    "N": int(lulc_row["N"].iloc[0]),
                })

        df_l = pd.DataFrame(loso_data)
    else:
        # Dev-only: placeholder mode
        df_l = pd.DataFrame({
            "param": ["COD", "DO", r"NH$_4^+$", "TN", "TSS"],
            "LULC":  [0.0, 0.0, 0.0, 0.0, 0.0],
            "Aug":   [0.0, 0.0, 0.0, 0.0, 0.0],
        })

    x_l = np.arange(len(df_l))
    w_l = 0.35

    ax.barh(x_l - w_l / 2, df_l["LULC"], w_l, color=CLR_SPATIAL, alpha=0.85,
            edgecolor="white", linewidth=0.3, label="LULC-only", zorder=3)
    ax.barh(x_l + w_l / 2, df_l["Aug"], w_l, color=CLR_TEMPORAL, alpha=0.85,
            edgecolor="white", linewidth=0.3, label="Augmented", zorder=3)

    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(x_l)
    ax.set_yticklabels(df_l["param"])
    ax.set_xlabel(r"LOSO R$^2$")
    ax.legend(fontsize=6.2,
              loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0,
              ncol=2, frameon=True, framealpha=0.9, edgecolor="none")
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], 0, color="#fee0d2", alpha=0.15, zorder=0)
    # Position "fails generalisation" text just below legend, above the plot data
    ax.text(0.02, 0.92, "fails\ngeneralisation",
            fontsize=6.0, color="#b2182b", ha="left", va="top",
            style="italic", transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    ax.invert_yaxis()
    panel_label(ax, "(b) Spatial prediction skill", fontsize=8, y=1.18)

    # ═══════════════════════════════════════════════════════════════════
    # (c) SHAP — READ from analysis CSV
    # ═══════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[2])

    require_file(SHAP_CSV, label="primary_model_shap_latest.csv (SHAP_CSV)")
    if SHAP_CSV.exists():
        shap_df = pd.read_csv(SHAP_CSV)
        # Top features for COD and DO from Primary model
        cod_shap = shap_df[
            (shap_df["Parameter"] == "COD") &
            (shap_df["Feature_Set"] == "Primary (LULC+Pop+Hydro)")
        ].sort_values("Rank").head(6)
        do_shap = shap_df[
            (shap_df["Parameter"] == "DO") &
            (shap_df["Feature_Set"] == "Primary (LULC+Pop+Hydro)")
        ].sort_values("Rank").head(6)

        all_feats = list(dict.fromkeys(
            list(cod_shap["Driver"].values) + list(do_shap["Driver"].values)))
        cod_vals = {r["Driver"]: r["Mean_Abs_SHAP"] for _, r in cod_shap.iterrows()}
        do_vals  = {r["Driver"]: r["Mean_Abs_SHAP"] for _, r in do_shap.iterrows()}

        for f in all_feats:
            metrics_rows.append({
                "Panel": "c_SHAP", "Parameter": "COD",
                "Feature": f, "Mean_Abs_SHAP": round(cod_vals.get(f, 0), 4),
            })
            metrics_rows.append({
                "Panel": "c_SHAP", "Parameter": "DO",
                "Feature": f, "Mean_Abs_SHAP": round(do_vals.get(f, 0), 4),
            })

        y_s = np.arange(len(all_feats))
        w_s = 0.35

        ax.barh(y_s - w_s / 2,
                [cod_vals.get(f, 0) for f in all_feats], w_s,
            color=PARAM_CLR.get("COD", "#e66101"), alpha=0.85, edgecolor="white", linewidth=0.3,
                label="COD", zorder=3)
        ax.barh(y_s + w_s / 2,
                [do_vals.get(f, 0) for f in all_feats], w_s,
            color="#1b9e77", alpha=0.9, edgecolor="white", linewidth=0.3,
                label="DO", zorder=3)

        ax.set_yticks(y_s)
        feat_labels = [f.replace("Tidal range (m)", "Tidal range")
                        .replace("Population density", "Pop. density")
                        .replace("Rainfall (dry z)", "Rainfall (dry)")
                        .replace("Rainfall (annual z)", "Rainfall (ann.)")
                        .replace("Discharge (My Thuan, annual z)", "Discharge")
                        for f in all_feats]
        ax.set_yticklabels(feat_labels, fontsize=7.2)
        ax.set_xlabel("Mean |SHAP|")
        ax.legend(fontsize=6.5, loc="lower right", frameon=True, framealpha=0.9,
                  edgecolor="none")
        ax.invert_yaxis()
    else:
        # Dev-only: placeholder mode
        ax.text(0.5, 0.5, "SHAP data\nnot available", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#969696")
    panel_label(ax, "(c) Feature importance", fontsize=8, y=1.18)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    save_fig(fig, "Figure_5_Driver_Attribution", metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    figure_5()
