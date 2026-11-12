#!/usr/bin/env python3
"""
CEE Figure 6 — Spatiotemporal Variance, Cross-Validation, SHAP, and Fixed Effects
=================================================================================
Cloned directly from `nw_fig4_attribution.py` to preserve 100% of the tuned 2x2
GridSpec (180 x 155 mm) and subplot `d` vertical forest-plot layout.
Surgical upgrades:
  1. Adds chloride-immune `BOD_5` alongside `COD` in Panels (a) and (b).
  2. Uses the parsimonious single-discharge station fixed-effects specification
     (`H1_COD_StationFE_All19Stns` from `cee_fixed_effects_comprehensive_suite.csv`)
     with station-clustered 95% CIs in Panel (d), eliminating dual-Q collinearity.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D

from _cee_common import (
    apply_style, load_master, load_bt, load_wq,
    MM, SHAP_CSV, VP_CSV, PARAM_CLR,
    CLR_SPATIAL, CLR_TEMPORAL, CLR_RESIDUAL, CEE_VERIFY_DIR,
    panel_label, save_fig,
)

apply_style()


def _clean_fe_term_label(term: str) -> str:
    label_map = {
        "Q_Ann_z":      "Q My Thuan (annual)",
        "Year_z":       "Year",
        "Salinity_z":   "Salinity",
        "Urban_PD_z":   "Urban PD",
        "Aqua_ED_z":    "Aquaculture ED",
        "Rice_PD_z":    "Rice PD",
        "Rain_Ann_z":   "Rain (annual)",
        "Rain_Dry_z2":  "Rain (dry)",
    }
    return label_map.get(str(term).strip(), str(term).replace("_z", "").replace("_", " "))


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.nanstd(x, ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - float(np.nanmean(x))) / sd


def main() -> None:
    master = load_master()
    bt = load_bt(master)
    wq = load_wq()
    bt_wq = wq[wq["Data source"] == "Ben Tre"].copy()

    metrics_rows = []

    fig = plt.figure(figsize=(180 * MM, 155 * MM))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.38,
                           width_ratios=[1.0, 1.2], height_ratios=[1.0, 1.0])

    # ── (a) Variance decomposition (with chloride-immune BOD5) ──
    ax_a = fig.add_subplot(gs[0, 0])

    params_vd = [
        ("COD (mg/L)_annual",  "COD",           "master"),
        ("BOD5 (mg/L)",        r"BOD$_5$",      "wq"),
        ("DO (mg/L)_annual",   "DO",            "master"),
        ("NH4-N (mg/L)_annual", r"NH$_4^+$-N",  "master"),
        ("TSS (mg/L)_annual",  "TSS",           "master"),
    ]
    results_vd = []
    for col, label, src in params_vd:
        if src == "master":
            if col not in bt.columns:
                continue
            sub = bt[["stationName", "year", col]].dropna().rename(columns={col: "val"})
        else:
            if col not in bt_wq.columns:
                continue
            sub = (
                bt_wq.groupby(["stationName", "year"], as_index=False)[col]
                .mean()
                .dropna()
                .rename(columns={col: "val"})
            )
        if len(sub) < 20:
            continue
        gm = sub["val"].mean()
        ss_total = ((sub["val"] - gm) ** 2).sum()
        ss_stn = sum(
            n * (m - gm) ** 2
            for m, n in zip(sub.groupby("stationName")["val"].mean(),
                            sub.groupby("stationName")["val"].count()))
        ss_yr = sum(
            n * (m - gm) ** 2
            for m, n in zip(sub.groupby("year")["val"].mean(),
                            sub.groupby("year")["val"].count()))
        ss_res = max(0.0, ss_total - ss_stn - ss_yr)
        frac_stn = ss_stn / ss_total if ss_total else 0.0
        frac_yr  = ss_yr / ss_total if ss_total else 0.0
        frac_res = ss_res / ss_total if ss_total else 0.0
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
    x_vd = np.arange(len(df_vd))
    w_vd = 0.65
    stn_vals = df_vd["stn"].to_numpy(dtype=float)
    yr_vals = df_vd["yr"].to_numpy(dtype=float)
    res_vals = df_vd["res"].to_numpy(dtype=float)

    ax_a.barh(x_vd, stn_vals, w_vd, color=CLR_SPATIAL, alpha=0.85,
              edgecolor="white", linewidth=0.3, label="Spatial", zorder=3)
    ax_a.barh(x_vd, yr_vals, w_vd, left=stn_vals, color=CLR_TEMPORAL, alpha=0.85,
              edgecolor="white", linewidth=0.3, label="Temporal", zorder=3)
    ax_a.barh(x_vd, res_vals, w_vd, left=(stn_vals + yr_vals), color=CLR_RESIDUAL, alpha=0.7,
              edgecolor="white", linewidth=0.3, label="Residual", zorder=3)

    for i, (_, row) in enumerate(df_vd.iterrows()):
        if row["stn"] > 0.08:
            x_pos = row["stn"] / 2 + (0.008 if row["stn"] < 0.13 else 0.0)
            ax_a.text(x_pos, i, f"{row['stn']:.0%}", ha="center", va="center",
                      fontsize=7.2, fontweight="bold", color="black")
        if row["yr"] > 0.08:
            ax_a.text(row["stn"] + row["yr"] / 2, i, f"{row['yr']:.0%}", ha="center", va="center",
                      fontsize=7.2, fontweight="bold", color="black")

    ax_a.set_yticks(x_vd)
    ax_a.set_yticklabels(df_vd["param"], fontsize=8.4)
    ax_a.set_xlabel("Fraction of variance", fontsize=9.6)
    ax_a.set_xlim(0, 1.02)
    ax_a.tick_params(axis="x", labelsize=8.4)
    ax_a.legend(fontsize=7.8, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                borderaxespad=0.0, ncol=3, frameon=False, handlelength=1.2, columnspacing=1.0)
    ax_a.invert_yaxis()
    panel_label(ax_a, "a", x=-0.12, y=1.05)

    # ── (b) LOSO R² (with chloride-immune BOD5) ──
    ax_b = fig.add_subplot(gs[0, 1])

    loso_data = {"param": [], "LULC": [], "Aug": []}
    cv_csv = CEE_VERIFY_DIR / "cee_loyo_vs_loso_cv_matrix.csv"
    cv_df = pd.read_csv(cv_csv) if cv_csv.exists() else pd.DataFrame()
    vp = pd.read_csv(VP_CSV) if VP_CSV.exists() else pd.DataFrame()

    param_specs = [
        ("COD",   "COD",           "vp"),
        ("BOD5",  r"BOD$_5$",      "cv"),
        ("DO",    "DO",            "vp"),
        ("NH4-N", r"NH$_4^+$-N",   "vp"),
        ("TSS",   "TSS",           "vp"),
    ]
    for csv_name, display_name, src in param_specs:
        if src == "vp" and not vp.empty:
            lulc_row = vp[(vp["Parameter"] == csv_name) & (vp["Predictor_Set"] == "LULC-only")]
            aug_row = vp[(vp["Parameter"] == csv_name) & (vp["Predictor_Set"] == "Primary (LULC+Pop+Hydro)")]
            if not lulc_row.empty and not aug_row.empty:
                v_l = float(lulc_row["LOSO_R2_Ridge"].iloc[0])
                v_a = float(aug_row["LOSO_R2_Ridge"].iloc[0])
                n_obs = int(lulc_row["N"].iloc[0])
                loso_data["param"].append(display_name)
                loso_data["LULC"].append(v_l)
                loso_data["Aug"].append(v_a)
                metrics_rows.append({
                    "Panel": "b_LOSO_R2", "Parameter": display_name,
                    "LULC_only_LOSO_R2": round(v_l, 4),
                    "Augmented_LOSO_R2": round(v_a, 4),
                    "N": n_obs,
                })
        elif src == "cv" and not cv_df.empty:
            lulc_row = cv_df[(cv_df["Parameter"] == csv_name) & (cv_df["Feature_Set"] == "LULC_Only_6")]
            aug_row = cv_df[(cv_df["Parameter"] == csv_name) & (cv_df["Feature_Set"] == "Primary_Augmented_11")]
            if not lulc_row.empty and not aug_row.empty:
                v_l = float(lulc_row["LOSO_R2_Ridge"].iloc[0])
                v_a = float(aug_row["LOSO_R2_Ridge"].iloc[0])
                n_obs = int(lulc_row["N"].iloc[0])
                loso_data["param"].append(display_name)
                loso_data["LULC"].append(v_l)
                loso_data["Aug"].append(v_a)
                metrics_rows.append({
                    "Panel": "b_LOSO_R2", "Parameter": display_name,
                    "LULC_only_LOSO_R2": round(v_l, 4),
                    "Augmented_LOSO_R2": round(v_a, 4),
                    "N": n_obs,
                })

    df_l = pd.DataFrame(loso_data)
    x_l = np.arange(len(df_l))
    w_l = 0.35

    ax_b.barh(x_l - w_l / 2, df_l["LULC"], w_l, color=CLR_SPATIAL, alpha=0.85,
              edgecolor="white", linewidth=0.3, label="LULC-only", zorder=3)
    ax_b.barh(x_l + w_l / 2, df_l["Aug"], w_l, color=CLR_TEMPORAL, alpha=0.85,
              edgecolor="white", linewidth=0.3, label="Augmented", zorder=3)

    ax_b.axvline(0, color="black", lw=0.5)
    ax_b.set_yticks(x_l)
    ax_b.set_yticklabels(df_l["param"], fontsize=8.4)
    ax_b.set_xlabel(r"LOSO $R^2$ (Ridge)", fontsize=9.6)
    ax_b.tick_params(axis="x", labelsize=8.4)
    ax_b.legend(fontsize=7.8, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                borderaxespad=0.0, ncol=2, frameon=False, handlelength=1.2, columnspacing=1.0)
    xlim = ax_b.get_xlim()
    ax_b.axvspan(xlim[0], 0, color="#fee0d2", alpha=0.15, zorder=0)
    ax_b.text(
        xlim[0] * 0.56, 2.72,
        "Fails to transfer\n($R^2 < 0$)",
        fontsize=8.2, color="#b2182b", ha="center", va="center", style="italic",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2),
    )
    ax_b.invert_yaxis()
    panel_label(ax_b, "b", x=-0.12, y=1.05)

    # ── (c) SHAP ──
    ax_c = fig.add_subplot(gs[1, 0])

    if SHAP_CSV.exists():
        shap_df = pd.read_csv(SHAP_CSV)
        cod_full = shap_df[(shap_df["Parameter"] == "COD") & (shap_df["Feature_Set"] == "Primary (LULC+Pop+Hydro)")].sort_values("Rank")
        do_full = shap_df[(shap_df["Parameter"] == "DO") & (shap_df["Feature_Set"] == "Primary (LULC+Pop+Hydro)")].sort_values("Rank")

        all_feats = list(dict.fromkeys(list(cod_full["Driver"].values) + list(do_full["Driver"].values)))
        cod_vals = {r["Driver"]: float(r["Mean_Abs_SHAP"]) for _, r in cod_full.iterrows()}
        do_vals  = {r["Driver"]: float(r["Mean_Abs_SHAP"]) for _, r in do_full.iterrows()}

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

        ax_c.barh(y_s - w_s / 2, [cod_vals.get(f, 0) for f in all_feats], w_s,
                  color=PARAM_CLR.get("COD", "#e66101"), alpha=0.85,
                  edgecolor="white", linewidth=0.3, label="COD", zorder=3)
        ax_c.barh(y_s + w_s / 2, [do_vals.get(f, 0) for f in all_feats], w_s,
                  color="#1b9e77", alpha=0.9,
                  edgecolor="white", linewidth=0.3, label="DO", zorder=3)

        ax_c.set_yticks(y_s)
        feat_labels = [
            f.replace("Tidal range (m)", "Tidal range")
             .replace("Population density", "Pop. density")
             .replace("Rainfall (dry z)", "Rainfall (dry)")
             .replace("Rainfall (annual z)", "Rainfall (ann.)")
             .replace("Discharge (My Thuan, annual z)", "Discharge")
            for f in all_feats
        ]
        ax_c.set_yticklabels(feat_labels, fontsize=8.4)
        ax_c.set_xlabel("Mean |SHAP|", fontsize=9.6)
        ax_c.tick_params(axis="x", labelsize=8.4)
        ax_c.legend(fontsize=7.8, loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
        ax_c.invert_yaxis()
    panel_label(ax_c, "c", x=-0.14, y=1.05)

    # ── (d) Fixed-effects forest plot (Parsimonious single-Q station-clustered CIs) ──
    ax_d = fig.add_subplot(gs[1, 1])
    pos_d = ax_d.get_position()
    ax_d.set_position([pos_d.x0, pos_d.y0, pos_d.width, pos_d.height * 0.86])

    cod_col = "COD (mg/L)_annual"
    lulc_features = [
        ("Aquaculture_composition_pct",       "aquaculture (%)"),
        ("Residential_Urban_composition_pct", "residential (%)"),
        ("Rice_Cultivation_composition_pct",  "rice (%)"),
    ]
    naive_slopes_sd: dict[str, float] = {}

    if cod_col in bt.columns:
        available_lulc = [f[0] for f in lulc_features if f[0] in bt.columns]
        stn_means = bt.groupby("stationName").agg(
            {col: "mean" for col in [cod_col] + available_lulc if col in bt.columns}
        ).dropna(subset=[cod_col])
        for lulc_col, lulc_label in lulc_features:
            if lulc_col in stn_means.columns:
                x_val = stn_means[lulc_col].to_numpy(dtype=float)
                y_val = stn_means[cod_col].to_numpy(dtype=float)
                x_z = _zscore(x_val)
                lr_sd = np.asarray(stats.linregress(x_z, y_val), dtype=float)
                naive_slopes_sd[lulc_label] = float(lr_sd[0])

    fe_suite_csv = CEE_VERIFY_DIR / "cee_fixed_effects_comprehensive_suite.csv"
    if fe_suite_csv.exists():
        df_fe = pd.read_csv(fe_suite_csv)
        sub = df_fe[df_fe["Model"] == "H1_COD_StationFE_All19Stns"].copy().reset_index(drop=True)
        terms = sub["Term"].tolist()
        x_pos = np.arange(len(terms))

        for i, (_, r) in enumerate(sub.iterrows()):
            lo = float(r["CI95_lo_Cluster_Stn"])
            hi = float(r["CI95_hi_Cluster_Stn"])
            p_val = float(r["p_Cluster_Stn"])
            p_twoway = float(r["p_TwoWay"]) if "p_TwoWay" in r and pd.notna(r["p_TwoWay"]) else 1.0
            coef = float(r["Beta"])
            sig = p_val < 0.05
            sig_twoway = p_twoway < 0.05
            dot_color = "#1b3a5c" if sig else "#4C78A8"
            dot_size = 50 if sig else 30
            ax_d.plot([i, i], [lo, hi], color="#4C78A8", lw=1.5, zorder=3)
            ax_d.scatter([i], [coef], color=dot_color, s=dot_size, marker="o",
                         zorder=4, edgecolors="white", linewidths=0.4)
            if sig_twoway:
                ax_d.text(i, hi + 0.35, "*", ha="center", va="bottom",
                          fontsize=10.5, fontweight="bold", color="#1b3a5c", zorder=7)
            metrics_rows.append({
                "Panel": "d_FE_H1",
                "Term": str(r["Term"]),
                "Beta": round(coef, 4),
                "CI95_lo_Stn": round(lo, 4),
                "CI95_hi_Stn": round(hi, 4),
                "p_Cluster_Stn": round(p_val, 6),
                "p_TwoWay": round(p_twoway, 6),
            })

        ax_d.axhline(0.0, color="black", lw=1.8, zorder=1)
        ax_d.axhspan(-0.5, 0.5, color="#f0f0f0", alpha=0.5, zorder=0)
        ax_d.set_ylim(-4.5, 12.8)
        ax_d.set_yticks([-4, 0, 4, 8, 12])
        clean_terms = [_clean_fe_term_label(t) for t in terms]
        ax_d.set_xticks(x_pos)
        ax_d.set_xticklabels(clean_terms, rotation=45, ha="right", fontsize=8.4)
        ax_d.set_ylabel(r"Standardized effect on COD (mg $\mathrm{L^{-1}}$ per 1 s.d.)", fontsize=8.6)
        ax_d.tick_params(axis="y", labelsize=8.4)

        key_map = {"urban": "residential (%)", "aqua": "aquaculture (%)", "rice": "rice (%)"}
        for i, term in enumerate(terms):
            t = str(term).lower()
            ghost_key = None
            for k, lbl in key_map.items():
                if k in t:
                    ghost_key = lbl
                    break
            if ghost_key is None or ghost_key not in naive_slopes_sd:
                continue
            ghost_y = float(naive_slopes_sd[ghost_key])
            fe_y = float(sub.iloc[i]["Beta"])
            ax_d.scatter([i], [ghost_y], s=60, marker="D", color="#e6550d",
                         edgecolors="#333333", linewidths=0.6, alpha=0.9, zorder=6)
            ax_d.annotate(
                "", xy=(i, fe_y), xytext=(i, ghost_y),
                arrowprops=dict(arrowstyle="-|>", color="#e6550d", lw=1.8, alpha=0.9, mutation_scale=12),
                zorder=5,
            )

        handles = [
            Line2D([0], [0], color="#4C78A8", lw=1.5, marker="o", markersize=5,
                   markerfacecolor="#4C78A8", markeredgecolor="white", markeredgewidth=0.3,
                   label="FE estimate (n.s.)"),
            Line2D([0], [0], color="#1b3a5c", lw=1.5, marker="o", markersize=7,
                   markerfacecolor="#1b3a5c", markeredgecolor="white", markeredgewidth=0.3,
                   label=r"FE ($p<0.05$; * two-way)"),
            Line2D([0], [0], color="#e6550d", lw=0, marker="D", markersize=6.0,
                   markerfacecolor="#e6550d", markeredgecolor="#333333", markeredgewidth=0.5,
                   label="Naive bivariate slope"),
            Line2D([0], [0], color="#e6550d", lw=1.2, label="Arrow: naive → FE"),
        ]
        ax_d.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.08, 1.02),
                    borderaxespad=0.0, frameon=False, fontsize=7.8, ncol=2,
                    columnspacing=0.5, handletextpad=0.4)

        model_r2 = float(sub["R2"].iloc[0]) if "R2" in sub.columns else np.nan
        if np.isfinite(model_r2):
            ax_d.text(0.98, 0.03, f"Station FE $R^2$ = {model_r2:.3f}",
                      transform=ax_d.transAxes, fontsize=8.4, va="bottom", ha="right",
                      bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))

    panel_label(ax_d, "d", x=-0.12, y=1.22)
    save_fig(fig, "CEE_Figure_6_Driver_Attribution", metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    main()
