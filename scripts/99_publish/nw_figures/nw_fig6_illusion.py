#!/usr/bin/env python3
"""
Figure 6 — Attribution Illusion
==================================
(a) Naïve scatter plots: COD vs 4 LULC composition metrics
(b) Two-way panel FE coefficient forest plot

ALL values read from pre-computed analysis CSVs.

Exported metrics CSV:
  - Naïve R² for each LULC variable
  - FE coefficient, SE, p-value for each term
  - FE-only R² and FE+LULC R² (ΔR²)
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D

from _nw_common import (
    apply_style, load_master, load_bt,
    MM, MRC_STATIONS, FE_CSV, ROOT,
    panel_label, save_fig,
)

apply_style()


def _strip_q_wrapper(term: str) -> str:
    """Normalize patsy/statsmodels term names for plotting."""
    t = str(term).strip()
    if t.startswith("Q(") and t.endswith(")"):
        t = t[2:-1].strip()
    # remove surrounding quotes from Q('x')
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        t = t[1:-1]
    return t


def _clean_fe_term_label(term: str) -> str:
    """Human-readable y-axis label for panel (b)."""
    t = _strip_q_wrapper(term)
    # remove standardization suffixes
    t = t.replace("_z", "")

    label_map = {
        "Urban_PD": "Urban PD",
        "Aqua_ED": "Aquaculture ED",
        "Rice_PD": "Rice PD",
        "Salinity": "Salinity",
        "Year": "Year",
        "Q_branch_AnnualMean": "Q My Thuan (annual)",
        "Q_branch_DrySeasonMean": "Q My Thuan (dry)",
        "Q_MyThuan_AnnualMean": "Q My Thuan (annual)",
        "Q_MyThuan_DrySeasonMean": "Q My Thuan (dry)",
        "Rain_Annual": "Rain (annual)",
        "Rain_Dry": "Rain (dry)",
    }

    # fall back to a light-touch cleanup if a term is unexpected
    return label_map.get(t, t.replace("_", " "))


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.nanstd(x, ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - float(np.nanmean(x))) / sd


def figure_6() -> None:
    master = load_master()
    bt = load_bt(master)  # Ben Tre only (exclude MRC permanganate COD)

    metrics_rows = []

    fig = plt.figure(figsize=(240 * MM, 240 * MM))
    gs_main = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35,
                                height_ratios=[1.2, 0.7])

    # ═══════════════════════════════════════════════════════════════════
    # (a) 2×2 naive scatter: COD vs 4 LULC features
    # ═══════════════════════════════════════════════════════════════════
    gs_scatter = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[0],
                                                  hspace=0.30, wspace=0.35)
    cod_col = "COD (mg/L)_annual"
    lulc_features = [
        ("Aquaculture_composition_pct", "Aquaculture (%)"),
        ("Residential_Urban_composition_pct", "Residential (%)"),
        ("Rice_Cultivation_composition_pct", "Rice (%)"),
        ("Wetland_composition_pct", "Wetland (%)"),
    ]

    scatter_colors = ["#3288bd", "#d53e4f", "#66c2a5", "#bf812d"]
    stn_means = pd.DataFrame()
    # Store naive *per-SD* slopes for panel (b) overlays:
    # y units are mg/L; x is standardized by its across-station SD.
    naive_slopes_sd: dict[str, float] = {}

    if cod_col in bt.columns:
        available_lulc = [f[0] for f in lulc_features if f[0] in bt.columns]
        stn_means = bt.groupby("stationName").agg(
            {col: "mean" for col in [cod_col] + available_lulc
             if col in bt.columns}
        ).dropna(subset=[cod_col])

    for idx, (lulc_col, lulc_label) in enumerate(lulc_features):
        row_i, col_i = divmod(idx, 2)
        ax = fig.add_subplot(gs_scatter[row_i, col_i])

        if lulc_col in stn_means.columns:
            x_val = stn_means[lulc_col].to_numpy(dtype=float)
            y_val = stn_means[cod_col].to_numpy(dtype=float)

            ax.scatter(x_val, y_val, s=25, c=scatter_colors[idx],
                       edgecolors="#333333", linewidths=0.4, alpha=0.85, zorder=3)

            if len(x_val) > 3:
                lr = np.asarray(stats.linregress(x_val, y_val), dtype=float)
                sl, ic, r, p, se = float(lr[0]), float(lr[1]), float(lr[2]), float(lr[3]), float(lr[4])
                xs = np.linspace(x_val.min(), x_val.max(), 50)
                ax.plot(xs, ic + sl * xs, "--", color=scatter_colors[idx],
                        lw=1.2, alpha=0.7, zorder=2)
                sig_mark = "*" if p < 0.05 else ""
                ax.text(0.97, 0.97,
                        f"R$^2$ = {r**2:.2f}{sig_mark}\np = {p:.3f}",
                        transform=ax.transAxes, fontsize=10, va="top",
                        ha="right",
                        bbox=dict(facecolor="white", alpha=0.8,
                                  edgecolor="none", pad=1.5))
                metrics_rows.append({
                    "Panel": "a_naive_scatter",
                    "LULC_Variable": lulc_label,
                    "Naive_R2": round(r**2, 4),
                    "Slope": round(sl, 4),
                    "p_value": round(p, 6),
                    "N_stations": int(len(x_val)),
                })
                # For panel (b) overlay: slope of y on standardized x (mg/L per 1 SD of x)
                x_z = _zscore(x_val)
                lr_sd = np.asarray(stats.linregress(x_z, y_val), dtype=float)
                naive_slopes_sd[lulc_label.lower()] = float(lr_sd[0])

        ax.set_xlabel(lulc_label, fontsize=11)
        ax.set_ylabel(r"COD (mg $\mathrm{L^{-1}}$)" if col_i == 0 else "", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)

    # Overall label
    ax_label = fig.add_subplot(gs_main[0], frame_on=False)
    ax_label.set_xticks([]); ax_label.set_yticks([])
    ax_label.set_title("(a) Na\u00efve cross-sectional regressions",
                       fontsize=12, fontweight="bold", loc="left", pad=8)

    # ═══════════════════════════════════════════════════════════════════
    # (b) Fixed-effects coefficient forest plot — from CSV
    # ═══════════════════════════════════════════════════════════════════
    ax_fe = fig.add_subplot(gs_main[1])

    fe_loaded = False
    # Primary source: analysis outputs (FE_CSV)
    if FE_CSV.exists():
        df_fe = pd.read_csv(FE_CSV)
        needed = ["Model", "term", "coef", "se_HC3"]
        if all(c in df_fe.columns for c in needed):
            fe_loaded = True
            df_fe = df_fe[
                ~df_fe["term"].astype(str).str.startswith("C(stationName)")
            ].copy()
            df_fe = df_fe[df_fe["term"].astype(str) != "Intercept"].copy()
            df_fe = df_fe[~df_fe["term"].astype(str).str.startswith("C(year)")].copy()
            df_fe["lo"] = df_fe["coef"] - 1.96 * df_fe["se_HC3"]
            df_fe["hi"] = df_fe["coef"] + 1.96 * df_fe["se_HC3"]

            model_choice = "HydroClimate" if "HydroClimate" in df_fe["Model"].unique() else df_fe["Model"].iloc[0]
            sub = df_fe[df_fe["Model"] == model_choice].copy()
            terms = sub["term"].tolist()
            x_pos = np.arange(len(terms))

            for i, (_, r) in enumerate(sub.iterrows()):
                ax_fe.plot([i, i], [float(r["lo"]), float(r["hi"])],
                           color="#4C78A8", lw=1.5, zorder=3)
                p_val = float(r.get("p", 1.0))
                sig = p_val < 0.05
                dot_color = "#1b3a5c" if sig else "#4C78A8"
                dot_size = 50 if sig else 30
                ax_fe.scatter([i], [float(r["coef"])],
                              color=dot_color, s=dot_size, marker="o",
                              zorder=4, edgecolors="white", linewidths=0.4)

                metrics_rows.append({
                    "Panel": "b_FE_coefficients",
                    "Parameter": "COD",
                    "Model": model_choice,
                    "Term": str(r["term"]),
                    "Coefficient": round(float(r["coef"]), 4),
                    "SE": round(float(r["se_HC3"]), 4),
                    "p_value": round(float(r.get("p", np.nan)), 6) if pd.notna(r.get("p", np.nan)) else None,
                    "Significant": p_val < 0.05,
                })

            ax_fe.axhline(0.0, color="black", lw=1.8, zorder=1)
            ax_fe.axhspan(-0.5, 0.5, color="#f0f0f0", alpha=0.5, zorder=0)
            ax_fe.set_yscale("symlog", linthresh=2)
            clean_terms = [_clean_fe_term_label(t) for t in terms]
            ax_fe.set_xticks(x_pos)
            ax_fe.set_xticklabels(clean_terms, rotation=45, ha="right", fontsize=11)
            ax_fe.set_ylabel(r"Standardised coefficient (mg $\mathrm{L^{-1}}$)", fontsize=12)
            ax_fe.tick_params(axis="y", labelsize=10)

            # Ghost naive coefficients from panel (a), with arrows to FE estimates.
            key_map = {
                "urban": "residential (%)",
                "aqua": "aquaculture (%)",
                "rice": "rice (%)",
                "wet": "wetland (%)",
            }
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
                fe_y = float(sub.iloc[i]["coef"])
                ax_fe.scatter([i], [ghost_y], s=70, marker="D", color="#e6550d",
                              edgecolors="#333333", linewidths=0.6, alpha=0.9, zorder=6)
                ax_fe.annotate(
                    "",
                    xy=(i, fe_y), xytext=(i, ghost_y),
                    arrowprops=dict(arrowstyle="-|>", color="#e6550d",
                                    lw=2.0, alpha=0.9,
                                    mutation_scale=14),
                    zorder=5,
                )

            # Legend explaining symbols (panel b)
            handles = [
                Line2D([0], [0], color="#4C78A8", lw=1.5, marker="o",
                       markersize=5, markerfacecolor="#4C78A8",
                       markeredgecolor="white", markeredgewidth=0.3,
                       label="FE estimate (n.s.)"),
                Line2D([0], [0], color="#1b3a5c", lw=1.5, marker="o",
                       markersize=7, markerfacecolor="#1b3a5c",
                       markeredgecolor="white", markeredgewidth=0.3,
                       label=r"FE estimate ($p<0.05$)"),
                Line2D([0], [0], color="#e6550d", lw=0, marker="D",
                       markersize=6.0, markerfacecolor="#e6550d",
                       markeredgecolor="#333333", markeredgewidth=0.5,
                       label="Naïve slope (panel a)"),
                Line2D([0], [0], color="#e6550d", lw=1.2,
                       label="Arrow: naïve → FE"),
            ]
            ax_fe.legend(handles=handles, loc="upper right", frameon=False,
                        fontsize=10, ncol=2)

            model_r2 = float(sub["R2"].iloc[0]) if "R2" in sub.columns else np.nan
            n = int(sub["N"].iloc[0]) if "N" in sub.columns else np.nan
            if np.isfinite(model_r2) and np.isfinite(n):
                ax_fe.text(0.98, 0.03,
                           f"{model_choice} R$^2$ = {model_r2:.3f}",
                           transform=ax_fe.transAxes, fontsize=10.5,
                           va="bottom", ha="right",
                           bbox=dict(facecolor="white", alpha=0.8,
                                     edgecolor="none", pad=1.5))
                metrics_rows.append({
                    "Panel": "b_FE_model_summary",
                    "Parameter": "COD",
                    "Model": model_choice,
                    "Term": "MODEL_R2_FE_plus_LULC",
                    "Coefficient": round(model_r2, 4),
                    "SE": None, "p_value": None, "Significant": None,
                })

    if not fe_loaded:
        raise FileNotFoundError(
            f"FE coefficient CSV not found at {FE_CSV}. "
            "Run the analysis pipeline first to generate it."
        )

    ax_fe.set_title("(b) Fixed-effects regression coefficients",
                    fontsize=12, fontweight="bold", loc="left", pad=8)

    fig.tight_layout()
    save_fig(fig, "Figure_6_Attribution_Illusion", metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    figure_6()
