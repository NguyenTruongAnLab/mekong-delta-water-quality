#!/usr/bin/env python3
"""
Extended Data Figure 8 — Reactor Diagnostics Across All Parameters
===================================================================

Purpose
-------
Provide a complete-parameter companion to Figure 2 by summarizing:
  (a) Phase 1 station-level trend slopes
  (b) Phase 2 station-level trend slopes
  (c) Slope shift (Phase 2 - Phase 1)
  (d) Q-normalized mass-balance phase means and trends (where available)

Exports a comprehensive metrics CSV for cross-checking all values.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects  # For text halo
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, kendalltau

from _nw_common import (
    apply_style, load_wq, MM, MRC_STATIONS,
    BREAK_YEAR, MB_CSV, panel_label, save_fig,
)

apply_style()


def _sig_mark(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _annot_heatmap(ax, arr: np.ndarray, p_arr: np.ndarray | None = None, fmt: str = "{:+.1f}") -> None:
    """
    Annotate heatmap with values and significance markers.
    Now with LARGER text and WHITE HALO for better visibility on dark backgrounds.
    """
    n_r, n_c = arr.shape
    for i in range(n_r):
        for j in range(n_c):
            v = arr[i, j]
            if not np.isfinite(v):
                txt = "NA"
                clr = "#666666"
            else:
                sig = ""
                if p_arr is not None and np.isfinite(p_arr[i, j]):
                    sig = _sig_mark(float(p_arr[i, j]))
                txt = f"{fmt.format(v)}{sig}"
                clr = "#222222"  # Dark text with white halo works on all backgrounds

            # Create text with increased font size
            text_obj = ax.text(j, i, txt, ha="center", va="center", 
                             fontsize=6.5,  # INCREASED from 4.9 to 6.5
                             color=clr, 
                             weight='bold')  # Bold for better readability

            # Add white halo/outline effect for visibility on dark colors
            text_obj.set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground='white'),  # White outline
                path_effects.Normal()  # Original text on top
            ])


def ed_figure_10() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()

    stations = [s for s in MRC_STATIONS if s in set(mrc["stationName"].dropna().unique())]
    candidate_params = [
        "COD (mg/L)", "DO (mg/L)", "NH4-N (mg/L)", "NO3-N (mg/L)",
        "TN (mg/L)", "TP (mg/L)", "TSS (mg/L)", "pH"
    ]
    params = [p for p in candidate_params if p in mrc.columns]

    if not params or not stations:
        raise RuntimeError("No valid parameters/stations available for ED Figure 10.")

    n_p, n_s = len(params), len(stations)
    p1_slope = np.full((n_p, n_s), np.nan, dtype=float)
    p2_slope = np.full((n_p, n_s), np.nan, dtype=float)
    p1_pval = np.full((n_p, n_s), np.nan, dtype=float)
    p2_pval = np.full((n_p, n_s), np.nan, dtype=float)

    metrics_rows: list[dict] = []

    for pi, pcol in enumerate(params):
        p_name = pcol.split(" (")[0]
        for si, stn in enumerate(stations):
            sub = mrc[mrc["stationName"] == stn]
            annual = sub.groupby("year")[pcol].mean().dropna()
            if len(annual) < 6:
                continue

            x = annual.index.to_numpy(dtype=float)
            y = annual.to_numpy(dtype=float)

            for phase_label, mask in [("Phase 1 (≤2004)", x <= BREAK_YEAR), ("Phase 2 (>2004)", x > BREAK_YEAR)]:
                if np.sum(mask) < 6:
                    continue
                slope = float(np.asarray(theilslopes(y[mask], x[mask]), dtype=float)[0])
                pval = float(np.asarray(kendalltau(x[mask], y[mask]), dtype=float)[1])
                metrics_rows.append({
                    "Panel": "trends_allparams",
                    "Station": stn,
                    "Parameter": p_name,
                    "Phase": phase_label,
                    "Slope": round(slope, 6),
                    "Slope_x1000": round(slope * 1000.0, 4),
                    "Kendall_p": round(pval, 6),
                    "n_years": int(np.sum(mask)),
                })

                if phase_label.startswith("Phase 1"):
                    p1_slope[pi, si] = slope * 1000.0
                    p1_pval[pi, si] = pval
                else:
                    p2_slope[pi, si] = slope * 1000.0
                    p2_pval[pi, si] = pval

    delta = p2_slope - p1_slope

    # Mass-balance summary (available subset only)
    mb_summary = pd.DataFrame(columns=[
        "Parameter", "Phase1_NCF_pct", "Phase2_NCF_pct", "Delta_pct", "Trend_slope_pct_yr", "Kendall_p", "n_years"
    ])
    if MB_CSV.exists():
        mb = pd.read_csv(MB_CSV)
        rows = []
        for p_name in sorted(mb["parameter"].dropna().unique()):
            sub = mb[mb["parameter"] == p_name].dropna(subset=["nonconservative_pct"])
            if len(sub) < 5:
                continue
            annual = sub.groupby("year")["nonconservative_pct"].mean().dropna()
            xa = annual.index.to_numpy(dtype=float)
            ya = annual.to_numpy(dtype=float)
            slope = float(np.asarray(theilslopes(ya, xa), dtype=float)[0])
            pval = float(np.asarray(kendalltau(xa, ya), dtype=float)[1])
            p1 = float(np.mean(ya[xa <= BREAK_YEAR])) if (xa <= BREAK_YEAR).any() else np.nan
            p2 = float(np.mean(ya[xa > BREAK_YEAR])) if (xa > BREAK_YEAR).any() else np.nan
            row = {
                "Parameter": p_name,
                "Phase1_NCF_pct": p1,
                "Phase2_NCF_pct": p2,
                "Delta_pct": p2 - p1 if np.isfinite(p1) and np.isfinite(p2) else np.nan,
                "Trend_slope_pct_yr": slope,
                "Kendall_p": pval,
                "n_years": int(len(annual)),
            }
            rows.append(row)
            metrics_rows.append({"Panel": "mass_balance_allparams", **{k: (round(v, 6) if isinstance(v, float) and np.isfinite(v) else v) for k, v in row.items()}})
        if rows:
            mb_summary = pd.DataFrame(rows)

    # Figure layout
    fig = plt.figure(figsize=(180 * MM, 150 * MM))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.40, hspace=0.28)

    # COLOR SCALING: SymLogNorm to handle mixed magnitude values
    lim = float(np.nanpercentile(np.abs(np.r_[p1_slope.flatten(), p2_slope.flatten()]), 95))
    lim = lim if np.isfinite(lim) and lim > 0 else 1.0

    # linthresh=10.0: Linear range for small values, logarithmic for large values
    norm = mcolors.SymLogNorm(linthresh=10.0, linscale=1.0, vmin=-lim, vmax=lim, base=10)

    # Panel A: Phase 1
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(p1_slope, cmap="RdBu_r", norm=norm, aspect="auto")
    _annot_heatmap(ax_a, p1_slope, p_arr=p1_pval)
    ax_a.set_title("(a) Phase 1 slopes (×10$^{-3}$ mg L$^{-1}$ yr$^{-1}$)", fontsize=7, fontweight="bold", loc="left")
    ax_a.set_xticks(np.arange(n_s))
    ax_a.set_xticklabels(stations, rotation=20, ha="right", fontsize=5.8)
    ax_a.set_yticks(np.arange(n_p))
    ax_a.set_yticklabels([p.split(" (")[0] for p in params], fontsize=5.8)
    cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.02)
    cb_a.ax.tick_params(labelsize=5)
    cb_a.set_label("SymLog Scale (Linear < ±10)", fontsize=5)

    # Panel B: Phase 2
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(p2_slope, cmap="RdBu_r", norm=norm, aspect="auto")
    _annot_heatmap(ax_b, p2_slope, p_arr=p2_pval)
    ax_b.set_title("(b) Phase 2 slopes (×10$^{-3}$ mg L$^{-1}$ yr$^{-1}$)", fontsize=7, fontweight="bold", loc="left")
    ax_b.set_xticks(np.arange(n_s))
    ax_b.set_xticklabels(stations, rotation=20, ha="right", fontsize=5.8)
    ax_b.set_yticks(np.arange(n_p))
    ax_b.set_yticklabels([p.split(" (")[0] for p in params], fontsize=5.8)
    cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.02)
    cb_b.ax.tick_params(labelsize=5)
    cb_b.set_label("SymLog Scale (Linear < ±10)", fontsize=5)

    # Panel C: Slope Shift
    ax_c = fig.add_subplot(gs[1, 0])
    lim_d = float(np.nanpercentile(np.abs(delta.flatten()), 95))
    lim_d = lim_d if np.isfinite(lim_d) and lim_d > 0 else 1.0
    norm_d = mcolors.SymLogNorm(linthresh=10.0, linscale=1.0, vmin=-lim_d, vmax=lim_d, base=10)

    im_c = ax_c.imshow(delta, cmap="PiYG", norm=norm_d, aspect="auto")
    _annot_heatmap(ax_c, delta, p_arr=None)
    ax_c.set_title("(c) Slope shift (Phase 2 − Phase 1)", fontsize=7, fontweight="bold", loc="left")
    ax_c.set_xticks(np.arange(n_s))
    ax_c.set_xticklabels(stations, rotation=20, ha="right", fontsize=5.8)
    ax_c.set_yticks(np.arange(n_p))
    ax_c.set_yticklabels([p.split(" (")[0] for p in params], fontsize=5.8)
    cb_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.02)
    cb_c.ax.tick_params(labelsize=5)
    cb_c.set_label("SymLog Scale (Linear < ±10)", fontsize=5)

    # Panel D: Mass Balance
    ax_d = fig.add_subplot(gs[1, 1])
    if mb_summary.empty:
        ax_d.text(0.5, 0.5, "No mass-balance inputs found", ha="center", va="center", fontsize=7)
    else:
        # Force panel-d parameter order to match a/b/c and display unavailable terms as NA.
        ordered_param_names = [p.split(" (")[0] for p in params]
        mbp = pd.DataFrame({"Parameter": ordered_param_names})
        mbp = mbp.merge(mb_summary, on="Parameter", how="left")

        xpos = np.arange(len(mbp))
        w = 0.34
        p1_vals = mbp["Phase1_NCF_pct"].to_numpy(dtype=float)
        p2_vals = mbp["Phase2_NCF_pct"].to_numpy(dtype=float)
        has_mb = np.isfinite(p1_vals) & np.isfinite(p2_vals)

        ax_d.bar(xpos - w / 2, np.nan_to_num(p1_vals, nan=0.0), w,
                 color="#6baed6", edgecolor="white", linewidth=0.5, label=f"Phase 1 (≤{BREAK_YEAR})", zorder=3)
        ax_d.bar(xpos + w / 2, np.nan_to_num(p2_vals, nan=0.0), w,
                 color="#fb6a4a", edgecolor="white", linewidth=0.5, label=f"Phase 2 (>{BREAK_YEAR})", zorder=3)
        ax_d.axhline(0, color="#444444", lw=0.7, zorder=1)

        y_limit_tracker = 0

        for i, (_, row) in enumerate(mbp.iterrows()):
            xi = float(i)
            if not bool(has_mb[i]):
                ax_d.text(xi, 2.0, "NA", ha="center", va="bottom", fontsize=5.6,
                          color="#666666", fontweight="bold")
                continue

            delta_txt = row["Delta_pct"]
            sig = _sig_mark(float(row["Kendall_p"])) if np.isfinite(row["Kendall_p"]) else ""

            h_val = max(row["Phase1_NCF_pct"], row["Phase2_NCF_pct"])
            if h_val < 0: h_val = 0

            ax_d.text(xi, h_val + 5,
                      f"Δ={delta_txt:+.1f}{sig}", ha="center", va="bottom", fontsize=6)

            y_limit_tracker = max(y_limit_tracker, h_val)

        ax_d.set_ylim(top=y_limit_tracker * 1.35 if y_limit_tracker > 0 else 100)

        ax_d.set_xticks(xpos)
        ax_d.set_xticklabels(mbp["Parameter"].tolist(), fontsize=5.8)
        ax_d.set_ylabel("Non-conservative fraction (%)", fontsize=6.5)
        ax_d.text(0.01, 0.03, "NA = no transect load record for parameter",
              transform=ax_d.transAxes, fontsize=5.2, color="#666666", style="italic")
        ax_d.legend(fontsize=5.2, frameon=True, framealpha=0.9, edgecolor="none", loc="upper right")

    ax_d.set_title("(d) Mass balance by parameter", fontsize=7, fontweight="bold", loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    save_fig(fig, "ED_Figure_8_Reactor_All_Parameters", metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    ed_figure_10()
