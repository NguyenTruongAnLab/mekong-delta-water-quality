#!/usr/bin/env python3
"""
Figure 3 — Hydrodynamic Decoupling
====================================
(a) Chemostatic index (CI) bar chart at Tan Chau, Phase 1 vs Phase 2
(b–e) C–Q scatter plots at Tan Chau for COD, DO, NH$_4$-N, NO$_3$-N

Exported metrics CSV:
  - Station, Parameter, Phase, beta (slope), CI, p-value
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.ticker import FuncFormatter

from _nw_common import (
    apply_style, load_wq, load_hydro_monthly, cq_metrics,
    MM, MRC_STATIONS, STATION_CLR, STN_Q_MAP,
    PHASE1_CLR, PHASE2_CLR, BREAK_YEAR,
    panel_label, save_fig,
)

apply_style()


def _log_binned_means(q: np.ndarray, c: np.ndarray, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Return log-binned mean Q and C (filters empty/invalid bins)."""
    valid = np.isfinite(q) & np.isfinite(c) & (q > 0) & (c > 0)
    qv = q[valid]
    cv = c[valid]
    if len(qv) < 8:
        return np.array([]), np.array([])

    qmin = float(np.nanmin(qv))
    qmax = float(np.nanmax(qv))
    if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin:
        return np.array([]), np.array([])

    bins = np.logspace(np.log10(qmin), np.log10(qmax), n_bins)
    inds = np.digitize(qv, bins)

    bx, by = [], []
    for i in range(1, len(bins)):
        m = inds == i
        if np.sum(m) < 2:
            continue
        q_mean = float(np.nanmean(qv[m]))
        c_mean = float(np.nanmean(cv[m]))
        if np.isfinite(q_mean) and np.isfinite(c_mean):
            bx.append(q_mean)
            by.append(c_mean)

    return np.asarray(bx, dtype=float), np.asarray(by, dtype=float)


def figure_3() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()
    q_monthly = load_hydro_monthly()

    metrics_rows = []

    # Core parameters matching manuscript narrative (COD, DO, NH4, NO3)
    params_ci = [
        ("COD (mg/L)", "COD"),
        ("DO (mg/L)",  "DO"),
        ("NH4-N (mg/L)", r"NH$_4^+$"),
        ("NO3-N (mg/L)", r"NO$_3^-$"),
    ]

    # Layout: 2 rows × 3 columns
    # (a) CI bars at Tan Chau occupies top row spanning 2 columns
    # (b–e) C–Q scatter for 4 parameters at Tan Chau only (has independent gauge)
    fig = plt.figure(figsize=(180 * MM, 130 * MM))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[0.95, 0.95],
        hspace=0.20, wspace=0.35,
    )

    # ── (a) CI bars at Tan Chau ──
    ax_ci = fig.add_subplot(gs[0, :2])

    merged_tc = mrc[mrc["stationName"] == "Tan Chau"].merge(
        q_monthly[["ym", "TanChau"]], on="ym", how="inner")
    ci_p1, ci_p2 = [], []
    labels_ci = []

    for pcol, plabel in params_ci:
        if pcol not in merged_tc.columns:
            continue
        years = merged_tc["year"].to_numpy(dtype=float)
        c = merged_tc[pcol].to_numpy(dtype=float)
        q = merged_tc["TanChau"].to_numpy(dtype=float)

        r1 = np.asarray(cq_metrics(c[years <= BREAK_YEAR], q[years <= BREAK_YEAR]), dtype=float)
        r2 = np.asarray(cq_metrics(c[years > BREAK_YEAR],  q[years > BREAK_YEAR]), dtype=float)
        beta1, se1, ci1, p1 = float(r1[0]), float(r1[1]), float(r1[2]), float(r1[3])
        beta2, se2, ci2, p2 = float(r2[0]), float(r2[1]), float(r2[2]), float(r2[3])
        ci_p1.append(ci1)
        ci_p2.append(ci2)
        labels_ci.append(plabel)

        metrics_rows.append({
            "Panel": "a_CI", "Station": "Tan Chau", "Parameter": plabel,
            "Phase1_beta": round(beta1, 4) if np.isfinite(beta1) else None,
            "Phase1_CI": round(ci1, 4) if np.isfinite(ci1) else None,
            "Phase1_p": round(p1, 6) if np.isfinite(p1) else None,
            "Phase2_beta": round(beta2, 4) if np.isfinite(beta2) else None,
            "Phase2_CI": round(ci2, 4) if np.isfinite(ci2) else None,
            "Phase2_p": round(p2, 6) if np.isfinite(p2) else None,
            "Delta_CI": round(ci2 - ci1, 4) if np.isfinite(ci1) and np.isfinite(ci2) else None,
        })

    x = np.arange(len(labels_ci))
    w = 0.35
    ax_ci.bar(x - w / 2, ci_p1, w, color=PHASE1_CLR, alpha=0.85,
              edgecolor="white", linewidth=0.3,
              label=rf"Phase 1 ($\leq${BREAK_YEAR})", zorder=3)
    ax_ci.bar(x + w / 2, ci_p2, w, color=PHASE2_CLR, alpha=0.85,
              edgecolor="white", linewidth=0.3,
              label=rf"Phase 2 (>{BREAK_YEAR})", zorder=3)

    ax_ci.axhline(1.0, color="#969696", lw=0.8, ls=":", alpha=0.7, zorder=1)
    ax_ci.text(0.02, 0.88, "Transport-limited", transform=ax_ci.transAxes,
               fontsize=5.5, color="#525252", style="italic")
    # Multiline to avoid overlap with bars/legend and adjacent panel
    ax_ci.text(0.32, 0.25, "Source-limited /\nchemostatic", transform=ax_ci.transAxes,
               fontsize=5.5, color="#525252", style="italic")

    for i, (c1, c2) in enumerate(zip(ci_p1, ci_p2)):
        if np.isfinite(c1) and np.isfinite(c2):
            delta = c2 - c1
            arrow = r"$\downarrow$" if delta < 0 else r"$\uparrow$"
            clr = PHASE2_CLR if abs(delta) > 0.1 else "#969696"
            ax_ci.text(i, max(c1, c2) + 0.08,
                       f"{arrow}{abs(delta):.2f}",
                       ha="center", fontsize=5, color=clr, fontweight="bold")

    ax_ci.set_xticks(x)
    ax_ci.set_xticklabels(labels_ci, fontsize=7)
    ax_ci.set_ylabel("Chemostatic index (CV$_C$ / CV$_Q$)", fontsize=7)
    ax_ci.legend(fontsize=5.5, frameon=True, framealpha=0.9, edgecolor="none",
                 loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8)
    ax_ci.set_ylim(0, max(max(ci_p1), max(ci_p2)) * 1.35 if (ci_p1 and ci_p2) else 1.5)
    panel_label(ax_ci, "a")

    # ── (b–e) C–Q scatter for 4 core parameters at Tan Chau only ──
    cq_params = [
        ("COD (mg/L)", r"COD (mg $\mathrm{L^{-1}}$)"),
        ("DO (mg/L)",  r"DO (mg $\mathrm{L^{-1}}$)"),
        ("NH4-N (mg/L)", r"NH$_4^+$ (mg $\mathrm{L^{-1}}$)"),
        ("NO3-N (mg/L)", r"NO$_3^-$ (mg $\mathrm{L^{-1}}$)"),
    ]

    # Panel positions: bottom row (3 panels) + top-right (1 panel)
    panel_positions = [
        (0, 2),  # top-right for COD
        (1, 0),  # bottom-left for DO
        (1, 1),  # bottom-center for NH4
        (1, 2),  # bottom-right for NO3
    ]
    panel_labels = ["b", "c", "d", "e"]

    stn = "Tan Chau"
    q_col = "TanChau"
    stn_data = mrc[mrc["stationName"] == stn].merge(
        q_monthly[["ym", q_col]], on="ym", how="inner")

    for idx, ((pcol, ylabel), (row_i, col_i), plabel) in enumerate(zip(cq_params, panel_positions, panel_labels)):
        ax = fig.add_subplot(gs[row_i, col_i])

        if pcol not in stn_data.columns:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            panel_label(ax, plabel)
            continue

        c = stn_data[pcol].to_numpy(dtype=float)
        q = stn_data[q_col].to_numpy(dtype=float)
        years = stn_data["year"].to_numpy(dtype=float)
        valid = np.isfinite(c) & np.isfinite(q) & (c > 0) & (q > 0)
        c, q, years = c[valid], q[valid], years[valid]

        m1 = years <= BREAK_YEAR
        m2 = years > BREAK_YEAR

        ax.scatter(q[m1], c[m1], c=PHASE1_CLR, alpha=0.24, s=10,
                   edgecolors="none", zorder=2)
        ax.scatter(q[m2], c[m2], c=PHASE2_CLR, alpha=0.24, s=10,
                   edgecolors="none", zorder=2)

        # Log-binned means: reveal central C–Q tendency through overplotting noise
        for mask, clr in [(m1, PHASE1_CLR), (m2, PHASE2_CLR)]:
            bx, by = _log_binned_means(q[mask], c[mask], n_bins=15)
            if len(bx) >= 2:
                ax.plot(
                    bx, by,
                    color=clr, lw=2.4, marker="o", markersize=4.0,
                    markeredgecolor="white", markeredgewidth=0.7,
                    alpha=0.95, zorder=5,
                )

        for mask, clr, phase_lbl in [(m1, PHASE1_CLR, "P1"),
                                      (m2, PHASE2_CLR, "P2")]:
            if mask.sum() > 10:
                lq = np.log10(q[mask])
                lc = np.log10(c[mask])
                lr = np.asarray(stats.linregress(lq, lc), dtype=float)
                sl, ic, r, p, se = float(lr[0]), float(lr[1]), float(lr[2]), float(lr[3]), float(lr[4])
                q_fit = np.logspace(lq.min(), lq.max(), 50)
                c_fit = 10 ** (ic + sl * np.log10(q_fit))
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                ax.plot(q_fit, c_fit, color=clr, lw=1.5, alpha=0.9,
                        label=f"{phase_lbl}: $\\beta$={sl:+.2f}{sig}", zorder=4)

                metrics_rows.append({
                    "Panel": f"cq_{pcol.split(' ')[0]}", "Station": stn,
                    "Parameter": pcol.split(" (")[0], "Phase": phase_lbl,
                    "beta": round(sl, 4), "SE": round(se, 4),
                    "R2": round(r**2, 4), "p": round(p, 6),
                    "n": int(mask.sum()),
                })

        # Keep beta estimation in log-space, but plot on linear axes for readability
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        if len(q) > 0:
            q_min, q_max = float(np.min(q)), float(np.max(q))
            pad = 0.06 * (q_max - q_min) if q_max > q_min else 100.0
            ax.set_xlim(max(0.0, q_min - pad), q_max + pad)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel(r"Q ($\mathrm{m^3\,s^{-1}}$)" if row_i == 1 else "", fontsize=6.5)
        # Show y-axis labels on all four panels (b–e), per user request
        ax.set_ylabel(ylabel, fontsize=6.5)
        ax.tick_params(axis="both", labelsize=5.5)
        ax.legend(fontsize=5, loc="lower right", frameon=True,
                  framealpha=0.9, edgecolor="none",
                  handlelength=1.2, handletextpad=0.3)
        panel_label(ax, plabel)

    fig.tight_layout(pad=0.6)
    save_fig(fig, "Figure_3_Hydrodynamic_Decoupling", metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    figure_3()
