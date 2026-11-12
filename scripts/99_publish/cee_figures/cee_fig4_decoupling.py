#!/usr/bin/env python3
"""
CEE Figure 4 — Hydrodynamic Decoupling at Tan Chau
==================================================
Cloned directly from `nw_fig3_decoupling.py` to preserve 100% of the tuned
2x4 GridSpec: (a) Chemostatic index (`CV_C / CV_Q`) across COD, DO, NH4+-N,
and NO3--N, aligned directly above (b-e) linear-axis C-Q scatters with
log-binned means and log-log power-law fits.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

from _cee_common import (
    apply_style, load_wq, load_hydro_monthly, cq_metrics,
    MM, PHASE1_CLR, PHASE2_CLR, BREAK_YEAR,
    panel_label, save_fig,
)

apply_style()


def _log_binned_means(q: np.ndarray, c: np.ndarray, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray]:
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


def figure_4() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()
    q_monthly = load_hydro_monthly()

    metrics_rows = []
    params_ci = [
        ("COD (mg/L)",   "COD"),
        ("DO (mg/L)",    "DO"),
        ("NH4-N (mg/L)", r"NH$_4^+$-N"),
        ("NO3-N (mg/L)", r"NO$_3^-$-N"),
    ]

    fig = plt.figure(figsize=(180 * MM, 122 * MM))
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        height_ratios=[1.0, 1.05],
        left=0.08, right=0.985,
        bottom=0.09, top=0.92,
        hspace=0.40, wspace=0.46,
    )

    ax_ci = fig.add_subplot(gs[0, :])
    axes_cq = [fig.add_subplot(gs[1, i]) for i in range(4)]

    fig.canvas.draw()
    bbox_ci = ax_ci.get_position()
    centers_fig = [(ax.get_position().x0 + ax.get_position().x1) / 2.0 for ax in axes_cq]
    centers_ci = [(cx - bbox_ci.x0) / bbox_ci.width for cx in centers_fig]

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

    ax_ci.set_xlim(0, 1)
    bar_w = 0.046
    gap = 0.004
    for i, (cx, c1, c2) in enumerate(zip(centers_ci, ci_p1, ci_p2)):
        ax_ci.bar(cx - bar_w / 2 - gap / 2, c1, bar_w, color=PHASE1_CLR, alpha=0.88,
                  edgecolor="white", linewidth=0.4,
                  label=rf"Phase 1 ($\leq$ {BREAK_YEAR})" if i == 0 else "", zorder=3)
        ax_ci.bar(cx + bar_w / 2 + gap / 2, c2, bar_w, color=PHASE2_CLR, alpha=0.88,
                  edgecolor="white", linewidth=0.4,
                  label=rf"Phase 2 (> {BREAK_YEAR})" if i == 0 else "", zorder=3)

        if np.isfinite(c1) and np.isfinite(c2):
            delta = c2 - c1
            arrow = r"$\downarrow$" if delta < 0 else r"$\uparrow$"
            y_off = 0.032 if i == 1 else 0.055
            ax_ci.text(cx, max(c1, c2) + y_off,
                       f"{arrow} {abs(delta):.2f}",
                       ha="center", fontsize=8.3, color="black", fontweight="bold")

    ax_ci.axhline(1.0, color="#737373", lw=0.75, ls=":", alpha=0.8, zorder=1)
    ax_ci.axhline(0.5, color="#737373", lw=0.75, ls="--", alpha=0.7, zorder=1)

    regime_x = float(centers_ci[1])
    ax_ci.text(regime_x, 0.535, "Chemodynamic (flow-sensitive)", transform=ax_ci.get_yaxis_transform(),
               fontsize=8.0, color="black", style="italic", va="bottom", ha="center")
    ax_ci.text(regime_x, 0.465, "Chemostatic (flow-buffered)", transform=ax_ci.get_yaxis_transform(),
               fontsize=8.0, color="black", style="italic", va="top", ha="center")

    ax_ci.set_xticks(centers_ci)
    ax_ci.set_xticklabels(labels_ci, fontsize=8.5, fontweight="bold")
    ax_ci.tick_params(axis="y", labelsize=8.5)
    ax_ci.set_ylabel(r"Chemostatic index ($\mathrm{CV}_C / \mathrm{CV}_Q$)", fontsize=8.8)
    ax_ci.set_ylim(0, 1.62)

    ax_ci.legend(fontsize=8.5, frameon=True, framealpha=0.92, edgecolor="none",
                 loc="upper left", bbox_to_anchor=(0.015, 0.98),
                 ncol=1, handlelength=1.1, handletextpad=0.4, borderpad=0.4)
    panel_label(ax_ci, "a", x=-0.045, y=1.04)

    cq_params = [
        ("COD (mg/L)",   r"COD (mg $\mathrm{L^{-1}}$)"),
        ("DO (mg/L)",    r"DO (mg $\mathrm{L^{-1}}$)"),
        ("NH4-N (mg/L)", r"NH$_4^+$-N (mg $\mathrm{L^{-1}}$)"),
        ("NO3-N (mg/L)", r"NO$_3^-$-N (mg $\mathrm{L^{-1}}$)"),
    ]
    panel_labels = ["b", "c", "d", "e"]

    stn = "Tan Chau"
    q_col = "TanChau"
    stn_data = mrc[mrc["stationName"] == stn].merge(
        q_monthly[["ym", q_col]], on="ym", how="inner")

    for ax, (pcol, ylabel), plabel in zip(axes_cq, cq_params, panel_labels):
        if pcol not in stn_data.columns:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            panel_label(ax, plabel, x=-0.28, y=1.12)
            continue

        c = stn_data[pcol].to_numpy(dtype=float)
        q = stn_data[q_col].to_numpy(dtype=float)
        years = stn_data["year"].to_numpy(dtype=float)
        valid = np.isfinite(c) & np.isfinite(q) & (c > 0) & (q > 0)
        c, q, years = c[valid], q[valid], years[valid]

        m1 = years <= BREAK_YEAR
        m2 = years > BREAK_YEAR

        ax.scatter(q[m1], c[m1], c=PHASE1_CLR, alpha=0.20, s=8,
                   edgecolors="none", zorder=2)
        ax.scatter(q[m2], c[m2], c=PHASE2_CLR, alpha=0.20, s=8,
                   edgecolors="none", zorder=2)

        for mask, clr in [(m1, PHASE1_CLR), (m2, PHASE2_CLR)]:
            bx, by = _log_binned_means(q[mask], c[mask], n_bins=15)
            if len(bx) >= 2:
                ax.plot(
                    bx, by,
                    color=clr, lw=1.9, marker="o", markersize=3.6,
                    markeredgecolor="white", markeredgewidth=0.6,
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
                ax.plot(q_fit, c_fit, color=clr, lw=1.3, alpha=0.9,
                        label=f"{phase_lbl}: $\\beta$={sl:+.2f}{sig}", zorder=4)

                metrics_rows.append({
                    "Panel": f"cq_{pcol.split(' ')[0]}", "Station": stn,
                    "Parameter": pcol.split(" (")[0], "Phase": phase_lbl,
                    "beta": round(sl, 4), "SE": round(se, 4),
                    "R2": round(r**2, 4), "p": round(p, 6),
                    "n": int(mask.sum()),
                })

        ax.set_xscale("linear")
        ax.set_yscale("linear")
        if len(q) > 0:
            q_min, q_max = float(np.min(q)), float(np.max(q))
            pad = 0.06 * (q_max - q_min) if q_max > q_min else 100.0
            ax.set_xlim(max(0.0, q_min - pad), q_max + pad)
            ax.set_xticks([0, 10000, 20000])
            ax.set_xticklabels(["0", "10000", "20000"], fontsize=8.5)

        if plabel == "b":
            ax.set_yticks([0.0, 1.5, 3.0, 4.5, 6.0])
            ax.set_ylim(0.0, 6.8)
        elif plabel == "c":
            ax.set_yticks([4.0, 5.0, 6.0, 7.0, 8.0])
            ax.set_ylim(4.2, 8.2)
        elif plabel == "d":
            ax.set_yticks([0.00, 0.04, 0.08, 0.12, 0.16])
            ax.set_ylim(-0.005, 0.165)
        elif plabel == "e":
            ax.set_yticks([0.00, 0.15, 0.30, 0.45, 0.60])
            ax.set_ylim(-0.02, 0.63)

        ax.set_xlabel(r"$Q$ ($\mathrm{m^3\,s^{-1}}$)", fontsize=8.8)
        ax.set_ylabel(ylabel, fontsize=8.8)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5)

        ax.legend(fontsize=7.4, loc="upper right", frameon=True,
                  framealpha=0.92, edgecolor="none",
                  handlelength=0.9, handletextpad=0.25, borderaxespad=0.35, borderpad=0.3)

        px = -0.32 if plabel in ("d", "e") else -0.28
        panel_label(ax, plabel, x=px, y=1.12)

    save_fig(fig, "CEE_Figure_4_Hydrodynamic_Decoupling", metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    figure_4()
