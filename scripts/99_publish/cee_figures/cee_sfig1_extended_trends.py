#!/usr/bin/env python3
"""
CEE Supplementary Figure 1 — Extended Multi-Parameter Trends (1985–2022)
========================================================================
Cloned directly from `nw_ed_fig1_extended_trends.py`.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import theilslopes, kendalltau

from _cee_common import (
    apply_style, load_wq, MM, MRC_STATIONS, STATION_CLR,
    PHASE1_CLR, PHASE2_CLR, BREAK_YEAR,
    panel_label, save_fig,
)

apply_style()


def main() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()
    metrics = []

    params = [
        ("COD (mg/L)",   r"COD (mg $\mathrm{L^{-1}}$)",        BREAK_YEAR),
        ("DO (mg/L)",    r"DO (mg $\mathrm{L^{-1}}$)",         BREAK_YEAR),
        ("pH",           "pH",                                 BREAK_YEAR),
        ("NH4-N (mg/L)", r"NH$_4^+$-N (mg $\mathrm{L^{-1}}$)", BREAK_YEAR),
        ("NO3-N (mg/L)", r"NO$_3^-$-N (mg $\mathrm{L^{-1}}$)", BREAK_YEAR),
        ("TN (mg/L)",    r"TN (mg $\mathrm{L^{-1}}$)",         BREAK_YEAR),
        ("TP (mg/L)",    r"TP (mg $\mathrm{L^{-1}}$)",         BREAK_YEAR),
        ("TSS (mg/L)",   r"TSS (mg $\mathrm{L^{-1}}$)",        BREAK_YEAR),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(180 * MM, 150 * MM), sharex=True)

    for idx, (pcol, ylabel, bp) in enumerate(params):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        for stn in MRC_STATIONS:
            sub = mrc[mrc["stationName"] == stn]
            if pcol not in sub.columns:
                continue
            annual = sub.groupby("year")[pcol].mean().dropna()
            if len(annual) < 5:
                continue

            ax.plot(annual.index, annual.values, "-o",
                    color=STATION_CLR[stn], lw=0.9, ms=2, alpha=0.85,
                    label=stn if idx == 1 else None, zorder=3)

            x_arr = annual.index.values.astype(float)
            y_arr = annual.values
            for phase, mask in [("Phase1", x_arr <= bp), ("Phase2", x_arr > bp)]:
                xm, ym = x_arr[mask], y_arr[mask]
                if len(xm) > 5:
                    slope, intercept, _, _ = theilslopes(ym, xm)
                    _, p = kendalltau(xm, ym)
                    ax.plot(xm, intercept + slope * xm, "-",
                            color=STATION_CLR[stn], lw=1.5, alpha=0.7)
                    metrics.append({
                        "Figure": "SFig1", "Station": stn,
                        "Parameter": pcol.split(" (")[0], "Phase": phase,
                        "Slope": round(slope, 6), "p": round(p, 6),
                    })

        ax.axvline(bp + 0.5, color="#525252", lw=0.8, ls="--", alpha=0.6)
        ax.axvspan(1985, bp + 0.5, color=PHASE1_CLR, alpha=0.02, zorder=0)
        ax.axvspan(bp + 0.5, 2022, color=PHASE2_CLR, alpha=0.02, zorder=0)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.locator_params(axis="y", nbins=4)
        if row < 3:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("")
        if idx == 1:
            ax.legend(fontsize=7.8, ncol=2, loc="lower left",
                      frameon=True, framealpha=0.9, edgecolor="none", handlelength=1.2)
        panel_label(ax, chr(ord("a") + idx))

    for ax in axes.flat:
        ax.set_xlim(1984, 2023)
        ax.set_xticks(range(1985, 2025, 5))

    fig.tight_layout(h_pad=0.6, w_pad=1.5)
    save_fig(fig, "CEE_SFig_1_Extended_Trends", pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
