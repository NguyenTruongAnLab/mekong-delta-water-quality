#!/usr/bin/env python3
"""
CEE Supplementary Figure 4 — Seasonal Hysteresis Loops at Tan Chau
==================================================================
Cloned directly from `nw_ed_fig3_hysteresis_decoupling.py`.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy import stats

from _cee_common import (
    apply_style, load_wq, load_hydro_monthly,
    MM, panel_label, save_fig,
    BREAK_YEAR, PHASE1_CLR, PHASE2_CLR,
)

apply_style()


def _signed_area(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3:
        return np.nan
    xx = np.r_[x, x[0]]
    yy = np.r_[y, y[0]]
    return 0.5 * float(np.sum(xx[:-1] * yy[1:] - xx[1:] * yy[:-1]))


def _monthly_summary(df: pd.DataFrame, pcol: str, qcol: str) -> pd.DataFrame:
    return (
        df[["month", pcol, qcol]]
        .dropna()
        .groupby("month")
        .agg(
            C_mean=(pcol, "mean"),
            C_sem=(pcol, lambda s: np.nanstd(s, ddof=1) / np.sqrt(max(1, len(s)))),
            Q_mean=(qcol, "mean"),
            Q_sem=(qcol, lambda s: np.nanstd(s, ddof=1) / np.sqrt(max(1, len(s)))),
            n=(pcol, "size"),
        )
        .reset_index()
        .sort_values("month")
    )


def _plot_phase_loop(ax: Axes, mon: pd.DataFrame, color: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qv = mon["Q_mean"].to_numpy(dtype=float)
    cv = mon["C_mean"].to_numpy(dtype=float)
    mv = mon["month"].to_numpy(dtype=int)

    ax.errorbar(qv, cv, xerr=mon["Q_sem"].to_numpy(dtype=float), yerr=mon["C_sem"].to_numpy(dtype=float),
                fmt="none", ecolor=color, alpha=0.28, elinewidth=0.8, capsize=0, zorder=2)
    ax.plot(qv, cv, color=color, lw=1.6, alpha=0.85, zorder=3)
    ax.plot([qv[-1], qv[0]], [cv[-1], cv[0]], color=color, lw=0.9, alpha=0.5, ls="--", zorder=2)

    for m_start, m_end in [(6, 7), (10, 11)]:
        idx1 = np.where(mv == m_start)[0]
        idx2 = np.where(mv == m_end)[0]
        if len(idx1) > 0 and len(idx2) > 0:
            x1, y1 = qv[idx1[0]], cv[idx1[0]]
            x2, y2 = qv[idx2[0]], cv[idx2[0]]
            dx, dy = x2 - x1, y2 - y1
            ax.annotate("", xy=(x1 + 0.65 * dx, y1 + 0.65 * dy),
                        xytext=(x1 + 0.35 * dx, y1 + 0.35 * dy),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=12),
                        zorder=6)

    ax.scatter(qv, cv, s=82, facecolor="white", edgecolor=color, linewidth=0.95, zorder=4)
    for q_m, c_m, m in zip(qv, cv, mv):
        ax.text(q_m, c_m, f"{int(m)}", ha="center", va="center",
                fontsize=6.6, color=color, zorder=5, fontweight="bold")
    return qv, cv, mv


def main() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()
    q_monthly = load_hydro_monthly()

    stn = "Tan Chau"
    q_col = "TanChau"
    stn_df = mrc[mrc["stationName"] == stn].merge(q_monthly[["ym", q_col]], on="ym", how="inner")

    params = [
        ("COD (mg/L)",   r"COD (mg $\mathrm{L^{-1}}$)"),
        ("DO (mg/L)",    r"DO (mg $\mathrm{L^{-1}}$)"),
        ("NH4-N (mg/L)", r"NH$_4^+$-N (mg $\mathrm{L^{-1}}$)"),
        ("NO3-N (mg/L)", r"NO$_3^-$-N (mg $\mathrm{L^{-1}}$)"),
    ]

    fig = plt.figure(figsize=(180 * MM, 155 * MM))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.20)
    metrics_rows: list[dict] = []

    for idx, (pcol, ylabel) in enumerate(params):
        ridx, cidx = divmod(idx, 2)
        ax = fig.add_subplot(gs[ridx, cidx])

        sub = stn_df[np.isfinite(stn_df[pcol]) & np.isfinite(stn_df[q_col])].copy()
        sub = sub[(sub[pcol] > 0) & (sub[q_col] > 0)]

        p1 = sub[sub["year"] <= BREAK_YEAR].copy()
        p2 = sub[sub["year"] > BREAK_YEAR].copy()
        all_q: list[np.ndarray] = []
        all_c: list[np.ndarray] = []

        for phase_df, phase_clr, phase_label in [
            (p1, PHASE1_CLR, f"Phase 1 (≤{BREAK_YEAR})"),
            (p2, PHASE2_CLR, f"Phase 2 (>{BREAK_YEAR})"),
        ]:
            mon = _monthly_summary(phase_df, pcol, q_col)
            if mon.empty or len(mon) < 3:
                continue
            qv, cv, _ = _plot_phase_loop(ax, mon, phase_clr)
            all_q.append(qv)
            all_c.append(cv)

            sr = np.asarray(stats.spearmanr(mon["Q_mean"], mon["C_mean"], nan_policy="omit"), dtype=float)
            rho = float(sr[0]) if sr.size > 0 else np.nan
            q_std, c_std = np.nanstd(qv), np.nanstd(cv)
            area_std = _signed_area((qv - np.nanmean(qv)) / q_std, (cv - np.nanmean(cv)) / c_std) if (q_std > 0 and c_std > 0) else np.nan
            metrics_rows.append({
                "Station": stn, "Parameter": pcol.split(" (")[0], "Phase": phase_label,
                "Spearman_rho_monthly": round(rho, 4) if np.isfinite(rho) else None,
                "LoopArea_std_signed": round(area_std, 4) if np.isfinite(area_std) else None,
            })

        if all_q and all_c:
            q_all = np.concatenate(all_q)
            c_all = np.concatenate(all_c)
            qpad = 0.06 * (float(np.nanmax(q_all)) - float(np.nanmin(q_all)) + 1.0)
            cpad = 0.10 * (float(np.nanmax(c_all)) - float(np.nanmin(c_all)) + 1e-6)
            ax.set_xlim(float(np.nanmin(q_all)) - qpad, float(np.nanmax(q_all)) + qpad)
            ax.set_ylim(max(0.0, float(np.nanmin(c_all)) - cpad), float(np.nanmax(c_all)) + cpad)

        if ridx == 1:
            ax.set_xlabel(r"Discharge ($\mathrm{m^3\,s^{-1}}$)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.locator_params(axis="both", nbins=6)
        panel_label(ax, chr(ord("a") + idx))

    legend_elements = [
        Line2D([0], [0], color=PHASE1_CLR, lw=2.0, label=rf"Phase 1 (≤{BREAK_YEAR})"),
        Line2D([0], [0], color=PHASE2_CLR, lw=2.0, label=rf"Phase 2 (>{BREAK_YEAR})"),
        Line2D([0], [0], marker="o", color="white", markeredgecolor="#333333",
               lw=0, markersize=5.5, label="Monthly centroid (1 = Jan)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.95), fontsize=8.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_fig(fig, "CEE_SFig_4_Monthly_Hysteresis", metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    main()
