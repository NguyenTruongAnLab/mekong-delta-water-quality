#!/usr/bin/env python3
"""
Extended Data Figure 7 — Seasonal Hysteresis Loops (Hydrodynamic Decoupling)
===========================================================================

Purpose
-------
Provide a low-overlap, mechanism-focused diagnostic complementary to Figure 3.
Instead of plotting all raw monthly points, this figure summarizes each month
within each study phase and connects Jan→Dec into a seasonal hysteresis loop.

Layout
------
Panels: COD, DO, NH4, NO3 (2 × 2)
Each panel overlays:
    • Phase 1 (≤ BREAK_YEAR, blue)
    • Phase 2 (> BREAK_YEAR, red)

Exports companion metrics CSV with loop area and monthly C–Q association
for each phase.
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

from _nw_common import (
    apply_style, load_wq, load_hydro_monthly,
    MM, panel_label, save_fig,
    BREAK_YEAR, PHASE1_CLR, PHASE2_CLR,
)

apply_style()

def _signed_area(x: np.ndarray, y: np.ndarray) -> float:
    """Signed polygon area via shoelace formula (positive = counterclockwise)."""
    if len(x) < 3 or len(y) < 3:
        return np.nan
    xx = np.r_[x, x[0]]
    yy = np.r_[y, y[0]]
    return 0.5 * float(np.sum(xx[:-1] * yy[1:] - xx[1:] * yy[:-1]))


def _monthly_summary(df: pd.DataFrame, pcol: str, qcol: str) -> pd.DataFrame:
    g = (
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
    return g


def _plot_phase_loop(
    ax: Axes,
    mon: pd.DataFrame,
    color: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qv = mon["Q_mean"].to_numpy(dtype=float)
    cv = mon["C_mean"].to_numpy(dtype=float)
    mv = mon["month"].to_numpy(dtype=int)

    ax.errorbar(
        qv, cv,
        xerr=mon["Q_sem"].to_numpy(dtype=float),
        yerr=mon["C_sem"].to_numpy(dtype=float),
        fmt="none", ecolor=color, alpha=0.28,
        elinewidth=0.8, capsize=0, zorder=2,
    )

    ax.plot(qv, cv, color=color, lw=1.6, alpha=0.85, zorder=3)
    ax.plot([qv[-1], qv[0]], [cv[-1], cv[0]], color=color, lw=0.9,
            alpha=0.5, ls="--", zorder=2)

    ax.scatter(qv, cv, s=105, facecolor="white", edgecolor=color,
               linewidth=1.0, zorder=4)
    for q_m, c_m, m in zip(qv, cv, mv):
        ax.text(q_m, c_m, f"{int(m)}", ha="center", va="center",
                fontsize=7.5, color=color, zorder=5, fontweight="bold")

    return qv, cv, mv


def ed_figure_9() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()
    q_monthly = load_hydro_monthly()

    stn = "Tan Chau"
    q_col = "TanChau"
    stn_df = mrc[mrc["stationName"] == stn].merge(
        q_monthly[["ym", q_col]], on="ym", how="inner"
    )

    params = [
        ("COD (mg/L)", r"COD (mg $\mathrm{L^{-1}}$)"),
        ("DO (mg/L)", r"DO (mg $\mathrm{L^{-1}}$)"),
        ("NH4-N (mg/L)", r"NH$_4^+$ (mg $\mathrm{L^{-1}}$)"),
        ("NO3-N (mg/L)", r"NO$_3^-$ (mg $\mathrm{L^{-1}}$)"),
    ]

    fig = plt.figure(figsize=(180 * MM, 155 * MM))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.20)

    metrics_rows: list[dict] = []

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for idx, (pcol, ylabel) in enumerate(params):
        ridx, cidx = divmod(idx, 2)
        ax = fig.add_subplot(gs[ridx, cidx])

        if pcol not in stn_df.columns:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7)
            continue

        sub = stn_df[np.isfinite(stn_df[pcol]) & np.isfinite(stn_df[q_col])].copy()
        sub = sub[(sub[pcol] > 0) & (sub[q_col] > 0)]

        p1 = sub[sub["year"] <= BREAK_YEAR].copy()
        p2 = sub[sub["year"] > BREAK_YEAR].copy()

        phase_specs = [
            ("Phase1", p1, PHASE1_CLR, f"Phase 1 (≤{BREAK_YEAR})"),
            ("Phase2", p2, PHASE2_CLR, f"Phase 2 (>{BREAK_YEAR})"),
        ]

        all_q: list[np.ndarray] = []
        all_c: list[np.ndarray] = []

        for phase_key, phase_df, phase_clr, phase_label in phase_specs:
            mon = _monthly_summary(phase_df, pcol, q_col)
            if mon.empty or len(mon) < 3:
                continue

            qv, cv, _ = _plot_phase_loop(ax, mon, phase_clr)
            all_q.append(qv)
            all_c.append(cv)

            rho = np.nan
            p_rho = np.nan
            if len(mon) >= 3:
                sr = np.asarray(
                    stats.spearmanr(
                        mon["Q_mean"].to_numpy(dtype=float),
                        mon["C_mean"].to_numpy(dtype=float),
                        nan_policy="omit",
                    ),
                    dtype=float,
                )
                rho = float(sr[0]) if sr.size > 0 else np.nan
                p_rho = float(sr[1]) if sr.size > 1 else np.nan

            q_std = np.nanstd(qv)
            c_std = np.nanstd(cv)
            if np.isfinite(q_std) and np.isfinite(c_std) and q_std > 0 and c_std > 0:
                qz = (qv - np.nanmean(qv)) / q_std
                cz = (cv - np.nanmean(cv)) / c_std
                area_std = _signed_area(qz, cz)
            else:
                area_std = np.nan

            row = {
                "Panel": panel_labels[idx],
                "Station": stn,
                "Parameter": pcol.split(" (")[0],
                "Phase": phase_label,
                "Year_Start": int(phase_df["year"].min()) if len(phase_df) else None,
                "Year_End": int(phase_df["year"].max()) if len(phase_df) else None,
                "n_obs": int(len(phase_df)),
                "n_months": int(len(mon)),
                "Spearman_rho_monthly": round(rho, 4) if np.isfinite(rho) else None,
                "Spearman_p_monthly": round(p_rho, 6) if np.isfinite(p_rho) else None,
                "LoopArea_std_signed": round(area_std, 4) if np.isfinite(area_std) else None,
            }

            if np.all(qv > 0) and np.all(cv > 0) and len(qv) >= 4:
                lr = np.asarray(stats.linregress(np.log10(qv), np.log10(cv)), dtype=float)
                row["beta_loglog_monthly"] = round(float(lr[0]), 4)
                row["beta_p_monthly"] = round(float(lr[3]), 6)
            else:
                row["beta_loglog_monthly"] = None
                row["beta_p_monthly"] = None

            metrics_rows.append(row)

        if not all_q or not all_c:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="#666666")

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

    legend_elements = [
        Line2D([0], [0], color=PHASE1_CLR, lw=2.0, label=rf"Phase 1 (≤{BREAK_YEAR})"),
        Line2D([0], [0], color=PHASE2_CLR, lw=2.0, label=rf"Phase 2 (>{BREAK_YEAR})"),
        Line2D([0], [0], marker="o", color="white", markeredgecolor="#333333",
               lw=0, markersize=5.5, label="Monthly centroid (1 = Jan)")
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.95), fontsize=8.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_fig(fig, "ED_Figure_7_Monthly_Hysteresis_Decoupling", metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    ed_figure_9()
