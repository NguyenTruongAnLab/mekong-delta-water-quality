#!/usr/bin/env python3
"""
Supplementary Figure S8 — WRTDS Flow-Normalized Concentration Trends
=====================================================================
Flow-normalized COD and NO₃-N concentrations at 4 MRC stations (1986–2023),
computed using EGRET/WRTDSKalman with a structural wall at the 2004 PELT
breakpoint.  The CQTC/QTC decomposition (inset text) shows that >98% of
concentration changes are driven by changes in the C–Q relationship, not
by discharge changes alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import _nw_common as nw

nw.apply_style()

WRTDS_DIR = nw.ROOT / "analysis" / "wrtds_egret" / "outputs"
BREAK_YEAR = nw.BREAK_YEAR

STATIONS = [
    ("TanChau", "Tan Chau"),
    ("MyThuan", "My Thuan"),
    ("CanTho",  "Can Tho"),
    ("MyTho",   "My Tho"),
]

PARAMS = [
    ("COD", r"COD (mg L$^{-1}$)"),
    ("NO3", r"NO$_3$-N (mg L$^{-1}$)"),
]

FLOW_COLOR = {
    "COD": "#1f77b4",   # blue
    "NO3": "#d62728",   # red
}


def load_annual(stn_id: str, par_code: str) -> pd.DataFrame:
    """Read WRTDS AnnualResults CSV for one station×parameter."""
    path = WRTDS_DIR / f"{stn_id}_{par_code}_AnnualResults.csv"
    nw.require_file(path, label=f"{stn_id}_{par_code} annual")
    df = pd.read_csv(path)
    df.columns = [c.split("[")[0].strip().replace(" ", "_") for c in df.columns]
    return df


def load_pair(stn_id: str, par_code: str) -> dict | None:
    """Read WRTDS PairResults CSV for CQTC/QTC decomposition."""
    path = WRTDS_DIR / f"{stn_id}_{par_code}_PairResults.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return {
        "total": float(df["TotalChange1"].iloc[0]),
        "cqtc":  float(df["CQTC1"].iloc[0]),
        "qtc":   float(df["QTC1"].iloc[0]),
    }


def _best_annotation_position(df: pd.DataFrame) -> tuple[float, float, str, str]:
    """Choose the least-overlap corner for annotation based on line occupancy.

    We evaluate four corners by counting how many Observed/FN points fall in
    each corner window (x-side + y-band). Lower count => less overlap.
    """
    if df.empty:
        return (0.97, 0.95, "right", "top")

    years = df["Year"].to_numpy(dtype=float)
    y_obs = df["Conc"].to_numpy(dtype=float)
    y_fn = df["FN_Conc"].to_numpy(dtype=float)

    x_min, x_max = np.nanmin(years), np.nanmax(years)
    y_all = np.concatenate([y_obs[np.isfinite(y_obs)], y_fn[np.isfinite(y_fn)]])
    if y_all.size == 0:
        return (0.97, 0.95, "right", "top")

    y_lo, y_hi = np.nanmin(y_all), np.nanmax(y_all)
    y_rng = max(y_hi - y_lo, 1e-9)
    x_cut = x_min + 0.20 * (x_max - x_min)
    y_low_band = y_lo + 0.35 * y_rng
    y_high_band = y_hi - 0.35 * y_rng

    candidates = [
        (0.03, 0.95, "left", "top", "left", "top"),
        (0.97, 0.95, "right", "top", "right", "top"),
        (0.03, 0.05, "left", "bottom", "left", "bottom"),
        (0.97, 0.05, "right", "bottom", "right", "bottom"),
    ]

    def _score(side_x: str, side_y: str) -> int:
        if side_x == "left":
            xm = years <= x_cut
        else:
            xm = years >= (x_max - (x_cut - x_min))

        if side_y == "top":
            ym_obs = y_obs >= y_high_band
            ym_fn = y_fn >= y_high_band
        else:
            ym_obs = y_obs <= y_low_band
            ym_fn = y_fn <= y_low_band

        return int(np.sum(xm & ym_obs) + np.sum(xm & ym_fn))

    best = min(candidates, key=lambda c: _score(c[2], c[3]))
    return best[0], best[1], best[4], best[5]


def main() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig, axes = plt.subplots(
        len(STATIONS), len(PARAMS),
        figsize=(180 * nw.MM, 200 * nw.MM),
        sharex=True,
    )

    metrics_rows = []

    for row, (stn_id, stn_name) in enumerate(STATIONS):
        for col, (par_code, par_label) in enumerate(PARAMS):
            ax = axes[row, col]
            df = load_annual(stn_id, par_code)

            # Restrict to data period (exclude extrapolation years)
            df = df[(df["Year"] >= 1986) & (df["Year"] <= 2023)]

            # Observed concentration
            ax.plot(df["Year"], df["Conc"], color="#8a8a8a", lw=1.15,
                    alpha=0.95, label="Observed", zorder=1)

            # Flow-normalized concentration
            ax.plot(df["Year"], df["FN_Conc"], color=FLOW_COLOR[par_code],
                    lw=1.6, label="Flow-normalized")

            # Breakpoint
            ax.axvline(BREAK_YEAR + 0.5, color="k", ls="--", lw=0.5, alpha=0.5)

            # Phase means
            pre = df[df["Year"] <= BREAK_YEAR]["FN_Conc"]
            post = df[df["Year"] > BREAK_YEAR]["FN_Conc"]
            if len(pre) > 0 and len(post) > 0:
                ax.axhline(pre.mean(), color=nw.PHASE1_CLR, ls=":", lw=0.5,
                           xmin=0, xmax=0.5, alpha=0.6)
                ax.axhline(post.mean(), color=nw.PHASE2_CLR, ls=":", lw=0.5,
                           xmin=0.5, xmax=1, alpha=0.6)

            # CQTC/QTC annotation
            pair = load_pair(stn_id, par_code)
            if pair is not None:
                total = pair["total"]
                cqtc_pct = (pair["cqtc"] / pair["total"]) * 100 if pair["total"] != 0 else 0
                qtc_pct = (pair["qtc"] / pair["total"]) * 100 if pair["total"] != 0 else 0
                sign = "+" if total > 0 else ""
                bx, by, bha, bva = _best_annotation_position(df)
                ax.text(bx, by,
                    f"$\\Delta$C = {sign}{total:.2f}\nCQTC = {cqtc_pct:.1f}%\nQTC = {qtc_pct:+.1f}%",
                        transform=ax.transAxes, fontsize=8.5,
                        ha=bha, va=bva,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="0.7", alpha=0.85))

            # Labels
            if col == 0:
                ax.set_ylabel(stn_name, fontsize=12, fontweight="bold")
            if row == 0:
                ax.set_title(par_label, fontsize=13, fontweight="bold")

            ax.set_xlim(1985, 2024)

            # Collect metrics
            if len(pre) > 0 and len(post) > 0:
                pct = (post.mean() - pre.mean()) / pre.mean() * 100
                metrics_rows.append({
                    "Station": stn_name, "Parameter": par_code,
                    "FN_Pre2004": round(pre.mean(), 3),
                    "FN_Post2004": round(post.mean(), 3),
                    "Pct_Change": round(pct, 1),
                    "CQTC_pct": round(cqtc_pct, 2) if pair else None,
                    "QTC_pct": round(qtc_pct, 2) if pair else None,
                })

    # Bottom legend with three columns.
    legend_handles = [
        Line2D([0], [0], color="#999999", lw=1.0, alpha=0.7, label="Observed"),
        Line2D([0], [0], color=FLOW_COLOR["COD"], lw=1.8, label="Flow-normalized COD"),
        Line2D([0], [0], color=FLOW_COLOR["NO3"], lw=1.8, label="Flow-normalized NO$_3$-N"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=True, framealpha=0.9, edgecolor="0.8",
               bbox_to_anchor=(0.5, 0.012))

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    metrics = pd.DataFrame(metrics_rows) if metrics_rows else None
    nw.save_fig(fig, "Figure_S8_WRTDS_FlowNormalization", metrics=metrics)


if __name__ == "__main__":
    main()
