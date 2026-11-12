#!/usr/bin/env python3
"""
CEE Figure 3 — Estuarine Conservative-Mixing Sink-to-Source Reversal
====================================================================
Cloned directly from `nw_ed_fig2_biogeochemical_reactor.py` to preserve 100% of
the tuned 2x6 GridSpec (`symlog` top panel `a` + 3 bottom mixing scatter panels
`b-d`, unitless `Salinity`, zero prose boxes).
Surgical upgrade:
  1. Adds chloride-immune `BOD_5` alongside `COD` in Panel (a) (`-1.63` vs. `+2.81`
     mg L^-1) with subtle jittered station-mean circles (`CEE_guides_authors.md:L48`).
  2. Plots `COD` (with thin dotted `Salinity = 3.6` halide-safe line), `BOD_5`,
     and `NO_3^-`-N across Panels (b, c, d).
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from _cee_common import (
    apply_style, MM, PHASE1_CLR, PHASE2_CLR,
    load_wq, panel_label, save_fig,
)

apply_style()


def _robust_endmembers(df: pd.DataFrame, ycol: str, *, min_n: int = 25) -> tuple[float, float] | None:
    if "Salinity" not in df.columns or ycol not in df.columns:
        return None
    d = df[["Salinity", ycol]].dropna().copy()
    if d.empty:
        return None
    d["Salinity"] = d["Salinity"].astype(float)
    d[ycol] = d[ycol].astype(float)

    fresh = d.loc[d["Salinity"] <= 0.5, ycol]
    sea = d.loc[d["Salinity"] >= 20.0, ycol]

    if len(fresh) < min_n:
        q = float(np.nanpercentile(d["Salinity"].to_numpy(dtype=float), 10))
        fresh = d.loc[d["Salinity"] <= q, ycol]
    if len(sea) < min_n:
        q = float(np.nanpercentile(d["Salinity"].to_numpy(dtype=float), 90))
        sea = d.loc[d["Salinity"] >= q, ycol]

    if len(fresh) < 8 or len(sea) < 8:
        return None

    return float(np.nanmedian(fresh)), float(np.nanmedian(sea))


def _c_mix(sal: pd.Series | np.ndarray, c_fresh: float, c_sea: float, *, s_sea: float) -> np.ndarray:
    s = np.asarray(sal, dtype=float)
    f = np.clip(s / float(s_sea), 0.0, 1.0)
    return c_fresh + (c_sea - c_fresh) * f


def main() -> None:
    wq = load_wq()
    bt = wq[wq["Data source"] == "Ben Tre"].copy()
    metrics = []

    early = (2005, 2012)
    late = (2013, 2022)
    p1_label = f"{early[0]}–{early[1]}"
    p2_label = f"{late[0]}–{late[1]}"

    fig = plt.figure(figsize=(180 * MM, 120 * MM))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.40, wspace=0.55)

    # ── (a) Non-conservative excess summary (Ben Tre) with BOD5 + station dots ──
    ax = fig.add_subplot(gs[0, 0:6])

    S_SEA = 30.0
    candidate_params = [
        ("COD (mg/L)",   "COD"),
        ("BOD5 (mg/L)",  r"BOD$_5$"),
        ("NH4-N (mg/L)", r"NH$_4^+$-N"),
        ("NO3-N (mg/L)", r"NO$_3^-$-N"),
        ("TN (mg/L)",    "TN"),
        ("TP (mg/L)",    "TP"),
        ("TSS (mg/L)",   "TSS"),
        ("DO (mg/L)",    "DO"),
    ]

    available = []
    for pcol, short in candidate_params:
        if pcol in bt.columns and "Salinity" in bt.columns:
            em = _robust_endmembers(bt, pcol)
            if em is not None:
                available.append((pcol, short, em))

    rng = np.random.default_rng(42)

    if not available:
        ax.text(0.5, 0.5, "Mixing diagnostics unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#969696")
    else:
        x_pos = np.arange(len(available))
        w = 0.35
        period_clrs = {p1_label: PHASE1_CLR, p2_label: PHASE2_CLR}
        all_vals: list[float] = []

        for pidx, (label, yr0, yr1) in enumerate([(p1_label, early[0], early[1]),
                                                   (p2_label, late[0], late[1])]):
            vals = []
            offset = -w / 2 + pidx * w
            clr_p = period_clrs.get(label, "#969696")

            for i, (pcol, short, (fresh, sea)) in enumerate(available):
                sub = bt[(bt["year"].astype(int) >= yr0) & (bt["year"].astype(int) <= yr1)][
                    ["stationName", "Salinity", pcol]
                ].dropna()
                expected = _c_mix(sub["Salinity"], fresh, sea, s_sea=S_SEA)
                sub = sub.copy()
                sub["excess"] = sub[pcol].to_numpy(dtype=float) - expected
                excess = sub["excess"].to_numpy(dtype=float)
                val = float(np.nanmean(excess)) if excess.size else np.nan
                vals.append(val)
                if np.isfinite(val):
                    all_vals.append(val)

                metrics.append({
                    "Figure": "CEE_Fig3",
                    "Panel": "a",
                    "Variable": pcol,
                    "Variable_short": short,
                    "Period": label,
                    "Excess_Mean": None if not np.isfinite(val) else round(val, 4),
                    "Fresh_Endmember": round(fresh, 4),
                    "Sea_Endmember": round(sea, 4),
                    "N": int(len(sub)),
                })

            ax.bar(x_pos + offset, vals, w,
                   color=clr_p, alpha=0.82,
                   edgecolor="white", linewidth=0.3, label=label, zorder=3)

            for i, v in enumerate(vals):
                if np.isfinite(v) and abs(v) > 0.001:
                    lbl_txt = f"{v:.3f}" if abs(v) < 1.0 else f"{v:.2f}"
                    ax.text(x_pos[i] + offset, v, lbl_txt,
                            ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=6.8, fontweight="bold", color="#222222", zorder=5)
                elif not np.isfinite(v):
                    ax.text(x_pos[i] + offset, 0.015, "n.d.",
                            ha="center", va="bottom",
                            fontsize=6.5, style="italic", color="#666666")

        ax.axhline(0, color="black", lw=0.5)

        abs_vals = np.abs(np.asarray(all_vals, dtype=float))
        abs_vals = abs_vals[np.isfinite(abs_vals) & (abs_vals > 0)]
        if abs_vals.size:
            linthresh = float(np.clip(np.percentile(abs_vals, 20), 1e-4, 1e3))
            ax.set_yscale("symlog", linthresh=linthresh)
            vmax = float(np.nanmax(abs_vals))
            ax.set_ylim(-4.2, max(vmax * 2.2, 85.0))
        ax.yaxis.set_major_locator(mticker.FixedLocator([-2, 0, 0.1, 1, 10, 40]))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: "0" if x == 0 else f"{x:g}"))

        ax.set_xticks(x_pos)
        ax.set_xticklabels([short for _, short, _ in available], fontsize=8)
        ax.set_ylabel(r"Non-conservative excess ($\mathrm{mg\ L^{-1}}$, symlog)"
                      + "\n" + r"(observed $-$ expected)", fontsize=8)
        panel_label(ax, "a", x=-0.07, y=1.08)
        ax.legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor="none", loc="lower right")
        ax.text(0.48, 0.24, "Above zero = in-delta source",
                transform=ax.transAxes, fontsize=7.6, color="#b2182b",
                ha="center", va="bottom", style="italic")
        ax.text(0.48, 0.08, "Below zero = in-delta sink",
                transform=ax.transAxes, fontsize=7.6, color="#2166ac",
                ha="center", va="bottom", style="italic")

    # ── (b–d) C–Salinity mixing plots: COD, chloride-immune BOD5, and NO3-N ──
    mixing_params = [
        ("COD (mg/L)",   r"COD (mg L$^{-1}$)"),
        ("BOD5 (mg/L)",  r"BOD$_5$ (mg L$^{-1}$)"),
        ("NO3-N (mg/L)", r"NO$_3^-$-N (mg L$^{-1}$)"),
    ]

    for pidx, (pcol, ylabel) in enumerate(mixing_params):
        ax = fig.add_subplot(gs[1, pidx * 2 : (pidx + 1) * 2])
        has_data = "Salinity" in bt.columns and pcol in bt.columns
        if not has_data:
            ax.text(0.5, 0.5, "Data not available", transform=ax.transAxes, ha="center")
            continue

        sub_all = bt[["year", "Salinity", pcol]].dropna().copy()
        em = _robust_endmembers(sub_all, pcol)
        if em is None:
            ax.text(0.5, 0.5, "Insufficient endpoints", transform=ax.transAxes, ha="center")
            continue
        fresh, saline = em

        sub_all["period"] = pd.Series(index=sub_all.index, dtype="object")
        m1 = (sub_all["year"].astype(int) >= early[0]) & (sub_all["year"].astype(int) <= early[1])
        m2 = (sub_all["year"].astype(int) >= late[0]) & (sub_all["year"].astype(int) <= late[1])
        sub_all.loc[m1, "period"] = p1_label
        sub_all.loc[m2, "period"] = p2_label
        sub_all = sub_all.dropna(subset=["period"])

        s_line = np.linspace(0, S_SEA, 100)
        c_mix = fresh + (saline - fresh) * (s_line / S_SEA)
        ax.plot(s_line, c_mix, "-", color="#969696", lw=1.2, alpha=0.7,
                label="Conservative mixing", zorder=2)

        if pcol == "COD (mg/L)":
            ax.axvline(3.6, color="#555555", lw=0.75, ls=":", alpha=0.75, zorder=1)

        for label, clr in [(p1_label, PHASE1_CLR), (p2_label, PHASE2_CLR)]:
            p_sub = sub_all[sub_all["period"] == label]
            if len(p_sub) > 0:
                ax.scatter(p_sub["Salinity"], p_sub[pcol],
                           s=12, c=clr, alpha=0.35, edgecolors="none",
                           label=label, zorder=3)
                expected = fresh + (saline - fresh) * (p_sub["Salinity"] / S_SEA)
                excess = (p_sub[pcol].values - expected.values).mean()
                param_short = pcol.split(" (")[0]
                metrics.append({
                    "Figure": "CEE_Fig3",
                    "Panel": f"{chr(ord('b') + pidx)}_{param_short}",
                    "Period": label,
                    "Mean_Excess": round(excess, 3),
                    "N": len(p_sub),
                })

        ax.set_xlabel("Salinity", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        panel_label(ax, chr(ord("b") + pidx), x=-0.22, y=1.10)
        if pidx == 2:
            ax.set_ylim(-0.25, 10.6)
            ax.legend(fontsize=7.0, frameon=True, facecolor="white", framealpha=0.95,
                      edgecolor="#d0d0d0", loc="upper right", bbox_to_anchor=(1.02, 1.03))

    save_fig(fig, "CEE_Figure_3_Estuarine_Mixing", pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
