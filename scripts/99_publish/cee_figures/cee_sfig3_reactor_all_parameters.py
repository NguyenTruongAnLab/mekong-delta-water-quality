#!/usr/bin/env python3
"""
CEE Supplementary Figure 3 — Reactor Diagnostics Across All Parameters
======================================================================
Cloned directly from `nw_ed_fig8_reactor_all_parameters.py`.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, kendalltau

from _cee_common import (
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
    n_r, n_c = arr.shape
    for i in range(n_r):
        for j in range(n_c):
            v = arr[i, j]
            if not np.isfinite(v):
                txt = "NA"
            else:
                sig = ""
                if p_arr is not None and np.isfinite(p_arr[i, j]):
                    sig = _sig_mark(float(p_arr[i, j]))
                txt = f"{fmt.format(v)}{sig}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=6.5, color="#000000", weight="bold")


def main() -> None:
    wq = load_wq()
    mrc = wq[wq["Data source"] == "MRC"].copy()

    param_latex = {
        "NH4-N": r"$\mathrm{NH}_4^+$-N",
        "NO3-N": r"$\mathrm{NO}_3^-$-N",
    }

    stations = [s for s in MRC_STATIONS if s in set(mrc["stationName"].dropna().unique())]
    candidate_params = [
        "COD (mg/L)", "DO (mg/L)", "NH4-N (mg/L)", "NO3-N (mg/L)",
        "TN (mg/L)", "TP (mg/L)", "TSS (mg/L)", "pH",
    ]
    params = [p for p in candidate_params if p in mrc.columns]

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
                    "Station": stn, "Parameter": p_name, "Phase": phase_label,
                    "Slope": round(slope, 6), "Slope_x1000": round(slope * 1000.0, 4),
                    "Kendall_p": round(pval, 6), "n_years": int(np.sum(mask)),
                })
                if phase_label.startswith("Phase 1"):
                    p1_slope[pi, si] = slope * 1000.0
                    p1_pval[pi, si] = pval
                else:
                    p2_slope[pi, si] = slope * 1000.0
                    p2_pval[pi, si] = pval

    delta = p2_slope - p1_slope

    mb_summary = pd.DataFrame(columns=[
        "Parameter", "Phase1_NCF_pct", "Phase2_NCF_pct", "Delta_pct", "Trend_slope_pct_yr", "Kendall_p", "n_years"
    ])
    mb_yearly: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(42)
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
            y1 = ya[xa <= BREAK_YEAR]
            y2 = ya[xa > BREAK_YEAR]
            mb_yearly[p_name] = (y1, y2)
            p1 = float(np.mean(y1)) if y1.size else np.nan
            p2 = float(np.mean(y2)) if y2.size else np.nan
            row = {
                "Parameter": p_name, "Phase1_NCF_pct": p1, "Phase2_NCF_pct": p2,
                "Delta_pct": p2 - p1 if np.isfinite(p1) and np.isfinite(p2) else np.nan,
                "Trend_slope_pct_yr": slope, "Kendall_p": pval, "n_years": int(len(annual)),
            }
            rows.append(row)
            metrics_rows.append({"Panel": "mass_balance_allparams", **{
                k: (round(v, 6) if isinstance(v, float) and np.isfinite(v) else v) for k, v in row.items()
            }})
        if rows:
            mb_summary = pd.DataFrame(rows)

    fig = plt.figure(figsize=(180 * MM, 150 * MM))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.40, hspace=0.28)

    lim = float(np.nanpercentile(np.abs(np.r_[p1_slope.flatten(), p2_slope.flatten()]), 95))
    lim = lim if np.isfinite(lim) and lim > 0 else 1.0
    norm = mcolors.SymLogNorm(linthresh=10.0, linscale=1.0, vmin=-lim, vmax=lim, base=10)

    cmap_base = plt.get_cmap("RdBu_r")
    colors = cmap_base(np.linspace(0, 1, 256))
    colors[:, :3] = 0.40 * colors[:, :3] + 0.60
    light_rdbu = mcolors.ListedColormap(colors)

    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(p1_slope, cmap=light_rdbu, norm=norm, aspect="auto")
    _annot_heatmap(ax_a, p1_slope, p_arr=p1_pval)
    panel_label(ax_a, "a", x=-0.14, y=1.05)
    ax_a.set_xticks(np.arange(n_s))
    ax_a.set_xticklabels(stations, rotation=20, ha="right", fontsize=6.5, color="#000000")
    ax_a.set_yticks(np.arange(n_p))
    ax_a.set_yticklabels([param_latex.get(p.split(" (")[0], p.split(" (")[0]) for p in params], fontsize=6.5, color="#000000")
    ax_a.tick_params(axis="both", labelsize=6.5, labelcolor="#000000")
    cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.02)
    cb_a.ax.tick_params(labelsize=6.5, labelcolor="#000000")
    cb_a.set_label(r"Phase 1 trend slope ($10^{-3}$ units yr$^{-1}$)", fontsize=7.8, color="#000000", weight="bold")

    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(p2_slope, cmap=light_rdbu, norm=norm, aspect="auto")
    _annot_heatmap(ax_b, p2_slope, p_arr=p2_pval)
    panel_label(ax_b, "b", x=-0.14, y=1.05)
    ax_b.set_xticks(np.arange(n_s))
    ax_b.set_xticklabels(stations, rotation=20, ha="right", fontsize=6.5, color="#000000")
    ax_b.set_yticks(np.arange(n_p))
    ax_b.set_yticklabels([param_latex.get(p.split(" (")[0], p.split(" (")[0]) for p in params], fontsize=6.5, color="#000000")
    ax_b.tick_params(axis="both", labelsize=6.5, labelcolor="#000000")
    cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.02)
    cb_b.ax.tick_params(labelsize=6.5, labelcolor="#000000")
    cb_b.set_label(r"Phase 2 trend slope ($10^{-3}$ units yr$^{-1}$)", fontsize=7.8, color="#000000", weight="bold")

    ax_c = fig.add_subplot(gs[1, 0])
    lim_d = float(np.nanpercentile(np.abs(delta.flatten()), 95))
    lim_d = lim_d if np.isfinite(lim_d) and lim_d > 0 else 1.0
    norm_d = mcolors.SymLogNorm(linthresh=10.0, linscale=1.0, vmin=-lim_d, vmax=lim_d, base=10)
    im_c = ax_c.imshow(delta, cmap=light_rdbu, norm=norm_d, aspect="auto")
    _annot_heatmap(ax_c, delta, p_arr=None)
    panel_label(ax_c, "c", x=-0.14, y=1.05)
    ax_c.set_xticks(np.arange(n_s))
    ax_c.set_xticklabels(stations, rotation=20, ha="right", fontsize=6.5, color="#000000")
    ax_c.set_yticks(np.arange(n_p))
    ax_c.set_yticklabels([param_latex.get(p.split(" (")[0], p.split(" (")[0]) for p in params], fontsize=6.5, color="#000000")
    ax_c.tick_params(axis="both", labelsize=6.5, labelcolor="#000000")
    cb_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.02)
    cb_c.ax.tick_params(labelsize=6.5, labelcolor="#000000")
    cb_c.set_label(r"Slope shift ($10^{-3}$ units yr$^{-1}$)", fontsize=7.8, color="#000000", weight="bold")

    ax_d = fig.add_subplot(gs[1, 1])
    if not mb_summary.empty:
        import matplotlib.ticker as mticker
        ordered_param_names = [p.split(" (")[0] for p in params]
        mbp = pd.DataFrame({"Parameter": ordered_param_names})
        mbp = mbp.merge(mb_summary, on="Parameter", how="left").dropna(subset=["Phase1_NCF_pct", "Phase2_NCF_pct"]).reset_index(drop=True)
        xpos = np.arange(len(mbp))
        w = 0.28
        p1_vals = mbp["Phase1_NCF_pct"].to_numpy(dtype=float)
        p2_vals = mbp["Phase2_NCF_pct"].to_numpy(dtype=float)
        has_mb = np.isfinite(p1_vals) & np.isfinite(p2_vals)

        for pos in np.arange(len(mbp) - 1) + 0.5:
            ax_d.axvline(pos, color="#e0e0e0", lw=0.6, ls="--", zorder=1)

        ax_d.bar(xpos - w / 2, np.nan_to_num(p1_vals, nan=0.0), w,
                 color="#6baed6", alpha=0.78, edgecolor="none", linewidth=0, label=f"Phase 1 (≤{BREAK_YEAR})", zorder=2)
        ax_d.bar(xpos + w / 2, np.nan_to_num(p2_vals, nan=0.0), w,
                 color="#fb6a4a", alpha=0.78, edgecolor="none", linewidth=0, label=f"Phase 2 (>{BREAK_YEAR})", zorder=2)
        ax_d.axhline(0, color="#444444", lw=0.7, zorder=2)

        for i, (_, row) in enumerate(mbp.iterrows()):
            if not bool(has_mb[i]):
                continue
            pname = str(row["Parameter"])
            y1, y2 = mb_yearly.get(pname, (np.array([]), np.array([])))
            top_pair = max(float(row["Phase1_NCF_pct"]), float(row["Phase2_NCF_pct"]), 0.0)
            for off, y_arr, clr_edge in [(-w / 2, y1, "#2171b5"), (+w / 2, y2, "#cb181d")]:
                if y_arr.size > 0:
                    jit = rng.uniform(-w * 0.26, w * 0.26, size=y_arr.size)
                    ax_d.scatter(np.full(y_arr.size, xpos[i] + off) + jit, y_arr,
                                 s=6.5, facecolors="white", edgecolors=clr_edge,
                                 linewidths=0.45, alpha=0.58, zorder=3)
                    m_val = float(np.mean(y_arr))
                    boots = rng.choice(y_arr, size=(2000, y_arr.size), replace=True).mean(axis=1)
                    lo, hi = [float(x) for x in np.percentile(boots, [2.5, 97.5])]
                    ax_d.errorbar(xpos[i] + off, m_val,
                                  yerr=[[max(0.0, m_val - lo)], [max(0.0, hi - m_val)]],
                                  fmt="none", ecolor="#222222", elinewidth=0.65,
                                  capsize=1.5, capthick=0.65, zorder=4)
                    top_pair = max(top_pair, hi, float(np.max(y_arr)))

            delta_txt = row["Delta_pct"]
            sig = _sig_mark(float(row["Kendall_p"])) if np.isfinite(row["Kendall_p"]) else ""
            y_lbl = top_pair * 1.24 if top_pair > 22.0 else top_pair + 4.5
            ax_d.text(float(i), y_lbl, f"Δ={delta_txt:+.1f}{sig}",
                      ha="center", va="bottom", fontsize=5.9, color="#000000",
                      bbox=dict(boxstyle="round,pad=0.06", facecolor="white", alpha=0.82, edgecolor="none"),
                      zorder=5)

        ax_d.set_yscale("symlog", linthresh=20.0)
        ax_d.set_ylim(-75.0, 1400.0)
        ax_d.yaxis.set_major_locator(mticker.FixedLocator([-40, -20, 0, 20, 50, 150]))
        ax_d.yaxis.set_minor_locator(mticker.NullLocator())
        ax_d.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: "0" if x == 0 else f"{x:g}"))
        ax_d.set_xticks(xpos)
        ax_d.set_xticklabels([param_latex.get(x, x) for x in mbp["Parameter"]], rotation=32, ha="right", fontsize=6.5, color="#000000")
        ax_d.set_ylabel("Non-conservative fraction (%, symlog)", fontsize=7.6, color="#000000", weight="bold")
        ax_d.tick_params(axis="both", labelsize=6.5, labelcolor="#000000")
        ax_d.text(0.03, 0.04, "* p < 0.05", transform=ax_d.transAxes, fontsize=6.5, color="#000000", style="italic")
        ax_d.legend(fontsize=6.0, ncol=2, frameon=True, framealpha=0.92, edgecolor="none",
                    loc="upper center", bbox_to_anchor=(0.50, 1.00), columnspacing=0.9, handlelength=1.2, labelcolor="#000000")

    panel_label(ax_d, "d", x=-0.16, y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, "CEE_SFig_3_Reactor_All_Parameters", metrics=pd.DataFrame(metrics_rows))


if __name__ == "__main__":
    main()
