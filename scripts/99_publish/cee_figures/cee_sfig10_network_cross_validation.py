#!/usr/bin/env python3
"""
CEE Supplementary Figure 10 — Network Cross-Validation (MRC vs Ben Tre)
=======================================================================
Cloned directly from `nw_sfig1_network_cross_validation.py`.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import stats

from _cee_common import (
    apply_style, load_wq, MM, STATION_CLR,
    panel_label, save_fig,
)

apply_style()


def main() -> None:
    wq = load_wq()
    metrics = []

    pairs = [
        {"MRC": "My Thuan", "BenTre": "CC01", "label": "My Thuan / CC01", "color": "#0072B2", "marker": "o"},
        {"MRC": "My Tho",   "BenTre": "MT01", "label": "My Tho / MT01",   "color": "#D55E00", "marker": "s"},
    ]
    params = [
        ("COD (mg/L)", "COD"), ("DO (mg/L)",  "DO"),
        ("NH4-N (mg/L)", r"NH$_4^+$-N"), ("NO3-N (mg/L)", r"NO$_3^-$-N"),
    ]

    fig = plt.figure(figsize=(180 * MM, 140 * MM))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.40)

    for pidx, (pcol, plabel) in enumerate(params):
        row, col = divmod(pidx, 2)
        ax = fig.add_subplot(gs[row, col])
        all_mrc, all_bt = [], []

        for pair in pairs:
            mrc_data = wq[(wq["stationName"] == pair["MRC"]) & (wq["Data source"] == "MRC")]
            bt_data = wq[(wq["stationName"] == pair["BenTre"]) & (wq["Data source"] == "Ben Tre")]
            if pcol not in mrc_data.columns or pcol not in bt_data.columns:
                continue

            mrc_ann = mrc_data.groupby("year")[pcol].mean().dropna()
            bt_ann = bt_data.groupby("year")[pcol].mean().dropna()
            common = mrc_ann.index.intersection(bt_ann.index)
            if len(common) < 3:
                continue

            m_vals = mrc_ann.loc[common].values
            b_vals = bt_ann.loc[common].values
            all_mrc.extend(m_vals); all_bt.extend(b_vals)

            r_pair, p_pair = stats.pearsonr(m_vals, b_vals)
            rho_pair, sp_pair = stats.spearmanr(m_vals, b_vals)
            metrics.append({
                "Figure": "CEE_SFig10",
                "Parameter": plabel,
                "Station_Pair": f"{pair['MRC']} vs. {pair['BenTre']}",
                "Pearson_r": round(float(r_pair), 4),
                "p_value": round(float(p_pair), 6),
                "Spearman_rho": round(float(rho_pair), 4),
                "Spearman_p": round(float(sp_pair), 6),
                "N_paired_years": len(m_vals),
            })

            clr = pair.get("color", STATION_CLR.get(pair["MRC"], "#333333"))
            mkr = pair.get("marker", "o")
            ax.scatter(m_vals, b_vals, s=22, c=clr, marker=mkr, edgecolors="#333333",
                       linewidths=0.35, alpha=0.85, label=pair["label"], zorder=3)

        if all_mrc and all_bt:
            mn = min(min(all_mrc), min(all_bt)) * 0.8
            mx = max(max(all_mrc), max(all_bt)) * 1.38
            ax.plot([mn, mx], [mn, mx], "--", color="#969696", lw=0.8, zorder=1)
            r_val, p_val = stats.pearsonr(all_mrc, all_bt)
            rho_val, sp_val = stats.spearmanr(all_mrc, all_bt)
            ax.text(0.05, 0.96, f"r = {r_val:.2f} (p = {p_val:.3f})\nn = {len(all_mrc)}",
                    transform=ax.transAxes, fontsize=9.0, va="top",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.2))
            metrics.append({
                "Figure": "CEE_SFig10",
                "Parameter": plabel,
                "Station_Pair": "Pooled (My Thuan/CC01 + My Tho/MT01)",
                "Pearson_r": round(float(r_val), 4),
                "p_value": round(float(p_val), 6),
                "Spearman_rho": round(float(rho_val), 4),
                "Spearman_p": round(float(sp_val), 6),
                "N_paired_years": len(all_mrc),
            })

        if pidx == 0:
            ax.set_xlabel(r"MRC $\mathrm{COD_{Mn}}$ ($\mathrm{mg\ L^{-1}}$)", fontsize=10.2)
            ax.set_ylabel(r"Ben Tre $\mathrm{COD_{Cr}}$ ($\mathrm{mg\ L^{-1}}$)", fontsize=10.2)
        else:
            ax.set_xlabel(rf"MRC {plabel} ($\mathrm{{mg\ L^{{-1}}}}$)", fontsize=10.2)
            ax.set_ylabel(rf"Ben Tre {plabel} ($\mathrm{{mg\ L^{{-1}}}}$)", fontsize=10.2)
        ax.tick_params(axis="both", which="major", labelsize=10.2)
        ax.set_aspect("equal", adjustable="datalim")
        if pidx == 0:
            ax.legend(fontsize=10.2, frameon=False, handletextpad=0.1, loc="lower right")
        panel_label(ax, chr(ord("a") + pidx), x=-0.20, y=1.08)

    save_fig(fig, "CEE_SFig_10_Network_CrossValidation", pd.DataFrame(metrics))


if __name__ == "__main__":
    main()
