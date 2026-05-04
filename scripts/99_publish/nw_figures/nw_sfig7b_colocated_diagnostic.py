#!/usr/bin/env python3
"""
Diagnostic Figure: Co-located MRC vs Ben Tre stations
=====================================================
Compares N–NO₃⁻ correlations at MRC My Thuan / My Tho (mainstream)
vs. co-located Ben Tre stations (distributary channels).

Key finding: MRC mainstream stations sample mid-channel where basin-scale
variability dominates; Ben Tre stations in distributary branches reflect
local agricultural drainage → opposite fertilizer–WQ signals at ~same location.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _nw_common as nw

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

ROOT = nw.ROOT
FAO_PATH = ROOT / "data" / "Fertilizers" / "FAO" / "Vietnam_FAOSTAT_data.csv"

WQ_COL = "NO3-N (mg/L)_annual"
BREAK = nw.BREAK_YEAR


def load_fao_n():
    fao = pd.read_csv(FAO_PATH)
    fao["Year"] = fao["Year"].astype(int)
    sub = fao[(fao["Item"] == "Nutrient nitrogen N (total)") &
              (fao["Element"] == "Use per area of cropland")]
    return sub[["Year", "Value"]].set_index("Year")["Value"].rename("N_kg_ha").sort_index()


def fd_corr(x, y, min_n=5):
    xd, yd = x.diff().dropna(), y.diff().dropna()
    idx = xd.index.intersection(yd.index)
    xd, yd = xd.reindex(idx), yd.reindex(idx)
    mask = xd.notna() & yd.notna()
    xd, yd = xd[mask], yd[mask]
    if len(xd) < min_n:
        return np.nan, np.nan, len(xd)
    r, p = stats.pearsonr(xd, yd)
    return r, p, len(xd)


def main():
    nw.apply_style()

    master = pd.read_csv(nw.MASTER_PATH)
    n_fao = load_fao_n()

    # Pairs: MRC station → co-located Ben Tre stations
    # My Thuan (MRC, Tien River) co-located with CC01-CC05 (Co Chien, <5 km)
    # My Tho (MRC, Tien River) co-located with MT01-MT05 (My Tho branch)
    # Also include the closest "upstream" MRC pairs: Can Tho ↔ HL (Ham Luong)
    groups = {
        "MRC My Thuan\nvs Co Chien": {
            "mrc": "My Thuan",
            "bt": ["CC01", "CC02", "CC03", "CC04"],
        },
        "MRC My Tho\nvs My Tho River": {
            "mrc": "My Tho",
            "bt": ["MT01", "MT02", "MT03"],
        },
    }

    # Also get within-river gradients for all 4 rivers
    rivers = {
        "Ba Lai": ["BL01", "BL02", "BL03", "BL04"],
        "Co Chien": ["CC01", "CC02", "CC03", "CC04"],
        "Ham Luong": ["HL01", "HL02", "HL03", "HL04"],
        "My Tho R.": ["MT01", "MT02", "MT03"],
    }

    # ── Panel A + B: Time series comparison ──
    fig = plt.figure(figsize=(180 * nw.MM, 200 * nw.MM))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.95, bottom=0.08)

    # Panel (a): My Thuan MRC vs Co Chien BT
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a2 = ax_a.twinx()  # for N_kg_ha
    for label, stn, color, ls in [
        ("MRC My Thuan", "My Thuan", nw.PHASE1_CLR, "-"),
        ("CC01", "CC01", "#66c2a5", "--"),
        ("CC02", "CC02", "#fc8d62", "--"),
        ("CC04", "CC04", "#8da0cb", "--"),
    ]:
        s = master[(master["stationName"] == stn) & (master[WQ_COL].notna())]
        ax_a.plot(s["year"], s[WQ_COL], ls, label=label, color=color, lw=1.0, marker=".", ms=3)

    ax_a2.plot(n_fao.index, n_fao.values, color="grey", alpha=0.4, lw=0.8, ls=":")
    ax_a2.set_ylabel("FAO N (kg ha⁻¹)", fontsize=6, color="grey")
    ax_a2.tick_params(labelsize=5, colors="grey")
    ax_a.set_ylabel(r"NO$_3^-$-N (mg L$^{-1}$)")
    ax_a.set_title("My Thuan (MRC) vs Co Chien (Ben Tre)", fontsize=7, fontweight="bold")
    ax_a.legend(fontsize=5, loc="upper left", frameon=False)
    nw.panel_label(ax_a, "a")

    # Panel (b): My Tho MRC vs My Tho River BT
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b2 = ax_b.twinx()
    for label, stn, color, ls in [
        ("MRC My Tho", "My Tho", nw.PHASE2_CLR, "-"),
        ("MT01", "MT01", "#66c2a5", "--"),
        ("MT02", "MT02", "#fc8d62", "--"),
        ("MT03", "MT03", "#8da0cb", "--"),
    ]:
        s = master[(master["stationName"] == stn) & (master[WQ_COL].notna())]
        ax_b.plot(s["year"], s[WQ_COL], ls, label=label, color=color, lw=1.0, marker=".", ms=3)

    ax_b2.plot(n_fao.index, n_fao.values, color="grey", alpha=0.4, lw=0.8, ls=":")
    ax_b2.set_ylabel("FAO N (kg ha⁻¹)", fontsize=6, color="grey")
    ax_b2.tick_params(labelsize=5, colors="grey")
    ax_b.set_ylabel(r"NO$_3^-$-N (mg L$^{-1}$)")
    ax_b.set_title("My Tho (MRC) vs My Tho River (Ben Tre)", fontsize=7, fontweight="bold")
    ax_b.legend(fontsize=5, loc="upper left", frameon=False)
    nw.panel_label(ax_b, "b")

    # Panel (c): First-diff scatter for MRC My Thuan vs CC04
    ax_c = fig.add_subplot(gs[1, 0])
    for stn, color, label in [
        ("My Thuan", nw.PHASE1_CLR, "MRC My Thuan"),
        ("CC04", "#8da0cb", "BT CC04"),
    ]:
        s = master[master["stationName"] == stn].copy()
        s = s.rename(columns={"year": "Year"}).set_index("Year").sort_index()
        s = s.join(n_fao, how="left")
        dx = s["N_kg_ha"].diff()
        dy = s[WQ_COL].diff()
        mask = dx.notna() & dy.notna()
        ax_c.scatter(dx[mask], dy[mask], s=15, alpha=0.7, color=color, label=label, edgecolors="none")
        if mask.sum() >= 5:
            r, p = stats.pearsonr(dx[mask], dy[mask])
            # regression line
            z = np.polyfit(dx[mask], dy[mask], 1)
            xline = np.linspace(dx[mask].min(), dx[mask].max(), 50)
            ax_c.plot(xline, np.polyval(z, xline), color=color, lw=0.8, ls="--")
            ax_c.text(0.98, 0.95 - (0.12 if stn == "CC04" else 0.0),
                      f"{label}: Δr={r:+.2f} (p={p:.3f})",
                      transform=ax_c.transAxes, fontsize=5.5, ha="right", va="top",
                      color=color, fontweight="bold")

    ax_c.set_xlabel("ΔN (kg ha⁻¹ yr⁻¹)")
    ax_c.set_ylabel(r"ΔNO$_3^-$-N (mg L$^{-1}$ yr$^{-1}$)")
    ax_c.axhline(0, color="grey", lw=0.4, ls=":")
    ax_c.axvline(0, color="grey", lw=0.4, ls=":")
    ax_c.set_title("First-difference scatter: My Thuan co-location", fontsize=7, fontweight="bold")
    nw.panel_label(ax_c, "c")

    # Panel (d): First-diff scatter for MRC My Tho vs MT02
    ax_d = fig.add_subplot(gs[1, 1])
    for stn, color, label in [
        ("My Tho", nw.PHASE2_CLR, "MRC My Tho"),
        ("MT02", "#fc8d62", "BT MT02"),
    ]:
        s = master[master["stationName"] == stn].copy()
        s = s.rename(columns={"year": "Year"}).set_index("Year").sort_index()
        s = s.join(n_fao, how="left")
        dx = s["N_kg_ha"].diff()
        dy = s[WQ_COL].diff()
        mask = dx.notna() & dy.notna()
        ax_d.scatter(dx[mask], dy[mask], s=15, alpha=0.7, color=color, label=label, edgecolors="none")
        if mask.sum() >= 5:
            r, p = stats.pearsonr(dx[mask], dy[mask])
            z = np.polyfit(dx[mask], dy[mask], 1)
            xline = np.linspace(dx[mask].min(), dx[mask].max(), 50)
            ax_d.plot(xline, np.polyval(z, xline), color=color, lw=0.8, ls="--")
            ax_d.text(0.98, 0.95 - (0.12 if stn == "MT02" else 0.0),
                      f"{label}: Δr={r:+.2f} (p={p:.3f})",
                      transform=ax_d.transAxes, fontsize=5.5, ha="right", va="top",
                      color=color, fontweight="bold")

    ax_d.set_xlabel("ΔN (kg ha⁻¹ yr⁻¹)")
    ax_d.set_ylabel(r"ΔNO$_3^-$-N (mg L$^{-1}$ yr$^{-1}$)")
    ax_d.axhline(0, color="grey", lw=0.4, ls=":")
    ax_d.axvline(0, color="grey", lw=0.4, ls=":")
    ax_d.set_title("First-difference scatter: My Tho co-location", fontsize=7, fontweight="bold")
    nw.panel_label(ax_d, "d")

    # Panel (e): Within-river gradient bar chart
    ax_e = fig.add_subplot(gs[2, :])
    river_colors = {"Ba Lai": "#e66101", "Co Chien": "#5e4fa2",
                    "Ham Luong": "#3288bd", "My Tho R.": "#d53e4f"}
    x_pos = 0
    xticks, xlabels = [], []
    for river, stns in rivers.items():
        for stn in stns:
            s = master[master["stationName"] == stn].copy()
            s = s.rename(columns={"year": "Year"}).set_index("Year").sort_index()
            s = s.join(n_fao, how="left")
            if WQ_COL not in s.columns:
                x_pos += 1
                continue
            r, p, n = fd_corr(s["N_kg_ha"], s[WQ_COL])
            if np.isnan(r):
                x_pos += 1
                continue
            color = river_colors.get(river, "grey")
            edgec = "gold" if (not np.isnan(p) and p < 0.05) else "none"
            ax_e.bar(x_pos, r, color=color, edgecolor=edgec, linewidth=1.5)
            xticks.append(x_pos)
            xlabels.append(stn)
            x_pos += 1
        x_pos += 0.5  # gap between rivers

    # Add MRC stations
    x_pos += 0.5
    for stn in ["Tan Chau", "Can Tho", "My Thuan", "My Tho"]:
        s = master[master["stationName"] == stn].copy()
        s = s.rename(columns={"year": "Year"}).set_index("Year").sort_index()
        s = s.join(n_fao, how="left")
        r, p, n = fd_corr(s["N_kg_ha"], s[WQ_COL])
        edgec = "gold" if (not np.isnan(p) and p < 0.05) else "none"
        ax_e.bar(x_pos, r, color="grey", edgecolor=edgec, linewidth=1.5)
        xticks.append(x_pos)
        xlabels.append(stn)
        x_pos += 1

    ax_e.set_xticks(xticks)
    ax_e.set_xticklabels(xlabels, fontsize=5.5, rotation=45, ha="right")
    ax_e.axhline(0, color="black", lw=0.5)
    ax_e.set_ylabel("First-difference Δr\n(N kg/ha vs NO₃⁻-N)")
    ax_e.set_title(
        "Within-network spatial gradient: Δr(N, NO₃⁻) by station\n"
        "(gold border = p < 0.05; coloured = Ben Tre river; grey = MRC mainstream)",
        fontsize=7, fontweight="bold")
    # Custom legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=r) for r, c in river_colors.items()]
    handles.append(mpatches.Patch(color="grey", label="MRC mainstream"))
    ax_e.legend(handles=handles, fontsize=5.5, loc="lower left", frameon=False, ncol=5)
    nw.panel_label(ax_e, "e")

    # Build metrics
    rows = []
    for stn in (nw.MRC_STATIONS + [s for stns in rivers.values() for s in stns]):
        s = master[master["stationName"] == stn].copy()
        s = s.rename(columns={"year": "Year"}).set_index("Year").sort_index()
        s = s.join(n_fao, how="left")
        if WQ_COL not in s.columns:
            continue
        r_full, p_full, n_full = fd_corr(s["N_kg_ha"], s[WQ_COL])
        s2 = s[s.index > BREAK]
        r_post, p_post, n_post = fd_corr(s2["N_kg_ha"], s2[WQ_COL])
        network = "MRC" if stn in nw.MRC_STATIONS else "Ben Tre"
        rows.append({"station": stn, "network": network,
                     "r_fd_full": r_full, "p_fd_full": p_full, "n_full": n_full,
                     "r_fd_post2004": r_post, "p_fd_post2004": p_post, "n_post": n_post})
    metrics = pd.DataFrame(rows)

    nw.save_fig(fig, "Figure_S7b_CoLocated_Diagnostic", metrics=metrics)
    print("\nDone. Key finding: co-located stations show OPPOSITE signals due to")
    print("mainstream (MRC) vs distributary channel (Ben Tre) sampling context.")


if __name__ == "__main__":
    main()
