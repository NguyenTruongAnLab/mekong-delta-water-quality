#!/usr/bin/env python3
"""Discussion conceptual figure: Flushing capacity decline and management implications.

Panel A: Flushing Capacity Index (HFI proxy: TR/Q) trend over 38 years
Panel B: Conceptual policy diagram — retention-aware management
         Shows why "flushing pulses" from upstream dams are needed

Outputs:
- manuscript/figures/nw_composites/Figure_Discussion_Concept.{png,pdf}
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from pathlib import Path
from _nw_common import (
    save_fig, FIG_OUT, ROOT as REPO, STATION_CLR, BREAK_YEAR,
    HFI_CSV, MB_CSV, require_file,
)

plt.rcParams["font.family"] = "Arial"

BLUE = "#2166ac"
RED = "#b2182b"
GREY = "#636363"
GREEN = "#1b7837"


def _panel_flushing_decline(ax):
    """Plot the HFI (TR/Q) trend showing flushing decline — data-driven."""
    # Strict: require HFI input. Prefer any cached per-station file if present,
    # otherwise fall back to the analysis-output HFI_CSV.
    hfi_csv_primary = REPO / "analysis" / "cache" / "hfi_annual_station.csv"
    if not hfi_csv_primary.exists():
        hfi_csv_primary = HFI_CSV
    require_file(hfi_csv_primary, label="HFI time series (TR/Q)")

    if hfi_csv_primary.exists():
        hfi = pd.read_csv(hfi_csv_primary)

        # Handle both known schemas: cache table vs analysis_outputs table.
        if {"Station", "year"}.issubset(hfi.columns):
            hfi_col = "HFI_TR_over_Q" if "HFI_TR_over_Q" in hfi.columns else hfi.columns[-1]
            for stn in hfi["Station"].unique():
                sub = hfi[hfi["Station"] == stn].sort_values("year")
                clr = STATION_CLR.get(stn, "#333333")
                ax.plot(sub["year"], sub[hfi_col] * 1000, "-o",
                        color=clr, lw=1.2, ms=3, alpha=0.85, label=stn, zorder=3)
            ax.set_ylabel(r"HFI: TR/Q ($\times 10^{-3}$ m/($\mathrm{m^3\,s^{-1}}$))",
                          fontsize=8)
        elif {"year", "hfi_tr_over_q"}.issubset({c.lower() for c in hfi.columns}):
            # Normalize common lowercase naming
            cols = {c.lower(): c for c in hfi.columns}
            ycol = cols["year"]
            vcol = cols["hfi_tr_over_q"]
            sub = hfi.sort_values(ycol)
            ax.plot(sub[ycol], sub[vcol] * 1000, "-o", color=RED, ms=3, lw=1.2,
                    alpha=0.85, label="HFI (dry-season)", zorder=3)
            ax.set_ylabel(r"HFI: TR/Q ($\times 10^{-3}$ m/($\mathrm{m^3\,s^{-1}}$))",
                          fontsize=8)
        else:
            raise ValueError(
                f"Unrecognized HFI schema in {hfi_csv_primary}. "
                f"Columns={list(hfi.columns)}"
            )

    ax.axvline(BREAK_YEAR + 0.5, color=GREY, lw=1, ls="--", alpha=0.6)
    ax.fill_betweenx(ax.get_ylim(), 1985, BREAK_YEAR + 0.5, color=BLUE, alpha=0.04)
    ax.fill_betweenx(ax.get_ylim(), BREAK_YEAR + 0.5, 2022, color=RED, alpha=0.04)
    ax.set_xlabel("Year", fontsize=8)
    ax.legend(fontsize=6, frameon=True, framealpha=0.9, edgecolor="none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    # Trend annotation
    ax.annotate(r"$\rightarrow$ increasing tidal dominance",
                xy=(2018, ax.get_ylim()[1] * 0.9),
                fontsize=6, color=RED, fontstyle="italic", ha="right")


def _panel_management(ax):
    """Conceptual diagram: retention-aware management implications."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")

    # Title
    ax.text(5, 8.2, "Retention-Aware Delta Management", fontsize=10,
            ha="center", fontweight="bold", color=GREY)

    # Current approach (top)
    box_curr = FancyBboxPatch((0.5, 5.5), 9, 2.2,
                               boxstyle="round,pad=0.2",
                               facecolor="#fee0d2", edgecolor=RED,
                               lw=1.5, alpha=0.8)
    ax.add_patch(box_curr)
    ax.text(5, 7.3, r"Current approach: Land-use regulation", fontsize=8,
            ha="center", fontweight="bold", color=RED)
    ax.text(5, 6.6, r"$\bullet$ Restrict aquaculture expansion", fontsize=7,
            ha="center", color="#333")
    ax.text(5, 6.1, r"$\bullet$ Assumption: LULC $\rightarrow$ WQ (confounded by time)",
            fontsize=7, ha="center", color="#333")
    ax.text(5, 5.6, r"$\bullet$ $\Delta$R$^2$ = 0.005 $\Rightarrow$ 98% temporal confounding",
            fontsize=7, ha="center", color=RED, fontweight="bold")

    # Arrow
    ax.annotate("", xy=(5, 5.0), xytext=(5, 5.5),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2,
                                mutation_scale=20))
    ax.text(6.5, 5.15, "Evidence-based\nredirection", fontsize=7,
            ha="center", color=GREY, fontstyle="italic")

    # Proposed approach (bottom)
    box_prop = FancyBboxPatch((0.5, 1.0), 9, 3.7,
                               boxstyle="round,pad=0.2",
                               facecolor="#deebf7", edgecolor=BLUE,
                               lw=1.5, alpha=0.8)
    ax.add_patch(box_prop)
    ax.text(5, 4.3, "Proposed: Hydrodynamic flushing management", fontsize=8,
            ha="center", fontweight="bold", color=BLUE)

    proposals = [
        r"1. Upstream reservoir flushing pulses during dry season",
        r"2. Sediment bypass at dams to restore natural transport",
        r"3. Tidal barrier management to optimize residence time ($\tau$)",
        r"4. Real-time Q-normalized WQ monitoring (not just concentration)",
    ]
    for i, p in enumerate(proposals):
        ax.text(1.0, 3.6 - i * 0.6, p, fontsize=7, ha="left", color="#333")

    # Key message
    ax.text(5, 0.5, r"$\Rightarrow$ Target the conveyor belt (hydrology), not just the cargo (land use)",
            fontsize=7.5, ha="center", fontweight="bold", color=BLUE,
            fontstyle="italic")


def _panel_attribution_path(ax):
    """Causal path diagram showing direct vs confounded attribution."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    # Nodes
    nodes = {
        "Time": (5, 4.5),
        "LULC": (2, 2.5),
        "Q": (8, 2.5),
        "WQ": (5, 0.5),
    }

    node_colors = {
        "Time": "#969696",
        "LULC": GREEN,
        "Q": BLUE,
        "WQ": RED,
    }

    for name, (x, y) in nodes.items():
        box = FancyBboxPatch((x - 0.8, y - 0.3), 1.6, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor=node_colors[name], edgecolor="white",
                              alpha=0.85, lw=1.5, zorder=5)
        ax.add_patch(box)
        label = name
        if name == "LULC":
            label = "Land Use"
        elif name == "WQ":
            label = "Water\nQuality"
        elif name == "Q":
            label = "Discharge"
        ax.text(x, y, label, fontsize=7, ha="center", va="center",
                color="white", fontweight="bold", zorder=6)

    # Arrows: Time → LULC
    ax.annotate("", xy=(2.5, 2.9), xytext=(4.5, 4.3),
                arrowprops=dict(arrowstyle="-|>", color="#969696",
                                lw=2, mutation_scale=15))
    # Time → Q
    ax.annotate("", xy=(7.5, 2.9), xytext=(5.5, 4.3),
                arrowprops=dict(arrowstyle="-|>", color="#969696",
                                lw=2, mutation_scale=15))

    # LULC → WQ (thin, weak = confounded)
    ax.annotate("", xy=(4.3, 0.7), xytext=(2.5, 2.2),
                arrowprops=dict(arrowstyle="-|>", color=GREEN,
                                lw=1, mutation_scale=10, alpha=0.5,
                                linestyle="dashed"))
    ax.text(2.5, 1.3, r"$\Delta$R$^2$=0.005", fontsize=6.5,
            color=GREEN, ha="center", fontstyle="italic", rotation=30)

    # Q → WQ (thick, strong = real driver)
    ax.annotate("", xy=(5.7, 0.7), xytext=(7.5, 2.2),
                arrowprops=dict(arrowstyle="-|>", color=BLUE,
                                lw=3, mutation_scale=18))
    ax.text(7.5, 1.3, "Primary\ndriver", fontsize=6.5,
            color=BLUE, ha="center", fontweight="bold", rotation=-30)

    # Caption
    ax.text(5, -0.3, "Causal attribution: temporal confounder (Time)\ndrives both LULC and Q, creating spurious LULC-WQ correlation",
            fontsize=6.5, ha="center", color=GREY, fontstyle="italic")


def main():
    fig = plt.figure(figsize=(180 / 25.4, 220 / 25.4), dpi=300)

    ax1 = fig.add_axes([0.10, 0.70, 0.85, 0.26])
    ax2 = fig.add_axes([0.02, 0.37, 0.50, 0.28])
    ax3 = fig.add_axes([0.50, 0.37, 0.50, 0.28])
    ax4 = fig.add_axes([0.02, 0.02, 0.96, 0.32])

    _panel_flushing_decline(ax1)
    ax1.set_title(r"$\mathbf{a}$  Flushing Capacity Decline (HFI: TR/Q)",
                  fontsize=9, fontweight="bold", loc="left")

    _panel_attribution_path(ax2)
    ax2.set_title(r"$\mathbf{b}$  Causal Pathway",
                  fontsize=9, fontweight="bold", loc="left")

    # Empty panel or additional metric
    ax3.axis("off")
    ax3.set_title(r"$\mathbf{c}$  Self-Purification Decline",
                  fontsize=9, fontweight="bold", loc="left")

    # Self-purification capacity decline (data-driven from mass-balance table)
    require_file(MB_CSV, label="Table_NW_Mass_Balance.csv")
    phases = ["Phase 1\n(1985-2004)", "Phase 2\n(2005-2022)"]
    cod_ncf = [np.nan, np.nan]
    tn_ncf = [np.nan, np.nan]
    if MB_CSV.exists():
        mb = pd.read_csv(MB_CSV)
        if {"parameter", "year", "nonconservative_pct"}.issubset(mb.columns):
            mb = mb.dropna(subset=["nonconservative_pct"])
            cod = mb[mb["parameter"] == "COD"].copy()
            tn = mb[mb["parameter"] == "TN"].copy()
            if not cod.empty:
                cod_ncf = [
                    float(cod[cod["year"] <= BREAK_YEAR]["nonconservative_pct"].mean()),
                    float(cod[cod["year"] > BREAK_YEAR]["nonconservative_pct"].mean()),
                ]
            if not tn.empty:
                tn_ncf = [
                    float(tn[tn["year"] <= BREAK_YEAR]["nonconservative_pct"].mean()),
                    float(tn[tn["year"] > BREAK_YEAR]["nonconservative_pct"].mean()),
                ]
        else:
            raise ValueError(
                f"Unexpected mass-balance schema in {MB_CSV}. Columns={list(mb.columns)}"
            )
    x = np.arange(2)
    w = 0.3
    ax3_inner = fig.add_axes([0.55, 0.42, 0.38, 0.20])
    ax3_inner.bar(x - w / 2, cod_ncf, w, color=RED, alpha=0.8,
                  edgecolor="white", label="COD")
    ax3_inner.bar(x + w / 2, tn_ncf, w, color=BLUE, alpha=0.8,
                  edgecolor="white", label="TN")
    ax3_inner.axhline(0, color="black", lw=0.5)
    ax3_inner.set_xticks(x)
    ax3_inner.set_xticklabels(phases, fontsize=7)
    ax3_inner.set_ylabel("Non-conservative\nfraction (%)", fontsize=7)
    ax3_inner.legend(fontsize=6, frameon=True)
    ax3_inner.spines["top"].set_visible(False)
    ax3_inner.spines["right"].set_visible(False)
    ax3_inner.tick_params(labelsize=6)
    ax3_inner.text(0, -13, r"consumption $\downarrow$", fontsize=5.5,
                   color=BLUE, ha="center")
    ax3_inner.text(1, 19, r"generation $\uparrow$", fontsize=5.5,
                   color=RED, ha="center")

    _panel_management(ax4)
    ax4.set_title(r"$\mathbf{d}$  Policy Implications",
                  fontsize=9, fontweight="bold", loc="left")

    save_fig(fig, "Figure_Discussion_Concept")
    plt.close(fig)


if __name__ == "__main__":
    main()
